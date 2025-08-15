import pandas as pd
import numpy as np
from scipy.stats import beta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import joblib
import warnings
from sklearn.tree import export_text, plot_tree

warnings.filterwarnings('ignore')

class IPLWinPredictor:
    def __init__(self):
        self.le_team = LabelEncoder()
        self.le_toss_decision = LabelEncoder()
        self.le_city = LabelEncoder() 
        self.scaler = StandardScaler()
        self.selector = None
        self.feature_importances_ = None

        # Use only RandomForestClassifier with the best parameters
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.known_teams = None
        self.known_cities = None
        self.known_toss_decisions = None
        self.team_stats = {}
        self.venue_stats = {}
        self.h2h_stats = {}  # Head-to-head stats
        
    def fit_encoders(self, df):
        """Fit the label encoders on the full dataset"""
        self.known_teams = sorted(set(df['team1'].unique()) | set(df['team2'].unique()))
        self.known_cities = sorted(df['city'].unique()) if 'city' in df.columns else []
        if self.known_cities:
            self.le_city.fit(self.known_cities)
        self.known_toss_decisions = sorted(df['toss_decision'].unique())
        self.le_team.fit(self.known_teams)
        self.le_toss_decision.fit(self.known_toss_decisions)
    
    def encode_with_unknown(self, series, encoder, known_values):
        if len(known_values) == 0:
            return np.zeros(len(series))
        series = series.map(lambda x: known_values[0] if x not in known_values else x)
        
        return encoder.transform(series)
        
    def calculate_team_stats(self, df):
        teams = set(df['team1'].unique()) | set(df['team2'].unique())
        
        for team in teams:
            # Team matches and wins
            team1_matches = df[df['team1'] == team]
            team1_wins = team1_matches[team1_matches['winner'] == team]
            team2_matches = df[df['team2'] == team]
            team2_wins = team2_matches[team2_matches['winner'] == team]
            total_matches = len(team1_matches) + len(team2_matches)
            total_wins = len(team1_wins) + len(team2_wins)
            
            # Basic win rate
            win_rate = total_wins / total_matches if total_matches > 0 else 0.5
            
            # Toss advantage
            toss_wins = df[(df['toss_winner'] == team)]
            toss_and_match_wins = toss_wins[toss_wins['winner'] == team]
            toss_win_advantage = len(toss_and_match_wins) / len(toss_wins) if len(toss_wins) > 0 else 0.5
            
            # Batting first performance
            batting_first = df[((df['toss_winner'] == team) & (df['toss_decision'] == 'bat')) | 
                              ((df['toss_winner'] != team) & (df['toss_decision'] == 'field')) & ((df['team1'] == team) |
                              (df['team2'] == team))]
            batting_first_wins = batting_first[batting_first['winner'] == team]
            batting_first_win_rate = len(batting_first_wins) / len(batting_first) if len(batting_first) > 0 else 0.5
            
            # Recent form (last 10 matches)
            recent_matches = df[(df['team1'] == team) | (df['team2'] == team)].tail(10)
            recent_wins = recent_matches[recent_matches['winner'] == team]
            recent_form = len(recent_wins) / len(recent_matches) if len(recent_matches) > 0 else 0.5
            
            self.team_stats[team] = {
                'win_rate': win_rate,
                'toss_win_advantage': toss_win_advantage,
                'batting_first_win_rate': batting_first_win_rate,
                'recent_form': recent_form
            }
        #print("Team statistics calculated successfully. Total teams:", (self.team_stats))

    def calculate_h2h_stats(self, df):
        """Calculate head-to-head win rate for each team pair"""
        teams = set(df['team1'].unique()) | set(df['team2'].unique())
        for team_a in teams:
            for team_b in teams:
                if team_a == team_b:
                    continue
                matches = df[((df['team1'] == team_a) & (df['team2'] == team_b)) | ((df['team1'] == team_b) & (df['team2'] == team_a))]
                if len(matches) == 0:
                    win_rate_a = 0.5  # Neutral if no matches
                else:
                    wins_a = matches[matches['winner'] == team_a]
                    win_rate_a = len(wins_a) / len(matches)
                self.h2h_stats[(team_a, team_b)] = win_rate_a

    def calculate_city_stats(self, df):
        if 'city' not in df.columns:
            return
            
        venues = df['city'].unique()
        
        for venue in venues:
            venue_matches = df[df['city'] == venue]
            
            # Batting first win rate at venue
            batting_first_wins = venue_matches[((venue_matches['toss_decision'] == 'bat') & 
                                            (venue_matches['toss_winner'] == venue_matches['winner'])) | 
                                           ((venue_matches['toss_decision'] == 'field') & 
                                            (venue_matches['toss_winner'] != venue_matches['winner']))]
            
            batting_first_win_rate = len(batting_first_wins) / len(venue_matches) if len(venue_matches) > 0 else 0.5
            
            # Average first innings score
            avg_first_innings_score = venue_matches['target_runs'].mean() if 'target_runs' in venue_matches.columns else 150
            
            self.venue_stats[venue] = {
                'batting_first_win_rate': batting_first_win_rate,
                'avg_first_innings_score': avg_first_innings_score
            }

    def prepare_features(self, df, is_training=True):
        """Prepare enhanced features for the model with improved relative stats and reduced redundancy"""
        if is_training:
            self.fit_encoders(df)
            self.calculate_team_stats(df)
            self.calculate_h2h_stats(df)
            self.calculate_city_stats(df)
        
        df_encoded = df.copy()
        try:
            # Encode categorical features
            df_encoded['team1_encoded'] = self.encode_with_unknown(df['team1'], self.le_team, self.known_teams)
            df_encoded['team2_encoded'] = self.encode_with_unknown(df['team2'], self.le_team, self.known_teams)
            df_encoded['toss_winner_encoded'] = self.encode_with_unknown(df['toss_winner'], self.le_team, self.known_teams)
            df_encoded['toss_decision_encoded'] = self.encode_with_unknown(df['toss_decision'], self.le_toss_decision, self.known_toss_decisions)  
            
            if 'city' in df.columns and self.known_cities:
                df_encoded['venue_encoded'] = self.encode_with_unknown(df['city'], self.le_city, self.known_cities)
            
            # Toss-related features removed
            
            # Team statistics features
            team_stats_columns = [
                'win_rate', 'toss_win_advantage', 
                'batting_first_win_rate', 'recent_form'
            ]
            
            for stat in team_stats_columns:
                # Default value for missing stats
                default_val = 0.5
                
                # Team1 stats
                df_encoded[f'team1_{stat}'] = df['team1'].map(
                    lambda x: self.team_stats.get(x, {}).get(stat, default_val)
                )
                
                # Team2 stats
                df_encoded[f'team2_{stat}'] = df['team2'].map(
                    lambda x: self.team_stats.get(x, {}).get(stat, default_val))
            
            # Head-to-head relative win rate
            def get_h2h(t1, t2):
                return self.h2h_stats.get((t1, t2), 0.5)
            df_encoded['relative_h2h_winrate'] = [get_h2h(row['team1'], row['team2']) for idx, row in df_encoded.iterrows()]
            df_encoded['relative_h2h_winrate_copy'] = df_encoded['relative_h2h_winrate'].copy()

            # Venue statistics features
            if 'city' in df.columns:
                venue_stats_columns = ['batting_first_win_rate', 'avg_first_innings_score']
                
                for stat in venue_stats_columns:
                    default_val = 0.5 if stat == 'batting_first_win_rate' else 150
                    
                    df_encoded[f'venue_{stat}'] = df['city'].map(
                        lambda x: self.venue_stats.get(x, {}).get(stat, default_val))
                
            # Match situation features
            if 'target_runs' in df.columns:
                df_encoded['normalized_target'] = df['target_runs'] / df_encoded.get('venue_avg_first_innings_score', 150)
                df_encoded['run_rate_required'] = df['target_runs'] / df['target_overs']
            
            if is_training:
                df_encoded['target'] = (df['team1'] == df['winner']).astype(int)
            df_encoded.to_csv('team_stats1.csv', index=False)
            # Collect all feature columns
            features = [col for col in df_encoded.columns 
                   if col not in ['match_id', 'city', 'player_of_match', 'venue', 
                                'team1', 'team2', 'toss_winner', 'toss_decision', 
                                'winner', 'result', 'remaining_overs', 'required_runs',
                                'wickets_lost','target','result_margin', 
                                'target_runs', 'target_overs'] and 
                   col in df_encoded.columns]
            
            X = df_encoded[features]
            
            if is_training:
                self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            if is_training:
                return X_scaled, df_encoded['target'], features
            return X_scaled
            
        except Exception as e:
            print(f"Error in prepare_features: {str(e)}")
            print("Input data:")
            print(df.head())
            raise

    def tune_model(self, X, y):
        """Tune RandomForest model using GridSearchCV"""
        param_grid = {
            'n_estimators': [100, 200,300],
            'max_depth': [8, 10, 15],
            'min_samples_split': [2, 3,5],
            'min_samples_leaf': [1, 2, 3]
        }
        print("Starting GridSearchCV for hyperparameter tuning...")
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        print("Best Params from GridSearchCV:", grid_search.best_params_)
        self.model = grid_search.best_estimator_
        return grid_search.best_params_

    def print_sample_tree(self, feature_names):
        """Print a sample decision tree from the RandomForest"""
        if hasattr(self.model, "estimators_") and len(self.model.estimators_) > 0:
            tree = self.model.estimators_[0]
            print("\nSample Decision Tree Structure (first tree):")
            print(export_text(tree, feature_names=feature_names))
            try:
                plt.figure(figsize=(20, 8))
                plot_tree(tree, feature_names=feature_names, filled=True, max_depth=3, fontsize=10)
                plt.title("Sample Decision Tree (first tree, max_depth=3 shown)")
                plt.show()
            except Exception as e:
                print(f"Error plotting tree: {e}")
        else:
            print("No trained trees found in model.")

    def train(self, df):
        """Train the model with enhanced features, tune model and print sample tree"""
        try:
            X, y, features = self.prepare_features(df, is_training=True)

            # Model Tuning
            best_params = self.tune_model(X, y)

            # Feature selection
            selector = SelectFromModel(
                RandomForestClassifier(
                    n_estimators=100, random_state=42, max_depth=10, 
                    min_samples_split=5, min_samples_leaf=2
                ), 
                threshold='median'
            )
            selector.fit(X, y)
            self.selector = selector
            X_selected = selector.transform(X)
            selected_features = [f for i, f in enumerate(features) if selector.get_support()[i]]
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_selected, y, cv=5, scoring='accuracy')
            print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            
            print(f"\nTest accuracy: {accuracy:.4f}")
            print("\nClassification report:")
            print(report)
            print("\nConfusion matrix:")
            print(conf_matrix)

            # Print a sample decision tree
            #self.print_sample_tree(selected_features)
            
            return {
                'accuracy': accuracy,
                'cv_scores': cv_scores,
                'report': report,
                'conf_matrix': conf_matrix
            }
            
        except Exception as e:
            print(f"Error in train method: {str(e)}")
            raise
    
    def plot_feature_importance(self, top_n=10):
        """Plot feature importance"""
        if self.feature_importances_ is None:
            print("No feature importances available")
            return
            
        # Sort features by importance
        sorted_features = sorted(self.feature_importances_.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n]
        features, importances = zip(*sorted_features)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Important Features ({self.model_type})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def _wicket_resources_factor(self, wickets_lost):
        """ Returns the batting resources factor based on number of wickets lost,
        roughly following the DLS method."""
        dls_lookup = {
        0: 1.00,
        1: 0.95,
        2: 0.90,
        3: 0.80,
        4: 0.70,
        5: 0.55,
        6: 0.40,
        7: 0.25,
        8: 0.12,
        9: 0.05,
        10: 0.00
        }
        return dls_lookup.get(wickets_lost, 0.0)
    
    def _calculate_pressure_factor(self, required_runs, remaining_overs, wickets_remaining):
        """Calculate the pressure factor for the chasing team"""
        # Calculate balls remaining (6 balls per over)
        balls_remaining = remaining_overs * 6
        
        # Basic pressure calculation
        if balls_remaining <= 0:
            return 0  # No balls left means no chance
            
        rpb_needed = required_runs / balls_remaining
        pressure = rpb_needed / (0.2 * wickets_remaining + 0.5)
        pressure_factor = np.exp(-pressure)
        
        return np.clip(pressure_factor, 0.1, 1.0)
    
    def predict_win_probability(self, match_info):
        """
        Predict win probability for a match.
        This version fixes logical errors and improves probability blending, 
        head-to-head handling, and batting first/chasing identification.
        """
        try:
            # Prepare features for this match
            match_data = pd.DataFrame([match_info])
            X_full = self.prepare_features(match_data, is_training=False)
            
            if self.selector is None:
                raise ValueError("Model has not been trained yet")
            X = self.selector.transform(X_full)
            
            # Get base probabilities (pre-match, model-based)
            probabilities = self.model.predict_proba(X)[0]
            team1_base_prob = probabilities[1]
            team2_base_prob = probabilities[0]
            print(f"Base Probabilities: {match_info['team1']}={team1_base_prob:.3f}, {match_info['team2']}={team2_base_prob:.3f}")

            # If live-match situation not provided, return base probabilities
            if not all(k in match_info for k in ['required_runs', 'remaining_overs', 'wickets_lost', 'target_runs', 'target_overs']):
                return {
                    'team1_win_probability': team1_base_prob,
                    'team2_win_probability': team2_base_prob,
                    'match_situation': None
                }

            # Determine which team is currently batting/chasing:
            toss_winner = match_info.get('toss_winner')
            toss_decision = match_info.get('toss_decision')
            team1 = match_info['team1']
            team2 = match_info['team2']

            # Batting first team logic (make robust)
            if toss_winner == team1 and toss_decision == 'bat':
                batting_first = team1
                chasing_team = team2
            elif toss_winner == team2 and toss_decision == 'bat':
                batting_first = team2
                chasing_team = team1
            elif toss_winner == team1 and toss_decision == 'field':
                batting_first = team2
                chasing_team = team1
            elif toss_winner == team2 and toss_decision == 'field':
                batting_first = team1
                chasing_team = team2
            else:
                # Fallback (should not occur!)
                batting_first = team1
                chasing_team = team2
            #print(f"batting team {batting_first} chasing team {chasing_team}")

            # Extract match situation variables
            required_runs = match_info['required_runs']
            remaining_overs = max(float(match_info['remaining_overs']), 0.1)  # Avoid division by zero
            wickets_lost = int(match_info['wickets_lost'])
            wickets_remaining = 10 - wickets_lost
            total_overs = float(match_info.get('target_overs', 20.0))  # T20 default
            target_runs = float(match_info.get('target_runs', 0))

            # Calculate required run rate and initial run rate
            required_rr = required_runs / remaining_overs
            initial_rr = target_runs / total_overs if target_runs > 0 else 0

            # Resource calculation (overs and wickets)
            overs_remaining_pct = remaining_overs / total_overs
            wickets_factor = self._wicket_resources_factor(wickets_lost)
            resources_remaining = overs_remaining_pct * wickets_factor

            # Run rate difficulty
            reference_rr = 7.5 if total_overs <= 20 else 5.5
            rr_difficulty = required_rr / (initial_rr if initial_rr > 0 else reference_rr)
            rr_difficulty = np.clip(rr_difficulty, 0.5, 3.0)

            # Chase difficulty metric
            chase_difficulty = rr_difficulty / (resources_remaining if resources_remaining > 0 else 0.1)
            match_progress = 1 - (remaining_overs / total_overs)

            # Beta distribution parameters
            alpha = 1 + (10 * match_progress)
            beta_param = 1 + (5 * chase_difficulty)
            # Use a scaled chase difficulty to get a smoother curve
            chase_win_prob = 1 - beta.cdf(np.clip(chase_difficulty / 5, 0, 1), alpha, beta_param)

            # End-game pressure
            if remaining_overs < 5:
                pressure_factor = self._calculate_pressure_factor(
                    required_runs, 
                    remaining_overs, 
                    wickets_remaining
                )
                chase_win_prob *= pressure_factor

            # Probability smoothing/blending
            pre_match_weight = max(0.2, 1 - match_progress)  # a bit more weight on model
            situation_weight = 1 - pre_match_weight

            # Assign probabilities according to which team is chasing
            if chasing_team == team1:
                team1_win_prob = (pre_match_weight * team1_base_prob) + (situation_weight * chase_win_prob)
                team2_win_prob = 1 - team1_win_prob
            else:
                team2_win_prob = (pre_match_weight * team2_base_prob) + (situation_weight * chase_win_prob)
                team1_win_prob = 1 - team2_win_prob

            # Final clipping to [0, 1]
            team1_win_prob = float(np.clip(team1_win_prob, 0.0, 1.0))
            team2_win_prob = float(np.clip(team2_win_prob, 0.0, 1.0))

            return {
                'team1_win_probability': team1_win_prob,
                'team2_win_probability': team2_win_prob,
                'match_situation': {
                    'required_rr': required_rr,
                    'resources_remaining': resources_remaining,
                    'chase_difficulty': chase_difficulty,
                    'pre_match_weight': pre_match_weight,
                    'chasing_team': chasing_team,
                    'batting_first': batting_first
                }
            }

        except Exception as e:
            print(f"Error in predict_win_probability: {str(e)}")
            print("Input match_info:")
            print(match_info)
            # If error, return base or 50-50
            try:
                return {
                    'team1_win_probability': probabilities[1],
                    'team2_win_probability': probabilities[0],
                    'match_situation': None
                }
            except Exception:
                return {
                    'team1_win_probability': 0.5,
                    'team2_win_probability': 0.5,
                    'match_situation': None
                }
            
        except Exception as e:
            print(f"Error in predict_win_probability: {str(e)}")
            print("Input match_info:")
            print(match_info)
            
            # Return base probabilities in case of error
            try:
                return {
                    'team1_win_probability': probabilities[1],
                    'team2_win_probability': probabilities[0],
                    'match_situation': None
                }
            except:
                # If everything fails, return 50-50
                return {
                    'team1_win_probability': 0.5,
                    'team2_win_probability': 0.5,
                    'match_situation': None
                }
    
    def save_model(self, filepath):
        """Save the trained model to a file"""
        try:
            joblib.dump(self, filepath)
            print(f"Model saved successfully to {filepath}")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
    
    @staticmethod
    def load_model(filepath):
        """Load a trained model from file"""
        try:
            model = joblib.load(filepath)
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

def main():
    # Load your cleaned IPL dataset
    try:
        df = pd.read_csv('output2.csv', skiprows=range(1, 858))  # Adjust as needed
        print("Data loaded successfully. Shape:", df.shape)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Directly use RandomForestClassifier
    predictor = IPLWinPredictor()
    results = predictor.train(df)
    
    # Print results
    print(f"\nFinal Model Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['report'])

    print("\nConfusion Matrix:")
    print(results['conf_matrix'])

    print("\nCross-Validation Scores:")
    print(results['cv_scores'])
    print(f"CV Mean Accuracy: {results['cv_scores'].mean():.4f}")
    
    # Example prediction
    match_info = {
        'team1': 'Chennai Super Kings',
        'team2': 'Royal Challengers Bengaluru',
        'city': 'Bengaluru',
        'target_runs': 214,
        'target_overs': 20,
        'required_runs': 40,
        'remaining_overs': 8,
        'wickets_lost': 2,
        'toss_winner': 'Chennai Super Kings',
        'toss_decision': 'field'
    }
    
    try:
        probabilities = predictor.predict_win_probability(match_info)
        print("\nWin Probabilities:")
        print(f"{match_info['team1']}: {probabilities['team1_win_probability']:.2%}")
        print(f"{match_info['team2']}: {probabilities['team2_win_probability']:.2%}")
        
        if probabilities['match_situation']:
            print("\nMatch Situation Analysis:")
            sit = probabilities['match_situation']
            print(f"Chasing Team: {sit['chasing_team']}")
            print(f"Required Run Rate: {sit['required_rr']:.2f} runs/over")
            print(f"Resources Remaining: {sit['resources_remaining']:.2%}")
            print(f"Chase Difficulty: {sit['chase_difficulty']:.2f}")
            print(f"Pre-match Weight: {sit['pre_match_weight']:.2f}")
    except Exception as e:
        print(f"Error in prediction: {str(e)}")

    predictor.save_model('ipl_win_predictor.pkl')

if __name__ == "__main__":
    main()