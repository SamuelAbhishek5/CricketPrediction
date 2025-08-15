import pandas as pd
import numpy as np
from scipy.stats import beta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from scipy.special import expit 
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
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=1,
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
            
            # Toss-related features
            # Removed: is_toss_winner_team1, is_batting_first, team1_batting_first, team2_batting_first
            
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
            df_encoded['relative_h2h_winrate_copy'] = df_encoded['relative_h2h_winrate'].copy()  # or 3, depending on how much you want to boost it

            # Venue statistics features
            if 'city' in df.columns:
                venue_stats_columns = ['batting_first_win_rate', 'avg_first_innings_score']
                
                for stat in venue_stats_columns:
                    default_val = 0.5 if stat == 'batting_first_win_rate' else 150
                    
                    df_encoded[f'venue_{stat}'] = df['city'].map(
                        lambda x: self.venue_stats.get(x, {}).get(stat, default_val))
                
                # Example: team1_venue_advantage (now computes only win rate difference between the two teams at venue if enough data, else 0)
                # You may customize further if you want to bring in venue+team stats
                # (Removed team1_batting_first etc)
            
            # Match situation features
            if 'target_runs' in df.columns:
                df_encoded['normalized_target'] = df['target_runs'] / df_encoded.get('venue_avg_first_innings_score', 150)
                df_encoded['run_rate_required'] = df['target_runs'] / df['target_overs']
            
            if is_training:
                df_encoded['target'] = (df['team1'] == df['winner']).astype(int)

            df_encoded.to_csv('team_stats.csv', index=False)

            # Collect all feature columns
            features = [col for col in df_encoded.columns 
                   if col not in ['match_id', 'city', 'player_of_match', 'venue', 
                                'team1', 'team2', 'toss_winner', 'toss_decision', 
                                'winner', 'result', 'remaining_overs', 'required_runs',
                                'wickets_lost','target','result_margin', 
                                'target_runs', 'target_overs'] and 
                   col in df_encoded.columns]
            
            X = df_encoded[features]# Filter out low importance features

            if is_training:
                self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            if is_training:
                return X_scaled, df_encoded['target']
            return X_scaled
            
        except Exception as e:
            print(f"Error in prepare_features: {str(e)}")
            print("Input data:")
            print(df.head())
            raise
    
    def train(self, df):
        """Train the model with enhanced features"""
        try:
            X, y = self.prepare_features(df, is_training=True)
            
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
    
    def get_dls_resource(self, overs_left, wickets_lost):
        """Estimate batting resources left using simplified DLS table, capped for valid range."""
        overs = int(np.clip(round(overs_left), 0, 20))
        wickets = int(np.clip(wickets_lost, 0, 9))
        DLS_RESOURCE_TABLE = {
        20: [100.0, 92.0, 83.8, 74.9, 65.0, 54.0, 41.7, 28.6, 15.0, 6.0],
        19: [96.1, 88.5, 80.3, 71.4, 61.5, 50.6, 38.3, 25.4, 12.8, 5.2],
        18: [92.3, 84.9, 76.8, 68.0, 58.0, 47.3, 35.0, 22.5, 11.0, 4.5],
        17: [88.5, 81.3, 73.3, 64.5, 54.5, 44.1, 31.9, 19.9, 9.5, 3.8],
        16: [84.7, 77.7, 69.8, 61.0, 51.0, 41.0, 29.0, 17.0, 8.0, 3.2],
        15: [81.0, 74.1, 66.3, 57.5, 47.5, 37.8, 26.3, 15.0, 7.0, 2.7],
        14: [77.2, 70.5, 62.8, 54.0, 44.0, 34.5, 23.7, 13.0, 6.0, 2.3],
        13: [73.4, 66.9, 59.3, 50.5, 40.5, 31.2, 21.2, 11.0, 5.0, 1.9],
        12: [69.6, 63.3, 55.8, 47.0, 37.0, 28.0, 18.7, 9.5, 4.2, 1.5],
        11: [65.8, 59.7, 52.3, 43.5, 33.5, 24.8, 16.2, 8.0, 3.5, 1.2],
        10: [62.0, 56.1, 48.8, 40.0, 30.0, 21.6, 13.7, 6.5, 2.8, 0.9],
        9:  [58.2, 52.5, 45.3, 36.5, 26.5, 18.4, 11.2, 5.0, 2.2, 0.7],
        8:  [54.4, 48.9, 41.8, 33.0, 23.0, 15.2, 8.7, 4.0, 1.7, 0.5],
        7:  [50.6, 45.3, 38.3, 29.5, 19.5, 12.0, 6.2, 3.0, 1.2, 0.3],
        6:  [46.8, 41.7, 34.8, 26.0, 16.0, 9.0, 4.2, 2.0, 0.9, 0.2],
        5:  [43.0, 38.1, 31.3, 22.5, 12.5, 6.5, 2.8, 1.5, 0.6, 0.1],
        4:  [39.2, 34.5, 27.8, 19.0, 9.5, 4.5, 1.7, 1.0, 0.4, 0.1],
        3:  [35.4, 30.9, 24.3, 15.5, 7.0, 3.0, 1.0, 0.6, 0.3, 0.1],
        2:  [31.6, 27.3, 20.8, 12.0, 5.0, 1.8, 0.7, 0.4, 0.2, 0.0],
        1:  [27.8, 23.7, 17.3, 8.5, 3.0, 1.0, 0.4, 0.2, 0.1, 0.0],
        0:  [0.0]*10
        }
        resource_percent = DLS_RESOURCE_TABLE.get(overs, [0.0]*10)[wickets]
        return resource_percent / 100.0

    def _calculate_pressure_factor(self, required_runs, balls_left, wickets_in_hand):
        """Simple model: returns a pressure penalty for tight chases. Returns 1.0 for comfortable, <1.0 for pressure."""
        # If required run rate < 8, wickets > 5, no pressure
        if balls_left <= 0 or wickets_in_hand <= 0:
            return 0  # match lost
        req_rr = required_runs / (balls_left / 6)
        if req_rr < 7.5 and wickets_in_hand > 5:
            return 1.0
        elif req_rr < 10 and wickets_in_hand > 3:
            return 0.9
        else:
            return 0.5  # high pressure

    def predict_win_probability(self, match_info):
        """Predict win probability for a match: improved chase calculation based on live match situation."""
        try:
            match_data = pd.DataFrame([match_info])
            X_full = self.prepare_features(match_data, is_training=False)
            if self.selector is None:
                raise ValueError("Model has not been trained yet")
            X = self.selector.transform(X_full)
            probabilities = self.model.predict_proba(X)[0]
            team1_base_prob, team2_base_prob = probabilities[1], probabilities[0]
            print(f"Base Probabilities: {match_info['team1']}={team1_base_prob:.3f}, {match_info['team2']}={team2_base_prob:.3f}")
            # If no match situation, return base
            if not all(k in match_info for k in ['required_runs', 'remaining_overs', 'wickets_lost', 'target_runs', 'target_overs']):
                return {
                'team1_win_probability': team1_base_prob,
                'team2_win_probability': team2_base_prob,
                'match_situation': None
                }
        # Determine chasing team
            batting_first = (((match_info.get('toss_winner') == match_info.get('team1')) & (match_info.get('toss_decision') == 'bat')) |
                         ((match_info.get('toss_winner') != match_info.get('team1')) & (match_info.get('toss_decision') == 'field')))
            chasing_team = match_info['team1'] if not batting_first else match_info['team2']
        # Inputs
            required_runs = match_info['required_runs']
            remaining_overs = max(match_info['remaining_overs'], 0.05)
            balls_left = int(remaining_overs * 6)
            wickets_lost = match_info['wickets_lost']
            wickets_in_hand = 10 - wickets_lost
            target_runs = match_info['target_runs']
            total_overs = match_info['target_overs']
            resources_remaining = self.get_dls_resource(remaining_overs, wickets_lost)
        # --- Improved chase win probability ---
        # If runs to get < balls left and wickets > 5, win prob is very high
            if (required_runs <= balls_left) and (wickets_in_hand > 5) and (required_runs / balls_left < 1.2):
                chase_win_prob = 0.99
        # If wickets < 3 and RR > 10, win prob low
            elif wickets_in_hand < 3 and required_runs / balls_left > 1.5:
                chase_win_prob = 0.10
            else:
            # Logistic model: base on runs/ball and wickets
                rr_ratio = (required_runs / balls_left) / (target_runs / (total_overs*6))
            # More wickets, more chance; more rr_ratio, less chance
                score = 3.0 - 8.0 * rr_ratio + 0.25 * wickets_in_hand + 7 * resources_remaining
                chase_win_prob = expit(score)
            # pressure penalty
                chase_win_prob *= self._calculate_pressure_factor(required_runs, balls_left, wickets_in_hand)
        # Blend pre-match and live
            match_progress = 1 - (remaining_overs / total_overs)
            pre_match_weight = max(0.05, 1 - match_progress)
            live_weight = 1 - pre_match_weight
            if chasing_team == match_info['team1']:
                team1_win_prob = pre_match_weight * team1_base_prob + live_weight * chase_win_prob
                team2_win_prob = 1 - team1_win_prob
            else:
                team2_win_prob = pre_match_weight * team2_base_prob + live_weight * chase_win_prob
                team1_win_prob = 1 - team2_win_prob
            return {
            'team1_win_probability': team1_win_prob,
            'team2_win_probability': team2_win_prob,
            'match_situation': {
                'required_rr': required_runs / (remaining_overs if remaining_overs else 1),
                'resources_remaining': resources_remaining,
                'chase_win_prob': chase_win_prob,
                'pre_match_weight': pre_match_weight,
                'chasing_team': chasing_team
                }
            }
        except Exception as e:
            print(f"Error in predict_win_probability: {str(e)}")
            print("Input match_info:")
            print(match_info)
            try:
                return {
                'team1_win_probability': probabilities[1],
                'team2_win_probability': probabilities[0],
                'match_situation': None
                }
            except:
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
        df = pd.read_csv('FEHomePage/output2.csv',skiprows=range(1,858))  # Adjust as needed
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
        'team1': 'Royal Challengers Bengaluru',
        'team2': 'Chennai Super Kings',
        'city': 'Bengaluru',
        'target_runs': 200,
        'target_overs': 20,
        'required_runs': 200,
        'remaining_overs': 20,
        'wickets_lost': 0,
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
            #print(f"Chase Difficulty: {sit['chase_difficulty']:.2f}")
            print(f"Chase Win Probability: {sit['chase_win_prob']:.2f}")
            print(f"Pre-match Weight: {sit['pre_match_weight']:.2f}")
    except Exception as e:
        print(f"Error in prediction: {str(e)}")

    predictor.save_model('ipl_win_predictor.pkl')

if __name__ == "__main__":
    main()