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
import warnings
warnings.filterwarnings('ignore')

class IPLWinPredictor:
    def __init__(self, model_type='xgboost'):
        self.le_team = LabelEncoder()
        #self.le_venue = LabelEncoder()
        self.le_toss_decision = LabelEncoder()
        self.le_city = LabelEncoder() 
        self.scaler = StandardScaler()
        self.selector = None
        
        # Model selection
        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                random_state=42
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=200,
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
        
    def fit_encoders(self, df):
        """Fit the label encoders on the full dataset"""
        self.known_teams = sorted(set(df['team1'].unique()) | set(df['team2'].unique()))
        #self.known_venues = sorted(df['venue'].unique()) if 'venue' in df.columns else []
        self.known_cities = sorted(df['city'].unique()) if 'city' in df.columns else []
        if self.known_cities:
            self.le_city.fit(self.known_cities)
        self.known_toss_decisions = sorted(df['toss_decision'].unique())
        self.le_team.fit(self.known_teams)
        #if self.known_venues:
        #    self.le_venue.fit(self.known_venues)
        self.le_toss_decision.fit(self.known_toss_decisions)
    
    def encode_with_unknown(self, series, encoder, known_values):
        if len(known_values) == 0:
            return np.zeros(len(series))
        series = series.map(lambda x: known_values[0] if x not in known_values else x)
        return encoder.transform(series)
    def calculate_team_stats(self, df):
        teams = set(df['team1'].unique()) | set(df['team2'].unique())
        
        for team in teams:
            # Matches as team1
            team1_matches = df[df['team1'] == team]
            team1_wins = team1_matches[team1_matches['winner'] == team]
            
            # Matches as team2
            team2_matches = df[df['team2'] == team]
            team2_wins = team2_matches[team2_matches['winner'] == team]
            
            # Calculate win rates
            total_matches = len(team1_matches) + len(team2_matches)
            total_wins = len(team1_wins) + len(team2_wins)
            
            win_rate = total_wins / total_matches if total_matches > 0 else 0.5
            
            # Toss win and match win correlation
            toss_wins = df[(df['toss_winner'] == team)]
            toss_and_match_wins = toss_wins[toss_wins['winner'] == team]
            toss_win_advantage = len(toss_and_match_wins) / len(toss_wins) if len(toss_wins) > 0 else 0.5
            
            # Batting first win rate
            batting_first = df[((df['toss_winner'] == team) & (df['toss_decision'] == 'bat')) | 
                              ((df['toss_winner'] != team) & (df['toss_decision'] == 'field'))]
            batting_first_wins = batting_first[batting_first['winner'] == team]
            batting_first_win_rate = len(batting_first_wins) / len(batting_first) if len(batting_first) > 0 else 0.5
            
            # Recent form (last 5 matches)
            recent_matches = df[(df['team1'] == team) | (df['team2'] == team)].tail(5)
            recent_wins = recent_matches[recent_matches['winner'] == team]
            recent_form = len(recent_wins) / len(recent_matches) if len(recent_matches) > 0 else 0.5
            
            self.team_stats[team] = {
                'win_rate': win_rate,
                'toss_win_advantage': toss_win_advantage,
                'batting_first_win_rate': batting_first_win_rate,
                'recent_form': recent_form
            }
    def calculate_city_stats(self, df):
        """Calculate venue-specific statistics"""
        if 'city' not in df.columns:
            return
            
        venues = df['city'].unique()
        
        for venue in venues:
            venue_matches = df[df['city'] == venue]
            
            # Batting first win rate at this venue
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
        """Prepare enhanced features for the model"""
        if is_training:
            self.fit_encoders(df)
            self.calculate_team_stats(df)
            self.calculate_city_stats(df)
        
        df_encoded = df.copy()
        try:
            # Basic encoding
            df_encoded['team1_encoded'] = self.encode_with_unknown(df['team1'], self.le_team, self.known_teams)
            df_encoded['team2_encoded'] = self.encode_with_unknown(df['team2'], self.le_team, self.known_teams)
            df_encoded['toss_winner_encoded'] = self.encode_with_unknown(df['toss_winner'], self.le_team, self.known_teams)
            df_encoded['toss_decision_encoded'] = self.encode_with_unknown(df['toss_decision'], self.le_toss_decision, self.known_toss_decisions)
            
            if 'city' in df.columns and self.known_cities:
                df_encoded['venue_encoded'] = self.encode_with_unknown(df['city'], self.le_city, self.known_cities)
            
            # Basic feature engineering
            df_encoded['is_toss_winner_team1'] = (df['toss_winner'] == df['team1']).astype(int)
            df_encoded['is_batting_first'] = (df['toss_decision'] == 'bat').astype(int)
            
            # Create batting/bowling order features
            df_encoded['team1_batting_first'] = (
                ((df['toss_winner'] == df['team1']) & (df['toss_decision'] == 'bat')) |
                ((df['toss_winner'] == df['team2']) & (df['toss_decision'] == 'field'))
            ).astype(int)
            df_encoded['team2_batting_first'] = 1 - df_encoded['team1_batting_first']
            
            # Add team statistics
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
                    lambda x: self.team_stats.get(x, {}).get(stat, default_val)
                )
                
                # Relative advantage
                df_encoded[f'relative_{stat}'] = df_encoded[f'team1_{stat}'] - df_encoded[f'team2_{stat}']
            
            # Add venue statistics if available
            if 'city' in df.columns:
                venue_stats_columns = ['batting_first_win_rate', 'avg_first_innings_score']
                
                for stat in venue_stats_columns:
                    default_val = 0.5 if stat == 'batting_first_win_rate' else 150
                    
                    df_encoded[f'venue_{stat}'] = df['city'].map(
                        lambda x: self.venue_stats.get(x, {}).get(stat, default_val)
                    )
                
                # Interaction between venue and team stats
                df_encoded['team1_venue_advantage'] = df_encoded['team1_batting_first'] * df_encoded['venue_batting_first_win_rate'] + \
                                                    (1 - df_encoded['team1_batting_first']) * (1 - df_encoded['venue_batting_first_win_rate'])
                df_encoded['team2_venue_advantage'] = df_encoded['team2_batting_first'] * df_encoded['venue_batting_first_win_rate'] + \
                                                    (1 - df_encoded['team2_batting_first']) * (1 - df_encoded['venue_batting_first_win_rate'])
            
            # Add match-specific features if available
            if 'target_runs' in df.columns:
                df_encoded['normalized_target'] = df['target_runs'] / df_encoded.get('venue_avg_first_innings_score', 150)
                df_encoded['run_rate_required'] = df['target_runs'] / df['target_overs']
            
            if is_training:
                df_encoded['target'] = (df['team1'] == df['winner']).astype(int)
            
            # Collect all feature columns
            features = [col for col in df_encoded.columns if col not in ['match_id', 'city', 'player_of_match', 'venue', 
                                        'team1', 'team2', 'toss_winner', 'toss_decision', 'winner', 'result', 'remaining_overs',
                                        'required_runs','wickets_lost','target',
                                        'result_margin', 'target_runs', 'target_overs']]
            
            X = df_encoded[features]
            
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
    def tune_model(self, X_train, y_train):
        """Tune hyperparameters using GridSearchCV"""
        if isinstance(self.model, xgb.XGBClassifier):
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        else:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        self.model = grid_search.best_estimator_
    
    def train(self, df, tune_hyperparams=False):
        """Train the model with enhanced features"""
        try:
            X, y = self.prepare_features(df, is_training=True)
            
            # Feature selection
            selector = SelectFromModel(
                xgb.XGBClassifier(n_estimators=100, random_state=42), 
                threshold='median'
            )
            selector.fit(X, y)
            self.selector = selector
            X_selected = selector.transform(X)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Tune hyperparameters if requested
            if tune_hyperparams:
                self.tune_model(X_train, y_train)
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_selected, y, cv=5, scoring='accuracy')
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Evaluate on test set
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            print(f"Test accuracy: {accuracy:.4f}")
            print("Classification report:")
            print(report)
            print("Confusion matrix:")
            print(conf_matrix)
            
            # Get feature importance
            if hasattr(self.model, 'feature_importances_'):
                feature_names = [f for f in df.columns if f not in ['match_id', 'city', 'player_of_match', 'venue', 'team1', 'team2', 'toss_winner', 'toss_decision', 'winner', 'result', 'result_margin', 'target_runs', 'target_overs']]
                feature_importance = dict(sorted(zip(
                    feature_names,
                    self.model.feature_importances_
                ), key=lambda x: x[1], reverse=True))
                
                print("Top 10 feature importances:")
                for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
                    print(f"{i+1}. {feature}: {importance:.4f}")
            
            return {
                'accuracy': accuracy,
                'cv_scores': cv_scores,
                'report': report,
                'conf_matrix': conf_matrix
            }
            
        except Exception as e:
            print(f"Error in train method: {str(e)}")
            raise
    def _wicket_resources_factor(self, wickets_lost):
        if wickets_lost >= 10:
            return 0.01  # Almost no resources left with all wickets gone
        
        return np.exp(-0.4 * wickets_lost)
    
    def _calculate_pressure_factor(self, required_runs, remaining_overs, wickets_remaining):
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
        """Predict win probability for a match"""
        try:
            match_data = pd.DataFrame([match_info])
            X_full = self.prepare_features(match_data, is_training=False)
            X = self.selector.transform(X_full)
            probabilities = self.model.predict_proba(X)[0]
            team1_base_prob, team2_base_prob = probabilities[1], probabilities[0]
            
            # Adjust probabilities based on match situation
            if not all(k in match_info for k in ['required_runs', 'remaining_overs', 'wickets_lost']):
                return {
                    'team1_win_probability': team1_base_prob,
                    'team2_win_probability': team2_base_prob
                }
            batting_first = (((match_info.get('toss_winner') == match_info.get('team1')) & (match_info.get('toss_decision') == 'bat')) | 
                              ((match_info.get('toss_winner') != match_info.get('team1')) & (match_info.get('toss_decision') == 'field')))
            chasing_team_index = 0 if batting_first else 1
            
            # Extract match situation variables
            required_runs = match_info['required_runs']
            remaining_overs = max(match_info['remaining_overs'], 0.1)  # Avoid division by zero
            wickets_lost = match_info['wickets_lost']
            wickets_remaining = 10 - wickets_lost
            total_overs = match_info.get('target_overs', 20.0)  # Default to T20 if not specified
            target_runs = match_info.get('target_runs', 0)
            
            # Step 3: Calculate required run rate and compare with initial/target run rate
            required_rr = required_runs / remaining_overs
            initial_rr = target_runs / total_overs if target_runs > 0 else 0
            overs_remaining_pct = remaining_overs / total_overs
            wickets_factor = self._wicket_resources_factor(wickets_lost)
            resources_remaining = overs_remaining_pct * wickets_factor
            
            # 4.2 Run rate difficulty factor
            # How difficult is the required run rate compared to what's been achieved or par?
            if initial_rr > 0:
                rr_difficulty = required_rr / initial_rr
            else:
                # If we don't have initial run rate, use a reference value
                # based on format (e.g., ~7.5 for T20, ~5.5 for ODI)
                reference_rr = 7.5 if total_overs <= 20 else 5.5
                rr_difficulty = required_rr / reference_rr
                
            # Cap the difficulty factor
            rr_difficulty = np.clip(rr_difficulty, 0.5, 3.0)
            chase_difficulty = rr_difficulty / resources_remaining
            
            # 4.4 Win probability adjustment using Beta distribution
            # Parameters shape the distribution based on match stage
            match_progress = 1 - (remaining_overs / total_overs)
            
            # At start of innings, rely more on pre-match model
            # As match progresses, increase weight of in-game factors
            alpha = 1 + (10 * match_progress)
            beta_param = 1 + (5 * chase_difficulty)
            
            # Generate win probability from Beta distribution
            chase_win_prob = 1 - beta.cdf(chase_difficulty / 5, alpha, beta_param)
            
            # 4.5 Apply pressure factors for end-game scenarios
            if remaining_overs < 5:
                # End-game pressure increases with fewer wickets and tighter situations
                pressure_factor = self._calculate_pressure_factor(
                    required_runs, 
                    remaining_overs, 
                    wickets_remaining
                )
                chase_win_prob *= pressure_factor
            
            # 4.6 Blend with pre-match model probabilities
            # Weight of pre-match model decreases as match progresses
            pre_match_weight = max(0.1, 1 - match_progress)
            situation_weight = 1 - pre_match_weight
            
            # Blend probabilities
            if chasing_team_index == 1:
                # Team 1 is chasing
                team1_win_prob = (pre_match_weight * team1_base_prob) + (situation_weight * chase_win_prob)
                team2_win_prob = 1 - team1_win_prob
            else:
                # Team 2 is chasing
                team2_win_prob = (pre_match_weight * team2_base_prob) + (situation_weight * chase_win_prob)
                team1_win_prob = 1 - team2_win_prob
            
            return {
                'team1_win_probability': team1_win_prob,
                'team2_win_probability': team2_win_prob,
                'match_situation': {
                    'required_rr': required_rr,
                    'resources_remaining': resources_remaining,
                    'chase_difficulty': chase_difficulty,
                    'pre_match_weight': pre_match_weight
                }
            }
            
        except Exception as e:
            print(f"Error in predict_win_probability: {str(e)}")
            print("Input match_info:")
            print(match_info)
            # Return base probabilities in case of error
            try:
                return {
                    'team1_win_probability': probabilities[1],
                    'team2_win_probability': probabilities[0]
                }
            except:
                # If everything fails, return 50-50
                return {
                    'team1_win_probability': 0.5,
                    'team2_win_probability': 0.5
                }

def main():
    # Load your cleaned IPL dataset
    df = pd.read_csv('output2.csv')
    
    # Initialize and train the predictor
    predictor = IPLWinPredictor(model_type='xgboost')
    results = predictor.train(df, tune_hyperparams=True)
    
    print(f"Model Accuracy: {results['accuracy']:.2f}")
    print("\nClassification Report:")
    print(results['report'])

    print("\nConfusion Matrix:")
    print(results['conf_matrix'])

    print("\nCross-Validation Scores:")
    print(results['cv_scores'])
    print(f"CV Mean Accuracy: {results['cv_scores'].mean():.4f}")
    
    # Example prediction
    """print("\n--- Over-by-Over Simulation ---")
    current_score = 0
    target = 180
    wickets_fallen = [0, 0, 1, 1, 1, 1, 1, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7]
    total_overs = 20
    runs_per_over = [8, 6, 10, 7, 12, 5, 9, 10, 6, 11, 4, 8, 9, 7, 5, 10, 3, 4, 6, 7]  # example match

    for over in range(1, total_overs + 1):
        current_score += runs_per_over[over - 1]
        remaining_overs = total_overs - over
        required_runs = max(0, target - current_score)
        wickets_lost=wickets_fallen[over-1]

        match_info = {
            'team1': 'Mumbai Indians',
            'team2': 'Chennai Super Kings',
            'target_runs': target,
            'target_overs': total_overs,
            'required_runs': required_runs,
            'remaining_overs': remaining_overs,
            'wickets_lost': wickets_lost,
            'toss_winner': 'Mumbai Indians',
            'toss_decision': 'bat',
            'city':'Mumbai',
            'venue':'Wankhede Stadium'
        }   

        try:
            probabilities = predictor.predict_win_probability(match_info)
            print(f"Over {over} - Score: {current_score}/{wickets_fallen[over-1]}")
            print(f"Required: {required_runs} off {remaining_overs} overs")
            print(f"Win % -> {match_info['team2']}: {probabilities['team2_win_probability']:.1%} | {match_info['team1']}: {probabilities['team1_win_probability']:.1%}\n")
        except Exception as e:
            print(f"Error in simulation at over {over}: {str(e)}")
            break"""
    match_info = {
        'team1': 'Mumbai Indians',
        'team2': 'Chennai Super Kings',
        'city': 'Mumbai',
        'venue': 'Wankhede Stadium',
        'target_runs': 180,
        'target_overs': 20.0,
        'required_runs': 60,
        'wickets_lost': 3,
        'remaining_overs': 5.0,
        'toss_winner': 'Mumbai Indians',
        'toss_decision': 'bat'  # Added toss decision
    }
    
    try:
        probabilities = predictor.predict_win_probability(match_info)
        print("\nWin Probabilities:")
        print(f"{match_info['team1']}: {probabilities['team1_win_probability']:.2%}")
        print(f"{match_info['team2']}: {probabilities['team2_win_probability']:.2%}")
    except Exception as e:
        print(f"Error in prediction: {str(e)}")

if __name__ == "__main__":
    main()