import pandas as pd
import numpy as np
from scipy.stats import beta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class IPLWinPredictor:
    def __init__(self):
        self.le_team = LabelEncoder()
        self.le_toss_decision = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.known_teams = None
        self.known_toss_decisions = None
        
    def fit_encoders(self, df):
        """Fit the label encoders on the full dataset"""
        self.known_teams = sorted(set(df['team1'].unique()) | set(df['team2'].unique()))
        self.known_toss_decisions = sorted(df['toss_decision'].unique())
        
        self.le_team.fit(self.known_teams)
        self.le_toss_decision.fit(self.known_toss_decisions)
    
    def encode_with_unknown(self, series, encoder, known_values):
        """Safely encode values, handling unknown categories"""
        series = series.map(lambda x: known_values[0] if x not in known_values else x)
        return encoder.transform(series)
    
    def prepare_features(self, df, is_training=True):
        """Prepare features for the model without venue information"""
        if is_training:
            self.fit_encoders(df)
        
        df_encoded = df.copy()
        try:
            # Basic encoding
            df_encoded['team1_encoded'] = self.encode_with_unknown(df['team1'], self.le_team, self.known_teams)
            df_encoded['team2_encoded'] = self.encode_with_unknown(df['team2'], self.le_team, self.known_teams)
            df_encoded['toss_winner_encoded'] = self.encode_with_unknown(df['toss_winner'], self.le_team, self.known_teams)
            df_encoded['toss_decision_encoded'] = self.encode_with_unknown(df['toss_decision'], self.le_toss_decision, self.known_toss_decisions)
            
            # Feature engineering
            df_encoded['is_toss_winner_team1'] = (df['toss_winner'] == df['team1']).astype(int)
            df_encoded['is_batting_first'] = (df['toss_decision'] == 'bat').astype(int)
            
            # Create batting/bowling order features with corrected logic
            df_encoded['team1_batting_first'] = (
                ((df['toss_winner'] == df['team1']) & (df['toss_decision'] == 'bat')) |
                ((df['toss_winner'] == df['team2']) & (df['toss_decision'] == 'field'))
            ).astype(int)
            df_encoded['team2_batting_first'] = 1 - df_encoded['team1_batting_first']
            
            if is_training:
                df_encoded['target'] = (df['team1'] == df['winner']).astype(int)
            
            features = [
                'team1_encoded', 'team2_encoded',
                'toss_winner_encoded', 'toss_decision_encoded',
                'is_toss_winner_team1', 'is_batting_first',
                'team1_batting_first', 'team2_batting_first'
            ]
            
            if 'target_runs' in df.columns:
                features.extend(['target_runs', 'target_overs'])
            
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
    
    def train(self, df):
        """Train the model"""
        try:
            X, y = self.prepare_features(df, is_training=True)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Get feature importance
            feature_importance = dict(zip(
                ['team1', 'team2', 'toss_winner', 'toss_decision',
                 'is_toss_winner_team1', 'is_batting_first', 
                 'team1_batting_first', 'team2_batting_first', 
                 'target_runs', 'target_overs'],
                self.model.feature_importances_
            ))
            
            return accuracy, report, feature_importance
            
        except Exception as e:
            print(f"Error in train method: {str(e)}")
            raise
    def _wicket_resources_factor(self, wickets_lost):
        """
        Calculate batting resources based on wickets lost
        Uses an exponential model that values last wickets higher
        
        Args:
            wickets_lost: Number of wickets lost (0-10)
            
        Returns:
            Resource factor between 0 and 1
        """
        # Wickets are more valuable as more are lost
        # First wickets have less impact than later wickets
        if wickets_lost >= 10:
            return 0.01  # Almost no resources left with all wickets gone
        
        # Exponential model for wicket resources
        # Exponential curve that rapidly decreases as wickets approach 10
        return np.exp(-0.4 * wickets_lost)
    
    def _calculate_pressure_factor(self, required_runs, remaining_overs, wickets_remaining):
        """
        Calculate pressure factor for end-game scenarios
        
        Args:
            required_runs: Runs needed to win
            remaining_overs: Overs remaining
            wickets_remaining: Wickets in hand
            
        Returns:
            Pressure factor between 0 and 1
        """
        # Calculate balls remaining (6 balls per over)
        balls_remaining = remaining_overs * 6
        
        # Basic pressure calculation
        if balls_remaining <= 0:
            return 0  # No balls left means no chance
            
        # Runs per ball needed
        rpb_needed = required_runs / balls_remaining
        
        # Pressure increases with higher runs per ball requirement
        # and decreases with more wickets in hand
        pressure = rpb_needed / (0.2 * wickets_remaining + 0.5)
        
        # Convert to a factor between 0 and 1
        # Higher pressure means lower win probability
        pressure_factor = np.exp(-pressure)
        
        return np.clip(pressure_factor, 0.1, 1.0)
    def predict_win_probability(self, match_info):
        """Predict win probability for a match"""
        try:
            match_data = pd.DataFrame([match_info])
            X = self.prepare_features(match_data, is_training=False)
            probabilities = self.model.predict_proba(X)[0]
            team1_base_prob, team2_base_prob = probabilities[1], probabilities[0]
            
            # Adjust probabilities based on match situation
            if not all(k in match_info for k in ['required_runs', 'remaining_overs', 'wickets_lost']):
                return {
                    'team1_win_probability': team1_base_prob,
                    'team2_win_probability': team2_base_prob
                }
            # Step 2: Calculate dynamic factors based on match situation
            
            # Identify batting and bowling teams
            batting_first = match_info.get('toss_decision') == 'bat'
            chasing_team_index = 1 if batting_first else 0
            bowling_team_index = 0 if batting_first else 1
            
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
            
            # Step 4: Advanced dynamic win probability calculation
            
            # 4.1 Resources calculation (based on simplified DLS concept)
            # This estimates percentage of batting resources remaining
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
            
            # 4.3 Win probability based on resources and difficulty
            # As resources decrease and difficulty increases, win probability decreases
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
                """required_rr = match_info['required_runs'] / max(match_info['remaining_overs'], 0.1)
                initial_rr = match_info['target_runs'] / match_info['target_overs']
                
                # Enhanced situation analysis with penalty
                rr_factor = np.clip(initial_rr / max(required_rr, 0.1), 0.5, 2.0)
                wickets_factor = match_info['required_wickets'] / 10.0
                batting_first = match_info['toss_decision'] == 'bat'
                batting_first_factor = 1.1 if batting_first else 0.9

                # Penalty factor for low overs and many wickets needed
                penalty_factor = 1.0
                if match_info['remaining_overs'] < 5 and match_info['required_wickets'] > 5:
                    penalty_factor = 0.85

                # Combined situation factor
                situation_factor = rr_factor * wickets_factor * batting_first_factor * penalty_factor
                #adjusted_prob = probabilities * situation_factor
                
                # Normalize probabilities
                # Identify chasing team (batting second)
                chasing_team_index = 1 if match_info['toss_decision'] == 'bat' else 0

                # Apply situation factor to only the chasing team
                adjusted_prob = probabilities.copy()
                adjusted_prob[chasing_team_index] *= situation_factor

                # Normalize the adjusted probabilities
                total_prob = np.sum(adjusted_prob)
                if total_prob > 0:
                    adjusted_prob = adjusted_prob / total_prob
                probabilities = adjusted_prob
            
            return {
                'team1_win_probability': probabilities[1],
                'team2_win_probability': probabilities[0]
            }
            
        except Exception as e:
            print(f"Error in predict_win_probability: {str(e)}")
            print("Input match_info:")
            print(match_info)
            raise"""

def main():
    # Load your cleaned IPL dataset
    df = pd.read_csv('output2.csv')
    
    # Initialize and train the predictor
    predictor = IPLWinPredictor()
    accuracy, report, feature_importance = predictor.train(df)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(report)
    print("\nFeature Importance:")
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance:.4f}")
    
    # Example prediction
    print("\n--- Over-by-Over Simulation ---")
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
            'toss_decision': 'bat'
        }   

        try:
            probabilities = predictor.predict_win_probability(match_info)
            print(f"Over {over} - Score: {current_score}/{wickets_fallen[over-1]}")
            print(f"Required: {required_runs} off {remaining_overs} overs")
            print(f"Win % -> {match_info['team2']}: {probabilities['team2_win_probability']:.1%} | {match_info['team1']}: {probabilities['team1_win_probability']:.1%}\n")
        except Exception as e:
            print(f"Error in simulation at over {over}: {str(e)}")
            break

if __name__ == "__main__":
    main()
