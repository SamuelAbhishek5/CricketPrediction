import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class IPLWinPredictor:
    def __init__(self):
        self.le_team = LabelEncoder()
        self.le_city = LabelEncoder()
        self.le_venue = LabelEncoder()
        self.le_toss_decision = LabelEncoder()
        self.scaler = StandardScaler()
        # Initialize with optimized parameters
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.known_teams = None
        self.known_cities = None
        self.known_venues = None
        self.known_toss_decisions = None
        
    def fit_encoders(self, df):
        """Fit the label encoders on the full dataset"""
        self.known_teams = sorted(set(df['team1'].unique()) | set(df['team2'].unique()))
        self.known_cities = sorted(df['city'].unique())
        self.known_venues = sorted(df['venue'].unique())
        self.known_toss_decisions = sorted(df['toss_decision'].unique())
        
        self.le_team.fit(self.known_teams)
        self.le_city.fit(self.known_cities)
        self.le_venue.fit(self.known_venues)
        self.le_toss_decision.fit(self.known_toss_decisions)
    
    def encode_with_unknown(self, series, encoder, known_values):
        """Safely encode values, handling unknown categories"""
        series = series.map(lambda x: known_values[0] if x not in known_values else x)
        return encoder.transform(series)
    
    def prepare_features(self, df, is_training=True):
        """Prepare features for the model with enhanced feature engineering"""
        if is_training:
            self.fit_encoders(df)
        
        df_encoded = df.copy()
        
        try:
            # Basic encoding
            df_encoded['team1_encoded'] = self.encode_with_unknown(df['team1'], self.le_team, self.known_teams)
            df_encoded['team2_encoded'] = self.encode_with_unknown(df['team2'], self.le_team, self.known_teams)
            df_encoded['city_encoded'] = self.encode_with_unknown(df['city'], self.le_city, self.known_cities)
            df_encoded['venue_encoded'] = self.encode_with_unknown(df['venue'], self.le_venue, self.known_venues)
            df_encoded['toss_winner_encoded'] = self.encode_with_unknown(df['toss_winner'], self.le_team, self.known_teams)
            df_encoded['toss_decision_encoded'] = self.encode_with_unknown(df['toss_decision'], self.le_toss_decision, self.known_toss_decisions)
            
            # Enhanced feature engineering
            df_encoded['is_toss_winner_team1'] = (df['toss_winner'] == df['team1']).astype(int)
            df_encoded['is_batting_first'] = (df['toss_decision'] == 'bat').astype(int)
            
            # Create batting/bowling order features
            df_encoded['team1_batting_first'] = ((df['toss_winner'] == df['team1']) & 
                                               (df['toss_decision'] == 'bat')).astype(int)
            df_encoded['team2_batting_first'] = ((df['toss_winner'] == df['team2']) & 
                                               (df['toss_decision'] == 'bat')).astype(int)
            
            if is_training:
                df_encoded['target'] = (df['team1'] == df['winner']).astype(int)
            
            features = [
                'team1_encoded', 'team2_encoded',
                'city_encoded', 'venue_encoded',
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
    
    def optimize_hyperparameters(self, X, y):
        """Optimize model hyperparameters using GridSearchCV"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        grid_search.fit(X, y)
        return grid_search.best_estimator_
    
    def train(self, df):
        """Train the model with hyperparameter optimization"""
        try:
            X, y = self.prepare_features(df, is_training=True)
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Optimize hyperparameters
            self.model = self.optimize_hyperparameters(X_train, y_train)
            
            # Train with optimized model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            # Get feature importance
            feature_importance = dict(zip(
                ['team1', 'team2', 'city', 'venue', 'toss_winner', 'toss_decision',
                 'is_toss_winner_team1', 'is_batting_first', 'team1_batting_first',
                 'team2_batting_first', 'target_runs', 'target_overs'],
                self.model.feature_importances_
            ))
            
            # Save the trained model
            joblib.dump(self.model, 'cricket_model.pkl')  # Save the model here
            
            return accuracy, report, feature_importance
        except Exception as e:
            print(f"Error in train method: {str(e)}")
            raise
    
    def predict_win_probability(self, match_info):
        """Predict win probability with enhanced match situation analysis"""
        try:
            match_data = pd.DataFrame([match_info])
            X = self.prepare_features(match_data, is_training=False)
            probabilities = self.model.predict_proba(X)[0]
            
            # Adjust probabilities based on match situation
            if all(k in match_info for k in ['required_runs', 'remaining_overs', 'required_wickets']):
                required_rr = match_info['required_runs'] / max(match_info['remaining_overs'], 0.1)
                initial_rr = match_info['target_runs'] / match_info['target_overs']
                
                # Enhanced situation analysis
                rr_factor = np.clip(initial_rr / max(required_rr, 0.1), 0.5, 2.0)
                wickets_factor = match_info['required_wickets'] / 10.0
                
                # Consider batting/bowling order
                batting_first = match_info['toss_decision'] == 'bat'
                batting_first_factor = 1.1 if batting_first else 0.9
                
                # Combined situation factor
                situation_factor = rr_factor * wickets_factor * batting_first_factor
                adjusted_prob = probabilities * situation_factor
                
                # Normalize probabilities
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
            raise

def main():
    # Load your cleaned IPL dataset
    df = pd.read_csv(r'/Users/samuel/Downloads/Project/codenew15/CricketPrediction/FEHomePage/output2.csv')
    
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
    match_info = {
        'team1': 'Mumbai Indians',
        'team2': 'Chennai Super Kings',
        'city': 'Mumbai',
        'venue': 'Wankhede Stadium',
        'target_runs': 180,
        'target_overs': 20.0,
        'required_runs': 60,
        'required_wickets': 7,
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
