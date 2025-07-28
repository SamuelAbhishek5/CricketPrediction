import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load dataset (assuming you've preprocessed and handled missing data)
df = pd.read_csv('output2.csv')

# Target: 1 if team1 won, else 0
df['team1_win'] = (df['winner'] == df['team1']).astype(int)

# Encode categorical variables
le_city = LabelEncoder()
df['city_enc'] = le_city.fit_transform(df['city'])

teams = pd.unique(df[['team1', 'team2', 'toss_winner']].values.ravel())
le_team = LabelEncoder().fit(teams)
df['team1_enc'] = le_team.transform(df['team1'])
df['team2_enc'] = le_team.transform(df['team2'])
df['toss_winner_enc'] = le_team.transform(df['toss_winner'])

le_decision = LabelEncoder()
df['toss_decision_enc'] = le_decision.fit_transform(df['toss_decision'])

# Features (without venue and without target_runs and target_overs)
features = ['city_enc', 'team1_enc', 'team2_enc',
            'toss_winner_enc', 'toss_decision_enc']
X = df[features]
y = df['team1_win']


# Model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred.round()))
print("Classification Report:\n", classification_report(y_test, y_pred.round()))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred.round()))

# Sample prediction input (during 2nd innings)
sample_input = {
    'team1': 'Chennai Super Kings',
    'team2': 'Royal Challengers Bengaluru',
    'city': 'Chennai',
    'toss_winner': 'Chennai Super Kings',
    'toss_decision': 'field'
}

# Encode the input
sample = pd.DataFrame([{
    'city_enc': le_city.transform([sample_input['city']])[0],
    'team1_enc': le_team.transform([sample_input['team1']])[0],
    'team2_enc': le_team.transform([sample_input['team2']])[0],
    'toss_winner_enc': le_team.transform([sample_input['toss_winner']])[0],
    'toss_decision_enc': le_decision.transform([sample_input['toss_decision']])[0]
}])

# Predict probabilities
win_prob = model.predict(sample)[0]
team1_prob = win_prob
team2_prob = 1 - win_prob

# Output
team1 = sample_input['team1']
team2 = sample_input['team2']
print(f"Winning probabilities:\n{team1}: {team1_prob:.2f}, {team2}: {team2_prob:.2f}")
print("Predicted Winning Team:", team1 if team1_prob > team2_prob else team2)

# Batch prediction for multiple matchups
test_matches = [
    {
        'team1': 'Mumbai Indians',
        'team2': 'Chennai Super Kings',
        'city': 'Mumbai',
        'toss_winner': 'Mumbai Indians',
        'toss_decision': 'bat'
    },
    {
        'team1': 'Delhi Capitals',
        'team2': 'Kolkata Knight Riders',
        'city': 'Delhi',
        'toss_winner': 'Kolkata Knight Riders',
        'toss_decision': 'field'
    },
    {
        'team1': 'Rajasthan Royals',
        'team2': 'Sunrisers Hyderabad',
        'city': 'Jaipur',
        'toss_winner': 'Rajasthan Royals',
        'toss_decision': 'field'
    },
    {
        'team1': 'Lucknow Super Giants',
        'team2': 'Gujarat Titans',
        'city': 'Lucknow',
        'toss_winner': 'Gujarat Titans',
        'toss_decision': 'field'
    }
]

print("\n--- Batch Prediction Results ---")
for match in test_matches:
    try:
        encoded = pd.DataFrame([{
            'city_enc': le_city.transform([match['city']])[0],
            'team1_enc': le_team.transform([match['team1']])[0],
            'team2_enc': le_team.transform([match['team2']])[0],
            'toss_winner_enc': le_team.transform([match['toss_winner']])[0],
            'toss_decision_enc': le_decision.transform([match['toss_decision']])[0]
        }])
        win_prob = model.predict(encoded)[0]
        team1_prob, team2_prob = win_prob, 1 - win_prob
        print(f"\nMatch: {match['team1']} vs {match['team2']} in {match['city']}")
        print(f"Winning Probabilities -> {match['team1']}: {team1_prob:.2f}, {match['team2']}: {team2_prob:.2f}")
        print("Predicted Winner:", match['team1'] if team1_prob > team2_prob else match['team2'])
    except Exception as e:
        print(f"Prediction failed for {match['team1']} vs {match['team2']}: {e}")