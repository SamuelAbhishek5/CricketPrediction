import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load data
data = {
    'rr_ratio': [1.20, 1.05, 0.90, 1.10, 0.95, 1.25, 1.00, 0.80, 1.15, 0.85, 1.05, 0.92, 1.18, 1.10, 0.88],
    'wickets_in_hand': [4, 6, 8, 5, 7, 3, 6, 9, 4, 7, 5, 6, 3, 5, 8],
    'resources_remaining': [0.35, 0.50, 0.65, 0.40, 0.60, 0.30, 0.55, 0.70, 0.38, 0.62, 0.50, 0.58, 0.32, 0.44, 0.68],
    'chasing_team_won': [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]
}
df = pd.DataFrame(data)

# Train logistic regression
X = df[['rr_ratio', 'wickets_in_hand', 'resources_remaining']]
y = df['chasing_team_won']

model = LogisticRegression()
model.fit(X, y)

# Coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)