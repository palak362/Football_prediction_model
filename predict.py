import joblib
import numpy as np

def predict_result(model_path, home_team, away_team, home_goals, away_goals):
    model = joblib.load(model_path)
    goal_diff = home_goals - away_goals
    sample = np.array([[home_team, away_team, home_goals, away_goals, goal_diff]])
    prediction = model.predict(sample)
    return prediction[0]

result = predict_result("outputs/model_randomforest.pkl", 1, 5, 2, 1)
print("Predicted Result:", result)
