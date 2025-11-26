import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):
    df = pd.read_csv(path)

    df['GoalDiff'] = df['HomeGoals'] - df['AwayGoals']

    X = df[['HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals', 'GoalDiff']]
    y = df['Result']

    le_home = LabelEncoder()
    le_away = LabelEncoder()
    X['HomeTeam'] = le_home.fit_transform(X['HomeTeam'])
    X['AwayTeam'] = le_away.fit_transform(X['AwayTeam'])

    return train_test_split(X, y, test_size=0.2, random_state=42)
