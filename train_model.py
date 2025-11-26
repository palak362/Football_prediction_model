import sys
sys.path.append("src")   # allow import from preprocessing.py

from preprocessing import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

X_train, X_test, y_train, y_test = load_and_preprocess("C:\\Users\\palak priya\\Downloads\\archive (1)\\English Premier League.csv")

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    joblib.dump(model, f"outputs/model_{name.replace(' ','').lower()}.pkl")
