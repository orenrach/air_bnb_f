#5️⃣ אימון מודל CatBoost + שמירת המודל + שמירת שמות העמודות

#שמירה כ־best_model_CatBoost.pkl, model_features.pkl

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from catboost import CatBoostRegressor
import joblib

# טעינת הקבצים (הקפד שהם בתיקייה הנכונה)
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()

# הגדרת מודל CatBoost
model = CatBoostRegressor(
    n_estimators=100,
    learning_rate=0.1,
    depth=6,
    verbose=0,
    random_state=42
)

# אימון המודל
model.fit(X_train, y_train)

# חיזוי וביצוע הערכה
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# תוצאות
print(f"\n✅ תוצאות המודל:")
print(f"MAE (שגיאה ממוצעת): ${mae:.2f}")
print(f"R² (מקדם החלטה): {r2:.3f}")

# שמירת המודל ושמות העמודות
joblib.dump(model, "models/best_model_CatBoost.pkl")
joblib.dump(list(X_train.columns), "models/model_features.pkl")

valid_room_types = joblib.load("models/valid_room_types.pkl")
valid_property_types = joblib.load("models/valid_property_types.pkl")

print("\n✅ המודל נשמר בשם: models/best_model_CatBoost.pkl")
print("✅ רשימת העמודות נשמרה בשם: model_features.pkl")
