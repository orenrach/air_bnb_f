#3️⃣ הנדסת מאפיינים (Feature Engineering)
#קידוד עמודות קטגוריאליות (One-Hot)

#שמירה כ־featured_data.csv




import pandas as pd

# טוען את הקובץ הנקי משלב 2
df = pd.read_csv("data/cleaned_data.csv")

# עמודות קטגוריאליות שצריך להמיר ל-One-Hot
categorical_cols = ['room_type', 'property_type']

# יצירת קידוד One-Hot
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# בדיקה שאין ערכים חסרים
assert df_encoded.isnull().sum().sum() == 0, "יש עדיין ערכים חסרים!"

# שמירת הקובץ עם המאפיינים החדשים
df_encoded.to_csv("data/featured_data.csv", index=False)

import pandas as pd
import joblib

# טוען את הקובץ המקודד
df = pd.read_csv("data/featured_data.csv")

# מחלץ שמות של עמודות קטגוריאליות מקודדות
room_type_cols = [col for col in df.columns if col.startswith("room_type_")]
property_type_cols = [col for col in df.columns if col.startswith("property_type_")]

# שמירת הרשימות לקבצים
joblib.dump(room_type_cols, "models/valid_room_types.pkl")
joblib.dump(property_type_cols, "models/valid_property_types.pkl")

print("✅ נשמרו הקבצים: valid_room_types.pkl ו־valid_property_types.pkl")

print("✅ הנדסת מאפיינים הושלמה. הקובץ featured_data.csv מוכן.")
