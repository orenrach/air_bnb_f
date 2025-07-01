#4️⃣ פיצול לסט אימון וסט בדיקה (Train/Test Split)

#שמירה כ־X_train.csv, X_test.csv, y_train.csv, y_test.csv


import pandas as pd
from sklearn.model_selection import train_test_split

# טעינת הנתונים המקודדים
df = pd.read_csv("data/featured_data.csv")

# הפרדת משתני קלט (features) ותג (target)
X = df.drop("price", axis=1)
y = df["price"]

# פיצול ל־80% אימון ו־20% בדיקה
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# שמירת הקבצים לפלט
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("✅ הנתונים חולקו לסטים של אימון ובדיקה ונשמרו לקבצים.")
