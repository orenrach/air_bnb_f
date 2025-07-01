#ניקוי נתונים (Data Cleaning)
#טעינת הנתונים
#הסרת עמודות מיותרות
#המרת מחירים ממחרוזות למספרים
#טיפול בחסרים
#סינון מחירים קיצוניים (למשל: $40–$1000)
#שמירה כ־cleaned_data.csv

import pandas as pd

# טוען את הקובץ
df = pd.read_csv("data/listings.csv")

# בוחר רק עמודות חשובות למודל
columns_to_keep = [
    'latitude', 'longitude', 'room_type', 'bathrooms', 'bedrooms',
    'beds', 'price', 'property_type', 'accommodates'
]
df = df[columns_to_keep]

# מסיר סימני דולר ופסיקים וממיר מחיר למספר
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)

# הסרת שורות עם ערכים חסרים בעמודות קריטיות
critical_cols = ['price', 'room_type', 'latitude', 'longitude']
df = df.dropna(subset=critical_cols)

# סינון שורות עם מחיר פחות מ־40 או יותר מ־1000 דולר
df = df[(df['price'] >= 40) & (df['price'] <= 1000)]

# מילוי ערכים חסרים בעמודות משניות עם ממוצע
for col in ['bathrooms', 'bedrooms', 'beds']:
    df[col] = df[col].fillna(df[col].median())

# שומר את הקובץ הנקי
df.to_csv("data/cleaned_data.csv", index=False)

print("✅ הנתונים נוקו ונשמרו בקובץ cleaned_data.csv")
