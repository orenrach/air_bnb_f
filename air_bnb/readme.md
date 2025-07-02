# 🏠 חיזוי מחיר לינה ב-Airbnb

פרויקט למידת מכונה לחיזוי מחיר לינה ב-Airbnb בסן פרנסיסקו באמצעות מודל CatBoost מתקדם.

## 📋 תיאור הפרויקט

פרויקט זה משלב טכניקות מתקדמות של מדע נתונים ולמידת מכונה לחיזוי מדויק של מחירי לינה ב-Airbnb. המערכת כוללת:

- **ניקוי ועיבוד נתונים** אוטומטי
- **הנדסת מאפיינים** מתקדמת
- **מודל למידת מכונה** מבוסס CatBoost
- **ממשק ווב** אינטראקטיבי עם Streamlit
- **בוט טלגרם** לחיזוי מהיר
- **הסבר מודל** באמצעות SHAP ו-OpenAI

## 🚀 התקנה

### דרישות מערכת
- Python 3.8+
- pip

### התקנת תלויות
```bash
# שכפול הפרויקט
git clone <repository-url>
cd air_bnb_f

# התקנת תלויות
pip install -r requirements.txt
```

### הגדרת משתני סביבה
צור קובץ `.env` בתיקיית `security/` עם המשתנים הבאים:
```env
OPENAI_API_KEY=your_openai_api_key_here
TELEGRAM_TOKEN=your_telegram_bot_token_here
```

## 📁 מבנה הפרויקט

```
air_bnb_f/
├── air_bnb/
│   ├── data/                    # קבצי נתונים
│   │   ├── listings.csv         # נתונים גולמיים
│   │   ├── cleaned_data.csv     # נתונים מנוקים
│   │   ├── featured_data.csv    # נתונים עם מאפיינים
│   │   ├── X_train.csv          # נתוני אימון
│   │   ├── X_test.csv           # נתוני בדיקה
│   │   ├── y_train.csv          # תגי אימון
│   │   └── y_test.csv           # תגי בדיקה
│   ├── models/                  # מודלים מאומנים
│   │   ├── best_model_CatBoost.pkl
│   │   ├── best_model_XGBoost.pkl
│   │   ├── model_features.pkl
│   │   ├── valid_room_types.pkl
│   │   └── valid_property_types.pkl
│   ├── scripts/                 # סקריפטים
│   │   ├── clean.py            # ניקוי נתונים
│   │   ├── feature_e.py        # הנדסת מאפיינים
│   │   ├── split_data.py       # פיצול נתונים
│   │   ├── train_model.py      # אימון מודל
│   │   ├── app.py              # אפליקציית Streamlit
│   │   └── telegrambot.py      # בוט טלגרם
│   ├── requirements.txt         # תלויות Python
│   └── readme.md               # קובץ זה
```

## 🔧 שימוש

### 1. עיבוד נתונים ואימון מודל

הרץ את הסקריפטים בסדר הבא:

```bash
cd air_bnb/scripts

# 1. ניקוי נתונים
python clean.py

# 2. הנדסת מאפיינים
python feature_e.py

# 3. פיצול נתונים
python split_data.py

# 4. אימון מודל
python train_model.py
```

### 2. הפעלת אפליקציית ווב

```bash
cd air_bnb/scripts
streamlit run app.py
```

האפליקציה תיפתח בדפדפן בכתובת `http://localhost:8501`

### 3. הפעלת בוט טלגרם

```bash
cd air_bnb/scripts
python telegrambot.py
```

## 💡 דוגמאות שימוש

### אפליקציית ווב
1. פתח את האפליקציה בדפדפן
2. בחר אזור בסן פרנסיסקו
3. הגדר פרמטרים:
   - מספר אורחים: 2
   - חדרי רחצה: 1
   - חדרי שינה: 1
   - מיטות: 1
   - סוג חדר: Private room
   - סוג נכס: Entire home
4. לחץ על "חשב מחיר והצג הסבר"

### בוט טלגרם
שלח הודעה לבוט:
```
/predict latitude=37.77, longitude=-122.42, accommodates=2, bathrooms=1, bedrooms=1, beds=1, room_type=Private room, property_type=Entire home
```

## 📊 ביצועי המודל

המודל הנוכחי משיג:
- **MAE (שגיאה ממוצעת)**: ~$45-55
- **R² (מקדם החלטה)**: ~0.75-0.85

## 🛠️ טכנולוגיות

- **Python**: שפת התכנות הראשית
- **Pandas**: עיבוד נתונים
- **Scikit-learn**: אלגוריתמי למידת מכונה
- **CatBoost**: מודל הגרדיאנט בוסטינג
- **Streamlit**: ממשק ווב
- **Folium**: מפות אינטראקטיביות
- **SHAP**: הסבר מודלים
- **OpenAI API**: הסבר מילולי
- **Telegram Bot API**: בוט טלגרם

## 🔍 מאפיינים עיקריים

### ניקוי נתונים
- הסרת ערכים חסרים
- טיפול במחירים קיצוניים
- המרת פורמטים

### הנדסת מאפיינים
- קידוד One-Hot לעמודות קטגוריאליות
- נרמול נתונים מספריים
- יצירת מאפיינים חדשים

### מודל למידת מכונה
- CatBoost Regressor
- אופטימיזציה של היפר-פרמטרים
- הערכת ביצועים מקיפה

### ממשק משתמש
- ממשק ווב אינטואיטיבי
- הצגת מפות אינטראקטיביות
- הסבר מודל ויזואלי ומילולי


---

**נבנה ע"י רינת בן הרוש ואורן רחמינוב בישראל**
