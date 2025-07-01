import os
import logging
import pandas as pd
import joblib
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# טען משתני סביבה
load_dotenv(dotenv_path="security/.env")
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("❌ TELEGRAM_TOKEN לא הוגדר בקובץ .env")

# הגדר לוג
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

# טען מודל וקבצים נלווים
model = joblib.load("models/best_model_XGBoost.pkl")
model_features = joblib.load("models/model_features.pkl")
room_type_options = joblib.load("models/valid_room_types.pkl")
property_type_options = joblib.load("models/valid_property_types.pkl")


# /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 שלום! אני בוט לחיזוי מחיר לילה ב־Airbnb.\n"
        "שלח לי את פרטי הנכס שלך עם הפקודה /predict\n\n"
        "🔍 דוגמה:\n"
        "/predict latitude=37.77, longitude=-122.42, accommodates=2, bathrooms=1, bedrooms=1, beds=1, room_type=Private room, property_type=Entire home"
    )


# /options
async def options(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📌 סוגי חדרים:\n{', '.join(room_type_options)}\n\n"
        f"🏠 סוגי נכסים:\n{', '.join(property_type_options)}"
    )


# פונקציית תחזית
def predict_price_from_text(text: str) -> str:
    try:
        parts = dict(
            item.strip().split("=")
            for item in text.split(",")
            if "=" in item
        )

        for key in ["latitude", "longitude", "accommodates", "bathrooms", "bedrooms", "beds"]:
            parts[key] = float(parts[key]) if "." in parts[key] else int(parts[key])

        df = pd.DataFrame([{
            "latitude": parts["latitude"],
            "longitude": parts["longitude"],
            "accommodates": parts["accommodates"],
            "bathrooms": parts["bathrooms"],
            "bedrooms": parts["bedrooms"],
            "beds": parts["beds"],
            "room_type": parts["room_type"],
            "property_type": parts["property_type"],
        }])

        df = pd.get_dummies(df)

        for col in model_features:
            if col not in df.columns:
                df[col] = 0

        df = df[model_features]
        pred = model.predict(df)[0]
        return f"המחיר החזוי ללילה: ${pred:.2f}"
    except Exception as e:
        return f"❌ שגיאה: {str(e)}"


# /predict
async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_msg = update.message.text
    logging.info(f"📥 קלט מהמשתמש: {user_msg}")
    await update.message.reply_text("🔄 מחשב תחזית...")

    try:
        text_data = user_msg.split(" ", 1)[1]
        prediction = predict_price_from_text(text_data)
    except IndexError:
        prediction = "❗ פורמט שגוי. נסה לשלוח את פרטי הדירה אחרי /predict."

    await update.message.reply_text(prediction)


# הודעות רגילות
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("📌 השתמש ב־/start או /predict כדי להתחיל.")


if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("options", options))
    app.add_handler(CommandHandler("predict", predict_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 הבוט רץ... מחכה להודעות.")
    app.run_polling()
