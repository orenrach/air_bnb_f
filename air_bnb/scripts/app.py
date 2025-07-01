import os
import streamlit as st
import pandas as pd
import joblib
import folium
import shap
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from dotenv import load_dotenv
from openai import OpenAI

# טען משתני סביבה
load_dotenv("security/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# טען את המודל והרשימות
model = joblib.load("models/best_model_XGBoost.pkl")
model_features = joblib.load("models/model_features.pkl")
room_type_options = joblib.load("models/valid_room_types.pkl")
property_type_options = joblib.load("models/valid_property_types.pkl")

st.title("🔮 חיזוי מחיר לינה ב־Airbnb")

with st.form("prediction_form"):
    area_dict = {
        "Downtown": (37.7879, -122.4075),
        "Mission District": (37.7599, -122.4148),
        "North Beach": (37.8060, -122.4103),
        "Castro": (37.7609, -122.4350),
        "Haight-Ashbury": (37.7692, -122.4481),
        "Marina District": (37.8015, -122.4368),
        "Pacific Heights": (37.7925, -122.4382),
        "SoMa": (37.7785, -122.4056),
        "Nob Hill": (37.7930, -122.4161),
        "Fisherman's Wharf": (37.8080, -122.4177)
    }
    
    area = st.selectbox(
        "בחר אזור בסאן פרנסיסקו",
        list(area_dict.keys())
    )
    
    latitude, longitude = area_dict[area]
    accommodates = st.slider("מספר אורחים", 1, 16, 2)
    bathrooms = st.number_input("חדרי רחצה", min_value=0, max_value=10, step=1, value=1)
    bedrooms = st.number_input("חדרי שינה", min_value=0, max_value=10, step=1, value=1)
    beds = st.number_input("מיטות", min_value=0, max_value=10, step=1, value=1)

    room_type = st.selectbox("Room Type", room_type_options)
    property_type = st.selectbox("Property Type", property_type_options)

    submitted = st.form_submit_button("🔍 חשב מחיר והצג הסבר")

if submitted:


    # יצירת קלט לחיזוי
    base_input = {
        "latitude": latitude,
        "longitude": longitude,
        "accommodates": accommodates,
        "bathrooms": bathrooms,
        "bedrooms": bedrooms,
        "beds": beds,
        "room_type": room_type,
        "property_type": property_type,
    }
    df = pd.DataFrame([base_input])
    df = pd.get_dummies(df)

    for col in model_features:
        if col not in df.columns:
            df[col] = 0
    df = df[model_features]

    # חיזוי מחיר
    pred = model.predict(df)[0]
    st.success(f"המחיר החזוי ללילה הוא: ${pred:.2f}")

    # הסבר עם SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(df)

    st.subheader("📊 השפעת פרמטרים (SHAP)")
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0], show=False)
    st.pyplot(fig)

    # הסבר עם OpenAI על סמך תכונות חשובות
    important_features = sorted(
        zip(df.columns, shap_values[0].values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    important_text = ", ".join(
        [f"{name} ({value:.2f})" for name, value in important_features]
    )

    prompt = (
        f"המחיר שחושב הוא {pred:.2f} דולר ללילה. "
        f"הפרמטרים שהשפיעו הכי הרבה הם: {important_text}. "
        f"האם תוכל להסביר מדוע המחיר כזה – האם הוא נמוך, גבוה או סביר?"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = response.choices[0].message.content.strip()
        st.subheader("💬 הסבר מילולי (OpenAI)")
        st.write(explanation)
    except Exception as e:
        st.error(f"שגיאה בקבלת הסבר מ־OpenAI: {str(e)}")
    # הצגת מפה עם המיקום
    st.subheader("📍 מיקום על המפה")
    map_data = pd.DataFrame({
        'lat': [base_input['latitude']],
        'lon': [base_input['longitude']]
    })
    st.map(map_data)

    # כפתור לרענון
    if st.button("🔁 נקה והתחל מחדש"):
        st.experimental_rerun()
