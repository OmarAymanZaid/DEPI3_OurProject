import os
import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Stroke Health Analytics",
    layout="wide",
    page_icon="🧠"
)

st.title("🧠 Stroke Health Predictive Analytics Dashboard")
st.write("An interactive dashboard that visualizes stroke risk factors across 10 major health dimensions.")

# Load your dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "..", "..", "..", "Data", "stroke_data_cleaned.csv")

# Read the CSV
stroke_data = pd.read_csv(DATA_PATH)


# ---------------------------------------------------------
# GLOBAL REUSABLE CHART FUNCTION
# ---------------------------------------------------------
def bar_chart(df, x, y, title, x_label, subtitle_labels=None):
    df = df.copy()
    df['label'] = df[y].map(lambda v: f"{v:.2f}%")

    if subtitle_labels is not None:
        df['label'] = df.apply(lambda row:
                               f"{row[y]:.2f}%<br>{subtitle_labels[row[x]]}", axis=1)

    fig = px.bar(
        df,
        x=x,
        y=y,
        text='label',
        color=x,
        title=title,
        height=450,
        template="plotly_white"
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="Stroke Rate (%)")
    return fig


# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "📊 Analytics (1–5)",
        "📈 Analytics (6–10)",
        "🤖 Prediction Model"
    ]
)


# =========================================================
#  SECTION 1: OVERVIEW + KPIs
# =========================================================
if menu == "🏠 Overview":
    st.header("📌 Overview & Key Indicators")

    col1, col2, col3 = st.columns(3)

    total_people = len(stroke_data)
    total_strokes = stroke_data['stroke'].sum()
    stroke_rate_total = (stroke_data['stroke'].mean() * 100)

    col1.metric("Total Records", f"{total_people:,}")
    col2.metric("Total Stroke Cases", f"{int(total_strokes):,}")
    col3.metric("Overall Stroke Rate", f"{stroke_rate_total:.2f}%")

    st.write("Use the sidebar to navigate between sections.")


# =========================================================
#  SECTION 2: ANALYTICS 1–5
# =========================================================
elif menu == "📊 Analytics (1–5)":

    st.header("📊 Stroke Analysis — Part 1 (Questions 1–5)")

    # 1. Residence Type
    with st.expander("🏠 Stroke Rate by Residence Type", expanded=True):
        df = stroke_data.groupby("Residence_type")["stroke"].mean().reset_index()
        df["stroke"] *= 100
        fig = bar_chart(df, "Residence_type", "stroke",
                        "Stroke Rate by Residence Type", "Residence Type")
        st.plotly_chart(fig, use_container_width=True)

    # 2. Smoking Status
    with st.expander("🚬 Stroke Rate by Smoking Status"):
        df = stroke_data.groupby("smoking_status")["stroke"].mean().reset_index()
        df["stroke"] *= 100
        fig = bar_chart(df, "smoking_status", "stroke",
                        "Stroke Rate by Smoking Status", "Smoking Status")
        st.plotly_chart(fig, use_container_width=True)

    # 3. Work Type
    with st.expander("💼 Stroke Rate by Work Type"):
        df = stroke_data.groupby("work_type")["stroke"].mean().reset_index()
        df["stroke"] *= 100
        fig = bar_chart(df, "work_type", "stroke",
                        "Stroke Rate by Work Type", "Work Type")
        st.plotly_chart(fig, use_container_width=True)

    # 4. Gender
    with st.expander("👤 Stroke Rate by Gender"):
        df = stroke_data.groupby("gender")["stroke"].mean().reset_index()
        df["stroke"] *= 100
        fig = bar_chart(df, "gender", "stroke",
                        "Stroke Rate by Gender", "Gender")
        st.plotly_chart(fig, use_container_width=True)

    # 5. Hypertension
    with st.expander("❤️ Stroke Rate by Hypertension"):
        df = stroke_data.groupby("hypertension")["stroke"].mean().reset_index()
        df["hypertension"] = df["hypertension"].map({0: "No Hypertension", 1: "Hypertension"})
        df["stroke"] *= 100
        fig = bar_chart(df, "hypertension", "stroke",
                        "Stroke Rate by Hypertension", "Hypertension Status")
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# SECTION 3: ANALYTICS 6–10
# =========================================================
elif menu == "📈 Analytics (6–10)":

    st.header("📈 Stroke Analysis — Part 2 (Questions 6–10)")

    # 6. Ever Married
    with st.expander("💍 Stroke Rate by Marital Status", expanded=True):
        df = stroke_data.groupby("ever_married")["stroke"].mean().reset_index()
        df["stroke"] *= 100
        fig = bar_chart(df, "ever_married", "stroke",
                        "Stroke Rate by Marital Status", "Marital Status")
        st.plotly_chart(fig, use_container_width=True)

    # 7. BMI Group
    with st.expander("⚖️ Stroke Rate by BMI Group"):

        # --- Ensure BMI groups exist ---
        if 'BMI_Group' not in stroke_data.columns:
            stroke_data['BMI_Group'] = pd.cut(
                stroke_data['bmi'],
                bins=[-float('inf'), -0.5, 0.5, 1.5, float('inf')],
                labels=['Underweight', 'Normal', 'Overweight', 'Obese']
            )

        # --- Map readable BMI ranges ---
        bmi_ranges = {
            'Underweight': '(<25.2 kg/m²)',
            'Normal': '(25.2–32.3 kg/m²)',
            'Overweight': '(32.3–39.4 kg/m²)',
            'Obese': '(>39.4 kg/m²)'
        }

        # --- Group by BMI_Group and compute stroke rate ---
        df = stroke_data.groupby('BMI_Group')['stroke'].mean().reset_index()
        df['stroke'] *= 100

        # --- Create interactive bar chart ---
        fig = bar_chart(
            df,
            x='BMI_Group',
            y='stroke',
            title='Stroke Rate by BMI Group',
            x_label='BMI Group',
            subtitle_labels=bmi_ranges
        )

        st.plotly_chart(fig, use_container_width=True)



    # 8. Heart Disease
    with st.expander("❤️ Stroke Rate by Heart Disease"):
        df = stroke_data.groupby("heart_disease")["stroke"].mean().reset_index()
        df["heart_disease"] = df["heart_disease"].map({0: "No Heart Disease", 1: "Heart Disease"})
        df["stroke"] *= 100
        fig = bar_chart(df, "heart_disease", "stroke",
                        "Stroke Rate by Heart Disease", "Heart Disease Status")
        st.plotly_chart(fig, use_container_width=True)

    # 9. Glucose Level
    with st.expander("🩸 Stroke Rate by Glucose Level"):

        if 'Glucose_Level_Group' not in stroke_data.columns:
            stroke_data["Glucose_Level_Group"] = pd.cut(
                stroke_data["avg_glucose_level"],
                bins=[0, 90, 140, float("inf")],
                labels=["Low", "Normal", "High"]
            )

        glucose_ranges = {
            "Low": "<85 mg/dL",
            "Normal": "85–125 mg/dL",
            "High": ">125 mg/dL"
        }

        df = stroke_data.groupby("Glucose_Level_Group")["stroke"].mean().reset_index()
        df["stroke"] *= 100

        fig = bar_chart(
            df,
            "Glucose_Level_Group",
            "stroke",
            "Stroke Rate by Glucose Level",
            "Glucose Level",
            subtitle_labels=glucose_ranges
        )
        st.plotly_chart(fig, use_container_width=True)

    # 10. Age Group
    with st.expander("👵 Stroke Rate by Age Group"):
        # Create Age_Group if it doesn't exist
        if 'Age_Group' not in stroke_data.columns:
            stroke_data['Age_Group'] = pd.cut(
                stroke_data['age'],
                bins=[0, 32, 55, float('inf')],
                labels=['Younger', 'Middle-aged', 'Older']
            )

        age_ranges = {
            "Younger": "<32 yrs",
            "Middle-aged": "32–55 yrs",
            "Older": ">55 yrs"
        }

        df = stroke_data.groupby("Age_Group")["stroke"].mean().reset_index()
        df["stroke"] *= 100

        fig = bar_chart(
            df,
            "Age_Group",
            "stroke",
            "Stroke Rate by Age Group",
            "Age Group",
            subtitle_labels=age_ranges
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
#  SECTION 4: PREDICTION MODEL PAGE (LOCAL PREPROCESSING)
# =========================================================
elif menu == "🤖 Prediction Model":

    st.header("🤖 Stroke Risk Prediction")

    st.write("Enter patient information below to get a predicted probability of stroke using the trained machine learning model.")

    # ----------------------------------------
    # FORM INPUTS
    # ----------------------------------------
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            ever_married = st.selectbox("Ever Married", ["No", "Yes"])

        with col2:
            work_type = st.selectbox(
                "Work Type",
                ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"]
            )
            Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
            avg_glucose_level = st.number_input("Avg Glucose Level", 40.0, 300.0, 105.0)
            bmi = st.number_input("BMI", 10.0, 60.0, 28.5)
            smoking_status = st.selectbox(
                "Smoking Status",
                ["formerly smoked", "never smoked", "smokes", "Unknown"]
            )

        submitted = st.form_submit_button("Predict Stroke Risk")

    # ----------------------------------------
    # ON SUBMIT: LOCAL PREPROCESSING + PREDICTION
    # ----------------------------------------
    if submitted:

        # -----------------------------
        # Load model only
        # -----------------------------
        model = joblib.load("/home/omar/DEPI3_OurProject/healthcare_predictions/Models/Logistic_Regression_Tuned.pkl")  # Replace with your model path

        # -----------------------------
        # Numeric columns and scaling parameters
        # -----------------------------
        numeric_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        numeric_values = np.array([
            age,
            1 if hypertension == "Yes" else 0,
            1 if heart_disease == "Yes" else 0,
            avg_glucose_level,
            bmi
        ]).reshape(1, -1)

        # Example scaling parameters (replace with your training mean/std)
        numeric_mean = np.array([45.0, 0.1, 0.05, 105.0, 28.0])
        numeric_std = np.array([18.0, 0.3, 0.2, 45.0, 5.5])
        numeric_scaled = (numeric_values - numeric_mean) / numeric_std

        # -----------------------------
        # Categorical columns and one-hot encoding
        # -----------------------------
        categorical_cols = ["gender", "work_type", "ever_married", "Residence_type", "smoking_status"]

        categories = {
            "gender": ["Male", "Female", "Other"],
            "work_type": ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"],
            "ever_married": ["No", "Yes"],
            "Residence_type": ["Urban", "Rural"],
            "smoking_status": ["formerly smoked", "never smoked", "smokes", "Unknown"]
        }

        # Collect all categorical inputs in a dict
        input_dict = {
            "gender": gender,
            "work_type": work_type,
            "ever_married": ever_married,
            "Residence_type": Residence_type,
            "smoking_status": smoking_status
        }

        cat_array = []
        for col in categorical_cols:
            arr = [1 if input_dict[col] == cat else 0 for cat in categories[col]]
            cat_array.extend(arr)
        cat_array = np.array(cat_array).reshape(1, -1)

        # -----------------------------
        # Combine numeric + categorical
        # -----------------------------
        X_input = np.hstack([numeric_scaled, cat_array])

        # -----------------------------
        # Predict
        # -----------------------------
        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1]

        # -----------------------------
        # Display results
        # -----------------------------
        st.subheader("📊 Prediction Result")
        st.success(f"Estimated Stroke Probability: **{probability*100:.2f}%**")

        if probability < 0.10:
            st.info("🟢 *Low Risk*: Continue maintaining healthy habits.")
        elif probability < 0.30:
            st.warning("🟡 *Moderate Risk*: Consider routine medical checkups.")
        else:
            st.error("🔴 *High Risk*: Consult a healthcare professional promptly.")




