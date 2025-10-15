# ==========================================
# 🩺 Healthcare Predictive Analytics Dashboard
# ==========================================
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------
# PAGE SETUP
# ---------------------------
st.set_page_config(
    page_title="Healthcare Predictive Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom style
st.markdown("""
    <style>
    .block-container { padding-top: 1rem; }
    h2 { color: #2E8B57; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data/stroke_data_cleaned.csv")
    return df

stroke_data = load_data()

# ---------------------------
# HEADER
# ---------------------------
st.title("🏥 Healthcare Predictive Analytics: Stroke Dataset")
st.markdown("""
This interactive dashboard presents insights from the **Stroke Prediction Dataset**.
Each section explores an analytical question through statistical reasoning and visual evidence.
Use this tool to explore how demographic and health-related factors correlate with stroke occurrence.
""")

st.divider()

# ---------------------------
# DATA OVERVIEW
# ---------------------------
st.header("📊 Dataset Overview")
st.markdown("Preview the cleaned dataset below:")
st.dataframe(stroke_data.head())

st.markdown(f"**Total records:** {stroke_data.shape[0]} | **Columns:** {stroke_data.shape[1]}")

st.divider()

# ---------------------------
# INTERACTIVE QUESTIONS
# ---------------------------

# --- Question 1 ---
with st.expander("🧩 **Question 1: Do urban or rural residents experience more strokes?**", expanded=True):
    st.markdown("""
    **Why:** Lifestyle and healthcare accessibility may differ between urban and rural populations, potentially influencing stroke rates.
    """)
    
    stroke_rate = stroke_data.groupby('Residence_type')['stroke'].mean().reset_index()
    stroke_rate['stroke'] *= 100

    st.write("**Stroke rate by residence type (%):**")
    st.dataframe(stroke_rate)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=stroke_rate, x='Residence_type', y='stroke', palette='crest', ax=ax)
    ax.set_title('Stroke Rate by Residence Type')
    ax.set_ylabel('Stroke Rate (%)')
    ax.set_xlabel('Residence Type')
    for i, row in stroke_rate.iterrows():
        ax.text(i, row['stroke'] + 0.5, f"{row['stroke']:.2f}%", ha='center')
    st.pyplot(fig)

# --- Question 2 ---
with st.expander("🧩 **Question 2: Does stroke risk rise with age?**"):
    st.markdown("**Why:** Age is one of the most significant risk factors for stroke.")
    fig, ax = plt.subplots(figsize=(7,4))
    sns.boxplot(x='stroke', y='age', data=stroke_data, hue='stroke', palette='crest', legend=False, ax=ax)
    ax.set_title("Age vs Stroke Occurrence")
    ax.set_xlabel("Stroke (0 = No, 1 = Yes)")
    ax.set_ylabel("Age")
    st.pyplot(fig)

# --- Question 3 ---
with st.expander("🧩 **Question 3: Is high glucose level linked to stroke?**"):
    st.markdown("**Why:** Elevated glucose levels can indicate diabetes, which increases stroke risk.")
    fig, ax = plt.subplots(figsize=(7,4))
    sns.boxplot(x='stroke', y='avg_glucose_level', data=stroke_data, hue='stroke', palette='crest', legend=False, ax=ax)
    ax.set_title("Average Glucose Level vs Stroke Occurrence")
    ax.set_xlabel("Stroke (0 = No, 1 = Yes)")
    ax.set_ylabel("Average Glucose Level")
    st.pyplot(fig)

# --- Question 4 ---
with st.expander("🧩 **Question 4: Does marital status influence stroke risk?**"):
    st.markdown("**Why:** Marital status can influence lifestyle, stress levels, and healthcare engagement.")
    stroke_rate = stroke_data.groupby('ever_married')['stroke'].mean().reset_index()
    stroke_rate['stroke'] *= 100

    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=stroke_rate, x='ever_married', y='stroke', palette='crest', ax=ax)
    ax.set_title("Stroke Rate by Marital Status")
    ax.set_ylabel("Stroke Rate (%)")
    ax.set_xlabel("Ever Married")
    for i, row in stroke_rate.iterrows():
        ax.text(i, row['stroke'] + 0.5, f"{row['stroke']:.2f}%", ha='center')
    st.pyplot(fig)

# --- Question 5 ---
with st.expander("🧩 **Question 5: Does smoking increase stroke risk?**"):
    st.markdown("**Why:** Smoking contributes to vascular damage and is a known cardiovascular risk factor.")
    stroke_rate = stroke_data.groupby('smoking_status')['stroke'].mean().reset_index()
    stroke_rate['stroke'] *= 100

    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(data=stroke_rate, x='smoking_status', y='stroke', palette='crest', ax=ax)
    ax.set_title("Stroke Rate by Smoking Status")
    ax.set_ylabel("Stroke Rate (%)")
    ax.set_xlabel("Smoking Status")
    plt.xticks(rotation=25)
    st.pyplot(fig)

# --- Add additional questions ---
# You can easily extend this format up to Q10
# Example placeholders below:
with st.expander("🧩 **Question 6: Does work type correlate with stroke occurrence?**"):
    st.markdown("**Why:** Occupational environment and stress may impact stroke risk.")
    # Add your code here (similar to Q5)
    st.info("Add visualization here...")

with st.expander("🧩 **Question 7: Does gender affect stroke likelihood?**"):
    st.markdown("**Why:** Biological and hormonal differences may affect cardiovascular health.")
    st.info("Add visualization here...")

# --- Summary ---
st.divider()
st.header("📈 Summary")
st.markdown("""
This dashboard demonstrates exploratory findings from the **Stroke Prediction Dataset**.
The visual analyses reveal potential relationships between demographic, lifestyle, and health factors that may contribute to stroke risk.
Further preprocessing and machine learning modeling will be performed in **Milestone 2** to validate and quantify these relationships.
""")
