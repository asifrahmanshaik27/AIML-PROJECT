import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="Asif's Salary Predictor",
    page_icon="💼",
    layout="wide"
)

# ---------------- DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 15% 10%, rgba(94, 234, 212, 0.18), transparent 28%),
        radial-gradient(circle at 85% 15%, rgba(139, 92, 246, 0.24), transparent 30%),
        linear-gradient(135deg, #071225 0%, #111b3d 50%, #1c1240 100%);
    color: #f8fafc;
}

.block-container {
    max-width: 1000px;
    padding-top: 4rem;
    padding-bottom: 3rem;
}

.app-title {
    font-size: 3.2rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #67e8f9, #c4b5fd, #f9a8d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 1.15rem;
    margin-bottom: 2rem;
}

.glass-card {
    background: rgba(255, 255, 255, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 22px;
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    padding: 28px;
    margin: 20px 0;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
}

.metric-card {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.20);
    border-radius: 16px;
    padding: 16px;
    text-align: center;
}

.metric-label {
    color: #cbd5e1;
    font-size: 0.9rem;
}

.metric-value {
    color: white;
    font-size: 1.6rem;
    font-weight: bold;
    margin-top: 5px;
}

.stButton > button {
    width: 100%;
    border: none;
    border-radius: 12px;
    padding: 0.8rem;
    font-size: 1.05rem;
    font-weight: bold;
    color: #071225;
    background: linear-gradient(90deg, #67e8f9, #c4b5fd);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(103, 232, 249, 0.30);
}

label, p, .stMarkdown {
    color: #e2e8f0 !important;
}

.stTextInput input {
    background: rgba(255, 255, 255, 0.12) !important;
    color: white !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- PAGE STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------------- LOAD AND TRAIN MODEL ----------------
@st.cache_resource
def train_model():
    df = pd.read_csv("preprocessed_data.csv")

    features = [
        "Age",
        "Experience_Years",
        "Education_Level",
        "Job_Level",
        "Department",
        "Performance_Score",
        "Weekly_Working_Hours"
    ]

    X = df[features]
    y = df["Salary_LPA"]

    categorical_columns = [
        "Education_Level",
        "Job_Level",
        "Department"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            )
        ],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("random_forest", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return df, model, mae, rmse, r2

# ---------------- HOME PAGE ----------------
if st.session_state.page == "home":

    st.markdown(
        '<h1 class="app-title">💼 Asif\'s Salary Predictor</h1>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Predict employee salaries using machine learning.</p>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="glass-card">
        <h2>Welcome 👋</h2>
        <p>
            This web application predicts an employee's annual salary in
            <b>LPA (Lakhs Per Annum)</b>.
        </p>
        <p>
            The model uses information such as age, work experience, education,
            job level, department, performance score, and weekly working hours.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h2>🤖 Model Used</h2>
        <p>
            This application uses <b>Random Forest Regression</b>, a machine
            learning model that combines many decision trees to make more
            reliable salary predictions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("✨ Predict New Employee Salary"):
        st.session_state.page = "predict"
        st.rerun()

# ---------------- PREDICTION PAGE ----------------
elif st.session_state.page == "predict":

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown(
        '<h1 class="app-title">Salary Prediction</h1>',
        unsafe_allow_html=True
    )

    df, model, mae, rmse, r2 = train_model()

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Average Error</div>'
            f'<div class="metric-value">{mae:.2f} LPA</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">RMSE</div>'
            f'<div class="metric-value">{rmse:.2f} LPA</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">R² Score</div>'
            f'<div class="metric-value">{r2:.2f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔮 Enter Employee Details")

    col1, col2 = st.columns(2)

    with col1:
        age_text = st.text_input("Age", placeholder="Example: 25")
        experience_text = st.text_input(
            "Experience Years",
            placeholder="Example: 3"
        )
        education = st.text_input(
            "Education Level",
            placeholder="Example: Bachelor's"
        )
        job_level = st.text_input(
            "Job Level",
            placeholder="Example: Junior"
        )

    with col2:
        department = st.text_input(
            "Department",
            placeholder="Example: IT"
        )
        performance_text = st.text_input(
            "Performance Score",
            placeholder="Example: 4.5"
        )
        hours_text = st.text_input(
            "Weekly Working Hours",
            placeholder="Example: 40"
        )

    if st.button("✨ Predict Salary"):
        try:
            age = int(age_text)
            experience = float(experience_text)
            performance = float(performance_text)
            hours = float(hours_text)

            new_employee = pd.DataFrame([{
                "Age": age,
                "Experience_Years": experience,
                "Education_Level": education,
                "Job_Level": job_level,
                "Department": department,
                "Performance_Score": performance,
                "Weekly_Working_Hours": hours
            }])

            predicted_salary = model.predict(new_employee)[0]

            st.success(
                f"🎉 Predicted Employee Salary: ₹ {predicted_salary:.2f} LPA"
            )

        except ValueError:
            st.error(
                "Enter valid numbers for Age, Experience, "
                "Performance Score, and Working Hours."
            )

    st.markdown("</div>", unsafe_allow_html=True)