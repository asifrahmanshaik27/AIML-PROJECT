import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Asif's Salary Predictor",
    page_icon="💼",
    layout="wide"
)


# =========================================================
# DESIGN
# =========================================================

st.markdown("""
<style>

.app-title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    margin-bottom: 35px;
}

.section-title {
    font-size: 28px;
    font-weight: bold;
    margin-top: 25px;
    margin-bottom: 15px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
    border: 1px solid rgba(128,128,128,0.3);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PAGE STATE
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


# =========================================================
# TRAIN MODEL
# =========================================================

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

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                )
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    r2 = r2_score(y_test, y_pred)

    return model, mae, rmse, r2


# =========================================================
# HOME PAGE
# =========================================================

if st.session_state.page == "home":

    st.markdown(
        "<h1 class='app-title'>💼 Asif's Salary Predictor</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='subtitle'>"
        "Predict employee salaries using Random Forest Machine Learning."
        "</p>",
        unsafe_allow_html=True
    )

    st.subheader("Welcome 👋")

    st.write(
        "This application predicts an employee's annual salary "
        "in LPA (Lakhs Per Annum)."
    )

    st.write(
        "Enter employee information such as age, experience, "
        "education, job level, department, performance score "
        "and weekly working hours."
    )

    st.subheader("🤖 Model Used")

    st.write(
        "Random Forest Regression"
    )

    st.write(
        "Random Forest combines multiple decision trees "
        "to make salary predictions."
    )

    st.write("")

    if st.button(
        "✨ Predict New Employee Salary",
        use_container_width=True
    ):

        st.session_state.page = "predict"
        st.rerun()


# =========================================================
# PREDICTION PAGE
# =========================================================

elif st.session_state.page == "predict":

    # Back button
    if st.button("← Back to Home"):

        st.session_state.page = "home"
        st.rerun()


    st.markdown(
        "<h1 class='app-title'>Salary Prediction</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='subtitle'>"
        "Enter employee details to predict salary."
        "</p>",
        unsafe_allow_html=True
    )


    # =====================================================
    # LOAD MODEL
    # =====================================================

    try:

        model, mae, rmse, r2 = train_model()

    except FileNotFoundError:

        st.error(
            "preprocessed_data.csv was not found. "
            "Please upload it to the same folder as app.py."
        )

        st.stop()

    except Exception as e:

        st.error(
            f"Error loading the model: {e}"
        )

        st.stop()


    # =====================================================
    # MODEL PERFORMANCE - ABOVE EMPLOYEE DETAILS
    # =====================================================

    st.markdown(
        "<div class='section-title'>📊 Model Performance</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            label="Average Error",
            value=f"{mae:.2f} LPA"
        )

    with col2:

        st.metric(
            label="RMSE",
            value=f"{rmse:.2f} LPA"
        )

    with col3:

        st.metric(
            label="R² Score",
            value=f"{r2:.2f}"
        )


    st.divider()


    # =====================================================
    # EMPLOYEE DETAILS
    # =====================================================

    st.markdown(
        "<div class='section-title'>👤 Employee Details</div>",
        unsafe_allow_html=True
    )


    # Employee Name FIRST

    employee_name = st.text_input(
        "Employee Name",
        placeholder="Enter employee name"
    )


    st.markdown("### 🔮 Enter Employee Information")


    col1, col2 = st.columns(2)


    # =====================================================
    # NUMBER INPUTS
    # =====================================================

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=70,
            value=25,
            step=1
        )

        experience = st.number_input(
            "Experience Years",
            min_value=0.0,
            max_value=50.0,
            value=3.0,
            step=0.5
        )

        performance = st.number_input(
            "Performance Score",
            min_value=0.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )

        hours = st.number_input(
            "Weekly Working Hours",
            min_value=10.0,
            max_value=100.0,
            value=40.0,
            step=1.0
        )


    # =====================================================
    # DROPDOWNS
    # =====================================================

    with col2:

        education = st.selectbox(
            "Education Level",
            [
                "Bachelor",
                "Master",
                "PhD"
            ]
        )

        job_level = st.selectbox(
            "Job Level",
            [
                1,
                2,
                3,
                4,
                5,
                6
            ]
        )

        department = st.selectbox(
            "Department",
            [
                "HR",
                "Marketing",
                "Finance",
                "IT",
                "Operations"
            ]
        )


    st.write("")


    # =====================================================
    # PREDICT BUTTON
    # =====================================================

    if st.button(
        "✨ Predict Salary",
        use_container_width=True
    ):

        if employee_name.strip() == "":

            st.warning(
                "⚠️ Please enter the employee name."
            )

        else:

            new_employee = pd.DataFrame([
                {
                    "Age": age,
                    "Experience_Years": experience,
                    "Education_Level": education,
                    "Job_Level": job_level,
                    "Department": department,
                    "Performance_Score": performance,
                    "Weekly_Working_Hours": hours
                }
            ])


            predicted_salary = model.predict(
                new_employee
            )[0]


            # =================================================
            # CONGRATULATIONS
            # =================================================

            st.balloons()

            st.markdown(
                f"""
                <div class="result-box">

                <h2>🎉 Congratulations, {employee_name}!</h2>

                <p style="font-size:20px;">
                Based on the information you provided,
                you are eligible for an estimated salary of:
                </p>

                <h1>
                ₹ {predicted_salary:.2f} LPA
                </h1>

                <p>
                🌟 Keep improving your skills and performance!
                </p>

                </div>
                """,
                unsafe_allow_html=True
            )