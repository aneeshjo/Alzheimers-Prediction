# ============================================================
# Alzheimer's Disease Prediction System
# ============================================================

# ============================================================
# Import Libraries
# ============================================================

import streamlit as st
import pandas as pd

from src.pipeline.predict_pipeline import (
    PredictPipeline,
    CustomData
)

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Alzheimer's Disease Prediction",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Load Prediction Pipeline
# ============================================================

@st.cache_resource
def load_pipeline():
    return PredictPipeline()

pipeline = load_pipeline()

# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>

/* ===========================================
Main Page
=========================================== */

.stApp{

    background:#0E1117;

}

/* ===========================================
Header
=========================================== */

.main-title{

    font-size:42px;

    font-weight:700;

    color:white;

    margin-bottom:10px;

}

.sub-title{

    color:#B0B3B8;

    font-size:18px;

    margin-bottom:35px;

}

/* ===========================================
Sidebar
=========================================== */

[data-testid="stSidebar"]{

    background:#111827;

}

[data-testid="stSidebar"] h1{

    color:white;

}

[data-testid="stSidebar"] h2{

    color:white;

}

[data-testid="stSidebar"] h3{

    color:white;

}

/* ===========================================
Tabs
=========================================== */

button[data-baseweb="tab"]{

    font-size:17px;

    font-weight:600;

}

/* ===========================================
Predict Button
=========================================== */

.stButton>button{

    width:100%;

    height:55px;

    border-radius:12px;

    font-size:20px;

    font-weight:bold;

    background:linear-gradient(90deg,#8E2DE2,#4A00E0);

    color:white;

    border:none;

}

.stButton>button:hover{

    background:linear-gradient(90deg,#9D50FF,#6E48AA);

}

/* ===========================================
Result Cards
=========================================== */

.card{

    border-radius:18px;

    padding:25px;

    color:white;

    min-height:220px;

    box-shadow:0px 6px 15px rgba(0,0,0,.35);

}

/* Red */

.red{

    background:linear-gradient(135deg,#43171d,#241114);

    border:1px solid #ff4b4b;

}

/* Blue */

.blue{

    background:linear-gradient(135deg,#10294a,#121827);

    border:1px solid #2d8cff;

}

/* Orange */

.orange{

    background:linear-gradient(135deg,#443014,#221710);

    border:1px solid #ffb020;

}

.card-title{

    font-size:20px;

    color:#d0d0d0;

}

.card-value{

    font-size:36px;

    font-weight:700;

    margin-top:18px;

}

.card-desc{

    margin-top:20px;

    color:#d9d9d9;

    font-size:17px;

}

/* ===========================================
Recommendation Card
=========================================== */

.recommend-card{

    background:#0F2D20;

    border:1px solid #16C784;

    border-radius:18px;

    padding:25px;

}

/* ===========================================
Alert Card
=========================================== */

.alert-card{

    background:#401C1F;

    border:1px solid #ff4d4f;

    border-radius:16px;

    padding:18px;

    color:white;

    font-size:18px;

}

/* ===========================================
Patient Summary
=========================================== */

.summary-card{

    background:#151A23;

    border-radius:15px;

    padding:25px;

    border:1px solid #2A2E39;

}

</style>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar
# ============================================================

st.sidebar.markdown("# 🧠 Alzheimer's")

st.sidebar.markdown("## Prediction System")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📋 Instructions")

st.sidebar.write("1. Enter patient information.")

st.sidebar.write("2. Navigate through the tabs.")

st.sidebar.write("3. Click Predict Alzheimer's Risk.")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📂 Navigation")

st.sidebar.write("👤 Demographics")

st.sidebar.write("🏃 Lifestyle Factors")

st.sidebar.write("🏥 Medical History")

st.sidebar.write("🩺 Clinical Measurements")

st.sidebar.write("🧠 Cognitive Assessment")

st.sidebar.markdown("---")

st.sidebar.markdown("### 👨‍💻 Developed By")

st.sidebar.write("**Aneesh Jose**")

st.sidebar.write("Machine Learning Model")

st.sidebar.success("LightGBM")

# ============================================================
# Main Header
# ============================================================

st.markdown(
"""
<div class='main-title'>

🧠 Alzheimer's Disease Prediction System

</div>

<div class='sub-title'>

Predict the likelihood of Alzheimer's Disease using
a trained LightGBM Machine Learning model.

</div>
""",
unsafe_allow_html=True
)

# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([

    "👤 Demographics",

    "🏃 Lifestyle",

    "🏥 Medical History",

    "🩺 Clinical Measurements",

    "🧠 Cognitive Assessment"

])

# ============================================================
# 👤 Demographics
# ============================================================

with tab1:

    st.subheader("👤 Demographic Information")

    st.caption("Enter the patient's demographic information.")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=40,
            max_value=100,
            value=65
        )

    with col2:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

    with col3:

        ethnicity = st.selectbox(
            "Ethnicity",
            [
                "White",
                "Black",
                "Asian",
                "Hispanic"
            ]
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        education_level = st.selectbox(
            "Education Level",
            [
                "No Formal Education",
                "Primary",
                "Secondary",
                "Higher"
            ]
        )

    with col2:

        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=10.0,
            max_value=50.0,
            value=24.5,
            step=0.1
        )

    # Encode values

    gender = 1 if gender == "Male" else 0

    ethnicity = {

        "White":0,

        "Black":1,

        "Asian":2,

        "Hispanic":3

    }[ethnicity]

    education_level = {

        "No Formal Education":0,

        "Primary":1,

        "Secondary":2,

        "Higher":3

    }[education_level]

# ============================================================
# 🏃 Lifestyle Factors
# ============================================================

with tab2:

    st.subheader("🏃 Lifestyle Factors")

    col1, col2, col3 = st.columns(3)

    with col1:

        smoking = st.toggle("Smoking")

    with col2:

        alcohol_consumption = st.number_input(

            "Alcohol Consumption",

            min_value=0.0,

            max_value=20.0,

            value=5.0,

            step=0.1

        )

    with col3:

        physical_activity = st.number_input(

            "Physical Activity",

            min_value=0.0,

            max_value=10.0,

            value=5.0,

            step=0.1

        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        diet_quality = st.number_input(

            "Diet Quality",

            min_value=0.0,

            max_value=10.0,

            value=5.0,

            step=0.1

        )

    with col2:

        sleep_quality = st.number_input(

            "Sleep Quality",

            min_value=0.0,

            max_value=10.0,

            value=5.0,

            step=0.1

        )

    smoking = int(smoking)
# ============================================================
# 🏥 Medical History
# ============================================================

with tab3:

    st.subheader("🏥 Medical History")

    st.caption(
        "Select the patient's known medical conditions and family history."
    )

    # --------------------------------------------------------
    # Row 1
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        family_history_alzheimers = st.toggle(
            "Family History of Alzheimer's"
        )

    with col2:

        cardiovascular_disease = st.toggle(
            "Cardiovascular Disease"
        )

    with col3:

        diabetes = st.toggle(
            "Diabetes"
        )

    # --------------------------------------------------------
    # Row 2
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        depression = st.toggle(
            "Depression"
        )

    with col2:

        head_injury = st.toggle(
            "Head Injury"
        )

    with col3:

        hypertension = st.toggle(
            "Hypertension"
        )

    # --------------------------------------------------------
    # Convert Boolean to Integer
    # --------------------------------------------------------

    family_history_alzheimers = int(
        family_history_alzheimers
    )

    cardiovascular_disease = int(
        cardiovascular_disease
    )

    diabetes = int(
        diabetes
    )

    depression = int(
        depression
    )

    head_injury = int(
        head_injury
    )

    hypertension = int(
        hypertension
    )
# ============================================================
# 🩺 Clinical Measurements
# ============================================================

with tab4:

    st.subheader("🩺 Clinical Measurements")

    st.caption(
        "Enter the patient's vital signs and laboratory measurements."
    )

    # --------------------------------------------------------
    # Blood Pressure
    # --------------------------------------------------------

    st.markdown("#### 🩸 Blood Pressure")

    col1, col2 = st.columns(2)

    with col1:

        systolic_bp = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=80,
            max_value=220,
            value=120,
            step=1
        )

    with col2:

        diastolic_bp = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=40,
            max_value=140,
            value=80,
            step=1
        )

    st.divider()

    # --------------------------------------------------------
    # Cholesterol Profile
    # --------------------------------------------------------

    st.markdown("#### 🧪 Cholesterol Profile")

    col1, col2, col3 = st.columns(3)

    with col1:

        cholesterol_total = st.number_input(
            "Total Cholesterol",
            min_value=100.0,
            max_value=400.0,
            value=200.0,
            step=1.0
        )

        cholesterol_ldl = st.number_input(
            "LDL Cholesterol",
            min_value=20.0,
            max_value=300.0,
            value=120.0,
            step=1.0
        )

    with col2:

        cholesterol_hdl = st.number_input(
            "HDL Cholesterol",
            min_value=10.0,
            max_value=120.0,
            value=50.0,
            step=1.0
        )

        cholesterol_triglycerides = st.number_input(
            "Triglycerides",
            min_value=50.0,
            max_value=500.0,
            value=150.0,
            step=1.0
        )

    with col3:

        st.info(
            """
**Reference Ranges**

🩸 Total Cholesterol  
< 200 mg/dL

🟢 HDL  
> 40 mg/dL

🔴 LDL  
< 100 mg/dL

🟡 Triglycerides  
< 150 mg/dL
"""
        )
# ============================================================
# 🧠 Cognitive Assessment
# ============================================================

with tab5:

    st.subheader("🧠 Cognitive Assessment")

    st.caption(
        "Enter the patient's cognitive assessment scores and observed symptoms."
    )

    # --------------------------------------------------------
    # Assessment Scores
    # --------------------------------------------------------

    st.markdown("#### 📊 Assessment Scores")

    col1, col2, col3 = st.columns(3)

    with col1:

        mmse = st.number_input(
            "MMSE Score",
            min_value=0.0,
            max_value=30.0,
            value=20.0,
            step=0.1
        )

    with col2:

        functional_assessment = st.number_input(
            "Functional Assessment",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

    with col3:

        adl = st.number_input(
            "Activities of Daily Living (ADL)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )

    st.divider()

    # --------------------------------------------------------
    # Cognitive Symptoms
    # --------------------------------------------------------

    st.markdown("#### 📝 Cognitive Symptoms")

    col1, col2 = st.columns(2)

    # ---------------- Left Column ----------------

    with col1:

        memory_complaints = st.toggle(
            "Memory Complaints"
        )

        behavioral_problems = st.toggle(
            "Behavioral Problems"
        )

        confusion = st.toggle(
            "Confusion"
        )

        disorientation = st.toggle(
            "Disorientation"
        )

    # ---------------- Right Column ----------------

    with col2:

        personality_changes = st.toggle(
            "Personality Changes"
        )

        difficulty_completing_tasks = st.toggle(
            "Difficulty Completing Tasks"
        )

        forgetfulness = st.toggle(
            "Forgetfulness"
        )

    # --------------------------------------------------------
    # Convert Boolean Values to Integers
    # --------------------------------------------------------

    memory_complaints = int(memory_complaints)

    behavioral_problems = int(behavioral_problems)

    confusion = int(confusion)

    disorientation = int(disorientation)

    personality_changes = int(personality_changes)

    difficulty_completing_tasks = int(
        difficulty_completing_tasks
    )

    forgetfulness = int(forgetfulness)

    st.divider()

    st.info(
        """
**Clinical Notes**

• **MMSE** evaluates cognitive function (0–30).

• **Functional Assessment** measures the patient's ability to perform daily tasks independently.

• **ADL** (Activities of Daily Living) reflects the patient's capacity for routine self-care activities.

These assessments are commonly used alongside clinical evaluation and should not be interpreted in isolation.
"""
    )

# ============================================================
# Create User Input Dictionary
# ============================================================

user_inputs = {

    # Demographics
    "Age": age,
    "Gender": gender,
    "Ethnicity": ethnicity,
    "EducationLevel": education_level,

    # Lifestyle
    "BMI": bmi,
    "Smoking": smoking,
    "AlcoholConsumption": alcohol_consumption,
    "PhysicalActivity": physical_activity,
    "DietQuality": diet_quality,
    "SleepQuality": sleep_quality,

    # Medical History
    "FamilyHistoryAlzheimers": family_history_alzheimers,
    "CardiovascularDisease": cardiovascular_disease,
    "Diabetes": diabetes,
    "Depression": depression,
    "HeadInjury": head_injury,
    "Hypertension": hypertension,

    # Clinical Measurements
    "SystolicBP": systolic_bp,
    "DiastolicBP": diastolic_bp,
    "CholesterolTotal": cholesterol_total,
    "CholesterolLDL": cholesterol_ldl,
    "CholesterolHDL": cholesterol_hdl,
    "CholesterolTriglycerides": cholesterol_triglycerides,

    # Cognitive Assessment
    "MMSE": mmse,
    "FunctionalAssessment": functional_assessment,
    "ADL": adl,

    # Symptoms
    "MemoryComplaints": memory_complaints,
    "BehavioralProblems": behavioral_problems,
    "Confusion": confusion,
    "Disorientation": disorientation,
    "PersonalityChanges": personality_changes,
    "DifficultyCompletingTasks": difficulty_completing_tasks,
    "Forgetfulness": forgetfulness
}
st.markdown("---")

predict = st.button(
    "🧠 Predict Alzheimer's Disease",
    use_container_width=True
)
if predict:

    try:

        # ============================================================
        # Convert Input into DataFrame
        # ============================================================

        custom_data = CustomData(user_inputs)
        prediction_df = custom_data.get_data_as_dataframe()

        # ============================================================
        # Make Prediction
        # ============================================================

        result = pipeline.predict(prediction_df)

        prediction = result["prediction"]
        probability = result["probability"]
        risk = result["risk_level"]

        # ============================================================
        # Prediction Text
        # ============================================================

        if prediction == 1:

            prediction_title = "⚠ High Likelihood"

            prediction_description = (
                "The model predicts a high likelihood "
                "of Alzheimer's Disease."
            )

        else:

            prediction_title = "✅ Low Likelihood"

            prediction_description = (
                "The model predicts a low likelihood "
                "of Alzheimer's Disease."
            )

        # ============================================================
        # AI Diagnosis Summary
        # ============================================================

        st.markdown("---")

        st.header("🧠 AI Diagnosis Summary")

        col1, col2, col3 = st.columns(3)

        # --------------------------------------------------------
        # Prediction
        # --------------------------------------------------------

        with col1:

            st.metric(
                "Prediction",
                prediction_title
            )

        # --------------------------------------------------------
        # Probability
        # --------------------------------------------------------

        with col2:

            st.metric(
                "Probability",
                f"{probability:.2%}"
            )

        # --------------------------------------------------------
        # Risk
        # --------------------------------------------------------

        with col3:

            st.metric(
                "Risk Level",
                risk
            )

        # ============================================================
        # Confidence
        # ============================================================

        st.markdown("---")

        st.subheader("📈 Prediction Confidence")

        st.progress(float(probability))

        st.write(
            f"Confidence Score : **{probability:.2%}**"
        )

        # ============================================================
        # Alert
        # ============================================================

        if prediction == 1:

            st.error(
                """
⚠ High likelihood of Alzheimer's Disease.

This prediction is generated by the AI model and
should not replace professional medical diagnosis.
"""
            )

        else:

            st.success(
                """
✅ Low likelihood of Alzheimer's Disease.
"""
            )

        # ============================================================
        # Recommendations
        # ============================================================

        st.markdown("---")

        st.subheader("📋 Clinical Recommendations")

        if prediction == 1:

            st.warning("""
• Consult a Neurologist

• Perform a Cognitive Assessment

• Review Current Medication

• Encourage Healthy Lifestyle

• Monitor Cognitive Function
""")

        else:

            st.success("""
• Maintain Healthy Lifestyle

• Continue Regular Medical Check-ups

• Stay Physically Active

• Monitor Cognitive Health
""")

        # ============================================================
        # Patient Summary
        # ============================================================

        st.markdown("---")

        st.subheader("👤 Patient Summary")

        summary1, summary2, summary3 = st.columns(3)

        with summary1:

            st.metric("Age", age)
            st.metric("BMI", bmi)
            st.metric("MMSE", mmse)

        with summary2:

            st.metric(
                "Gender",
                "Male" if gender else "Female"
            )

            st.metric(
                "Blood Pressure",
                f"{systolic_bp}/{diastolic_bp}"
            )

            st.metric(
                "ADL",
                adl
            )

        with summary3:

            st.metric(
                "Functional Assessment",
                functional_assessment
            )

            st.metric(
                "Sleep Quality",
                sleep_quality
            )

            st.metric(
                "Diet Quality",
                diet_quality
            )

        # ============================================================
        # Disclaimer
        # ============================================================

        st.markdown("---")

        st.info("""
### ⚠ Medical Disclaimer

This application is intended for educational purposes only.

Predictions generated by this model should not
be considered as a medical diagnosis.

Always consult a qualified healthcare professional.
""")

        st.markdown("---")

        st.caption("""
🧠 Alzheimer's Disease Prediction System

Developed by **Aneesh Jose**

Machine Learning Model : **LightGBM**
""")

    except Exception as e:

        st.error(f"Prediction Failed\n\n{str(e)}")