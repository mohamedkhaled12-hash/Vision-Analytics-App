import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. Page Config
st.set_page_config(page_title="Vision Analytics AI", page_icon="✨", layout="wide")

# ==========================================
# 🎨 Premium UI/UX
# ==========================================
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #020617 100%) !important;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background-color: transparent !important; }
    h1, h2, h3, label, p, li {
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif;
    }
    .gradient-text {
        background: linear-gradient(135deg, #A855F7 0%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3.2rem;
        text-align: center;
        margin-top: -20px;
    }
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stForm"], .metric-card {
        background: rgba(15, 23, 42, 0.45) !important;
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        animation: fadeInUp 0.8s ease-out forwards;
    }
    div[data-baseweb="select"] > div, div[data-baseweb="base-input"] > input {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    .stRadio label:has(input:checked) {
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. Load Models
@st.cache_resource
def load_assets():
    risk_model = joblib.load('risk_model_pipeline.pkl')
    app_model = joblib.load('app_behavior_model_xgb (1).pkl')
    encoder = joblib.load('label_encoder.pkl')
    return risk_model, app_model, encoder

try:
    risk_model, app_model, encoder = load_assets()
except Exception as e:
    st.error(f"⚠️ Failed to load models: {e}")
    st.stop()

# ==========================================
# 🚀 Navigation Header
# ==========================================
st.markdown("<div class='gradient-text'>Vision Analytics</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A3B8 !important;'>Empowered by Advanced Machine Learning</p>", unsafe_allow_html=True)

page = st.radio("", ["Student Risk Analysis", "App Behavior Analysis"], horizontal=True, label_visibility="collapsed")

# ------------------------------------------------------------------
# Page 1: Student Risk Analysis
# ------------------------------------------------------------------
if page == "Student Risk Analysis":
    st.markdown("<h3 style='color: #E9D5FF !important;'>🧠 Student Risk Intelligence</h3>", unsafe_allow_html=True)
    
    stress_map = {"Very Calm": 1.0, "Normal Stress": 4.0, "Highly Stressed": 7.0, "Extremely Stressed": 10.0}
    anxiety_map = {"Stable": 1.0, "Mild Anxiety": 3.5, "Constant Tension": 7.0, "Severe Panic": 10.0}
    support_map = {"Completely Isolated": 1.0, "Limited Support": 4.0, "Good Support": 7.5, "Strong Support": 10.0}
    dep_map = {"Optimistic & Energetic": 1.0, "Occasional Sadness": 4.5, "Frequent Low Mood": 7.5, "Severe Despair": 10.0}
    sleep_map = {"< 4 hours": 3.0, "4-6 hours": 5.0, "7-9 hours": 8.0, "> 9 hours": 10.0}
    exam_map = {"No Immediate Exams": 1.0, "Manageable Workload": 4.0, "High Academic Stress": 7.5, "Overwhelming Pressure": 10.0}

    with st.form("risk_form"):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            stress = st.selectbox("Stress Level:", options=list(stress_map.keys()))
            anxiety = st.selectbox("Anxiety Level:", options=list(anxiety_map.keys()))
            depression = st.selectbox("Mood & Energy Levels:", options=list(dep_map.keys()))
        with col2:
            support = st.selectbox("Social Support Network:", list(support_map.keys()))
            sleep = st.selectbox("Average Daily Sleep:", list(sleep_map.keys()))
            exams = st.selectbox("Academic Workload:", options=list(exam_map.keys()))
        
        submit_risk = st.form_submit_button("Initiate AI Analysis", use_container_width=True)

    if submit_risk:
        features = pd.DataFrame([[
            stress_map[stress], anxiety_map[anxiety], dep_map[depression],
            support_map[support], sleep_map[sleep], exam_map[exams]
        ]], columns=['stress_level', 'anxiety_score', 'depression_score', 'social_support', 'sleep_hours', 'exam_pressure'])

        probs = risk_model.predict_proba(features)[0]
        clean_classes = [str(c).strip().title() for c in encoder.classes_]
        
        prob_dict = dict(zip(clean_classes, probs))
        if prob_dict.get('High', 0) >= 0.25: final_label = 'High'
        elif prob_dict.get('Medium', 0) >= 0.35: final_label = 'Medium'
        else: final_label = clean_classes[np.argmax(probs)]

        res_col1, res_col2 = st.columns([1, 1.5], gap="large")
        with res_col1:
            st.markdown('<div class="metric-card" style="text-align: center;">', unsafe_allow_html=True)
            color = "#F43F5E" if "High" in final_label else "#FBBF24" if "Medium" in final_label else "#34D399"
            st.markdown(f"<h2 style='color:{color} !important;'>{final_label.upper()} RISK</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with res_col2:
            fig = px.bar(x=probs*100, y=clean_classes, orientation='h', title="AI Confidence %")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", height=200)
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Page 2: App Behavior Analysis (MODIFIED TO FIX FLOAT ERROR)
# ------------------------------------------------------------------
else:
    st.markdown("<h3 style='color: #67E8F9 !important;'>📱 App Behavior Tech-Metrics</h3>", unsafe_allow_html=True)

    with st.expander("💡 Guide: How to find these metrics?"):
        st.write("Check Digital Wellbeing (Android) or Screen Time (iOS) in your settings.")

    with st.form("app_behavior_form"):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            age = st.number_input("Age:", 10, 100, 20)
            gender = st.selectbox("Gender:", ["Male", "Female"])
            num_apps = st.number_input("Number of Apps Installed:", 0, 500, 50)
        with col2:
            screen_time = st.number_input("Screen On Time (hours/day):", 0.0, 24.0, 5.0)
            battery = st.number_input("Battery Drain (mAh/day):", 0, 10000, 2000)
            data_usage = st.number_input("Data Usage (MB/day):", 0, 50000, 1000)
            app_usage = st.number_input("App Usage Time (min/day):", 0, 1440, 300)

        submit_app = st.form_submit_button("ANALYZE USER BEHAVIOR", use_container_width=True)

    if submit_app:
        try:
            # 1. تحديد أسماء الأعمدة المتوقعة
            if hasattr(app_model, 'feature_names_in_'):
                expected_cols = list(app_model.feature_names_in_)
            else:
                expected_cols = ['App Usage Time (min/day)', 'Screen On Time (hours/day)', 
                                 'Battery Drain (mAh/day)', 'Number of Apps Installed', 
                                 'Data Usage (MB/day)', 'Age', 'Gender']

            # 2. تحويل الجنس لرقم (0 أو 1) فوراً لمنع خطأ float64
            gender_val = 1.0 if gender == "Male" else 0.0

            # 3. تجهيز قاموس البيانات مع التأكد من النوع float
            input_data = {
                'app usage time': float(app_usage),
                'screen on time': float(screen_time),
                'battery drain': float(battery),
                'number of apps': float(num_apps),
                'data usage': float(data_usage),
                'age': float(age),
                'gender': gender_val
            }

            # 4. بناء الـ DataFrame بناءً على ترتيب الموديل
            df_app = pd.DataFrame(columns=expected_cols)
            df_app.loc[0] = 0.0

            for col in expected_cols:
                for key, val in input_data.items():
                    if key in col.lower():
                        df_app.at[0, col] = val
            
            # التأكيد على أن كل القيم أرقام
            df_app = df_app.apply(pd.to_numeric)

            with st.spinner("Analyzing..."):
                pred = app_model.predict(df_app)[0]
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; margin-top:25px;">
                    <h3 style='color:#CBD5E1;'>Predicted Class</h3>
                    <h1 style='color:#22D3EE; font-size:3.5rem;'>{pred}</h1>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
