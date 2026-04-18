import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# 1. Page Config
st.set_page_config(page_title="Vision Analytics AI", page_icon="✨", layout="wide")

# ==========================================
# 🎨 Premium UI/UX: Animations & Refined Glassmorphism
# ==========================================
st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #020617 100%) !important; background-attachment: fixed; }
    [data-testid="stHeader"] { background-color: transparent !important; }
    [data-testid="stHeader"] * { color: #F8FAFC !important; }
    h1, h2, h3, label, p, li { color: #F8FAFC !important; font-family: 'Inter', sans-serif; font-weight: 600 !important; }
    .gradient-text {
        background: linear-gradient(135deg, #A855F7 0%, #38BDF8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 900; font-size: 3.2rem; text-align: center; margin-top: -20px; text-shadow: 0px 4px 20px rgba(168, 85, 247, 0.3);
    }
    @keyframes fadeInUp { 0% { opacity: 0; transform: translateY(30px); } 100% { opacity: 1; transform: translateY(0); } }
    @keyframes pulseGlow { 0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.5); } 70% { box-shadow: 0 0 0 12px rgba(59, 130, 246, 0); } 100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); } }
    [data-testid="stForm"], .metric-card {
        background: rgba(15, 23, 42, 0.45) !important; backdrop-filter: blur(20px); border-radius: 20px; padding: 30px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4); border: 1px solid rgba(255, 255, 255, 0.08); animation: fadeInUp 0.8s ease-out forwards;
    }
    [data-testid="stForm"]:hover, .metric-card:hover { transform: translateY(-5px); box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6); border: 1px solid rgba(255, 255, 255, 0.15); }
    div[data-baseweb="select"] > div, div[data-baseweb="base-input"] > input {
        background-color: #F8FAFC !important; color: #0F172A !important; font-weight: 700 !important; font-size: 15px !important;
        border-radius: 10px !important; border: 2px solid transparent !important; -webkit-text-fill-color: #0F172A !important; box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
    }
    div[role="radiogroup"] {
        display: flex; justify-content: center !important; gap: 8px; background: rgba(15, 23, 42, 0.6); padding: 6px; border-radius: 100px;
        width: fit-content; margin: 0 auto 35px auto; animation: fadeInUp 0.5s ease-out forwards;
    }
    .stRadio [role="radio"] { display: none !important; }
    .stRadio label { background: transparent !important; padding: 10px 30px !important; border-radius: 100px !important; cursor: pointer; margin: 0 !important; border: none !important; }
    .stRadio label:has(input:checked) { background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%) !important; box-shadow: 0 4px 15px rgba(168, 85, 247, 0.5); }
    .stRadio label:has(input:checked) div { color: #FFFFFF !important; font-weight: 800 !important; }
    [data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%) !important; border: none !important; padding: 16px 24px !important;
        border-radius: 12px !important; animation: pulseGlow 2.5s infinite;
    }
    [data-testid="baseButton-secondary"] * { color: #FFFFFF !important; font-weight: 800 !important; text-transform: uppercase; }
    [data-testid="baseButton-secondary"]:hover { transform: translateY(-3px) scale(1.02); animation: none; }
</style>
""", unsafe_allow_html=True)

# 2. Load Models
@st.cache_resource
def load_assets():
    risk_model = joblib.load('risk_model_pipeline.pkl')
    app_model = joblib.load('app_behavior_model_xgb.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return risk_model, app_model, encoder

try:
    risk_model, app_model, encoder = load_assets()
except Exception as e:
    st.error(f"⚠️ Failed to load models. Error: {e}")
    st.stop()

st.markdown("<div class='gradient-text'>Vision Analytics</div>", unsafe_allow_html=True)
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
    sleep_map = {"< 4 hours (Severely Deprived)": 3.0, "4-6 hours (Insufficient)": 5.0, "7-9 hours (Healthy)": 8.0, "> 9 hours (Oversleeping)": 10.0}
    exam_map = {"No Immediate Exams": 1.0, "Manageable Workload": 4.0, "High Academic Stress": 7.5, "Overwhelming Pressure": 10.0}

    with st.form("risk_form"):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            stress = st.selectbox("Stress Level:", list(stress_map.keys()))
            anxiety = st.selectbox("Anxiety Level:", list(anxiety_map.keys()))
            depression = st.selectbox("Mood & Energy:", list(dep_map.keys()))
        with col2:
            support = st.selectbox("Social Support:", list(support_map.keys()))
            sleep = st.selectbox("Daily Sleep:", list(sleep_map.keys()))
            exams = st.selectbox("Academic Workload:", list(exam_map.keys()))
        submit_risk = st.form_submit_button("Initiate AI Analysis", use_container_width=True)

    if submit_risk:
        features = pd.DataFrame([{
            'stress_level': float(stress_map[stress]), 'anxiety_score': float(anxiety_map[anxiety]),
            'depression_score': float(dep_map[depression]), 'social_support': float(support_map[support]),
            'sleep_hours': float(sleep_map[sleep]), 'exam_pressure': float(exam_map[exams])
        }])

        with st.spinner("Processing..."):
            probs = risk_model.predict_proba(features)[0]
            clean_classes = [str(c).strip().title() for c in encoder.classes_]
            prob_dict = {c: p for c, p in zip(clean_classes, probs)}
            
            if prob_dict.get('High', 0.0) >= 0.25: final_label = 'High'
            elif prob_dict.get('Medium', 0.0) >= 0.35: final_label = 'Medium'
            else: final_label = clean_classes[np.argmax(probs)]

        res_col1, res_col2 = st.columns([1, 1.5])
        with res_col1:
            st.markdown('<div class="metric-card" style="text-align: center;">', unsafe_allow_html=True)
            if 'High' in final_label: st.markdown("<h2 style='color:#F43F5E;'>🚨 HIGH RISK</h2>", unsafe_allow_html=True)
            elif 'Medium' in final_label: st.markdown("<h2 style='color:#FBBF24;'>🟡 MEDIUM RISK</h2>", unsafe_allow_html=True)
            else: st.markdown("<h2 style='color:#34D399;'>✅ LOW RISK</h2>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with res_col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            fig = px.bar(x=probs*100, y=clean_classes, orientation='h', color=clean_classes, color_discrete_map={'High':'#F43F5E', 'Medium':'#FBBF24', 'Low':'#34D399'})
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'), height=200, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Page 2: App Behavior Analysis (النسخة الذكية لاكتشاف الأخطاء)
# ------------------------------------------------------------------
else:
    st.markdown("<h3 style='color: #67E8F9 !important;'>📱 App Behavior Tech-Metrics</h3>", unsafe_allow_html=True)
    
    with st.form("app_behavior_form"):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            age = st.number_input("Age:", 10, 100, 20)
            gender = st.selectbox("Gender:", ["Male", "Female"])
            num_apps = st.number_input("Apps Installed:", 0, 500, 50)
        with col2:
            screen_time = st.number_input("Screen Time (hours):", 0.0, 24.0, 5.0)
            battery = st.number_input("Battery Drain (mAh):", 0, 10000, 2000)
            data_usage = st.number_input("Data (MB):", 0, 50000, 1000)
            app_usage = st.number_input("Usage Time (min):", 0, 1440, 300)
        submit_app = st.form_submit_button("ANALYZE USER BEHAVIOR", use_container_width=True)

    if submit_app:
        with st.spinner("Analyzing..."):
            # بنجهز البيانات بـ Gender كنص زي ما هي (عشان Pipeline كولاب)
            raw_data = {
                'App Usage Time (min/day)': [float(app_usage)],
                'Screen On Time (hours/day)': [float(screen_time)],
                'Battery Drain (mAh/day)': [float(battery)],
                'Number of Apps Installed': [float(num_apps)],
                'Data Usage (MB/day)': [float(data_usage)],
                'Age': [float(age)],
                'Gender': [gender] # نص!
            }
            
            df_app = pd.DataFrame(raw_data)
            
            # ترتيب الأعمدة لو الموديل طالب كده
            if hasattr(app_model, 'feature_names_in_'):
                expected_cols = list(app_model.feature_names_in_)
                
                # لو الموديل متدرب بـ OneHotEncoding هنديله اللي هو عايزه
                if 'Gender_Male' in expected_cols:
                    df_app['Gender_Male'] = 1.0 if gender == "Male" else 0.0
                    df_app['Gender_Female'] = 1.0 if gender == "Female" else 0.0
                    if 'Gender' in df_app.columns:
                        df_app = df_app.drop(columns=['Gender'])
                
                # إعادة ترتيب آمنة
                df_app = df_app.reindex(columns=expected_cols, fill_value=0.0)

            try:
                # المحاولة الأولى: زي ما هي
                pred = app_model.predict(df_app)[0]
                success = True
            except Exception as e1:
                try:
                    # المحاولة التانية: لو فشل، هنحول الـ Gender لرقم ونجبر الداتا كلها تبقى أرقام
                    df_app['Gender'] = 1.0 if gender == "Male" else 0.0
                    df_app = df_app.astype(float)
                    pred = app_model.predict(df_app)[0]
                    success = True
                except Exception as e2:
                    success = False
                    st.error("⚠️ الموديل رافض شكل البيانات! دي رسالة الخطأ:")
                    st.code(str(e2))
                    
                    if hasattr(app_model, 'feature_names_in_'):
                        st.warning("🔍 الموديل متدرب ومستني الأعمدة دي بالظبط (انسخها عشان نعرف المشكلة):")
                        st.write(list(app_model.feature_names_in_))
            
            if success:
                st.markdown(f"""
                    <div class="metric-card" style="text-align: center; margin-top:25px;">
                        <h3 style='color:#CBD5E1 !important;'>Predicted Class</h3>
                        <h1 style='color:#22D3EE !important; font-size:3.5rem; font-weight:900;'>{int(pred)}</h1>
                    </div>
                """, unsafe_allow_html=True)
