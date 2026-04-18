import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# 1. Page Config
st.set_page_config(page_title="Vision Analytics AI", page_icon="✨", layout="wide")

# ==========================================
# 🎨 Premium UI/UX: Animations & Glow
# ==========================================
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #020617 100%) !important;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background-color: transparent !important; }
    h1, h2, h3, label, p, li { color: #F8FAFC !important; font-family: 'Inter', sans-serif; }
    .gradient-text {
        background: linear-gradient(135deg, #A855F7 0%, #38BDF8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900; font-size: 3.2rem; text-align: center;
        text-shadow: 0px 4px 20px rgba(168, 85, 247, 0.3);
    }
    [data-testid="stForm"], .metric-card {
        background: rgba(15, 23, 42, 0.45) !important;
        backdrop-filter: blur(20px); border-radius: 20px; padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 100%) !important;
        border: none !important; padding: 16px 24px !important; border-radius: 12px !important;
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
    st.error(f"⚠️ Model Load Error: {e}")
    st.stop()

st.markdown("<div class='gradient-text'>Vision Analytics</div>", unsafe_allow_html=True)
page = st.radio("", ["Student Risk Analysis", "App Behavior Analysis"], horizontal=True, label_visibility="collapsed")

# ------------------------------------------------------------------
# Page 1: Student Risk
# ------------------------------------------------------------------
if page == "Student Risk Analysis":
    st.markdown("<h3>🧠 Student Risk Intelligence</h3>", unsafe_allow_html=True)
    stress_map = {"Very Calm": 1.0, "Normal Stress": 4.0, "Highly Stressed": 7.0, "Extremely Stressed": 10.0}
    anxiety_map = {"Stable": 1.0, "Mild Anxiety": 3.5, "Constant Tension": 7.0, "Severe Panic": 10.0}
    support_map = {"Isolated": 1.0, "Limited": 4.0, "Good": 7.5, "Strong": 10.0}
    dep_map = {"Optimistic": 1.0, "Occasional Sadness": 4.5, "Frequent Low": 7.5, "Severe Despair": 10.0}
    sleep_map = {"< 4h": 3.0, "4-6h": 5.0, "7-9h": 8.0, "> 9h": 10.0}
    exam_map = {"No Exams": 1.0, "Manageable": 4.0, "High Stress": 7.5, "Overwhelming": 10.0}

    with st.form("risk_form"):
        c1, c2 = st.columns(2)
        with c1:
            stress = st.selectbox("Stress:", list(stress_map.keys()))
            anxiety = st.selectbox("Anxiety:", list(anxiety_map.keys()))
            depression = st.selectbox("Mood:", list(dep_map.keys()))
        with c2:
            support = st.selectbox("Support:", list(support_map.keys()))
            sleep = st.selectbox("Sleep:", list(sleep_map.keys()))
            exams = st.selectbox("Exams:", list(exam_map.keys()))
        if st.form_submit_button("Initiate AI Analysis", use_container_width=True):
            features = pd.DataFrame([[stress_map[stress], anxiety_map[anxiety], dep_map[depression], support_map[support], sleep_map[sleep], exam_map[exams]]], 
                                     columns=['stress_level', 'anxiety_score', 'depression_score', 'social_support', 'sleep_hours', 'exam_pressure'])
            probs = risk_model.predict_proba(features)[0]
            clean_classes = [str(c).strip().title() for c in encoder.classes_]
            prob_dict = {c: p for c, p in zip(clean_classes, probs)}
            if prob_dict.get('High', 0.0) >= 0.25: final_label = 'High'
            elif prob_dict.get('Medium', 0.0) >= 0.35: final_label = 'Medium'
            else: final_label = clean_classes[np.argmax(probs)]
            
            res_c1, res_c2 = st.columns([1, 1.5])
            with res_c1:
                st.markdown(f"<div class='metric-card' style='text-align:center;'><h2>{final_label.upper()} RISK</h2></div>", unsafe_allow_html=True)
            with res_c2:
                fig = px.bar(x=probs*100, y=clean_classes, orientation='h', color=clean_classes, color_discrete_map={'High':'#F43F5E','Medium':'#FBBF24','Low':'#34D399'})
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#F8FAFC'), height=200)
                st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Page 2: App Behavior (The Final Fix)
# ------------------------------------------------------------------
else:
    st.markdown("<h3 style='color: #67E8F9 !important;'>📱 App Behavior Tech-Metrics</h3>", unsafe_allow_html=True)
    with st.form("app_behavior_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age:", 10, 100, 20)
            gender = st.selectbox("Gender:", ["Male", "Female"])
            num_apps = st.number_input("Apps:", 0, 500, 50)
        with c2:
            screen_time = st.number_input("Screen Time (h):", 0.0, 24.0, 5.0)
            battery = st.number_input("Battery (mAh):", 0, 10000, 2000)
            data_usage = st.number_input("Data (MB):", 0, 50000, 1000)
            app_usage = st.number_input("Usage (min):", 0, 1440, 300)
        
        if st.form_submit_button("ANALYZE USER BEHAVIOR", use_container_width=True):
            try:
                # 1. تجهيز البيانات كـ Dictionary
                gender_val = 1.0 if gender == "Male" else 0.0
                raw_dict = {
                    'App Usage Time (min/day)': float(app_usage),
                    'Screen On Time (hours/day)': float(screen_time),
                    'Battery Drain (mAh/day)': float(battery),
                    'Number of Apps Installed': float(num_apps),
                    'Data Usage (MB/day)': float(data_usage),
                    'Age': float(age),
                    'Gender': gender_val
                }
                
                # 2. التأكد من الترتيب
                if hasattr(app_model, 'feature_names_in_'):
                    expected_cols = list(app_model.feature_names_in_)
                    # بناء الداتا فريم بالترتيب المطلوب
                    df_final = pd.DataFrame([raw_dict])[expected_cols]
                else:
                    df_final = pd.DataFrame([raw_dict])

                # 🌟 THE NUCLEAR FIX: 
                # تحويل الداتا فريم لمصفوفة Numpy ثم إعادتها كـ DataFrame بنوع بيانات موحد
                # هذا يكسر أي تعارض في الـ Internal Dtypes ويحل مشكلة isnan للأبد
                clean_array = df_final.to_numpy(dtype=np.float64)
                df_clean = pd.DataFrame(clean_array, columns=df_final.columns)

                with st.spinner("Analyzing..."):
                    pred = app_model.predict(df_clean)[0]
                    st.markdown(f"<div class='metric-card' style='text-align:center;'><h3>Predicted Class</h3><h1>{int(pred)}</h1></div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction Error: {e}")
