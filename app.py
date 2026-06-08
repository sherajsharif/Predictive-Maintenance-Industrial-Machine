import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Industrial Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .sub-title {
        font-size: 1.2rem;
        color: #94a3b8;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card Container */
    .card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 1.5rem;
        padding: 2rem;
        margin-bottom: 1rem;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    
    .metric-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .prediction-card {
        border-radius: 1.5rem;
        padding: 2.5rem;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .success-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.2) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .warning-card {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(217, 119, 6, 0.2) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .danger-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(185, 28, 28, 0.2) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(to right, #38bdf8, #818cf8) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        padding: 0.75rem 2rem !important;
        border-radius: 0.75rem !important;
        width: 100%;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.4);
    }

    /* Input Styling */
    .stSlider [data-testid="stSliderTickBar"] {
        display: none;
    }
    
    /* Make slider labels more visible */
    [data-testid="stWidgetLabel"] p {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    /* Hide Streamlit components */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom spacing */
    .block-container {
        padding-top: 2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_resource
def load_assets():
    try:
        # Using relative paths for better portability
        models_dir = 'models'
        with open(os.path.join(models_dir, 'binary_xgb.pkl'), 'rb') as f:
            binary_model = pickle.load(f)
        with open(os.path.join(models_dir, 'multi_xgb.pkl'), 'rb') as f:
            multi_model = pickle.load(f)
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        return binary_model, multi_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# --- Main App ---
def main():
    st.markdown('<h1 class="main-title">Predictive Maintenance AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Real-time industrial sensor analysis & failure prevention</p>', unsafe_allow_html=True)

    # Load Assets
    binary_model, multi_model, scaler = load_assets()
    if not (binary_model and multi_model and scaler):
        st.error("Missing model files in 'models/' directory.")
        return

    # --- Dashboard Layout ---
    # Top Section: Inputs
    st.markdown("### 🛠️ Configuration & Sensors")
    
    input_col1, input_col2, input_col3 = st.columns([1.5, 2, 2])
    
    with input_col1:
        st.markdown("**Product Grade**")
        type_map = {'Low (L)': 0, 'Medium (M)': 1, 'High (H)': 2}
        type_input = st.selectbox("Select quality type", list(type_map.keys()), index=1, label_visibility="collapsed")
        
        st.markdown("**Tool Wear**")
        tool_wear = st.slider("Wear time (min)", 0, 260, 100, label_visibility="collapsed")

    with input_col2:
        st.markdown("**Temperatures [K]**")
        air_temp = st.slider("Air Temperature", 295.0, 305.0, 300.0, 0.1)
        process_temp = st.slider("Process Temperature", 305.0, 315.0, 310.0, 0.1)

    with input_col3:
        st.markdown("**Mechanical Parameters**")
        speed = st.slider("Rotational Speed (RPM)", 1100, 3000, 1500)
        torque = st.slider("Torque (Nm)", 3.0, 80.0, 40.0, 0.1)

    # Middle Section: Analysis
    st.markdown("---")
    
    col_results, col_metrics = st.columns([1.2, 1])

    with col_results:
        st.markdown("### 🔮 AI Diagnostics")
        
        # Prepare Input Data
        input_sensors = np.array([[air_temp, process_temp, speed, torque, tool_wear]])
        scaled_sensors = scaler.transform(input_sensors)
        
        type_val = type_map[type_input]
        final_features = np.zeros((1, 6))
        final_features[0, 0] = type_val
        final_features[0, 1:] = scaled_sensors[0]
        
        feature_names = ['Type', 'Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
        df_input = pd.DataFrame(final_features, columns=feature_names)

        # Run Prediction
        if st.button("RUN DIAGNOSTICS"):
            with st.spinner("Analyzing sensor patterns..."):
                time.sleep(0.5)
                
                # Model predictions
                is_failure = binary_model.predict(df_input)[0]
                failure_type_idx = multi_model.predict(df_input)[0]
                class_labels = ['No Failure', 'Power Failure', 'Tool Wear Failure', 'Overstrain Failure', 'Heat Dissipation Failure']
                
                # Get probabilities
                try:
                    binary_prob = binary_model.predict_proba(df_input)[0][1]
                    multi_probs = multi_model.predict_proba(df_input)[0]
                except:
                    binary_prob = 1.0 if is_failure else 0.0
                    multi_probs = np.zeros(len(class_labels))
                    multi_probs[failure_type_idx] = 1.0

                # Refine logic: If binary model detects a failure but multi-model says 'No Failure' (0),
                # we force it to show the most likely failure type from the other classes.
                if is_failure and failure_type_idx == 0:
                    # Find the index of the highest probability among the actual failure types (indices 1-4)
                    failure_type_idx = np.argmax(multi_probs[1:]) + 1
                
                failure_name = class_labels[failure_type_idx]

                if not is_failure and failure_type_idx == 0:
                    st.markdown(f"""
                        <div class="prediction-card success-card">
                            <div class="status-badge" style="background: #10b981; color: white;">HEALTHY</div>
                            <h2 style="margin: 0; color: #10b981;">{failure_name}</h2>
                            <p style="color: #94a3b8; margin-top: 1rem;">Machine is operating within normal parameters.<br><b>Confidence: {100-binary_prob*100:.1f}%</b></p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    card_class = "danger-card"
                    badge_color = "#ef4444"
                    
                    st.markdown(f"""
                        <div class="prediction-card {card_class}">
                            <div class="status-badge" style="background: {badge_color}; color: white;">FAILURE DETECTED</div>
                            <h2 style="margin: 0; color: {badge_color};">{failure_name}</h2>
                            <p style="color: #94a3b8; margin-top: 1rem;">Anomaly detected in sensor stream.<br><b>Confidence: {binary_prob*100:.1f}%</b></p>
                            <p style="font-size: 0.9rem; margin-top: 0.5rem; color: #cbd5e1;"><b>Action Required:</b> Immediate maintenance inspection.</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Probability Distribution Chart
                st.markdown("#### Confidence Analysis")
                prob_df = pd.DataFrame({
                    'Failure Type': class_labels,
                    'Probability': multi_probs
                })
                st.bar_chart(prob_df.set_index('Failure Type'))
                
        else:
            st.info("Adjust the parameters above and click 'Run Diagnostics' to see the AI analysis.")

    with col_metrics:
        st.markdown("### 📡 Live Readings")
        
        # Use a single markdown block for the metrics to keep them inside a card
        st.markdown(f"""
            <div class="card">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <p class="metric-label">Air Temp</p><p class="metric-value">{air_temp} K</p>
                        <p class="metric-label">Process Temp</p><p class="metric-value">{process_temp} K</p>
                        <p class="metric-label">Speed</p><p class="metric-value">{speed} RPM</p>
                    </div>
                    <div>
                        <p class="metric-label">Torque</p><p class="metric-value">{torque} Nm</p>
                        <p class="metric-label">Tool Wear</p><p class="metric-value">{tool_wear} min</p>
                        <p class="metric-label">Quality</p><p class="metric-value">{type_input.split()[0]}</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
