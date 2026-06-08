import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

# --- Page Configuration (MUST be first Streamlit call) ---
st.set_page_config(
    page_title="PredictX — Industrial AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Premium CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #050a10;
    color: #e2e8f0;
}

/* ── Animated grid background ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(56,189,248,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56,189,248,0.04) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* ── Hide Streamlit chrome ── */
header {visibility: hidden;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.block-container { padding-top: 2rem !important; position: relative; z-index: 1; }

/* ── Hero Header ── */
.hero-wrap {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    color: #38bdf8;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(100deg, #f0f9ff 30%, #38bdf8 60%, #818cf8 90%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
    letter-spacing: -0.03em;
}
.hero-sub {
    color: #64748b;
    font-size: 1rem;
    font-family: 'DM Mono', monospace;
    font-weight: 300;
}

/* ── Section headings ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    color: #38bdf8;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,0.4), transparent);
}

/* ── Glass card ── */
.glass-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(56,189,248,0.12);
    border-radius: 1.25rem;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 4px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
}

/* ── Slider label ── */
[data-testid="stWidgetLabel"] p {
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.05em !important;
}

/* ── Selectbox ── */
[data-baseweb="select"] {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(56,189,248,0.2) !important;
    border-radius: 0.6rem !important;
}

/* ── Slider track color ── */
[data-testid="stSlider"] [role="slider"] {
    background: #38bdf8 !important;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    color: #fff !important;
    border: none !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 2rem !important;
    border-radius: 0.75rem !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 0 24px rgba(56,189,248,0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(56,189,248,0.4) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid rgba(56,189,248,0.1) !important;
    margin: 1.5rem 0 !important;
}

/* ── Result cards ── */
.result-card {
    border-radius: 1.25rem;
    padding: 2rem 1.8rem;
    text-align: center;
    margin-bottom: 1rem;
    border: 1px solid;
}
.result-healthy {
    background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(5,150,105,0.15));
    border-color: rgba(16,185,129,0.35);
}
.result-failure {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(185,28,28,0.15));
    border-color: rgba(239,68,68,0.35);
}
.result-icon { font-size: 2.5rem; margin-bottom: 0.6rem; }
.result-badge {
    display: inline-block;
    padding: 0.3rem 1.2rem;
    border-radius: 9999px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 0.8rem;
}
.badge-green  { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid rgba(16,185,129,0.4); }
.badge-red    { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
.result-title { font-size: 1.6rem; font-weight: 800; margin: 0 0 0.5rem; }
.result-title-green { color: #34d399; }
.result-title-red   { color: #f87171; }
.result-sub { color: #64748b; font-family: 'DM Mono', monospace; font-size: 0.8rem; }
.result-conf { font-size: 0.9rem; margin-top: 0.5rem; color: #94a3b8; }

/* ── Live metric grid ── */
.metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.9rem;
}
.metric-tile {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(56,189,248,0.1);
    border-radius: 0.9rem;
    padding: 0.9rem 1rem;
    transition: border-color 0.3s;
}
.metric-tile:hover { border-color: rgba(56,189,248,0.3); }
.m-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.66rem;
    color: #475569;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.m-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e2e8f0;
    line-height: 1;
}
.m-unit {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #38bdf8;
    margin-left: 0.2rem;
}

/* ── Status strip ── */
.status-strip {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #38bdf8;
    padding: 0.6rem 1rem;
    background: rgba(56,189,248,0.05);
    border: 1px solid rgba(56,189,248,0.12);
    border-radius: 0.6rem;
    margin-bottom: 1.2rem;
    letter-spacing: 0.08em;
}
.pulse-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #38bdf8;
    animation: pulse 1.8s infinite;
    flex-shrink: 0;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.85); }
}

/* ── Bar chart overrides ── */
[data-testid="stVegaLiteChart"] canvas,
[data-testid="stArrowVegaLiteChart"] { border-radius: 0.75rem; }

/* ── Info box ── */
[data-testid="stAlert"] {
    background: rgba(56,189,248,0.06) !important;
    border: 1px solid rgba(56,189,248,0.18) !important;
    border-radius: 0.75rem !important;
    color: #94a3b8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    """Load pickle models. Returns (binary_model, multi_model, scaler) or (None,None,None)."""
    try:
        models_dir = 'models'
        with open(os.path.join(models_dir, 'binary_xgb.pkl'), 'rb') as f:
            binary_model = pickle.load(f)
        with open(os.path.join(models_dir, 'multi_xgb.pkl'), 'rb') as f:
            multi_model = pickle.load(f)
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        return binary_model, multi_model, scaler
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}. Make sure the `models/` folder is present.")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Unexpected error loading models: {e}")
        return None, None, None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Hero ──
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">⚙ Industrial Intelligence Platform</div>
        <h1 class="hero-title">PredictX Maintenance AI</h1>
        <p class="hero-sub">Real-time sensor fusion · Failure classification · Root cause analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load models ──
    binary_model, multi_model, scaler = load_assets()
    if binary_model is None or multi_model is None or scaler is None:
        st.stop()

    # ── Status strip ──
    st.markdown("""
    <div class="status-strip">
        <span class="pulse-dot"></span>
        SYSTEM ONLINE &nbsp;·&nbsp; MODELS LOADED &nbsp;·&nbsp; SENSOR FEED ACTIVE
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════════
    #  INPUT SECTION
    # ════════════════════════════════════════════
    st.markdown('<div class="section-label">01 — Configuration & Sensor Input</div>', unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([1.4, 2, 2], gap="medium")

    with col_a:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Product Grade**")
        type_map = {'Low (L)': 0, 'Medium (M)': 1, 'High (H)': 2}
        type_input = st.selectbox(
            "Product quality grade",
            list(type_map.keys()),
            index=1,
            label_visibility="collapsed",
        )
        st.markdown("&nbsp;", unsafe_allow_html=True)
        st.markdown("**Tool Wear (min)**")
        tool_wear = st.slider("Tool wear time", 0, 260, 100, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Temperatures [Kelvin]**")
        air_temp     = st.slider("Air Temperature [K]",     295.0, 305.0, 300.0, 0.1)
        process_temp = st.slider("Process Temperature [K]", 305.0, 315.0, 310.0, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Mechanical Parameters**")
        speed  = st.slider("Rotational Speed [RPM]", 1100, 3000, 1500)
        torque = st.slider("Torque [Nm]",             3.0,  80.0,  40.0, 0.1)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ════════════════════════════════════════════
    #  RESULTS SECTION
    # ════════════════════════════════════════════
    col_res, col_live = st.columns([1.3, 1], gap="large")

    # ── Prepare features (always, so live panel updates) ──
    input_sensors  = np.array([[air_temp, process_temp, speed, torque, tool_wear]])
    scaled_sensors = scaler.transform(input_sensors)

    type_val       = type_map[type_input]
    final_features = np.zeros((1, 6))
    final_features[0, 0]  = type_val
    final_features[0, 1:] = scaled_sensors[0]

    feature_names = ['Type', 'Air temperature', 'Process temperature',
                     'Rotational speed', 'Torque', 'Tool wear']
    df_input = pd.DataFrame(final_features, columns=feature_names)

    CLASS_LABELS = [
        'No Failure',
        'Power Failure',
        'Tool Wear Failure',
        'Overstrain Failure',
        'Heat Dissipation Failure',
    ]

    with col_res:
        st.markdown('<div class="section-label">02 — AI Diagnostics</div>', unsafe_allow_html=True)

        if st.button("⚡  RUN DIAGNOSTICS", use_container_width=True):
            with st.spinner("Analyzing sensor stream…"):
                time.sleep(0.5)

                is_failure      = int(binary_model.predict(df_input)[0])
                failure_type_idx = int(multi_model.predict(df_input)[0])

                # Safe probability extraction
                try:
                    binary_prob = float(binary_model.predict_proba(df_input)[0][1])
                    multi_probs = multi_model.predict_proba(df_input)[0]
                except Exception:
                    binary_prob = 1.0 if is_failure else 0.0
                    multi_probs = np.zeros(len(CLASS_LABELS))
                    multi_probs[failure_type_idx] = 1.0

                # Reconcile models: if binary says failure but multi says "No Failure"
                if is_failure and failure_type_idx == 0:
                    failure_type_idx = int(np.argmax(multi_probs[1:]) + 1)

                failure_name = CLASS_LABELS[failure_type_idx]
                healthy = (not is_failure) and (failure_type_idx == 0)

                if healthy:
                    conf_pct = (1 - binary_prob) * 100
                    st.markdown(f"""
                    <div class="result-card result-healthy">
                        <div class="result-icon">✅</div>
                        <div class="result-badge badge-green">System Healthy</div>
                        <div class="result-title result-title-green">{failure_name}</div>
                        <div class="result-sub">All parameters within nominal range</div>
                        <div class="result-conf">Confidence: <b>{conf_pct:.1f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    conf_pct = binary_prob * 100
                    st.markdown(f"""
                    <div class="result-card result-failure">
                        <div class="result-icon">🚨</div>
                        <div class="result-badge badge-red">Failure Detected</div>
                        <div class="result-title result-title-red">{failure_name}</div>
                        <div class="result-sub">Anomaly detected — immediate inspection required</div>
                        <div class="result-conf">Confidence: <b>{conf_pct:.1f}%</b></div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Probability distribution ──
                st.markdown('<div class="section-label" style="margin-top:1.2rem;">Confidence Distribution</div>',
                            unsafe_allow_html=True)
                prob_df = pd.DataFrame({
                    'Failure Type': CLASS_LABELS,
                    'Probability':  [float(p) for p in multi_probs],
                })
                st.bar_chart(prob_df.set_index('Failure Type'), height=220, use_container_width=True)

        else:
            st.info("🔬 Adjust parameters and click **RUN DIAGNOSTICS** to begin analysis.")

    # ── Live readings panel ──
    with col_live:
        st.markdown('<div class="section-label">03 — Live Sensor Readings</div>', unsafe_allow_html=True)
        temp_delta = round(process_temp - air_temp, 1)
        power_kw   = round((speed * torque) / 9550, 2)

        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-grid">
                <div class="metric-tile">
                    <div class="m-label">Air Temp</div>
                    <div class="m-value">{air_temp:.1f}<span class="m-unit">K</span></div>
                </div>
                <div class="metric-tile">
                    <div class="m-label">Process Temp</div>
                    <div class="m-value">{process_temp:.1f}<span class="m-unit">K</span></div>
                </div>
                <div class="metric-tile">
                    <div class="m-label">Rotational Speed</div>
                    <div class="m-value">{speed}<span class="m-unit">RPM</span></div>
                </div>
                <div class="metric-tile">
                    <div class="m-label">Torque</div>
                    <div class="m-value">{torque:.1f}<span class="m-unit">Nm</span></div>
                </div>
                <div class="metric-tile">
                    <div class="m-label">Tool Wear</div>
                    <div class="m-value">{tool_wear}<span class="m-unit">min</span></div>
                </div>
                <div class="metric-tile">
                    <div class="m-label">Quality Grade</div>
                    <div class="m-value">{type_input.split()[0]}</div>
                </div>
                <div class="metric-tile">
                    <div class="m-label">Temp Delta</div>
                    <div class="m-value">+{temp_delta}<span class="m-unit">K</span></div>
                </div>
                <div class="metric-tile">
                    <div class="m-label">Est. Power</div>
                    <div class="m-value">{power_kw}<span class="m-unit">kW</span></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Quick health indicators ──
        st.markdown('<div class="section-label" style="margin-top:0.5rem;">Quick Health Flags</div>',
                    unsafe_allow_html=True)

        flags = []
        if tool_wear > 200:
            flags.append(("⚠️ High Tool Wear", "#f59e0b"))
        if temp_delta > 12:
            flags.append(("⚠️ Large Temp Delta", "#f59e0b"))
        if torque > 65:
            flags.append(("⚠️ High Torque", "#ef4444"))
        if speed > 2500:
            flags.append(("⚠️ High RPM", "#ef4444"))
        if not flags:
            flags.append(("✅ All Parameters Normal", "#10b981"))

        for label, color in flags:
            st.markdown(
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.8rem;color:{color};'
                f'padding:0.4rem 0.8rem;background:rgba(255,255,255,0.03);border-radius:0.5rem;'
                f'margin-bottom:0.4rem;border:1px solid {color}33;">{label}</div>',
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
