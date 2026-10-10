"""
app.py
------
Quantum AI Real-Time Fraud Detection System
Interactive Academic Demonstration Dashboard built with Streamlit & Qiskit.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Ensure root directory in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.preprocessing import QUANTUM_FEATURES, ALL_FEATURES
from quantum.circuit import build_full_quantum_circuit, get_circuit_summary, draw_circuit_figure
from utils.helpers import (
    load_system_artifacts,
    analyze_transaction,
    generate_random_simulated_transaction,
    record_transaction_history,
    load_transaction_history,
    clear_transaction_history
)

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Quantum AI Fraud Detection",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Academic Presentation
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-card {
        background-color: #F3F4F6;
        border-left: 5px solid #2563EB;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 1.2rem;
    }
    .metric-container {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .high-risk {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
        border-radius: 8px;
        padding: 15px;
    }
    .medium-risk {
        background-color: #FEF3C7;
        border: 2px solid #F59E0B;
        border-radius: 8px;
        padding: 15px;
    }
    .low-risk {
        background-color: #DCFCE7;
        border: 2px solid #10B981;
        border-radius: 8px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached Artifact Loading
# ---------------------------------------------------------
@st.cache_resource
def get_artifacts():
    try:
        return load_system_artifacts(), None
    except Exception as e:
        return None, str(e)

artifacts, load_error = get_artifacts()

# ---------------------------------------------------------
# Header & System Status
# ---------------------------------------------------------
st.markdown('<div class="main-title">⚛️ QUANTUM AI REAL-TIME FRAUD DETECTION SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hybrid Quantum-Classical Financial Transaction Monitoring | Academic Prototype</div>', unsafe_allow_html=True)

# Status Bar
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**🟢 Quantum Simulator**  \n`StatevectorSampler (Aer)`")
with c2:
    st.markdown("**🟢 Quantum Architecture**  \n`4 Qubits | Angle Encoding`")
with c3:
    st.markdown("**🟢 Classifier**  \n`Variational Quantum Classifier (VQC)`")
with c4:
    if artifacts:
        st.markdown("**🟢 System Status**  \n`MODELS TRAINED & ACTIVE`")
    else:
        st.markdown("**🔴 System Status**  \n`ARTIFACTS MISSING (Run train_models.py)`")

st.divider()

# ---------------------------------------------------------
# Sidebar Controls & Information
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Risk Thresholds")
    thresh_low = st.slider("Low Risk Upper Bound (%)", min_value=10.0, max_value=45.0, value=30.0, step=5.0)
    thresh_high = st.slider("High Risk Lower Bound (%)", min_value=55.0, max_value=90.0, value=70.0, step=5.0)
    
    st.divider()
    st.subheader("📚 How Quantum AI Works")
    st.markdown("""
    1. **Preprocessing:** Transaction features are normalized.
    2. **Quantum Feature Map:** 4 key indicators are mapped into single-qubit rotations $R_y(\\theta_i)$.
    3. **Angle Encoding:** Continuous classical variables become quantum superposition states.
    4. **Entanglement:** CNOT cascades correlate features non-linearly across qubits.
    5. **Variational Classifier:** Classical optimizer (COBYLA) tunes variational rotation parameters.
    6. **Measurement:** Parity measurement yields quantum class state.
    7. **Risk Scoring:** Model expectation translates to a real-time Fraud Risk Score.
    """)
    
    st.divider()
    st.subheader("⚠️ Academic Notice")
    st.caption("""
    This prototype uses synthetic transaction data for academic demonstration. 
    It is running on a local quantum statevector simulator on classical hardware. 
    Quantum advantage is not claimed.
    """)

if not artifacts:
    st.error(f"Model artifacts not found or failed to load!\n\nDetails: `{load_error}`\n\nPlease ensure `models/` directory with `quantum_vqc.dill` and `.joblib` files are committed and pushed to GitHub.")
    st.stop()

# ---------------------------------------------------------
# Tabs Layout
# ---------------------------------------------------------
tab_analysis, tab_circuit, tab_performance, tab_history = st.tabs([
    "🔍 Real-Time Analysis",
    "⚛️ Quantum Circuit Architecture",
    "📊 Model Performance & Baseline",
    "📜 Transaction History"
])

# Initialize Session States
if "txn_input" not in st.session_state:
    st.session_state.txn_input = {
        "transaction_amount": 1200.0,
        "transaction_frequency": 2,
        "location_risk": 0,
        "device_changed": 0,
        "previous_fraud": 0,
        "distance_from_usual_location": 3.0,
        "transaction_hour": 14,
        "merchant_risk": 0,
        "failed_attempts": 0,
        "international_transaction": 0,
        "account_age_days": 850
    }

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

# ---------------------------------------------------------
# TAB 1: Real-Time Transaction Analysis
# ---------------------------------------------------------
with tab_analysis:
    st.subheader("1. Transaction Parameter Input")
    
    # Preset Demo Buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        if st.button("🟢 Load Demo 1: Legitimate Profile"):
            st.session_state.txn_input = {
                "transaction_amount": 1200.0,
                "transaction_frequency": 2,
                "location_risk": 0,
                "device_changed": 0,
                "previous_fraud": 0,
                "distance_from_usual_location": 3.0,
                "transaction_hour": 14,
                "merchant_risk": 0,
                "failed_attempts": 0,
                "international_transaction": 0,
                "account_age_days": 850
            }
            st.rerun()
            
    with btn_col2:
        if st.button("🔴 Load Demo 2: Suspicious Profile"):
            st.session_state.txn_input = {
                "transaction_amount": 75000.0,
                "transaction_frequency": 16,
                "location_risk": 2,
                "device_changed": 1,
                "previous_fraud": 1,
                "distance_from_usual_location": 850.0,
                "transaction_hour": 2,
                "merchant_risk": 2,
                "failed_attempts": 4,
                "international_transaction": 1,
                "account_age_days": 45
            }
            st.rerun()

    # Form with inputs
    inp = st.session_state.txn_input

    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        txn_amount = st.number_input("Transaction Amount (₹)", min_value=50.0, max_value=200000.0, value=float(inp["transaction_amount"]), step=500.0)
        txn_freq = st.slider("Daily Transaction Frequency", min_value=1, max_value=30, value=int(inp["transaction_frequency"]))
        loc_risk_str = st.selectbox("Location Risk", ["Low (0)", "Medium (1)", "High (2)"], index=int(inp["location_risk"]))
        loc_risk = int(loc_risk_str.split("(")[1][0])
        device_str = st.selectbox("Device Changed?", ["No (0)", "Yes (1)"], index=int(inp["device_changed"]))
        device_changed = int(device_str.split("(")[1][0])

    with col_b:
        prev_fraud_str = st.selectbox("Previous Fraud on Account?", ["No (0)", "Yes (1)"], index=int(inp["previous_fraud"]))
        prev_fraud = int(prev_fraud_str.split("(")[1][0])
        dist_val = st.number_input("Distance From Usual Location (km)", min_value=0.1, max_value=3000.0, value=float(inp["distance_from_usual_location"]), step=10.0)
        txn_hour = st.slider("Transaction Hour (24h)", min_value=0, max_value=23, value=int(inp["transaction_hour"]))
        merch_risk_str = st.selectbox("Merchant Risk Category", ["Low (0)", "Medium (1)", "High (2)"], index=int(inp["merchant_risk"]))
        merch_risk = int(merch_risk_str.split("(")[1][0])

    with col_c:
        failed_att = st.slider("Prior Failed PIN/OTP Attempts", min_value=0, max_value=5, value=int(inp["failed_attempts"]))
        intl_str = st.selectbox("International Transaction?", ["No (0)", "Yes (1)"], index=int(inp["international_transaction"]))
        intl = int(intl_str.split("(")[1][0])
        acct_age = st.number_input("Account Age (Days)", min_value=1, max_value=3650, value=int(inp["account_age_days"]), step=30)

    # Action Buttons
    action_col1, action_col2 = st.columns(2)
    
    with action_col1:
        analyze_clicked = st.button("🔍 ANALYZE TRANSACTION", type="primary", use_container_width=True)
    with action_col2:
        simulate_clicked = st.button("⚡ SIMULATE INCOMING TRANSACTION", use_container_width=True)

    # Process Simulation Click
    if simulate_clicked:
        sim_txn = generate_random_simulated_transaction("random")
        st.session_state.txn_input = sim_txn
        res = analyze_transaction(
            sim_txn,
            artifacts["quantum_model"],
            artifacts["quantum_scaler"],
            artifacts["classical_model"],
            artifacts["full_scaler"],
            risk_threshold_low=thresh_low,
            risk_threshold_high=thresh_high
        )
        record_transaction_history(res)
        st.session_state.last_analysis = res
        st.rerun()

    # Process Manual Analyze Click
    if analyze_clicked:
        current_txn = {
            "transaction_id": f"TXN-{np.random.randint(1000, 9999)}",
            "transaction_amount": txn_amount,
            "transaction_frequency": txn_freq,
            "location_risk": loc_risk,
            "device_changed": device_changed,
            "previous_fraud": prev_fraud,
            "distance_from_usual_location": dist_val,
            "transaction_hour": txn_hour,
            "merchant_risk": merch_risk,
            "failed_attempts": failed_att,
            "international_transaction": intl,
            "account_age_days": acct_age
        }
        res = analyze_transaction(
            current_txn,
            artifacts["quantum_model"],
            artifacts["quantum_scaler"],
            artifacts["classical_model"],
            artifacts["full_scaler"],
            risk_threshold_low=thresh_low,
            risk_threshold_high=thresh_high
        )
        record_transaction_history(res)
        st.session_state.last_analysis = res

    # Display Analysis Result
    if st.session_state.last_analysis:
        res = st.session_state.last_analysis
        st.markdown("---")
        st.subheader(f"2. Quantum AI Detection Output — {res['transaction_id']}")
        
        # Color-coded box
        risk_class = "low-risk" if res["risk_level"] == "LOW" else ("medium-risk" if res["risk_level"] == "MEDIUM" else "high-risk")
        
        st.markdown(f"""
        <div class="{risk_class}">
            <h3 style="margin-top:0;">{res['alert_icon']} {res['classification'].upper()} TRANSACTION</h3>
            <p style="font-size:1.15rem; margin-bottom:5px;">
                <b>Fraud Risk Score:</b> <span style="font-size:1.4rem; font-weight:800;">{res['fraud_risk_score']}%</span> &nbsp;|&nbsp; 
                <b>Risk Level:</b> {res['risk_level']} &nbsp;|&nbsp; 
                <b>Recommended Action:</b> <u>{res['recommended_action']}</u>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk Score Progress Bar
        st.write("")
        st.progress(float(res["fraud_risk_score"]) / 100.0)

        # Comparison with Classical ML & Quantum Parity
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Fraud Risk Score", f"{res['fraud_risk_score']}%")
        with m_col2:
            st.metric("Raw Quantum Parity Prob", f"{res.get('raw_quantum_prob', res['fraud_risk_score'])}%")
        with m_col3:
            st.metric("Classical RF Score", f"{res['classical_risk_score']}%")
        with m_col4:
            st.metric("Quantum Register", "4 Qubits")

        # Quantum Features Angle Rotation display
        with st.expander("🔬 View Quantum State Angle Encodings (θ in radians)"):
            st.write("Feature values mapped to $[0, \\pi]$ for $R_y(\\theta)$ rotations:")
            q_cols = st.columns(4)
            q_names = ["Amount", "Frequency", "Location Risk", "Device Changed"]
            for i, val in enumerate(res["quantum_scaled_features"]):
                with q_cols[i]:
                    st.metric(f"q[{i}]: {q_names[i]}", f"{val:.3f} rad")

        # Risk Indicators Explanation
        st.subheader("3. Transaction Risk Indicators")
        st.caption("Transparent heuristic risk indicators based on transaction properties (independent of quantum weights):")
        for indicator in res["risk_indicators"]:
            st.markdown(f"- {indicator}")

# ---------------------------------------------------------
# TAB 2: Quantum Circuit Architecture
# ---------------------------------------------------------
with tab_circuit:
    st.subheader("Genuine Parameterized Quantum Circuit (Qiskit)")
    st.caption("Real 4-qubit circuit architecture combining Angle Encoding and hardware-efficient linear CNOT entanglement.")
    
    q_model = artifacts["quantum_model"]
    full_circuit = q_model.get_full_circuit()
    summary = get_circuit_summary(full_circuit)
    
    # Metadata badges
    q_b1, q_b2, q_b3, q_b4, q_b5 = st.columns(5)
    with q_b1:
        st.metric("Qubits", summary["num_qubits"])
    with q_b2:
        st.metric("Circuit Depth", summary["circuit_depth"])
    with q_b3:
        st.metric("Variational Parameters", summary["num_parameters"])
    with q_b4:
        st.metric("Total Gates", summary["total_gates"])
    with q_b5:
        st.metric("Backend", "Statevector (Aer)")

    st.markdown("### Quantum Circuit Diagram")
    try:
        fig = draw_circuit_figure(full_circuit)
        st.pyplot(fig)
    except Exception as e:
        st.code(str(full_circuit), language="text")

    st.markdown("### Circuit Component Breakdown")
    c_info1, c_info2 = st.columns(2)
    with c_info1:
        st.markdown("""
        **1. Feature Map (Angle Encoding):**
        - Maps each classical normalized feature $x_i \\in [0, \\pi]$ into single-qubit rotations:
          - $q_0$: `transaction_amount` $\\to R_y(x_0)$
          - $q_1$: `transaction_frequency` $\\to R_y(x_1)$
          - $q_2$: `location_risk` $\\to R_y(x_2)$
          - $q_3$: `device_changed` $\\to R_y(x_3)$
        """)
    with c_info2:
        st.markdown("""
        **2. Variational Entangling Ansatz:**
        - Linear CNOT entanglement chain: $q_0 \\to q_1 \\to q_2 \\to q_3$.
        - Parameterized $R_y(w_j)$ rotation gates with 8 variational weights.
        - Optimized iteratively using classical COBYLA optimizer.
        """)

# ---------------------------------------------------------
# TAB 3: Model Performance & Classical Comparison
# ---------------------------------------------------------
with tab_performance:
    st.subheader("Model Performance Evaluation: Classical ML vs Quantum ML")
    st.caption("Measured performance on unseen test data from the synthetic financial transactions dataset.")

    # Load metrics from file
    if os.path.exists("results/metrics.csv"):
        metrics_df = pd.read_csv("results/metrics.csv")
        st.table(metrics_df)
    else:
        st.info("Metrics file not found. Run train_models.py to compute evaluation metrics.")

    # Confusion Matrix Image
    st.subheader("Side-by-Side Confusion Matrices (1,000 Unseen Test Transactions)")
    if os.path.exists("results/confusion_matrix.png"):
        st.image("results/confusion_matrix.png", use_container_width=True)
    else:
        st.info("Confusion matrix image not found.")

    # ROC Curves Image
    st.subheader("Receiver Operating Characteristic (ROC) Curves")
    if os.path.exists("results/roc_curves.png"):
        st.image("results/roc_curves.png", use_container_width=True)
    else:
        st.info("ROC curves image not found.")

    # Controlled Experimentation History
    if os.path.exists("results/experiments/vqc_experiment_results.csv"):
        st.subheader("🔬 Systematic Hyperparameter Experimentation History")
        st.caption("All configurations evaluated on the exact same 1,000 unseen test samples.")
        exp_table = pd.read_csv("results/experiments/vqc_experiment_results.csv")
        st.dataframe(exp_table, use_container_width=True)

    st.markdown("""
    ### Honest Academic Findings & Interpretation
    - **Class Imbalance Realities:** Fraud comprises ~7% of all transactions. Relying solely on raw Accuracy is misleading; **Recall** (catching true frauds) and **F1-Score** are the critical metrics.
    - **Controlled Experiments:** Balancing and stratifying the training data increased fraud recall from **7.14% to 40.00%**, catching 28 true fraudulent transactions out of 70.
    - **Classical Baseline (Random Forest):** Operates on all 11 features with deep decision trees, achieving superior Recall (~78.6%) and high ROC-AUC (0.9498).
    - **Quantum Prototype (VQC):** Operates on 4 selected features using an 8-parameter variational circuit. While it demonstrates genuine quantum feature encoding and classification principles, **it does not establish quantum advantage over mature classical ensemble models**.
    - **Significance:** This prototype establishes a working baseline for hybrid quantum-classical pipeline design on contemporary NISQ simulators without fabricating performance.
    """)

# ---------------------------------------------------------
# TAB 4: Transaction History
# ---------------------------------------------------------
with tab_history:
    st.subheader("Real-Time Transaction Audit Log")
    
    col_hist_action1, col_hist_action2 = st.columns([1, 4])
    with col_hist_action1:
        if st.button("🗑️ Clear History"):
            clear_transaction_history()
            st.rerun()

    history_df = load_transaction_history()
    if len(history_df) > 0:
        st.dataframe(history_df, use_container_width=True)
        # Download button
        csv_data = history_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export History as CSV",
            data=csv_data,
            file_name="quantum_fraud_transaction_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No transactions logged yet. Go to 'Real-Time Analysis' and analyze or simulate transactions to view audit history.")
