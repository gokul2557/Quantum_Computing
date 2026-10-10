"""
utils/helpers.py
----------------
Real-time prediction pipeline, feature scaling, risk scoring,
risk indicator rules, and synthetic transaction generator.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import joblib
import numpy as np
import pandas as pd
from models.quantum_model import QuantumFraudVQC
from models.classical_model import ClassicalFraudModel

HISTORY_FILE = "data/transaction_history.csv"

QUANTUM_FEATURES = [
    "failed_attempts",
    "location_risk",
    "account_age_days",
    "merchant_risk"
]

ALL_FEATURES = [
    "transaction_amount",
    "transaction_frequency",
    "location_risk",
    "device_changed",
    "previous_fraud",
    "distance_from_usual_location",
    "transaction_hour",
    "merchant_risk",
    "failed_attempts",
    "international_transaction",
    "account_age_days"
]

def load_system_artifacts():
    """
    Loads trained Quantum VQC, Classical Random Forest, and scalers.
    """
    q_scaler = joblib.load("models/quantum_scaler.joblib")
    full_scaler = joblib.load("models/full_scaler.joblib")
    
    q_model = QuantumFraudVQC.load("models")
    c_model = ClassicalFraudModel.load("models/random_forest.joblib")
    
    return {
        "quantum_model": q_model,
        "classical_model": c_model,
        "quantum_scaler": q_scaler,
        "full_scaler": full_scaler
    }

def analyze_transaction(
    txn_dict: dict,
    quantum_model: QuantumFraudVQC,
    quantum_scaler,
    classical_model: ClassicalFraudModel = None,
    full_scaler = None,
    risk_threshold_low: float = 30.0,
    risk_threshold_high: float = 70.0
) -> dict:
    """
    Full real-time inference pipeline for a single transaction.
    Runs the transaction through genuine Quantum VQC to get fraud risk score.
    """
    # 1. Prepare Quantum 4-feature DataFrame (Top 4 Informative Features)
    q_df = pd.DataFrame([{
        "failed_attempts": float(txn_dict.get("failed_attempts", 0)),
        "location_risk": float(txn_dict.get("location_risk", 0)),
        "account_age_days": float(txn_dict.get("account_age_days", 365)),
        "merchant_risk": float(txn_dict.get("merchant_risk", 0))
    }])
    
    # Scale to [0, pi] for angle encoding (strictly clipped to [0, pi] to prevent Bloch sphere wrap-around)
    q_scaled = np.clip(quantum_scaler.transform(q_df), 0.0, np.pi)
    
    # 2. Evaluate through Genuine Quantum VQC
    proba = quantum_model.predict_proba(q_scaled)[0]
    raw_quantum_prob = float(proba[1])  # Raw uncalibrated parity expectation [~0.28 to ~0.97]
    
    # Calibrate risk score to standard [0% - 100%] scale across the model's observed dynamic range
    # Centered on the VQC 0.50 parity threshold (p_min=0.38, p_max=0.62)
    p_min = 0.38
    p_max = 0.62
    calibrated_score = np.clip((raw_quantum_prob - p_min) / (p_max - p_min) * 100.0, 0.0, 100.0)
    fraud_risk_score = round(float(calibrated_score), 1)

    # 3. Decision & Recommended Action based on Calibrated Fraud Risk Score
    if fraud_risk_score < risk_threshold_low:
        risk_level = "LOW"
        classification = "Legitimate"
        action = "APPROVE TRANSACTION"
        alert_icon = "✅"
    elif fraud_risk_score < risk_threshold_high:
        risk_level = "MEDIUM"
        classification = "Suspicious"
        action = "MANUAL VERIFICATION REQUIRED"
        alert_icon = "⚠️"
    else:
        risk_level = "HIGH"
        classification = "Suspicious"
        action = "FLAG TRANSACTION FOR VERIFICATION"
        alert_icon = "🚨"

    # 4. Optional Classical baseline comparison for this specific transaction
    classical_score = None
    if classical_model and full_scaler:
        c_df = pd.DataFrame([{f: float(txn_dict[f]) for f in ALL_FEATURES}])
        c_scaled = full_scaler.transform(c_df)
        c_prob = classical_model.predict_proba(c_scaled)[0, 1]
        classical_score = round(c_prob * 100.0, 2)

    # 5. Domain Risk Indicators (Heuristic explanation, NOT fake quantum importances)
    indicators = extract_risk_indicators(txn_dict)

    return {
        "transaction_id": txn_dict.get("transaction_id", "TXN-MANUAL"),
        "raw_features": txn_dict,
        "quantum_scaled_features": q_scaled[0].tolist(),
        "fraud_risk_score": fraud_risk_score,
        "raw_quantum_prob": round(raw_quantum_prob * 100.0, 2),
        "classical_risk_score": classical_score,
        "risk_level": risk_level,
        "classification": classification,
        "recommended_action": action,
        "alert_icon": alert_icon,
        "risk_indicators": indicators
    }

def extract_risk_indicators(txn_dict: dict) -> list:
    """
    Identifies notable risk flags based on financial heuristics.
    Clearly labeled as 'Transaction Risk Indicators'.
    """
    indicators = []
    if float(txn_dict.get("transaction_amount", 0)) > 40000:
        indicators.append("Transaction amount is unusually high (> ₹40,000)")
    if int(txn_dict.get("device_changed", 0)) == 1:
        indicators.append("Device change detected (New device ID)")
    if int(txn_dict.get("location_risk", 0)) == 2:
        indicators.append("Transaction originated from a High-Risk geographical location")
    if int(txn_dict.get("previous_fraud", 0)) == 1:
        indicators.append("Historical fraud or dispute flagged on this account")
    if float(txn_dict.get("distance_from_usual_location", 0)) > 200:
        indicators.append(f"Unusual location distance ({txn_dict['distance_from_usual_location']} km from home base)")
    if int(txn_dict.get("failed_attempts", 0)) >= 2:
        indicators.append(f"Multiple failed PIN/OTP attempts detected ({txn_dict['failed_attempts']} prior attempts)")
    if int(txn_dict.get("international_transaction", 0)) == 1:
        indicators.append("Cross-border / international transaction channel")
    if int(txn_dict.get("transaction_hour", 12)) in [1, 2, 3, 4]:
        indicators.append(f"Off-hours transaction initiated at {txn_dict['transaction_hour']}:00 AM")
    if int(txn_dict.get("merchant_risk", 0)) == 2:
        indicators.append("High-risk merchant category code (MCC)")
    if float(txn_dict.get("transaction_frequency", 1)) > 12:
        indicators.append("High transaction frequency burst within 24 hours")
        
    if not indicators:
        indicators.append("Standard transaction profile with no high-risk anomalies detected.")
        
    return indicators

def generate_random_simulated_transaction(scenario: str = "random") -> dict:
    """
    Generates a realistic transaction dictionary for real-time simulation.
    Scenarios: 'random', 'normal', 'suspicious'
    """
    import random
    txn_id = f"TXN-{random.randint(1000, 9999)}"

    if scenario == "normal":
        return {
            "transaction_id": txn_id,
            "transaction_amount": float(random.randint(450, 4800)),
            "transaction_frequency": random.randint(1, 4),
            "location_risk": 0,
            "device_changed": 0,
            "previous_fraud": 0,
            "distance_from_usual_location": round(random.uniform(0.5, 15.0), 1),
            "transaction_hour": random.randint(9, 21),
            "merchant_risk": 0,
            "failed_attempts": 0,
            "international_transaction": 0,
            "account_age_days": random.randint(300, 2500)
        }
    elif scenario == "suspicious":
        return {
            "transaction_id": txn_id,
            "transaction_amount": float(random.randint(55000, 120000)),
            "transaction_frequency": random.randint(14, 25),
            "location_risk": 2,
            "device_changed": 1,
            "previous_fraud": 1,
            "distance_from_usual_location": round(random.uniform(450.0, 1800.0), 1),
            "transaction_hour": random.choice([1, 2, 3, 4]),
            "merchant_risk": 2,
            "failed_attempts": random.randint(2, 4),
            "international_transaction": 1,
            "account_age_days": random.randint(15, 120)
        }
    else:
        # 85% normal, 15% suspicious random generator
        if random.random() < 0.85:
            return generate_random_simulated_transaction("normal")
        else:
            return generate_random_simulated_transaction("suspicious")

def record_transaction_history(res: dict, history_file: str = HISTORY_FILE):
    """
    Appends an analyzed transaction record to local history storage.
    """
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    row = {
        "Transaction ID": res["transaction_id"],
        "Amount (₹)": f"₹{res['raw_features']['transaction_amount']:,.2f}",
        "Fraud Risk Score": f"{res['fraud_risk_score']:.1f}%",
        "Risk Level": res["risk_level"],
        "Classification": res["classification"],
        "Recommended Action": res["recommended_action"]
    }
    df_row = pd.DataFrame([row])
    if os.path.exists(history_file):
        df_row.to_csv(history_file, mode="a", header=False, index=False)
    else:
        df_row.to_csv(history_file, mode="w", header=True, index=False)

def load_transaction_history(history_file: str = HISTORY_FILE) -> pd.DataFrame:
    if os.path.exists(history_file):
        return pd.read_csv(history_file)
    return pd.DataFrame(columns=[
        "Transaction ID", "Amount (₹)", "Fraud Risk Score", "Risk Level", "Classification", "Recommended Action"
    ])

def clear_transaction_history(history_file: str = HISTORY_FILE):
    if os.path.exists(history_file):
        os.remove(history_file)

if __name__ == "__main__":
    artifacts = load_system_artifacts()
    demo_normal = generate_random_simulated_transaction("normal")
    res_normal = analyze_transaction(
        demo_normal,
        artifacts["quantum_model"],
        artifacts["quantum_scaler"],
        artifacts["classical_model"],
        artifacts["full_scaler"]
    )
    print("\n[Test] Normal Transaction Analysis Result:")
    print(f"ID: {res_normal['transaction_id']}")
    print(f"Risk Score: {res_normal['fraud_risk_score']}% ({res_normal['risk_level']})")
    print(f"Classification: {res_normal['classification']}")
    print(f"Action: {res_normal['recommended_action']}")

    demo_suspicious = generate_random_simulated_transaction("suspicious")
    res_suspicious = analyze_transaction(
        demo_suspicious,
        artifacts["quantum_model"],
        artifacts["quantum_scaler"],
        artifacts["classical_model"],
        artifacts["full_scaler"]
    )
    print("\n[Test] Suspicious Transaction Analysis Result:")
    print(f"ID: {res_suspicious['transaction_id']}")
    print(f"Risk Score: {res_suspicious['fraud_risk_score']}% ({res_suspicious['risk_level']})")
    print(f"Classification: {res_suspicious['classification']}")
    print(f"Action: {res_suspicious['recommended_action']}")
    print("Indicators:", [ind.replace("\u20b9", "INR ") for ind in res_suspicious["risk_indicators"]])
