import os
import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import models and metrics
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models import RecoveryDecision, PaymentRecord, AuditLog, CustomerOutreachEvent
from app.config import get_settings
from app.simulation_evidence import PHASE3_SIMULATION_EVIDENCE
from app.dashboard_metrics import (
    get_live_summary,
    get_experiment_metrics,
    get_economic_impact,
    get_guardrail_metrics
)

st.set_page_config(page_title="Recovery Opportunity Engine", layout="wide")

# --- DATABASE SETUP ---
@st.cache_resource
def get_db_session():
    settings = get_settings()
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return SessionLocal()

db = get_db_session()

# Fetch LIVE metrics safely
try:
    summary = get_live_summary(db)
    experiment = get_experiment_metrics(db)
    economic = get_economic_impact(db)
    guardrails = get_guardrail_metrics(db)
    db_connected = True
except Exception as e:
    st.error(f"Database error or missing tables: {e}")
    db_connected = False
    summary = {"total_failed": 0, "total_opportunities": 0, "links_executed": 0, "links_paid": 0, "recovered_revenue_inr": 0.0}
    experiment = {"control_n": 0, "control_recovered": 0, "control_rate": 0, "treatment_n": 0, "treatment_recovered": 0, "treatment_rate": 0, "intervention_rate": 0, "observed_uplift": None}
    economic = {"expected_incremental_net_inr": 0.0}
    guardrails = {"duplicate_blocks": 0, "cooldown_blocks": 0, "fallback_blocks": 0}

# --- SECTION 1: HEADER ---
st.title("Recovery Opportunity Engine")
st.subheader("Recover failed payments without blindly intervening.")

st.info(
    "When a payment fails, the engine estimates whether a customer would recover naturally "
    "and whether sending a Payment Link would materially improve the chance of recovery. "
    "It intervenes only when the expected economic value justifies it."
)

# --- SECTION 2: EXECUTIVE KPIs ---
st.header("Executive KPIs")
st.caption("Live Data from the Production Database")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Failed Payments", summary["total_failed"])
col2.metric("Treatment Decisions", experiment["treatment_n"])
col3.metric("Control Decisions", experiment["control_n"])
col4.metric("Links Executed", summary["links_executed"])
col5.metric("Outcomes Observed", summary["links_paid"])

st.metric("AI Intervention Rate (Treatment)", f"{experiment['intervention_rate']*100:.1f}%")

# --- SECTION 3: HOW THE AI DECIDES ---
st.header("How the AI Decides")

st.markdown("""
<div style="text-align: center; font-size: 1.1em; line-height: 1.6;">
    <b>FAILED PAYMENT</b><br>
    ⬇<br>
    <b>P0</b> — probability of natural recovery<br>
    ⬇<br>
    <b>P1</b> — probability of recovery with Payment Link<br>
    ⬇<br>
    <b>UPLIFT</b> = P1 - P0<br>
    ⬇<br>
    <b>EXPECTED INCREMENTAL NET VALUE</b><br>
    ⬇<br>
    <b>SEND PAYMENT LINK / NO ACTION</b><br>
    ⬇<br>
    <b>GUARDRAILS</b><br>
    ⬇<br>
    <b>EXECUTION</b>
</div>
""", unsafe_allow_html=True)

st.caption("""
**The "Sleeping Dog" Concept**: If P0 is already high, the customer may recover naturally and intervention can be unnecessary. The engine intervenes only when the predicted incremental economic value justifies it.
""")

# --- SECTION 4: LIVE ECONOMIC IMPACT ---
st.header("Live Economic Impact")
st.caption("Real performance metrics based on active production data.")

# Main economic impact metrics
ecol1, ecol2, ecol3, ecol4 = st.columns(4)
ecol1.metric("Opportunities", summary["total_opportunities"])
ecol2.metric("Links Sent", summary["links_executed"])
ecol3.metric("Links Paid", summary["links_paid"])
ecol4.metric("Guardrail Blocks", sum(guardrails.values()))

# Financials
fcol1, fcol2 = st.columns(2)
with fcol1:
    st.metric("Observed Recovered Revenue", f"₹{summary['recovered_revenue_inr']:,.2f}")
    st.caption("Actual gross revenue collected through recovery Payment Links.")
    
with fcol2:
    st.metric("Expected Incremental Net Value", f"₹{economic['expected_incremental_net_inr']:,.2f}")
    st.caption("ML-estimated incremental value after intervention cost.")

# --- SECTION 5: EXPERIMENTAL SIGNAL ---
st.header("Experimental Signal (Control vs Treatment)")
st.caption("Live A/B experiment monitoring the true causal uplift of the ML Policy.")

sigcol1, sigcol2, sigcol3 = st.columns(3)

# Treatment
with sigcol1:
    st.subheader("TREATMENT")
    st.write(f"N = {experiment['treatment_n']}")
    st.metric("Treatment Recovery", f"{experiment['treatment_rate']*100:.1f}%")

# Control
with sigcol2:
    st.subheader("CONTROL")
    st.write(f"N = {experiment['control_n']}")
    if experiment["control_n"] > 0:
        st.metric("Control Recovery", f"{experiment['control_rate']*100:.1f}%")
    else:
        st.write("Awaiting control observations")

# Uplift (Only report if N >= 30)
with sigcol3:
    st.subheader("UPLIFT")
    st.write("Difference (T - C)")
    
    if experiment["control_n"] >= 30 and experiment["treatment_n"] >= 30:
        uplift_pp = (experiment["treatment_rate"] - experiment["control_rate"]) * 100
        st.metric("Observed Experimental Uplift", f"{uplift_pp:+.1f} pp")
    else:
        st.warning("Insufficient sample size to report experimental uplift (minimum N=30 in both groups required).")

# --- SECTION 6: LIVE DECISION INSPECTOR ---
st.header("Live Decision Inspector")
if db_connected and summary["total_opportunities"] > 0:
    # Use raw SQL to only fetch the latest 10 decisions directly
    latest_query = """
        SELECT 
            p.amount / 100.0 as "Amount (INR)",
            d.experiment_variant as "Variant",
            d.predicted_p0 as "P0",
            d.predicted_p1 as "P1",
            d.predicted_uplift as "Uplift",
            d.expected_incremental_net_paise / 100.0 as "Expected Inc Net (INR)",
            d.selected_action as "Action",
            d.decision_status as "Status",
            d.outcome_observed_at as "Outcome Observed"
        FROM recovery_decisions d
        JOIN payment_records p ON d.payment_record_id = p.id
        ORDER BY d.created_at DESC
        LIMIT 10
    """
    inspector_df = pd.read_sql(latest_query, db.bind)
    
    st.dataframe(inspector_df.style.format({
        "P0": "{:.3f}",
        "P1": "{:.3f}",
        "Uplift": "{:.3f}",
        "Expected Inc Net (INR)": "{:.2f}",
        "Amount (INR)": "{:.2f}"
    }), use_container_width=True)
else:
    st.write("No decisions available yet.")

# --- SECTION 7: SAFETY ---
st.header("Safety & Guardrails")
st.write("These rules are intentionally separate from the economic model to ensure operational safety.")

st.write(f"- Duplicate executions prevented: {guardrails['duplicate_blocks']}")
st.write(f"- Cooldown blocks (48h rule): {guardrails['cooldown_blocks']}")
st.write(f"- Model fallbacks/downgrades: {guardrails['fallback_blocks']}")

# --- SECTION 8: END-TO-END DEMO ---
st.header("End-to-End Lifecycle")
st.markdown("""
1. `payment.failed` webhook is received from Razorpay.
2. ML Policy makes a decision.
3. Guardrails verify safety.
4. If authorized, a Payment Link is generated via Razorpay API.
5. Customer pays the link.
6. `payment_link.paid` webhook arrives.
7. System maps the payment back to the exact decision, setting `OUTCOME_OBSERVED`.
""")

# --- SECTION 9: ANALYTICAL SIMULATION EVIDENCE ---
st.divider()
with st.expander("Analytical Simulation Evidence (Phase 3 RCT)"):
    st.info(
        "Simulation evidence — "
        f"{PHASE3_SIMULATION_EVIDENCE.transaction_count:,} synthetic transactions "
        f"(Phase 3 evaluation seed {PHASE3_SIMULATION_EVIDENCE.evaluation_seed}). "
        "This is strictly for model validation and not indicative of live production revenue."
    )

    simulation_evidence = PHASE3_SIMULATION_EVIDENCE

    col_ei1, col_ei2, col_ei3 = st.columns(3)
    col_ei1.metric("AI Policy Incremental Net Value", f"{simulation_evidence.ai_policy_incremental_net_inr:,.0f} INR")
    col_ei2.metric("Always-Link Incremental Net Value", f"{simulation_evidence.always_link_incremental_net_inr:,.0f} INR")
    col_ei3.metric(
        "Difference (Value Added by AI)",
        f"+{simulation_evidence.incremental_value_vs_always_link_inr:,.0f} INR",
        delta=f"{simulation_evidence.incremental_value_vs_always_link_inr:,.0f}",
    )

    st.subheader("Solving the 'Sleeping Dog' Problem")
    st.write("A Sleeping Dog is a customer who would likely recover without intervention. Sending an unnecessary recovery message can create cost without incremental recovery, or worse, annoy them into abandoning the purchase.")

    chart_data = pd.DataFrame({
        "Policy": ["Always-Link (Naive)", "AI Policy (S-Learner)"],
        "Loss from Sleeping Dogs (INR)": [
            abs(simulation_evidence.always_link_sleeping_dog_loss_inr),
            abs(simulation_evidence.ai_policy_sleeping_dog_loss_inr),
        ],
    })

    st.bar_chart(chart_data, x="Policy", y="Loss from Sleeping Dogs (INR)", color="#ff4b4b")
    st.success(
        "The AI Policy reduces harmful Sleeping Dog interventions by 80%, avoiding over "
        f"{simulation_evidence.sleeping_dog_loss_avoided_inr:,.0f} INR in analytical losses."
    )
