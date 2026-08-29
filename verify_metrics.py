import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.dashboard_metrics import get_live_summary, get_experiment_metrics, get_economic_impact, get_guardrail_metrics

engine = create_engine("sqlite:///roe.db")
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

summary = get_live_summary(db)
exp = get_experiment_metrics(db)
econ = get_economic_impact(db)
guard = get_guardrail_metrics(db)

print(f"Summary: {summary}")
print(f"Experiment: {exp}")
print(f"Economic: {econ}")
print(f"Guardrails: {guard}")
