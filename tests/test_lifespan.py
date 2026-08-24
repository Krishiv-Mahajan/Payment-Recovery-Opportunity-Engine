"""
Tests for FastAPI lifespan model loading.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from app.main import create_app
from app.ml.policy import EconomicPolicyPredictor
from app.ml.predictor import PlaceholderPredictor
import joblib

def test_lifespan_loads_model(monkeypatch):
    class DummyModel:
        is_fitted = True
        def __init__(self):
            self.feature_names_in_ = ["amount_paise", "hour_of_day", "day_of_week"]

    
    def mock_load(path):
        return DummyModel()
        
    monkeypatch.setattr(joblib, "load", mock_load)
    
    app = create_app()
    # Mock settings so init_db doesn't affect production DB if not caught
    from app.config import get_settings, Settings
    get_settings.cache_clear()
    def mock_settings():
        return Settings(database_url="sqlite:///:memory:", app_env="test", razorpay_key_id="test", razorpay_key_secret="test", razorpay_webhook_secret="test")
    monkeypatch.setattr("app.main.get_settings", mock_settings)
    
    with TestClient(app) as client:
        assert isinstance(app.state.predictor, EconomicPolicyPredictor)
        assert hasattr(app.state, "guardrails")
        assert hasattr(app.state, "executor")

def test_lifespan_model_fallback(monkeypatch):
    def mock_load_error(path):
        raise FileNotFoundError("Model not found")
        
    monkeypatch.setattr(joblib, "load", mock_load_error)
    
    app = create_app()
    from app.config import get_settings, Settings
    get_settings.cache_clear()
    def mock_settings():
        return Settings(database_url="sqlite:///:memory:", app_env="test", razorpay_key_id="test", razorpay_key_secret="test", razorpay_webhook_secret="test")
    monkeypatch.setattr("app.main.get_settings", mock_settings)
    
    with TestClient(app) as client:
        assert isinstance(app.state.predictor, PlaceholderPredictor)
