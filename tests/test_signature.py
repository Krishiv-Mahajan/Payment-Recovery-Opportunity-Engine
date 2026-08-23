"""
Tests for webhook signature verification.

Covers:
  A. Valid webhook signature accepted
  B. Invalid webhook signature rejected
  C. Missing signature header rejected
  D. Correct HTTP status codes
"""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from app.security import verify_webhook_signature
from app.config import get_settings
from tests.conftest import TEST_WEBHOOK_SECRET, PAYMENT_FAILED_PAYLOAD, make_signed_request


# ── Unit tests for verify_webhook_signature ────────────────────────────────────


class TestVerifyWebhookSignature:
    """Unit tests for the signature verification function."""

    def _sign(self, body: bytes, secret: str = TEST_WEBHOOK_SECRET) -> str:
        return hmac.new(
            key=secret.encode("utf-8"),
            msg=body,
            digestmod=hashlib.sha256,
        ).hexdigest()

    def test_valid_signature_returns_true(self, monkeypatch):
        """A correctly signed payload must be accepted."""
        from app import config as config_module
        from app.config import Settings

        settings_override = Settings(
            razorpay_webhook_secret=TEST_WEBHOOK_SECRET,
            database_url="sqlite:///:memory:",
        )
        monkeypatch.setattr(config_module, "get_settings", lambda: settings_override)
        import app.security as sec
        monkeypatch.setattr(sec, "get_settings", lambda: settings_override)

        body = b'{"event":"payment.failed","id":"evt_001"}'
        sig = self._sign(body)

        assert verify_webhook_signature(body, sig) is True

    def test_invalid_signature_returns_false(self, monkeypatch):
        """A tampered payload signature must be rejected."""
        from app import config as config_module
        from app.config import Settings
        import app.security as sec

        settings_override = Settings(
            razorpay_webhook_secret=TEST_WEBHOOK_SECRET,
            database_url="sqlite:///:memory:",
        )
        monkeypatch.setattr(config_module, "get_settings", lambda: settings_override)
        monkeypatch.setattr(sec, "get_settings", lambda: settings_override)

        body = b'{"event":"payment.failed","id":"evt_001"}'
        wrong_sig = "deadbeef" * 8  # Clearly wrong hex string

        assert verify_webhook_signature(body, wrong_sig) is False

    def test_empty_signature_header_returns_false(self, monkeypatch):
        """Missing signature header must be rejected."""
        from app import config as config_module
        from app.config import Settings
        import app.security as sec

        settings_override = Settings(
            razorpay_webhook_secret=TEST_WEBHOOK_SECRET,
            database_url="sqlite:///:memory:",
        )
        monkeypatch.setattr(config_module, "get_settings", lambda: settings_override)
        monkeypatch.setattr(sec, "get_settings", lambda: settings_override)

        body = b'{"event":"payment.failed","id":"evt_001"}'
        assert verify_webhook_signature(body, "") is False

    def test_missing_webhook_secret_rejects_all(self, monkeypatch):
        """If no webhook secret is configured, all webhooks must be rejected (fail-secure)."""
        from app import config as config_module
        from app.config import Settings
        import app.security as sec

        settings_override = Settings(
            razorpay_webhook_secret="",  # No secret configured
            database_url="sqlite:///:memory:",
        )
        monkeypatch.setattr(config_module, "get_settings", lambda: settings_override)
        monkeypatch.setattr(sec, "get_settings", lambda: settings_override)

        body = b'{"event":"payment.failed","id":"evt_001"}'
        sig = self._sign(body)  # Technically correct but secret is empty

        # Must still reject — empty secret = reject all
        assert verify_webhook_signature(body, sig) is False

    def test_different_secret_returns_false(self, monkeypatch):
        """Signature produced with the wrong secret must be rejected."""
        from app import config as config_module
        from app.config import Settings
        import app.security as sec

        settings_override = Settings(
            razorpay_webhook_secret=TEST_WEBHOOK_SECRET,
            database_url="sqlite:///:memory:",
        )
        monkeypatch.setattr(config_module, "get_settings", lambda: settings_override)
        monkeypatch.setattr(sec, "get_settings", lambda: settings_override)

        body = b'{"event":"payment.failed","id":"evt_001"}'
        # Sign with a DIFFERENT secret
        wrong_sig = self._sign(body, secret="completely_different_secret")

        assert verify_webhook_signature(body, wrong_sig) is False


# ── Integration tests via HTTP endpoint ───────────────────────────────────────


class TestWebhookEndpointSignature:
    """Integration tests for the webhook endpoint signature handling."""

    def test_valid_signature_returns_200(self, client):
        """A correctly signed request must return HTTP 200."""
        response, _ = make_signed_request(client, PAYMENT_FAILED_PAYLOAD)
        assert response.status_code == 200

    def test_invalid_signature_returns_400(self, client):
        """A request with a wrong signature must return HTTP 400."""
        body = json.dumps(PAYMENT_FAILED_PAYLOAD, separators=(",", ":")).encode()
        response = client.post(
            "/webhooks/razorpay",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": "invalidsignature",
            },
        )
        assert response.status_code == 400

    def test_missing_signature_header_returns_400(self, client):
        """A request with no X-Razorpay-Signature header must return HTTP 400."""
        body = json.dumps(PAYMENT_FAILED_PAYLOAD, separators=(",", ":")).encode()
        response = client.post(
            "/webhooks/razorpay",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

    def test_empty_body_returns_400(self, client):
        """An empty request body must return HTTP 400."""
        response = client.post(
            "/webhooks/razorpay",
            content=b"",
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": "anything",
            },
        )
        assert response.status_code == 400

    def test_invalid_json_returns_400(self, client):
        """Malformed JSON that passes signature check (using matching sig) must return 400."""
        bad_json = b"this is not json {"
        import hashlib, hmac as hmac_mod
        sig = hmac_mod.new(
            key=TEST_WEBHOOK_SECRET.encode(),
            msg=bad_json,
            digestmod=hashlib.sha256,
        ).hexdigest()
        response = client.post(
            "/webhooks/razorpay",
            content=bad_json,
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": sig,
            },
        )
        assert response.status_code == 400
