"""
Webhook signature verification.

Razorpay signs webhook payloads using HMAC-SHA256.
The signature is delivered in the X-Razorpay-Signature HTTP header.

Verification process:
  1. Take the raw request body exactly as received (do NOT re-serialize).
  2. Compute HMAC-SHA256 of the raw body using the webhook secret as the key.
  3. Compare the computed digest to the X-Razorpay-Signature header value.
  4. Use hmac.compare_digest to prevent timing attacks.

Reference:
  https://razorpay.com/docs/webhooks/validate-test/

SECURITY NOTE:
  The webhook secret is read from the environment.
  It is NEVER logged, returned in responses, or included in any output.
"""

from __future__ import annotations

import hashlib
import hmac

from app.config import get_settings


def verify_webhook_signature(raw_body: bytes, signature_header: str) -> bool:
    """
    Verify that a Razorpay webhook payload has a valid signature.

    Args:
        raw_body: The exact raw bytes of the request body as received.
                  Must NOT be parsed/re-serialized JSON.
        signature_header: The value of the X-Razorpay-Signature header.

    Returns:
        True if the signature is valid, False otherwise.

    Security:
        Uses hmac.compare_digest to resist timing-based side-channel attacks.
        Does not raise exceptions for invalid signatures — returns False.
    """
    settings = get_settings()
    webhook_secret = settings.razorpay_webhook_secret

    if not webhook_secret:
        # If no secret is configured (e.g., during early development),
        # reject all webhooks. Do not silently accept unsigned payloads.
        return False

    if not signature_header:
        return False

    try:
        expected_digest = hmac.new(
            key=webhook_secret.encode("utf-8"),
            msg=raw_body,
            digestmod=hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected_digest, signature_header)

    except Exception:
        # Any unexpected error (encoding issues, etc.) is treated as invalid
        return False
