# Recovery Opportunity Engine

**Razorpay Buildathon Track 03 — Phase 5**

An AI-driven revenue recovery decision engine that evaluates failed Razorpay payments and decides whether and how to intervene.

> **Phase 5 scope**: This version is production-ready, featuring an S-Learner ML policy, operational safety guardrails, a simulated outbox executor, live Razorpay Payment Link integration, and Prometheus metrics.
> See [What is NOT implemented yet](#what-is-not-implemented-yet) for the full boundary.

---

## Core Concept

The system's central question is:

> **"Should we intervene?"** — before **"What should we do?"**

Not every failed payment benefits from intervention. The eventual system will estimate incremental recovery value:

```
Incremental Gross Recovery(action)
= P(recovery | context, action) − P(recovery | context, no_action)
× recoverable amount
− intervention cost
```

Treating **DO NOTHING** as a first-class, legitimate action.

---

## Phase 5: What Is Implemented

The system is now fully closed-loop:

Razorpay webhook (payment.failed)
    ↓
Signature verification & Idempotency Check
    ↓
Event normalization (PaymentRecord)
    ↓
Feature Extraction (customer history & payment dimensions)
    ↓
ML Policy Prediction (S-Learner predicts P0, P1, Uplift)
    ↓
GuardrailsEngine (blocks duplicates, handles model fallback, enforces 48h cooldown)
    ↓
RecoveryDecision (persisted as DECIDED)
    ↓
Action Execution (Mock or real Razorpay Payment Links API)
    ↓
Execution Correlation & Audit Logs
    ↓
payment_link.paid webhook closes the loop (OUTCOME_OBSERVED)

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/webhooks/razorpay` | Razorpay webhook receiver |
| `GET`  | `/health` | Liveness check |
| `GET`  | `/events/{event_id}` | Retrieve a webhook event by Razorpay event ID |
| `GET`  | `/metrics` | Prometheus operational metrics |

### Database Tables

| Table | Purpose |
|-------|---------|
| `webhook_events` | Raw event storage, signature status, processing lifecycle |
| `payment_records` | Normalized payment entity — updated by subsequent events |
| `recovery_decisions` | ML predictions, execution reference, and outcome tracking |
| `audit_logs` | Immutable append-only processing history |
| `alembic_version` | Tracks database migration state |
| `customer_outreach_events` | Tracks outreach actions for operational cooldown guardrails |

---

## What Is NOT Implemented Yet

| Feature | Reason deferred |
|---------|-----------------|
| A/B experimentation framework | Planned for Phase 6 |
| WhatsApp / SMS / email channels | Currently limited to Payment Links |
| Dashboard / frontend | Planned for Phase 7 |
| PostgreSQL migration | Architecture supports it via Alembic; not needed for local dev |
| Redis / Kafka | Outbox is simulated synchronously for simplicity |

---

## Local Setup

### Prerequisites

- Python 3.12+ (tested with 3.12.13)
- A Razorpay test-mode account (for real webhook testing; not required for unit tests)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd recovery_opportunity_engine

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your Razorpay test-mode credentials
```

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
RAZORPAY_KEY_ID=rzp_test_REPLACEME
RAZORPAY_KEY_SECRET=REPLACEME
RAZORPAY_WEBHOOK_SECRET=REPLACEME
DATABASE_URL=sqlite:///./roe.db
```

| Variable | Description |
|----------|-------------|
| `RAZORPAY_KEY_ID` | Razorpay test-mode API key ID |
| `RAZORPAY_KEY_SECRET` | Razorpay test-mode API key secret |
| `RAZORPAY_WEBHOOK_SECRET` | Webhook signing secret |
| `DATABASE_URL` | SQLAlchemy connection string |
| `EXECUTOR_MODE` | `mock` (default) or `razorpay` (requires valid keys) |
| `COOLDOWN_HOURS` | Number of hours before a customer can be contacted again (default: 48) |

---

## Running the Server

```bash
# Activate venv first
source .venv/bin/activate

# Start the development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Running Tests

Tests use an in-memory SQLite database and require **no live Razorpay credentials**.

```bash
# Activate venv first
source .venv/bin/activate

# Run all tests with verbose output
pytest -v

# Run a specific test file
pytest tests/test_signature.py -v
pytest tests/test_processing.py -v
pytest tests/test_features.py -v
pytest tests/test_policy.py -v
```

### Test Coverage

| Test File | Covers |
|-----------|--------|
| `test_signature.py` | Signature verification (valid, invalid, missing, no secret) |
| `test_processing.py` | Webhook processing, duplicate handling, event normalization, outcomes |
| `test_features.py` | Feature extraction: field values, history counts, None handling |
| `test_policy.py` | ML predictor interface |
| `test_guardrails.py` | Operational safety: Duplicate prevention, Model fallback, 48h Cooldown |
| `test_executor.py` | Mock executor determinism and abstraction |
| `test_razorpay_executor.py` | Razorpay API request building and error handling (mocked network) |
| `test_metrics.py` | Prometheus metrics formatting and high-cardinality absence |
| `test_migrations.py` | Alembic up/down migrations |
| `test_lifespan.py` | Startup configuration, executor selection, model loading |

---

## Webhook Testing Strategy

### Unit Tests (no external dependencies)

All tests in `tests/` use mocked settings and in-memory databases. Run `pytest` to verify the full pipeline without any Razorpay credentials.

### Manual Testing with a Local Tunnel

To test with real Razorpay test-mode webhook deliveries:

1. Start the server: `uvicorn app.main:app --reload --port 8000`
2. Open a public tunnel: `ngrok http 8000`
3. In Razorpay Dashboard → Settings → Webhooks, add:
   - URL: `https://<your-ngrok-subdomain>.ngrok-free.app/webhooks/razorpay`
   - Events: `payment.failed`, `payment.captured`, `order.paid`
4. Copy the webhook secret shown in the Dashboard into your `.env` as `RAZORPAY_WEBHOOK_SECRET`
5. Trigger a test payment in Razorpay test mode

### Manual Simulation (no Razorpay account needed)

You can simulate a correctly-signed webhook using the following Python snippet:

```python
import hashlib, hmac, json, requests

SECRET = "your_webhook_secret_here"
payload = {
    "entity": "event",
    "event": "payment.failed",
    "id": "evt_manual_test_001",
    "contains": ["payment"],
    "payload": {
        "payment": {
            "entity": {
                "id": "pay_manual_test_001",
                "entity": "payment",
                "amount": 50000,
                "currency": "INR",
                "status": "failed",
                "method": "card",
                "email": "test@example.com",
                "contact": "+919999999999",
                "error_code": "BAD_REQUEST_ERROR",
                "error_description": "Simulated failure",
                "error_source": "customer",
                "error_step": "payment_authentication",
                "error_reason": "payment_failed",
                "created_at": 1700000000
            }
        }
    },
    "created_at": 1700000000
}
# Sign the exact bytes sent in the request. Do not parse and re-serialize an
# incoming Razorpay webhook before verifying its signature.
body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
sig = hmac.new(SECRET.encode("utf-8"), body, hashlib.sha256).hexdigest()

response = requests.post(
    "http://localhost:8000/webhooks/razorpay",
    data=body,
    headers={"Content-Type": "application/json", "X-Razorpay-Signature": sig}
)
print(response.status_code, response.json())
```

---

## Security Notes

- **Secrets**: `RAZORPAY_KEY_SECRET` and `RAZORPAY_WEBHOOK_SECRET` are never logged, never included in API responses, and `.env` is gitignored.
- **Signature verification**: Performed on the raw request body bytes before any parsing. Uses `hmac.compare_digest` to prevent timing attacks.
- **Fail-secure**: If `RAZORPAY_WEBHOOK_SECRET` is not configured, all webhooks are rejected.
- **Idempotency**: Duplicate webhook deliveries are safely detected and acknowledged without reprocessing.
- **Input validation**: All webhook payloads are validated against Pydantic schemas before processing.
- **ORM**: All database queries use SQLAlchemy parameterized operations — no raw SQL string interpolation.
- **Audit trail**: Processing failures are recorded with status `FAILED` on the webhook event row, preserving the raw payload for investigation.

---

## Project Structure

```
recovery_opportunity_engine/
├── .env.example           # Template for environment variables
├── .gitignore             # Excludes .env, .venv, *.db, __pycache__
├── requirements.txt       # Minimal justified dependencies
├── pytest.ini             # Test configuration
├── README.md              # This file
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app factory + route registration
│   ├── config.py          # Environment-based settings (pydantic-settings)
│   ├── database.py        # SQLAlchemy engine + session factory
│   ├── models.py          # ORM models: WebhookEvent, PaymentRecord, RecoveryDecision, AuditLog
│   ├── schemas.py         # Pydantic validation schemas for webhook payloads
│   ├── security.py        # HMAC-SHA256 webhook signature verification
│   ├── crud.py            # Database operations (idempotent upserts)
│   ├── event_processor.py # Event routing and processing pipeline orchestration
│   ├── webhooks.py        # POST /webhooks/razorpay endpoint
│   └── ml/
│       ├── __init__.py
│       ├── features.py    # Deterministic feature extraction
│       └── predictor.py   # Placeholder policy interface
└── tests/
    ├── __init__.py
    ├── conftest.py         # Shared fixtures, test payloads, helpers
    ├── test_signature.py   # Signature verification tests
    ├── test_processing.py  # Webhook processing + normalization tests
    ├── test_features.py    # Feature extraction tests
    └── test_policy.py      # Placeholder predictor + health endpoint tests
```

---

## Razorpay API Assumptions

Phase 1 makes the following documented assumptions about Razorpay webhooks:

| Assumption | Source |
|------------|--------|
| Webhook signature uses HMAC-SHA256, key=webhook_secret, msg=raw_body | Razorpay docs: webhooks/validate-test |
| Signature delivered in `X-Razorpay-Signature` header | Razorpay docs: webhooks |
| Root-level `id` field is the unique event identifier | Razorpay webhook payload structure |
| `payment.failed` contains `payload.payment.entity` with error fields | Razorpay docs: payment entity |
| `payment.captured` contains `payload.payment.entity` | Razorpay docs: payment entity |
| `order.paid` may contain both `payload.payment.entity` and `payload.order.entity` | Razorpay docs: order entity |
| Razorpay retries on non-2xx responses (exponential backoff, up to 24h) | Razorpay docs: webhooks |

**No Razorpay API endpoints are invented.** Phase 1 does not call any Razorpay APIs — it only receives webhooks.
