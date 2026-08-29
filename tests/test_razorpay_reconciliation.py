import pytest
import httpx
from unittest.mock import patch, MagicMock

from app.executor_razorpay import RazorpayPaymentLinkProvider
from app.executor import TransientExecutionError, PermanentExecutionError, AmbiguousStateError, ReconciliationError
from app.models import RecoveryDecision, DecisionStatus, PaymentRecord


@pytest.fixture
def mock_decision():
    payment = PaymentRecord(id=101, amount=1000, currency="INR")
    return RecoveryDecision(id=102, payment_record=payment, decision_status=DecisionStatus.EXECUTING, execution_attempts=3)


@pytest.fixture
def provider():
    return RazorpayPaymentLinkProvider("test_id", "test_secret")


def mock_response(status_code, json_data):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    return resp


def test_reconciliation_succeeds_immediately(provider, mock_decision):
    with patch("httpx.Client.get") as mock_get:
        mock_get.return_value = mock_response(200, {"items": [{"id": "plink_123", "reference_id": "roe_rd_102", "amount": 1000, "currency": "INR"}]})
        
        plink_id = provider.reconcile_duplicate_reference(mock_decision, "exec_rd_102")
        
        assert plink_id == "plink_123"
        assert mock_get.call_count == 1


@patch("time.sleep")
def test_reconciliation_succeeds_after_one_delay(mock_sleep, provider, mock_decision):
    with patch("httpx.Client.get") as mock_get:
        mock_get.side_effect = [
            mock_response(200, {"items": []}),  # 1st attempt empty
            mock_response(200, {"items": [{"id": "plink_123", "reference_id": "roe_rd_102", "amount": 1000, "currency": "INR"}]}),  # 2nd succeeds
        ]
        
        plink_id = provider.reconcile_duplicate_reference(mock_decision, "exec_rd_102")
        
        assert plink_id == "plink_123"
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once_with(1)


@patch("time.sleep")
def test_reconciliation_succeeds_after_two_delays(mock_sleep, provider, mock_decision):
    with patch("httpx.Client.get") as mock_get:
        mock_get.side_effect = [
            mock_response(200, {"items": []}),  # 1st empty
            mock_response(200, {"items": []}),  # 2nd empty
            mock_response(200, {"items": [{"id": "plink_123", "reference_id": "roe_rd_102", "amount": 1000, "currency": "INR"}]}),  # 3rd succeeds
        ]
        
        plink_id = provider.reconcile_duplicate_reference(mock_decision, "exec_rd_102")
        
        assert plink_id == "plink_123"
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2


@patch("time.sleep")
def test_reconciliation_never_appears(mock_sleep, provider, mock_decision):
    with patch("httpx.Client.get") as mock_get:
        mock_get.return_value = mock_response(200, {"items": []})  # always empty
        
        with pytest.raises(AmbiguousStateError) as exc_info:
            provider.reconcile_duplicate_reference(mock_decision, "exec_rd_102")
            
        assert "No payment link found" in str(exc_info.value)
        assert mock_get.call_count == 3
        assert mock_sleep.call_count == 2
        # Verify execution attempts not incremented
        assert mock_decision.execution_attempts == 3


def test_reconciliation_transient_failure(provider, mock_decision):
    with patch("httpx.Client.get") as mock_get:
        mock_get.side_effect = httpx.TimeoutException("Timeout")
        
        with pytest.raises(TransientExecutionError) as exc_info:
            provider.reconcile_duplicate_reference(mock_decision, "exec_rd_102")
            
        assert "timed out" in str(exc_info.value)
        assert mock_get.call_count == 1
        assert mock_decision.execution_attempts == 3


def test_reconciliation_integrity_mismatch(provider, mock_decision):
    with patch("httpx.Client.get") as mock_get:
        # Amount differs (2000 vs 1000)
        mock_get.return_value = mock_response(200, {"items": [{"id": "plink_123", "reference_id": "roe_rd_102", "amount": 2000, "currency": "INR"}]})
        
        with pytest.raises(ReconciliationError) as exc_info:
            provider.reconcile_duplicate_reference(mock_decision, "exec_rd_102")
            
        assert "Mismatched amount" in str(exc_info.value)
