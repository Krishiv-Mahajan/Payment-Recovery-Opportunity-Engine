import math
from datetime import datetime, timezone
import pytest

from app.retry_policy import (
    calculate_retry,
    MAX_EXECUTION_ATTEMPTS,
    BASE_DELAY_SECONDS,
    MAX_DELAY_SECONDS,
)
from app.executor import TransientExecutionError, PermanentExecutionError, ReconciliationError


def test_calculate_retry_attempt_progression():
    # Attempt 1 -> Retry
    res = calculate_retry(1, TransientExecutionError("mock", error_type="MOCK"))
    assert res.retryable is True
    assert res.is_final_failure is False

    # Attempt 4 -> Retry
    res = calculate_retry(4, TransientExecutionError("mock", error_type="MOCK"))
    assert res.retryable is True
    assert res.is_final_failure is False

    # Attempt 5 -> Final Failure (max attempts reached)
    res = calculate_retry(5, TransientExecutionError("mock", error_type="MOCK"))
    assert res.retryable is False
    assert res.is_final_failure is True
    assert "Maximum attempts" in res.reason


def test_calculate_retry_backoff_math():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    
    # Force no jitter to test exact math
    def no_jitter():
        return 0.5  # middle of 0.8-1.2 -> 1.0 multiplier

    # Attempt 1: 60 * 2^0 = 60s
    res = calculate_retry(1, TransientExecutionError("mock", error_type="MOCK"), now=now, random_func=no_jitter)
    assert (res.next_retry_at - now).total_seconds() == 60.0

    # Attempt 2: 60 * 2^1 = 120s
    res = calculate_retry(2, TransientExecutionError("mock", error_type="MOCK"), now=now, random_func=no_jitter)
    assert (res.next_retry_at - now).total_seconds() == 120.0

    # Attempt 3: 60 * 2^2 = 240s
    res = calculate_retry(3, TransientExecutionError("mock", error_type="MOCK"), now=now, random_func=no_jitter)
    assert (res.next_retry_at - now).total_seconds() == 240.0


def test_calculate_retry_maximum_delay():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    
    def no_jitter():
        return 0.5

    # A very high attempt (e.g. if MAX_EXECUTION_ATTEMPTS was higher)
    import app.retry_policy
    orig_max = app.retry_policy.MAX_EXECUTION_ATTEMPTS
    app.retry_policy.MAX_EXECUTION_ATTEMPTS = 20
    try:
        res = calculate_retry(10, TransientExecutionError("mock", error_type="MOCK"), now=now, random_func=no_jitter)
        assert (res.next_retry_at - now).total_seconds() == MAX_DELAY_SECONDS
    finally:
        app.retry_policy.MAX_EXECUTION_ATTEMPTS = orig_max


def test_calculate_retry_jitter():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # test min jitter
    res_min = calculate_retry(1, TransientExecutionError("mock", error_type="MOCK"), now=now, random_func=lambda: 0.0)
    assert (res_min.next_retry_at - now).total_seconds() == 60.0 * 0.8

    # test max jitter
    res_max = calculate_retry(1, TransientExecutionError("mock", error_type="MOCK"), now=now, random_func=lambda: 0.9999999)
    # almost 1.2
    assert math.isclose((res_max.next_retry_at - now).total_seconds(), 60.0 * 1.2, rel_tol=1e-5)


def test_calculate_retry_permanent_errors():
    res = calculate_retry(1, PermanentExecutionError("mock", error_type="MOCK"))
    assert res.retryable is False
    assert res.is_final_failure is True
    assert res.next_retry_at is None

    res2 = calculate_retry(1, ReconciliationError("mock", error_type="MOCK"))
    assert res2.retryable is False
    assert res2.is_final_failure is True


def test_calculate_retry_unknown_error():
    res = calculate_retry(1, ValueError("random exception"))
    assert res.retryable is False
    assert res.is_final_failure is True
    assert "Unknown error type: ValueError" in res.reason
