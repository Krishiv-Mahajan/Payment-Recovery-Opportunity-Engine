import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

from app.executor import PermanentExecutionError, TransientExecutionError, ReconciliationError


MAX_EXECUTION_ATTEMPTS = 5
BASE_DELAY_SECONDS = 60.0
MAX_DELAY_SECONDS = 1800.0  # 30 minutes
JITTER_MIN = 0.8
JITTER_MAX = 1.2


@dataclass
class RetryResult:
    retryable: bool
    next_retry_at: datetime | None
    is_final_failure: bool
    reason: str


def calculate_retry(
    attempt_number: int,
    error: Exception,
    now: datetime | None = None,
    random_func: Callable[[], float] | None = None,
) -> RetryResult:
    """
    Calculate the retry parameters for a given execution failure.

    Args:
        attempt_number: The 1-indexed attempt number that just failed (e.g. 1 means the first attempt failed).
        error: The exception that was raised during execution.
        now: The current timezone-aware UTC datetime. Defaults to datetime.now(timezone.utc).
        random_func: A function returning a float [0.0, 1.0) used for jitter. Defaults to random.random.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if random_func is None:
        random_func = random.random

    # 1. Determine if the error is conceptually retryable
    if isinstance(error, (PermanentExecutionError, ReconciliationError)):
        return RetryResult(
            retryable=False,
            next_retry_at=None,
            is_final_failure=True,
            reason="Permanent error",
        )
    
    if not isinstance(error, TransientExecutionError):
        # We err on the side of making unknown exceptions permanent to avoid infinite loops,
        # but in our specific architecture we only process expected structured errors for retry.
        # Unknown/unhandled exceptions leave the state in EXECUTING for crash recovery.
        # If it reaches here, it shouldn't be retried blindly.
        return RetryResult(
            retryable=False,
            next_retry_at=None,
            is_final_failure=True,
            reason=f"Unknown error type: {type(error).__name__}",
        )

    # 2. Check attempt limits
    if attempt_number >= MAX_EXECUTION_ATTEMPTS:
        return RetryResult(
            retryable=False,
            next_retry_at=None,
            is_final_failure=True,
            reason=f"Maximum attempts ({MAX_EXECUTION_ATTEMPTS}) reached",
        )

    # 3. Calculate exponential backoff
    # attempt_number = 1 -> base_delay * 2^0
    # attempt_number = 2 -> base_delay * 2^1
    exponent = attempt_number - 1
    delay = BASE_DELAY_SECONDS * math.pow(2, exponent)
    delay = min(delay, MAX_DELAY_SECONDS)

    # 4. Apply jitter
    jitter_factor = JITTER_MIN + (JITTER_MAX - JITTER_MIN) * random_func()
    final_delay = delay * jitter_factor

    next_retry_at = now + timedelta(seconds=final_delay)

    return RetryResult(
        retryable=True,
        next_retry_at=next_retry_at,
        is_final_failure=False,
        reason=f"Transient error, scheduling attempt {attempt_number + 1}",
    )
