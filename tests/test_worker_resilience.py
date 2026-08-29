import threading
import time
from unittest.mock import MagicMock, patch

from app.models import RecoveryDecision, DecisionStatus, PaymentRecord, RecoveryAction
from app.worker import start_worker
from app.config import get_settings


def test_worker_survives_cycle_exception():
    settings = get_settings()
    settings.worker_poll_interval_seconds = 0.1
    executor = MagicMock()
    stop_event = threading.Event()
    
    # Simulate first call raises exception, second call sets stop_event to break out cleanly
    call_count = 0
    
    def mock_run_cycle(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Simulated Database Connection Drop")
        if call_count == 2:
            stop_event.set()
            
    with patch("app.worker.run_worker_cycle", side_effect=mock_run_cycle):
        # This will block until stop_event is set (on cycle 2)
        start_worker(db_engine=MagicMock(), settings=settings, executor=executor, stop_event=stop_event)
        
    assert call_count == 2, "Worker should have survived the first exception and executed cycle 2"
