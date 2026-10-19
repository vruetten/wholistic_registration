import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from wholistic_registration.utils.runlog import log_progress


class RegistrationRunLogTests(unittest.TestCase):
    def test_appends_phase_frame_loss_and_elapsed_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "nested" / "registration.log"
            started = time.monotonic() - 0.02
            log_progress(
                log_path,
                started,
                "backward_frame_complete",
                frame=123,
                initial_mse=4.5,
                loss_mse=1.25,
            )
            log_progress(log_path, started, "registration_complete")

            records = [json.loads(line) for line in log_path.read_text().splitlines()]

        self.assertEqual(records[0]["phase"], "backward_frame_complete")
        self.assertEqual(records[0]["frame"], 123)
