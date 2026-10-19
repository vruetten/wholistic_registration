import tempfile
import unittest
from pathlib import Path

import numpy as np

from wholistic_registration.utils.registration_checkpoint import (
    checkpoint_elapsed_seconds,
    load_checkpoint,
    load_json_state,
    registration_fingerprint,
    save_checkpoint,
    save_json_state,
)


class RegistrationCheckpointTests(unittest.TestCase):
    def test_checkpoint_and_state_round_trip_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint_path = root / "nested" / "state.npz"
            state_path = root / "nested" / "run.json"
            save_checkpoint(
                checkpoint_path,
                {"next_chunk": 3, "elapsed_seconds": 12.5},
                motion_init=np.arange(6, dtype=np.float32).reshape(2, 3),
            )
            save_json_state(state_path, {"status": "running", "elapsed_seconds": 12.5})

            metadata, arrays = load_checkpoint(checkpoint_path)
            state = load_json_state(state_path)

            self.assertEqual(metadata["next_chunk"], 3)
            self.assertEqual(checkpoint_elapsed_seconds(checkpoint_path), 12.5)
            np.testing.assert_array_equal(arrays["motion_init"], np.arange(6).reshape(2, 3))
            self.assertEqual(state["status"], "running")
            self.assertEqual(list(root.rglob("*.tmp")), [])

    def test_checkpoint_rejects_object_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                save_checkpoint(Path(tmp) / "bad.npz", {}, unsafe=np.array([object()]))

    def test_fingerprint_changes_when_config_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.toml"
            input_file = root / "movie.nd2"
            code = root / "main.py"
            config.write_text("setting = 1\n", encoding="utf-8")
            input_file.write_bytes(b"movie")
            code.write_text("pass\n", encoding="utf-8")
            first = registration_fingerprint(config, input_file, [code])
            config.write_text("setting = 2\n", encoding="utf-8")
            second = registration_fingerprint(config, input_file, [code])
            self.assertNotEqual(first, second)

if __name__ == "__main__":
    unittest.main()
