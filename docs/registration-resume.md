# Resuming a registration run

The registration runner writes a small JSON run manifest and numeric checkpoints under the configured `registrated_path`. The middle reference block is checkpointed after each batch. Forward and backward processing have independent checkpoints, written after each complete reference chunk. A completed chunk is reused after restart; if interruption occurs inside the active chunk, that chunk is reprocessed from its beginning and already completed chunks remain untouched.

To continue an interrupted run, use the same TOML file, input ND2, output directory, and code revision, and call:

```python
from wholistic_registration.core.main_function import Registration_v3

Registration_v3("/path/to/config.toml", parallel=True, resume=True)
```

`resume=True` verifies a fingerprint of the configuration, input-file path/size/mtime, and registration source files before reading checkpoints. A mismatch is rejected to avoid mixing incompatible outputs. A completed run is not restarted by a resume call. Use the normal `resume=False` invocation to start a fresh run; that resets the configured output folders and checkpoint files.

Elapsed time is stored in the manifest and checkpoints and remains cumulative across resume attempts. Progress logging is a separate, optional change.
