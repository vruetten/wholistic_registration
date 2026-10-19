# Resuming a registration run

The registration runner writes an append-only `registration.log`, a small JSON run manifest, and numeric checkpoints under the configured `registrated_path`. The middle reference block is checkpointed after each batch. Forward and backward processing have independent checkpoints, written after each complete reference chunk. A completed chunk is reused after restart; if interruption occurs inside the active chunk, that chunk is reprocessed from its beginning and already completed chunks remain untouched.

To continue an interrupted run, use the same TOML file, input ND2, output directory, and code revision, and call:

```python
from wholistic_registration.core.main_function import Registration_v3

Registration_v3("/path/to/config.toml", parallel=True, resume=True)
```

`resume=True` verifies a fingerprint of the configuration, input-file path/size/mtime, and registration source files before reading checkpoints. A mismatch is rejected to avoid mixing incompatible outputs. A completed run is not restarted by a resume call. Use the normal `resume=False` invocation to start a fresh run; that resets the configured output folders and the registration log/checkpoint files.

The log's `elapsed_seconds` is cumulative across resume attempts. Frame records include the existing whole-frame pre-registration and post-registration MSE values (`initial_mse` and `loss_mse`); these are not local CFU-region errors.
