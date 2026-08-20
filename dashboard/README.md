# QC dashboard

Visualises and diagnoses the output of `run_F260517_0625_qc_cluster.sh`. Runs entirely on
**ws1**, which has `/nrs` mounted, an RTX A4000, and the `wholistic-registration` env on
the shared home. No packages are installed and the shared conda env is not written to.

## Launch

On ws1 (or over ssh from anywhere):

```bash
cd /groups/ahrens/home/ruttenv/python_packages/wholistic_registration/dashboard
setsid nohup /groups/ahrens/home/ruttenv/miniforge3/envs/wholistic-registration/bin/python \
  server.py \
  --run-dir /nrs/ahrens/Virginia_nrs/wVT/mesoscope/260517_ubbr_mkate_phox2b/registration_out/f260517_0625_qc_v4 \
  --cache-dir /tmp/qc_dash_cache \
  --port 8787 > /tmp/qc_dash.log 2>&1 < /dev/null &
```

Sitting at ws1: open `http://localhost:8787`.

From another machine, forward the port first, then open `http://localhost:8790`:

```bash
ssh -f -N -L 8790:127.0.0.1:8787 ws1
```

The server binds `127.0.0.1` only, so it is not reachable from the network without the
forward.

To stop it, find the process by port rather than by name — `pkill -f server.py` over ssh
also matches the remote shell's own command line and kills the connection:

```bash
ssh ws1 'ss -tlnp | grep 8787'   # read the pid, then kill that pid
```

## Editing

`static/index.html` is re-read from disk on every request, so UI edits need only an
`rsync`. Changes to `server.py`, `qc_bundle.py`, `render.py` or `reproject.py` need a
restart.

```bash
rsync -az dashboard/ ws1:/groups/ahrens/home/ruttenv/python_packages/wholistic_registration/dashboard/
```

## Modules

| File | Role | Selftest |
|---|---|---|
| `qc_bundle.py` | run-directory discovery, z-major cache, axis normalisation, mask and coverage unpacking | `--selftest` against the real run dir |
| `render.py` | arrays to PNG: contrast, colormaps, compositing, quiver, montage | `--selftest`, synthetic only |
| `reproject.py` | GPU projection and scatter, wrapping the pipeline's own functions | `--selftest`, GPU required |
| `server.py` | stdlib HTTP routes, PNG LRU cache, job store | `--selftest`, 22 checks |
| `static/index.html` | the page; inline CSS and JS, no build step, no CDN | none |

`SPEC.md` is the contract, including the axis-order table and the three corrections made
during the build. Read it before changing any module.

## Measured behaviour (ws1, run `f260517_0625_qc_v4`)

| Request | Time |
|---|---|
| `/api/manifest` | 0.001 s |
| `/img/motion` norm, PNG cache warm | 0.078 s |
| `/img/motion` norm, cold | 0.68-1.37 s |
| `/img/motion` with quiver, stride 22 | 0.66 s |
| `/img/refspace` one plane | 0.17 s |
| `/img/refspace` with anatomy overlay | 0.45 s |
| `/img/montage` axis=time, 5 tiles | 0.05-5.1 s |
| `/img/montage` axis=z, 20 tiles | 2.18 s |
| `POST /api/reproject` planes | 1.9-2.2 s compute, plus ~7.6 s cupy init on the first call |
| `/api/summary` | 5.1 s at 5 frames |

The z-major cache costs 2.2 s per frame the first time that frame is touched and is
reused afterwards.

## Known gaps

- `/api/summary` computes its statistics synchronously across every frame and plane. At
  5 frames it takes 5.1 s; at the 200-frame scale of a real run it would take minutes and
  exceed any reasonable client timeout. It needs to move into `qc_bundle.field_summary()`
  with background building and disk persistence (SPEC.md Correction 3).
- `qc_bundle.py` still implements the 5-frame interface: no `frames_for(kind)`,
  `plane_downsampled()`, `field_summary()` or `cache_state()`. Real 200-frame runs will
  have per-artifact frame sets from `PHASE_SAVE_STRIDE` and `REF_SPACE_SAVE_STRIDE` that
  are neither contiguous nor arithmetic, and the z-major cache is currently unbounded.
- The editable install of `wholistic_registration` in the shared env is broken on ws1: its
  `.pth` points at `/groups/ahrens/home/ruttenv/tmp/wr-audit`, which does not exist, so a
  bare `import wholistic_registration` raises `ModuleNotFoundError`. `reproject.py` works
  around it with a `sys.path` fallback; the env itself was not modified.
- The page has never been loaded in a browser by an agent — the Chrome extension was not
  connected. Every route was verified with curl, and every URL the page's builders emit
  was verified to return 200 with the right content type, but rendered layout, debounce
  behaviour and click-through interactions are unverified.
