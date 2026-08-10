# audit/ — findings ledgers

One file per review pass, produced by the process defined in
`plan/full-package-review.md` (which also defines the evidence protocol,
ID scheme, severity scale, and verification statuses).

| File | Pass | Status |
|---|---|---|
| `pass-0-inventory.md` | Inventory & liveness (dead files) | not started |
| `pass-1-bugs.md` | Bugs & errors | not started |
| `pass-2-math.md` | Math & numerics | not started |
| `pass-3-performance.md` | Performance | not started |
| `pass-4-architecture.md` | Architecture | not started |

Repo-hygiene/packaging/tooling findings do **not** go here — append those to
the top-level `AUDIT.md` (2026-05-27) instead.

## Finding format

```markdown
### B-014 🟧 `utils/interp.py:88` — claim in one sentence
- **Commit:** <sha reviewed>
- **Claim:** ...
- **Failure scenario:** concrete inputs/state → wrong output/crash
- **Status:** CONFIRMED-run | CONFIRMED-read | PLAUSIBLE | REFUTED
- **Evidence:** repro snippet / adversarial-check notes / refutation
- **Test:** tests/regressions/test_b014_....py (if CONFIRMED-run)
```
