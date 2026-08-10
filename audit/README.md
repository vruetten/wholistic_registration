# audit/ — findings ledgers

One ledger per review pass. The process, evidence protocol, finding format,
severity scale, and statuses are defined in the `/full-package-review` skill
(`~/.claude/skills/full-package-review/SKILL.md`, user-level); the instance plan is
`plan/full-package-review.md`.

| File | Pass | Status |
|---|---|---|
| `pass-0-inventory.md` | Inventory & liveness (dead files) | **done 2026-08-10** |
| `pass-1-bugs.md` | Bugs & errors | **done 2026-08-10** (96 confirmed / 18 refuted; regression tests pending) |
| `pass-1-verification-log.md` | Full verifier evidence for Pass 1 | archive |
| `pass-2-math.md` | Math & numerics | not started |
| `pass-3-performance.md` | Performance | not started |
| `pass-4-architecture.md` | Architecture | not started |

Repo-hygiene/packaging/tooling findings go to the top-level `AUDIT.md`
(2026-05-27), not here.
