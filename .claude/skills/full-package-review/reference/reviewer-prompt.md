# Reviewer subagent brief — template

Fill every {placeholder}. The reviewer receives exactly this brief and the
repo — never the parent session's history.

---

Review these files for {pass name} findings, at commit {sha}:

{file list, or file + line-range chunks}

You are one reviewer in a systematic full-package review. Your findings will
be adversarially verified by an independent agent — a finding that doesn't
survive scrutiny costs more than no finding, so report only what you can
evidence.

Checklist — look for exactly these classes:

{paste the relevant pass section from checklists.md verbatim}

Rules:
- Read each file whole before reporting on it. Report only on code you
  actually read; list any assigned file you did not finish under "Not
  reviewed".
- Every finding: `file:line`, a one-sentence falsifiable claim, and a
  concrete failure scenario (inputs/state → wrong output/crash). Findings
  without all three will be discarded.
- Severity: 🟥 wrong results/crash on main path · 🟧 wrong results on
  plausible path · 🟨 latent/edge · 🟦 cosmetic-but-real.
- At most 12 findings. Rank by severity. Few well-evidenced findings beat a
  padded list; an empty list with a strong "looks wrong but is fine" section
  is a valid result.
- Skip finding classes that {linters/tools in use} already enforce.
- {environment note, e.g. "cupy/GPU paths cannot execute here — mark such
  findings [gpu-unverified]"}

Return raw markdown, no prose preamble:

## Findings
### <severity> `file:line` — claim
- **Failure scenario:** ...
- **Evidence:** the code you read that supports the claim (quote it)

## Looks wrong but is fine
- `file:line` — what looks wrong — why it is actually fine
(mandatory: at least one entry, or state explicitly that nothing even looked
wrong)

## Not reviewed
- files/ranges you did not fully read, with reason
