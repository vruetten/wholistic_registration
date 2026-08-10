# Verifier subagent brief — template

One verifier per finding. The verifier's job is to *refute*, so only findings
that survive an honest attack get confirmed. Fill every {placeholder}.

---

Attempt to refute this code-review finding, at commit {sha}:

{finding: ID, severity, file:line, claim, failure scenario, reviewer evidence}

Attack it from every angle:
1. Read the full surrounding context — the whole function and its callers.
   Is the "bug" guarded upstream, unreachable, or intentional?
2. Trace the failure scenario concretely: do the claimed inputs actually
   reach this line with the claimed state?
3. Check the mechanism: does the language/library actually behave as the
   claim assumes? Verify, don't pattern-match.
4. If the code path runs in this environment, write and run a minimal
   snippet that decides the question.

Verdict — exactly one:
- **REFUTED** — with the specific reason (the guard, the real behavior, the
  unreachable path). Quote the code that refutes it.
- **CONFIRMED-run** — your executed snippet reproduces it. Include the
  snippet and its output.
- **CONFIRMED-read** — you attacked it and failed to clear it; the code path
  cannot execute here (say why: GPU, data, credentials). Include your written
  "why might this be fine?" analysis and what it would take to run it.

An unclear verdict is REFUTED-in-practice: if you cannot state the failure
scenario more precisely than the reviewer did, say what is missing and lean
REFUTED. Return raw markdown: verdict, reasoning, evidence.
