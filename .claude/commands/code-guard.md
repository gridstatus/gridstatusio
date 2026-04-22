## Review code changes against CLAUDE.md guidelines to catch issues and ensure maintainability

Follow the steps below and only output the final Report Out section.

1. **Collect Inputs**
   - Review changes in this branch only
   - Base branch for the diff: `$ARGUMENTS` (default to `main` if empty)
   - Use `git diff <base>...HEAD` to show all changes
   - Note any context the user provided (goal, constraints, known tradeoffs)

2. **Load Guidelines**
   - Read the workspace `../CLAUDE.md` for monorepo-wide rules
   - Read `developing.md` and `README.md` for client conventions if touched areas need it
   - No repo-local `CLAUDE.md` exists here — rely on conventions evident in nearby code

3. **Build a Mental Model**
   - Identify what the change is trying to accomplish
   - Map what's touched: `GridStatusClient`, request helpers, pagination, retries, typing, examples, tests
   - Flag risk zones:
     - **Public API surface**: renamed/removed/re-typed methods or kwargs are breaking changes for users
     - **Retry and rate-limit behavior**: free tier is 1M rows/month; avoid accidental request amplification
     - **Pagination / large responses**: memory blow-ups, silent truncation
     - **Auth / API key handling**: never log or surface keys in errors
     - **Datetime handling**: tz-aware inputs and outputs consistent with `gridstatus`
     - **Return-type stability**: DataFrame columns/dtypes and order are part of the contract

4. **Review the Diff**
   - Before flagging a concern, weigh it against the PR's stated context — description, commit messages, inline comments. If the author's reasoning already addresses the concern, drop it.
   - Go file by file
   - Verify correctness and intent
   - Look for bugs, logic errors, regressions, and common AI-generated mistakes (unused abstractions, inconsistent naming, over-broad `except Exception`, dead code)
   - For new/changed public methods: confirm docstring, type hints, and example usage stay consistent
   - For tests: confirm they don't rely on unstable live data (prefer recorded/mocked responses when possible); slow tests gated by marker

5. **Validate Against CLAUDE.md + Best Practices**
   - Run `make lint` and surface any failures
   - Run `make type-check` (pyright) and surface any failures
   - Flag important best-practice issues not covered by guidelines
   - Cite the relevant guideline / convention for each issue when applicable

6. **Report Out**
   - Only include actual issues — omit anything the PR's description, commit messages, or inline comments already justify. No higher-level summary, no recap of what the PR does, no positive acknowledgments.
   - Keep each issue to one or two sentences: the problem, then the fix.
   - Order issues with sections by priority (blocking, important, nit) with global numbering so each issue can unambiguously be referenced. Omit any priority tier with no issues.
   - At the end, list each issue one by one with a concise (no more than 20 words) description, so the user can reply with numbers. Call this section "Next Steps". The last line should say "Which issue(s) do you want to fix first?".
   - Propose architectural changes only if they materially improve maintainability.
