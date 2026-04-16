# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. Guardrails

- Never overwrite user edits between reads.
- Never restore deleted code without confirmation.
- Make the smallest fix that solves the problem.
- No scope drift: no refactors, restyles, or extras unless asked.
- Fix root causes, not symptoms.
- Use web search for unstable or version-specific behavior; cite sources.
- State assumptions; ask only when blocked.
- Briefly narrate multi-step tool usage.
- Finish the full plan once started.

## 1. Non-Destructive File Handling

**Preserve user data. Prefer additive changes over destructive ones.**

- Never delete files or directories.
- Never run destructive commands or cleanup operations that remove user data.
- Do not use `rm`, `del`, `erase`, `rmdir`, `Remove-Item`, or equivalents.
- Do not remove files just because they look temporary, generated, redundant, cached, old, or replaceable.
- Do not delete files before recreating or renaming them.
- If replacement is needed, write a new file and leave the original intact.
- If renaming fails, keep the old file and create a new one with a different name.
- If deletion seems necessary, stop and propose a non-destructive alternative first.

Default rule: when in doubt, keep all existing files.

## 2. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 3. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 4. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 5. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
