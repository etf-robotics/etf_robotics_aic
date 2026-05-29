# aic_task — Skill Routing

Before reading any doc here, route through one skill. Each skill body
names the single doc and the single source file to read, and explicitly
lists the docs to **not** read for that task — that's where the savings
come from.

- **Editing the existing port-insertion task** (tighten a threshold,
  change a randomization range, tweak the DiffIK scale, add an
  observation term, swap the controlled body, swap the port, change
  `sim.dt`): invoke
  [`aic-task-edit`](../../../.claude/skills/aic-task-edit/SKILL.md).
- **Writing or editing a class/function under `mdp/`** (a new
  termination, a new event, a new command, or substantive changes to an
  existing term's logic): invoke
  [`aic-mdp-term-work`](../../../.claude/skills/aic-mdp-term-work/SKILL.md).
- **Adding a new Gym ID** (a second task subpackage alongside
  `AIC-Port-Insertion-v0`): invoke
  [`aic-task-add`](../../../.claude/skills/aic-task-add/SKILL.md).
- **Doc-anchor maintenance** (re-verify the `last_verified_commit` SHAs
  and file:line references after a source change): invoke
  [`docs-task-sync`](../../../.claude/skills/docs-task-sync/SKILL.md).

The skill bodies are short; the docs in [docs/](../docs/) do the heavy
lifting. Skipping the skill and reading docs cover-to-cover is the
expensive path — pick one skill first.
