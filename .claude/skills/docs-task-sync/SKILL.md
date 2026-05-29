---
name: docs-task-sync
description: Check whether source/aic_task/docs/ and the aic-task-* skill bodies are in sync with the current source tree, and report which need updating before editing. Use after touching files under source/aic_task/aic_task/tasks/manager_based/ or asset_specs/, or when an aic-task-* skill's pointers look stale.
---

# docs-task-sync

The `source/aic_task/docs/` set is anchor-heavy: tables call out
file:line locations, and each doc carries a `last_verified_commit` SHA
in its frontmatter. The
[aic-task-edit](../aic-task-edit/SKILL.md),
[aic-mdp-term-work](../aic-mdp-term-work/SKILL.md), and
[aic-task-add](../aic-task-add/SKILL.md) skills all point at those
anchors. When the source moves, both can drift.

This skill is the **sync check**: before trusting a skill's pointer or
before posting a docs PR, run the checklist below and flag everything
that needs to move.

Do **not** invoke this skill for ordinary edits. The aic-task-* skills
already give you the right routing for a single change; this one is for
when you suspect the routing itself is wrong.

## Run this checklist

1. Read each doc's frontmatter `last_verified_commit` SHA. Files of
   interest:
   - [01_package_structure.md](../../../source/aic_task/docs/01_package_structure.md)
   - [02_gym_registration.md](../../../source/aic_task/docs/02_gym_registration.md)
   - [03_port_insertion_overview.md](../../../source/aic_task/docs/03_port_insertion_overview.md)
   - [04_assembly_pattern.md](../../../source/aic_task/docs/04_assembly_pattern.md)
   - [05_mdp_terms.md](../../../source/aic_task/docs/05_mdp_terms.md)
   - [06_diff_ik_contract.md](../../../source/aic_task/docs/06_diff_ik_contract.md)
   - [08_adding_a_new_task.md](../../../source/aic_task/docs/08_adding_a_new_task.md)
   - [09_modifying_existing_task.md](../../../source/aic_task/docs/09_modifying_existing_task.md)
   - [10_glossary.md](../../../source/aic_task/docs/10_glossary.md)

2. For each doc whose SHA is behind `git rev-parse HEAD`, run:

   ```bash
   git log --oneline <doc-sha>..HEAD -- source/aic_task/aic_task/
   ```

   If the output is empty, the doc is still valid — just bump the SHA in
   its frontmatter as a maintenance pass.

3. For docs with non-empty diff, **check each anchored file:line**. The
   high-risk anchors:

   | Doc | Anchored at |
   |---|---|
   | 03, 09 | line numbers inside `specs.py`, `builders.py`, `port_insertion_env_cfg.py` |
   | 04 | the cheat-sheet table rows (must match `port_insertion/README.md`) |
   | 05 | every term's `mdp/*.py` line number |
   | 09 | every diff block's line context |
   | 10 | the canonical-example file:line per entry |

   Read the linked file at the cited line; if the cited symbol moved,
   record the new line. Re-anchoring is cheap; trusting a stale line
   number costs a debugging round-trip.

4. Check the cheat sheet in
   [port_insertion/README.md](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/README.md#what-lives-where-cheat-sheet)
   against the one reproduced in
   [04_assembly_pattern.md](../../../source/aic_task/docs/04_assembly_pattern.md#where-each-kind-of-change-belongs)
   and the worked-example list in
   [09_modifying_existing_task.md](../../../source/aic_task/docs/09_modifying_existing_task.md#cheat-sheet-entries).
   These three tables are duplicated by design — one is canonical (the
   README), the other two cite the README. If they have diverged, the
   docs lose. Pick the README as ground truth and bring the docs back.

5. Check the **aic-task-* skill bodies** for stale pointers:
   - [aic-task-edit/SKILL.md](../aic-task-edit/SKILL.md) embeds the
     cheat-sheet table by reference. If the README's table has new rows,
     reflect them here too.
   - [aic-mdp-term-work/SKILL.md](../aic-mdp-term-work/SKILL.md) lists
     the term names that route here. If a new
     `CommandTerm`/`ManagerTermBase` lands in
     [port_insertion/mdp/](../../../source/aic_task/aic_task/tasks/manager_based/port_insertion/mdp/),
     add it to the trigger list.
   - [aic-task-add/SKILL.md](../aic-task-add/SKILL.md) embeds the file
     list. If
     [08_adding_a_new_task.md](../../../source/aic_task/docs/08_adding_a_new_task.md)
     changes the file list, mirror it here.

6. When you have a complete list, propose the edits as a single doc-sync
   PR. Bump the `last_verified_commit` SHA on every doc whose anchors
   you re-checked, even if the body needed no other change — that is the
   signal future readers (and this skill) trust.

## What this skill does NOT do

- Read the docs cover-to-cover. The point is the *anchors*; if the body
  is bit-rot, that's a separate cleanup pass.
- Auto-rewrite anchors. Re-anchoring is a judgment call — the symbol
  might have moved, been renamed, or been split.
- Run any sim or test. Sync checks are purely repository-state.
- Apply to docs outside the
  [source/aic_task/docs/](../../../source/aic_task/docs/) set or the
  task-side READMEs. The asset-specs README is small enough to read
  whole; don't bother with anchored sync there.

## Quick start (most common case)

If you just edited one file under
[source/aic_task/aic_task/](../../../source/aic_task/aic_task/) and want
to confirm the docs survived:

```bash
git diff HEAD~1 -- source/aic_task/aic_task/
```

Grep that diff's file list against the doc set:

```bash
grep -l "<file_path>" source/aic_task/docs/*.md
```

If a doc mentions the file you changed, run steps 3–5 for that doc only.
If no doc mentions it, you're done.
