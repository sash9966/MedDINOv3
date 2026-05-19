## Workflow Orchestration
### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution
### 3. Self-Improvement Loop
- After ANY correction from the user: update 'tasks/lessons.md"
with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project
### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness
### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it
### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how
## Task Management
1. **Plan First**: Write plan to "tasks/todo.md" with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to "tasks/todo.md"
**Capture Lessons**: Update "tasks/lessons.md' after corrections
## Command Formatting
- **All shell commands must be copy-pastable as-is.** No invisible indentation, no extra leading spaces, no smart quotes.
- Multi-line commands: use a single backslash `\` at the end of each continued line with NO trailing space after it, and NO indentation on continuation lines:
  ```
  python feature_viewer_3d.py \
  --nifti image.nii.gz \
  --checkpoint ckpt.pth
  ```
- Never put continuation lines inside an indented code block where the indent would be included in a paste.
- When in doubt, put the entire command on one line rather than risk a broken multi-line paste.
- **Never use `python3 -c "..."` with comments (`#`) or newlines inside the string.** This triggers security prompts and breaks. Always write a temporary `.py` file instead and run it with `python3 tmpscript.py`.

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary Avoid introducing bugs.
## Project Dashboard

Maintain a single `dashboard.html` at the project root — a self-contained, interactive overview updated continuously as the project evolves. It should include:

- **Data**: datasets used, formats, sizes, sources, and preprocessing notes
- **Architecture**: model/system design with a visual diagram if applicable
- **Papers**: linked references with one-line summaries
- **Repos**: GitHub links with branch/experiment status badges
- **Branches & Experiments**: a live table of active branches, their purpose, current status, and key results (editable inline)
- **To-Do**: a persistent checklist (state saved in `localStorage`) with priority tags
- **Bug Tracker**: a lightweight log of known issues with status (open/resolved)

Use vanilla HTML/CSS/JS only — no build step, no dependencies. All state (todos, bugs, results) persists via `localStorage`. Re-render and update `dashboard.html` whenever architecture, data, experiments, or results change. Keep it skimmable: favor tables and collapsible sections over prose.
