# PaperLab autoresearch — JITMem (untrained read-time curator slice)

This is an experiment to have the LLM agent improve a **minimal faithful JITMem** implementation autonomously, under Karpathy-style autoresearch rules adapted to Paper Lab.

**Paper:** Just-in-Time Memory (Salesforce AI Research), arXiv:2609.27334v1.  
**Scope for this loop:** Untrained read-time curator + BM25 (k=3) + raw trajectory bank + frozen executor on a **fixed tiny ALFWorld subset**. **No GRPO**, no full 140-task eval, no WebShop/τ²-bench until the human expands `prepare.py`.

**There is no official JITMem code.** You reimplement a minimal slice from the paper brief / Appendix prompts. Do not claim full paper reproduction.

Human steers via this `program.md`. You may edit **only** `experiment.py`.

---

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `oct2`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main/master.
3. **Read the in-scope files**:
   - `program.md` — these instructions (human-owned; do not rewrite policy without asking).
   - `prepare.py` — fixed constants, ALFWorld subset, wall-clock budget, eval harness, LLM client hooks. **Do not modify.**
   - `experiment.py` — the **only** file you modify (curator/executor prompts, retrieval/bank logic, method variants).
   - `JITMem-extract.md` (parent PaperLab folder) — paper claims, prompts pointers, numbers. Read-only context.
4. **Verify env ready**: Confirm ALFWorld assets / subset manifest exist as `prepare.py` expects. If not, tell the human to run `uv run prepare.py`. Do not invent tasks outside `FIXED_TASK_IDS`.
5. **Initialize `results.tsv`**: Create with header row only. Baseline recorded after first run. **Leave `results.tsv` untracked** (never git-commit it).
6. **Confirm and go**: Confirm setup looks good with the human once. After confirmation, **do not stop to ask permission** again.

Once you get confirmation, kick off experimentation.

---

## Experimentation

Each experiment runs under a **fixed wall-clock budget of 20 minutes** (`EXPERIMENT_BUDGET_SEC = 1200` in `prepare.py`, excluding import/startup if the harness says so). Launch:

```bash
uv run experiment.py > run.log 2>&1
```

**Why 20 minutes (not 5):** LLM-agent episodes are sequential (curator call + many executor steps × N tasks × API latency on Mac mini). Five minutes cannot finish a meaningful fixed subset; twenty minutes still allows ~2–3 comparable runs per hour and overnight iteration while forcing a small N.

If a run exceeds **25 minutes** total wall time, kill it and treat as `crash` (discard + revert).

### What you CAN do

- Modify **`experiment.py` only**. Fair game inside that file:
  - Curator prompt wording (stay faithful to read-time, task-conditioned curation).
  - Payload structure / formatting.
  - Executor prompt tweaks that do not change the success metric definition.
  - Bank admission heuristics (still: prefer successful trajectories).
  - Light retrieval post-processing (still BM25 over task text unless prepare hard-codes k).
  - Method switch constants: `no_memory`, `jitmem_base`, ablations like `no_task_adapt` / `distilled_bank` **if** implemented entirely inside `experiment.py` and still evaluated by the same harness.
  - Caching, retries, logging verbosity.

### What you CANNOT do

- Modify **`prepare.py`**. Read-only. Contains fixed evaluation, `FIXED_TASK_IDS`, time budget, max env steps, metric.
- Install new packages or edit `pyproject.toml`. Use only frozen deps.
- Change the evaluation harness or metric definition (`evaluate_success_rate` / whatever prepare exports).
- Expand or swap the task subset to inflate SR.
- Shorten `MAX_ENV_STEPS` for one method only, or give one method extra tools/oracle labels.
- Hand-edit `run.log` or fabricate `success_rate`.
- Claim paper table numbers (77.4, +16.2, etc.) as your result unless your log literally matches a full paper setup (it will not in this slice).

### Goal

**Maximize `success_rate`** on the fixed ALFWorld subset (higher is better).

Secondary soft goals (do not override SR):

- Fewer env steps / tokens when SR is equal.
- **Simplicity criterion:** all else equal, simpler is better. Tiny SR gain with ugly complexity → discard. Equal SR from deleting code → keep.

### The first run

Always establish the **baseline** first: run `experiment.py` as-is configured for **`no_memory`** (or whatever prepare/docs declare as baseline). Record it as `keep`. Only then try `jitmem_base` and further ideas.

Core thesis to confirm: **read-time task-adaptive curation (`jitmem_base`) beats `no_memory`** on the same fixed subset under the same budget.

---

## Output format

When the script finishes it must print a summary block that `prepare.py`'s helpers or your `experiment.py` emit, including lines grep-able as:

```
---
success_rate:     0.500000
n_success:        4
n_tasks:          8
wall_seconds:     1180.2
method:           no_memory
---
```

Extract:

```bash
grep -E '^success_rate:|^n_tasks:|^wall_seconds:|^method:' run.log
```

If `success_rate:` is missing → treat as crash; `tail -n 50 run.log`.

`n_tasks` must equal `len(FIXED_TASK_IDS)` from prepare. Mismatch → crash (possible harness bug or illicit subset change).

---

## Logging results

Log every attempt to `results.tsv` (tab-separated, **not** CSV).

Header and columns:

```
commit	success_rate	n_tasks	status	description
```

1. git commit hash (short, 7 chars)
2. success_rate (e.g. `0.500000`) — use `0.000000` for crashes
3. n_tasks (integer) — use `0` for crashes
4. status: `keep`, `discard`, or `crash`
5. short description of what this experiment tried (**no tabs**; avoid commas)

Example:

```
commit	success_rate	n_tasks	status	description
a1b2c3d	0.375000	8	keep	baseline no_memory
b2c3d4e	0.500000	8	keep	jitmem_base BM25 k=3 raw bank untrained curator
c3d4e5f	0.375000	8	discard	curator without task text in prompt
d4e5f6g	0.000000	0	crash	bank JSON serialize bug
```

Do **not** commit `results.tsv`.

---

## The experiment loop

Runs on dedicated branch `autoresearch/<tag>`.

**LOOP FOREVER:**

1. Note current branch/commit (best-so-far).
2. Edit `experiment.py` with one clear experimental idea.
3. `git add experiment.py && git commit -m "..."` 
4. Run: `uv run experiment.py > run.log 2>&1`
5. Grep metrics from `run.log`.
6. If missing metrics → crash path: inspect `tail -n 50 run.log`. Trivial fix (typo/import) → fix and re-run once. Fundamental break → log `crash`, reset, move on.
7. Append row to `results.tsv` (untracked).
8. If `success_rate` **strictly greater** than best on this branch → **advance** (keep commit).
9. If equal or worse → **`git reset --hard`** to pre-change commit; status `discard`.

You are an autonomous researcher: try ideas, keep improvements, discard failures. Rewind sparingly.

**NEVER STOP** after the loop has begun: do not ask "should I continue?" The human may be asleep. If out of ideas, re-read the extract (ablations, Fig. 3 payload structure), try simplifying prompts, try bank-quality filters, try payload format changes, try `no_task_adapt` as a *negative* control (expect worse — if it somehow wins, investigate for eval bugs rather than celebrating).

**Crashes:** Fix dumb mistakes; skip fundamentally broken ideas.

**API / cost:** Respect any kill switch or `MAX_API_CALLS` in `prepare.py`. If the harness aborts for budget, log `crash` or `discard` as appropriate and try a cheaper configuration inside `experiment.py` (shorter curator output, fewer retries) — never by editing prepare.

---

## Faithfulness checklist (every keep)

Before marking `keep` on a JITMem variant, confirm the run still matches the paper's **minimal** design:

- [ ] Bank stores **raw** trajectories (not write-time distilled skills) unless this experiment *intentionally* ablates that.
- [ ] Retrieval is over **task descriptions**; top-k as prepare allows (default 3).
- [ ] Curator sees **current task + retrieved raw trajs** and emits a short payload (unless ablating task adaptivity).
- [ ] Executor is frozen and sees **payload**, not the full raw bank dump (unless no_memory).
- [ ] Metric came only from prepare's evaluator on `FIXED_TASK_IDS`.

---

## Out of scope (do not start inside this loop)

- GRPO / RL training of the curator (paper: 100 steps, ~21 h on 8×H200).
- Full ALFWorld 140 / WebShop 500 / τ²-bench.
- Reimplementing SkillOS, ReasoningBank, MemP as full systems (optional thin write-time ablation only if it fits the 20-min budget).
- Claiming absolute SR parity with Table 1.

When the human wants those, they will change `program.md` / `prepare.py` and start a new run tag.
