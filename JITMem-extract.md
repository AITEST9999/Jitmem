# JITMem Paper Extract & Minimal Autoresearch Plan

**Paper:** Just-in-Time Memory: Learning to Curate Task-Adaptive Memory for LLM Agents  
**Authors:** Yefan Zhou*, Yang Li*, Zeyu Leo Liu, Semih Yavuz, Shafiq Joty (* equal contribution)  
**Affiliation:** Salesforce AI Research (`{yefan.zhou,yli2,sjoty}@salesforce.com`)  
**arXiv:** 2609.27334v1 [cs.AI] 23 Sep 2026  
**Local text:** `/workspace/PaperLab/JITMem.txt` · **PDF:** `/workspace/PaperLab/JITMem-arXiv-2609.27334.pdf`  
**Official code / GitHub / HF weights:** **none found** (checked arXiv abs/html, Papers with Code, HF papers page, GitHub keyword search for JitMem/JITMem + Salesforce authors). Only incidental HF link in appendix is WebShop eval script `zhangdw/webshop` (baseline infra, not JITMem).

---

## 1. Paper brief

### Problem + thesis (one paragraph)

Most agentic memory systems **curate at write time**: after a task finishes they distill the trajectory into a fixed artifact (reflection, skill, workflow, reasoning strategy) that is later retrieved by similarity. That commits to what to remember *before* the future query is known, irreversibly discards detail, and forces one query-independent summary to serve many downstream tasks. Learning write-time curators also faces long-horizon credit assignment (utility of a store decision may appear only many tasks later), often requiring artificial task grouping (e.g. SkillOS). **JITMem instead stores raw trajectories losslessly and defers curation to read time**: given the current task and BM25-retrieved traces, a memory curator synthesizes a compact **task-adaptive payload** consumed immediately by a frozen executor. Because the payload is graded by same-task success, the curator can be trained with immediate reward (GRPO), with no delayed returns or task grouping. Empirically this beats no-memory and write-time baselines on ALFWorld, WebShop, and τ²-bench; even an **untrained** curator is already competitive, showing that read-time task-adaptive curation itself is a major source of gain.

### Core claims (numbered)

1. **Read-time curation enables task-adaptive memory.** Same stored trajectory yields different payloads for different tasks (Fig. 3; §1 contributions; §4.3). Write-time curators cannot provide this.
2. **Read-time curation simplifies credit assignment.** Payload is consumed on the same task → immediate reward; no task-grouping scaffolds (§1, §3).
3. **JITMem beats strongest baselines by large absolute SR margins** (Table 1–2, §4.1): **+16.2** ALFWorld (77.4 vs SkillOS 61.2, Qwen3-8B), **+16.3** WebShop SR (32.8 vs SkillOS 16.5, Qwen3-8B), **+3.9** τ²-bench micro-avg (75.6 vs ReasoningBank-GPT 71.7, GPT-5.4 executor / JITMem-gpt).
4. **Untrained read-time curator is already competitive / often superior** to write-time methods on the same base model (§abstract, §4.1). E.g. JITMem-base 60.5 vs ReasoningBank 55.7 / SkillOS-base 53.1 on ALFWorld (Qwen); JITMem-gemini 61.0 WebShop SR vs SkillOS-gemini 41.0.
5. **RL training compounds gains** and the trained curator **transfers** across executors without retraining (Table 3: transfer gap 1.4 SR on ALFWorld to GPT-5.4).
6. **Compact payloads**: vs write-time methods, input tokens −50.3%–56.3% and executor steps −28.4%–31.4% (intro; Table 4 with GPT-5.4 on ALFWorld: JITMem 9.8K in / 11.6 steps vs ReasoningBank 19.7K / 16.2 and SkillOS-base 22.4K / 16.9).
7. **Ablations:** task-conditioned curation, success-filtered storage, and raw (not distilled) trajectories each contribute independently (§4.2, Fig. 2, Table 9). Removing retrieved traj. from trained curator drops SR by up to 14.8 (ALFWorld) / 15.2 (WebShop).

### Method

**Four components** (Fig. 1; §3). Only the curator is trainable.

| Component | Role |
|-----------|------|
| **Memory bank** `M` | Stores raw trajectories `ξ = (x, o1,a1,…,on,an)`. No write-time summarization. At deploy: append only if **executor-as-LLM-judge** says success. At train: fixed bank from base-executor successes using **ground-truth** labels. |
| **Retriever** `R` | **BM25** over **task descriptions only** (not trajectory body). Top-**k=3** (Appendix A; k∈{3,5} stable, Table 8). Untuned. |
| **Curator** `π_ϕ` | Prompted LLM: current task `xt` + k raw trajs → compact NL **payload** (relevant memories, strategies, task-specific guidance). Prompts in Appendix A. |
| **Executor** `π_L` | **Frozen** LLM. Payload prepended to executor prompt; never sees raw trajs. Also used as LLM-as-judge for bank update. |

**Inference flow:** Retrieve → Curate → Execute → Update bank if judge accepts. Payload is **ephemeral** (not stored).

**Training (full paper, not for first PaperLab run):** GRPO (Shao et al. 2024) on curator only. For each train task: retrieve from fixed bank; sample **G=8** candidate payloads; frozen executor scores each with ground-truth reward `rt∈[0,1]`; advantages `Âi = rt^(i) − mean_j rt^(j)` (no std norm, following Liu et al. 2025); update without value network. **100** GRPO steps, LR **1e-6**, batch **32**, curator init **Qwen3-8B non-thinking**. Train bank fixed; mild train/test shift studied via staged refresh (+2.8 SR best, Table 5).

**Eval protocol:** Test bank starts **empty** (cold start); warm-start negligible (Table 5). Batched streaming: batch size **10** ALFWorld/WebShop, **5** τ²-bench; bank updated after each batch. Mean±std over **3** orderings (ALFWorld/WebShop) or **4** (τ²-bench). Success via benchmark verifier; bank gate via LLM judge only.

### Experimental setup

**Benchmarks**
- **ALFWorld** (Shridhar et al. 2021): text embodied control, **140** test tasks; binary SR; max **30** turns; history window **3**.
- **WebShop** (Yao et al. 2022): product purchase, **500** test; SR + continuous score; max 30 turns.
- **τ²-bench** (Barres et al. 2025): airline / retail / telecom tool-use dialogues; max **200** turns; **no standard train split** → paper reports training-free variants only.

**Baselines:** No Memory; ReasoningBank; MemP; SkillOS (+ SkillOS-base, SkillOS-gpt/gemini).

**Models:** Executors & curators: **Qwen3-8B**, **Gemini-2.5-Pro**, **GPT-5.4**. Trained curator = Qwen3-8B (thinking disabled). Temp: executors 1.0 / max out 4096; Qwen curator train+eval 0.6 top-p 0.95 top-k 20. Qwen executor: thinking on ALFWorld, non-thinking on WebShop (+ WebShop search-guidance paragraph for conservative no-memory repro, Table 6).

**Infra (paper):** 8× NVIDIA H200, vLLM for Qwen3-8B (TP=1, DP=4, max len 40960). One GRPO run ≈ **21 h** ALFWorld, **≈27 h** WebShop.

### Metrics & headline numbers (verified from Tables 1–2)

Primary metric: **Success Rate (SR)**; WebShop also reports Score.

| Setting | Strongest baseline | JITMem | Δ |
|---------|-------------------|--------|---|
| ALFWorld, Qwen3-8B exec | SkillOS 61.2 | **77.4** | **+16.2** |
| WebShop SR, Qwen3-8B | SkillOS 16.5 | **32.8** | **+16.3** |
| τ²-bench micro, GPT-5.4 | ReasoningBank-GPT 71.7 | JITMem-gpt **75.6** | **+3.9** |

Untrained highlights: JITMem-base ALFWorld Qwen **60.5** (vs No Mem 47.9); with GPT-5.4 exec **79.3**; JITMem-gemini WebShop SR **61.0**.

### Ablations that matter for a minimal experiment

Priority order for a small faithful test (from §4.2 / Table 9, Qwen JITMem-base ALFWorld unless noted):

1. **Read-time + task adaptivity vs no-memory** (core claim) — base gain ~12.6 SR (60.5−47.9).
2. **w/o task adaptivity** (query-independent summarizer) → drops base by up to 3.1 ALFWorld / 4.6 WebShop; trained drops up to 11.4 / 10.4.
3. **w/o raw traj.** (ReasoningBank-style write-time distill into bank) → −1.7–2.9 ALFWorld, **−6.8–8.2** WebShop.
4. **w/o successful traj. filtering** → −1.5–2.9 ALFWorld, −2.3–3.4 WebShop.
5. **w/o retrieved traj.** (trained only) → catastrophic; proves RL learns to distill retrieval, not parametric hints.

For PaperLab v0: (1) alone is enough to test the thesis. Skip GRPO until the untrained slice is green.

### Limitations / compute (from §5 + Appendix A)

- BM25 may bottleneck as bank grows; curator = **extra LLM call per task**; payload format hand-designed per benchmark.
- Full repro needs **multi-GPU / large API budget** (8×H200, 21–27 h per GRPO train; Gemini/GPT API for strong executors; 140× multi-turn ALFWorld or 500 WebShop × 3 seeds).
- No released code → must reimplement from paper + Appendix prompts.
- τ²-bench: no train split; dual-control / tool APIs heavier to stand up.
- Qwen3-8B thinking vs non-thinking settings affect baseline match (Table 6: authors could not fully match SkillOS-reported no-memory without prompt tweaks).

---

## 2. Minimal faithful experiment / repro plan (autoresearch-shaped)

### Design principle (Karpathy autoresearch → PaperLab)

Adapt [karpathy/autoresearch](https://github.com/karpathy/autoresearch):

| Autoresearch | PaperLab / JITMem |
|--------------|-------------------|
| Human edits `program.md` | Human steers research org via `program.md` |
| Agent edits **only** `train.py` | Agent edits **only** `experiment.py` (alias `curator_loop.py` OK if one mutable file) |
| `prepare.py` immutable | `prepare.py` + eval harness + metric + task subset + wall budget **immutable** |
| Fixed ~5 min train budget | Fixed **wall-clock budget per experiment** (propose **20 min** — see below) |
| Single metric `val_bpb` ↓ | Single metric **`success_rate` ↑** on a **fixed tiny ALFWorld subset** |
| Baseline first | First run = **No Memory** or **JITMem-base** as declared baseline |
| keep / discard via git | Branch `autoresearch/<tag>`; advance only on metric improvement; else `git reset` |
| `results.tsv` | Same schema adapted: `commit`, `success_rate`, `n_tasks`, `status`, `description` |
| Never fake greens | Evidence = `run.log` lines only; no manual metric edits |
| Loop forever until stopped | Same once setup confirmed |

**Why not full GRPO / full 140-task eval first:** No official code; paper train needs ~21 h on 8×H200. Untrained curator already carries most of the conceptual gain. Autoresearch needs **many short comparable runs**, not one giant train.

### Recommended first benchmark: ALFWorld (tiny fixed subset)

**Why ALFWorld first**
- Smallest official test set (140) → easy to carve a **fixed** subset (e.g. N=8–12 tasks, pinned IDs in `prepare.py`).
- Text-only env; fewer moving parts than WebShop (product DB) or τ²-bench (tools + dual control).
- Paper’s untrained JITMem-base already shows clear lift vs No Memory (60.5 vs 47.9) — thesis testable without RL.
- Prompts fully in Appendix A.

**Primary metric (immutable):** `success_rate = (# tasks with env success) / N` on the fixed subset, **higher better**. Report also `n_success`, `n_tasks`, `wall_seconds` in `run.log` for audit — but **keep/discard uses only `success_rate`** (ties → discard / prefer simpler).

### Minimal architecture to implement (faithful slice, no GRPO)

1. Memory bank: list of raw `(task_desc, trajectory)` from **successful** episodes only (LLM-judge or env success for v0 — prefer env GT only for *scoring*; for bank gate use executor-judge if available, else env success with note).
2. Retriever: BM25 over task descriptions, **k=3**.
3. Curator: **untrained** prompted LLM (start with whatever Herdr routes: Claude/Codex API or local Qwen if available) — Appendix A ALFWorld curator prompt.
4. Executor: **frozen** same or separate LLM — Appendix A ALFWorld executor prompt; payload in `{retrieved_context}`; max 30 steps.
5. Baselines in `experiment.py` selectable by flag/constant: `no_memory` | `jitmem_base` (and later ablations: `no_task_adapt`, `distilled_bank`).

**Do not** implement GRPO / SkillOS / full WebShop until the autoresearch loop is proven on this slice.

### Success criteria (“green” with evidence)

| Gate | Criterion | Evidence required |
|------|-----------|-------------------|
| G0 harness | Env runs N tasks; logs SR for no_memory | `run.log` contains `success_rate:` and per-task outcomes |
| G1 core claim | `jitmem_base` SR **>** `no_memory` SR on **same** fixed subset, same seed/order | Both rows in `results.tsv` with `status=keep` for better; shares identical `prepare.py` subset |
| G2 stability | ≥2 seeds/orderings or ≥1 full 20-min budget rerun; direction of gain holds | Multiple TSV rows; no cherry-pick |
| G3 (optional later) | Ablation `w/o task adaptivity` ≤ jitmem_base | TSV + description |

**Not green:** higher SR from editing task list, shortening max steps only for one method, changing judge, or hand-editing logs. **Never fake greens.**

### Deps (expected)

- Python 3.10+
- `alfworld` (+ TextWorld / Fast Downward as upstream requires)
- `rank_bm25` or `bm25s` / whoosh — BM25 only
- LLM client used by Herdr (Anthropic/OpenAI/local) — **no new packages in agent loop** once `pyproject.toml` frozen
- `git`, `uv` recommended

### Compute estimate (Mac mini + API)

| Phase | Wall | $ / hardware |
|-------|------|----------------|
| Env install + subset smoke (1–2 tasks) | 30–90 min once | local CPU |
| One **20-min** experiment (N≈8–12, max 30 steps, 2 LLM calls/step worst case + 1 curator call/task) | **20 min** fixed | roughly tens of $ depending on model; or local LLM if Herdr provides |
| Overnight autoresearch @ ~2–3 runs/hour | ~40–60 runs / sleep | API budget must be capped in `prepare.py` (`MAX_API_CALLS` / kill switch) |
| Full paper GRPO | **skip** for v0 | 21 h × 8×H200 — out of scope |

**Why 20 min budget (not 5):** Agent episodes are sequential LLM+env steps; 5 min is too short for meaningful N on Mac mini / API latency. **20 min** still allows ~3 runs/hour and overnight iteration, while forcing a **tiny fixed N** so methods stay comparable. Soft timeout kill at **25 min** → `crash`. Justify in `prepare.py` constant `EXPERIMENT_BUDGET_SEC = 1200`.

### Step-by-step runbook (Mac mini via Herdr → Claude/Codex)

Workspace: `~/Documents/LatentFlux/PaperLab/`

```text
1. mkdir -p ~/Documents/LatentFlux/PaperLab/jitmem-lab && cd $_
2. Copy program.md, and scaffold:
     prepare.py      # IMMUTABLE after human review
     experiment.py   # ONLY file agent may edit
     pyproject.toml  # freeze deps before loop
     results.tsv     # untracked; header only
3. Human: implement prepare.py once (ALFWorld install helpers, FIXED_TASK_IDS,
   EXPERIMENT_BUDGET_SEC, evaluate_success_rate(), LLM client factory, seed).
4. Human: seed experiment.py with no_memory baseline + jitmem_base skeleton
   (BM25 k=3, raw bank, curator prompt from paper App. A).
5. uv sync && (one-time) uv run prepare.py   # downloads/checks ALFWorld assets
6. Manual smoke: uv run experiment.py > run.log 2>&1
   Confirm grep '^success_rate:' run.log
7. Point Herdr agent at program.md: "setup first, then confirm and go"
8. Agent creates branch autoresearch/<tag>, runs baseline, then LOOP
```

Standing Surendra rule: **all experiments on Mac mini; heavy LLM/coding via Herdr → Claude/Codex.**

### Risks / blockers

| Risk | Mitigation |
|------|------------|
| **No official JITMem code** | Reimplement minimal slice from paper + App. A prompts; do not claim full paper repro |
| ALFWorld install friction on macOS | Docker or pre-built env; document in prepare; if blocked, FYI Bruce |
| API cost / rate limits | Cap calls in prepare.py; prefer cheaper curator/executor for loop; log `$` estimate |
| Non-determinism (LLM sampling) | Fixed temp in prepare; report seed; keep only if SR clearly up (e.g. +≥1/N absolute) or mean over 2 orders inside budget — policy in program.md |
| Evaluating on full 140 too slow | Subset pinned; never expand N inside agent-editable file |
| Temptation to hack eval | prepare.py immutable; agent cannot touch metric |
| Closed Gemini/GPT-5.4 parity | Use available Herdr models; compare **relative** no_mem vs jitmem_base, not absolute paper numbers |

---

## 3. Autoresearch adaptation details

### Immutable vs mutable

**Immutable (human-owned):**
- `program.md` (human iterates org policy)
- `prepare.py` — `FIXED_TASK_IDS`, `EXPERIMENT_BUDGET_SEC`, `MAX_ENV_STEPS`, BM25 k default, `evaluate_success_rate`, data paths, LLM endpoint config hooks, API kill switch
- Eval harness / metric definition
- `pyproject.toml` once frozen for a run tag

**Mutable (agent-only):**
- `experiment.py` — curator prompt wording (within faithfulness), payload structure, executor prompt tweaks, retrieval post-processing, bank admission heuristics, hyperparams that are not in prepare.py, optional light ablations toggles

### results.tsv schema

```
commit	success_rate	n_tasks	status	description
```

- `commit`: short 7-char hash  
- `success_rate`: float in [0,1], e.g. `0.500000` (use `0.000000` for crash)  
- `n_tasks`: int (must equal `len(FIXED_TASK_IDS)` or crash)  
- `status`: `keep` | `discard` | `crash`  
- `description`: short text, **no commas** (TSV)

Do **not** commit `results.tsv` (leave untracked).

### Keep / discard rule

- Advance branch iff `success_rate` **strictly greater** than current best on branch.
- Equal or worse → `git reset --hard` to pre-experiment commit; status `discard`.
- Crash / timeout → status `crash`; reset unless trivial fix (typo/import) then one retry.
- **Simplicity criterion:** all else equal, simpler wins; small SR gain with large hacky complexity → discard; simplification with equal SR → keep.

### What “green” means (again)

A claim is green only if `results.tsv` + matching `run.log` show the metric under the immutable harness. Absolute match to paper’s 77.4 / 16.2 is **not** required for v0; **directional** confirmation of read-time curation > no-memory on the fixed slice is the v0 thesis test.

---

## 4. Recommended first action on Mac mini

Concrete next steps under `~/Documents/LatentFlux/PaperLab/`:

```bash
cd ~/Documents/LatentFlux/PaperLab
mkdir -p jitmem-lab
# Copy from box or sync:
#   JITMem-extract.md, program.md, JITMem.txt, PDF

cd jitmem-lab
git init
# Scaffold prepare.py + experiment.py + pyproject.toml from extract §2–3
# (Herdr/Claude: implement prepare.py from paper App. A prompts + FIXED_TASK_IDS)

# Example first commands after scaffold exists:
uv sync
uv run prepare.py          # verify ALFWorld assets; write subset manifest
uv run experiment.py > run.log 2>&1
grep -E '^success_rate:|^n_tasks:|^wall_seconds:' run.log
```

Then: open Herdr → Claude/Codex with cwd `jitmem-lab`, prompt:

> Read `program.md`. Agree on run tag for today, create `autoresearch/<tag>`, initialize `results.tsv`, establish **no_memory** baseline, then start the loop. Do not modify `prepare.py`.

**Out of scope until G1 green:** GRPO, WebShop, τ²-bench, SkillOS reimplementation, 8×H200 claims.

---

## 5. program.md

See companion file: `/workspace/PaperLab/program.md` (also intended to live at `~/Documents/LatentFlux/PaperLab/jitmem-lab/program.md`).
