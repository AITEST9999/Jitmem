"""PaperLab JITMem prepare.py — IMMUTABLE after first commit.

Fixed evaluation harness for the untrained read-time curator slice.
Agent may edit experiment.py only.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Immutable constants
# ---------------------------------------------------------------------------

EXPERIMENT_BUDGET_SEC = 1200  # 20 minutes wall-clock (excl. import if harness says so)
SOFT_KILL_SEC = 1500  # 25 minutes → crash
MAX_ENV_STEPS = 30
HISTORY_WINDOW = 3
BM25_K = 3
MAX_API_CALLS = 2500  # kill switch for overnight safety
LLM_MODEL = os.environ.get("JITMEM_LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 1.0
LLM_MAX_TOKENS = 1024
SEED = 42

# Paths
REPO_ROOT = Path(__file__).resolve().parent
ALFWORLD_DATA = Path(
    os.environ.get("ALFWORLD_DATA", str(Path.home() / ".cache" / "alfworld"))
)
VALID_SEEN = ALFWORLD_DATA / "json_2.1.1" / "valid_seen"

# Pinned ALFWorld subset (8 tasks, valid_seen, non-movable, non-Sliced).
# IDs are relative paths under VALID_SEEN (task_dir/trial_dir).
FIXED_TASK_IDS: list[str] = [
    "look_at_obj_in_light-AlarmClock-None-DeskLamp-323/trial_T20190909_044715_250790",
    "pick_and_place_simple-Book-None-SideTable-329/trial_T20190908_050633_745514",
    "pick_clean_then_place_in_recep-ButterKnife-None-CounterTop-8/trial_T20190909_105559_983897",
    "pick_cool_then_place_in_recep-Apple-None-CounterTop-14/trial_T20190909_044933_815840",
    "pick_heat_then_place_in_recep-Apple-None-DiningTable-26/trial_T20190907_060234_011675",
    "pick_two_obj_and_place-AlarmClock-None-Dresser-305/trial_T20190907_165826_194855",
    "look_at_obj_in_light-Bowl-None-DeskLamp-301/trial_T20190909_150719_492274",
    "pick_and_place_simple-Book-None-Sofa-229/trial_T20190907_042856_259139",
]

# Existing Mini credential file (do not invent secrets). Used only if env unset.
_EXISTING_OPENAI_ENV = Path(
    "/Users/surendra/Documents/Experiments/langgraph-research-agent/.env"
)


# ---------------------------------------------------------------------------
# Budget / API accounting
# ---------------------------------------------------------------------------


@dataclass
class RunState:
    t0: float = field(default_factory=time.time)
    api_calls: int = 0
    aborted: Optional[str] = None

    def elapsed(self) -> float:
        return time.time() - self.t0

    def check_budget(self) -> None:
        if self.elapsed() > EXPERIMENT_BUDGET_SEC:
            self.aborted = f"EXPERIMENT_BUDGET_SEC={EXPERIMENT_BUDGET_SEC} exceeded"
            raise TimeoutError(self.aborted)
        if self.api_calls >= MAX_API_CALLS:
            self.aborted = f"MAX_API_CALLS={MAX_API_CALLS} exceeded"
            raise RuntimeError(self.aborted)


RUN = RunState()


def reset_run_state() -> None:
    global RUN
    RUN = RunState()


# ---------------------------------------------------------------------------
# LLM client (OpenAI — uses existing Mini credentials)
# ---------------------------------------------------------------------------


def _ensure_openai_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    if _EXISTING_OPENAI_ENV.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(_EXISTING_OPENAI_ENV, override=False)
        except Exception:
            pass
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY not set and could not load from existing Mini .env"
        )


_client = None


def get_llm_client():
    global _client
    if _client is None:
        _ensure_openai_key()
        from openai import OpenAI

        _client = OpenAI()
    return _client


def llm_chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    model: Optional[str] = None,
) -> str:
    """Single chat completion. Counts against MAX_API_CALLS / budget."""
    RUN.check_budget()
    client = get_llm_client()
    RUN.api_calls += 1
    resp = client.chat.completions.create(
        model=model or LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()


# ---------------------------------------------------------------------------
# ALFWorld env helpers
# ---------------------------------------------------------------------------


def task_game_path(task_id: str) -> Path:
    p = VALID_SEEN / task_id / "game.tw-pddl"
    if not p.exists():
        raise FileNotFoundError(f"Missing ALFWorld game for {task_id}: {p}")
    return p


def task_description(task_id: str) -> str:
    traj = VALID_SEEN / task_id / "traj_data.json"
    td = json.loads(traj.read_text())
    # Prefer the goal line embedded in the env observation; turk desc is fine too.
    anns = td.get("turk_annotations", {}).get("anns", [])
    if anns and anns[0].get("task_desc"):
        return anns[0]["task_desc"].strip()
    return td.get("task_type", task_id)


def extract_goal_from_obs(obs: str) -> str:
    m = re.search(r"Your task is to:\s*(.+)", obs)
    if m:
        return m.group(1).strip()
    return ""


@dataclass
class EnvStep:
    observation: str
    admissible_actions: list[str]
    reward: float
    done: bool
    won: bool


class AlfworldTaskEnv:
    """Single-task TextWorld/ALFWorld wrapper (batch_size=1)."""

    def __init__(self, task_id: str, max_steps: int = MAX_ENV_STEPS):
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import (
            AlfredDemangler,
            AlfredInfos,
        )

        self.task_id = task_id
        self.max_steps = max_steps
        game = str(task_game_path(task_id))
        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            extras=["gamefile"],
            description=True,
            inventory=True,
        )
        wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
        env_id = textworld.gym.register_games(
            [game],
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=max_steps,
            wrappers=wrappers,
        )
        self._env = textworld.gym.make(env_id)
        self._step_count = 0
        self.goal = ""
        self._last_admissible: list[str] = []

    def reset(self) -> EnvStep:
        obs, infos = self._env.reset()
        obs0 = obs[0] if isinstance(obs, (list, tuple)) else obs
        adm = infos["admissible_commands"][0]
        won = bool(infos["won"][0])
        self._step_count = 0
        self.goal = extract_goal_from_obs(obs0) or task_description(self.task_id)
        self._last_admissible = list(adm)
        return EnvStep(obs0, list(adm), 0.0, False, won)

    def step(self, action: str) -> EnvStep:
        obs, scores, dones, infos = self._env.step([action])
        obs0 = obs[0]
        done = bool(dones[0])
        won = bool(infos["won"][0])
        adm = list(infos["admissible_commands"][0])
        reward = float(scores[0]) if scores is not None else (1.0 if won else 0.0)
        self._step_count += 1
        if self._step_count >= self.max_steps:
            done = True
        self._last_admissible = adm
        return EnvStep(obs0, adm, reward, done, won)

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass


def parse_action(text: str, admissible: list[str]) -> str:
    """Extract <action>...</action> or fuzzy-match an admissible command."""
    m = re.search(r"<action>\s*(.*?)\s*</action>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        cand = m.group(1).strip().strip('"').strip("'")
    else:
        # last non-empty line as fallback
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        cand = lines[-1] if lines else ""

    # Exact match
    if cand in admissible:
        return cand
    # Case-insensitive
    lower_map = {a.lower(): a for a in admissible}
    if cand.lower() in lower_map:
        return lower_map[cand.lower()]
    # Contained
    for a in admissible:
        if a.lower() in cand.lower() or cand.lower() in a.lower():
            return a
    # Default: look / first admissible
    for prefer in ("look", "inventory"):
        if prefer in admissible:
            return prefer
    return admissible[0] if admissible else "look"


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    task_id: str
    success: bool
    steps: int
    goal: str
    error: Optional[str] = None


def evaluate_success_rate(
    run_one_task: Callable[[str], TaskResult],
    task_ids: Optional[list[str]] = None,
    method: str = "no_memory",
) -> dict[str, Any]:
    """Run fixed subset; print grep-able summary; return metrics dict.

    `run_one_task(task_id) -> TaskResult` is provided by experiment.py.
    """
    ids = list(task_ids or FIXED_TASK_IDS)
    results: list[TaskResult] = []
    reset_run_state()

    print(f"method:           {method}")
    print(f"n_tasks_planned:  {len(ids)}")
    print(f"model:            {LLM_MODEL}")
    print(f"budget_sec:       {EXPERIMENT_BUDGET_SEC}")
    print("---")

    for i, tid in enumerate(ids):
        if RUN.aborted:
            break
        print(f"\n=== task {i+1}/{len(ids)}: {tid} ===")
        try:
            RUN.check_budget()
            tr = run_one_task(tid)
        except (TimeoutError, RuntimeError) as e:
            print(f"ABORT: {e}")
            tr = TaskResult(tid, False, 0, "", error=str(e))
            results.append(tr)
            break
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            tr = TaskResult(tid, False, 0, "", error=str(e))
        results.append(tr)
        print(
            f"task_result: success={int(tr.success)} steps={tr.steps} "
            f"goal={tr.goal!r} err={tr.error!r}"
        )

    n_tasks = len(ids)
    # If we aborted early, still report planned n_tasks for mismatch detection;
    # successes only among completed.
    n_success = sum(1 for r in results if r.success)
    # Per program.md: n_tasks must equal len(FIXED_TASK_IDS). We always report planned.
    success_rate = (n_success / n_tasks) if n_tasks else 0.0
    wall = RUN.elapsed()
    total_steps = sum(r.steps for r in results)

    print("\n---")
    print(f"success_rate:     {success_rate:.6f}")
    print(f"n_success:        {n_success}")
    print(f"n_tasks:          {n_tasks}")
    print(f"steps:            {total_steps}")
    print(f"wall_seconds:     {wall:.1f}")
    print(f"method:           {method}")
    print(f"api_calls:        {RUN.api_calls}")
    if RUN.aborted:
        print(f"aborted:          {RUN.aborted}")
    print("---")

    return {
        "success_rate": success_rate,
        "n_success": n_success,
        "n_tasks": n_tasks,
        "steps": total_steps,
        "wall_seconds": wall,
        "method": method,
        "api_calls": RUN.api_calls,
        "results": results,
        "aborted": RUN.aborted,
    }


def verify_assets() -> None:
    """One-shot check: ALFWorld data + FIXED_TASK_IDS exist."""
    missing = []
    for tid in FIXED_TASK_IDS:
        if not task_game_path(tid).exists():
            missing.append(tid)
    print(f"ALFWORLD_DATA={ALFWORLD_DATA}")
    print(f"FIXED_TASK_IDS={len(FIXED_TASK_IDS)}")
    if missing:
        raise SystemExit(f"Missing games: {missing}")
    print("verify_assets: OK")
    for tid in FIXED_TASK_IDS:
        print(f"  - {tid} :: {task_description(tid)}")


if __name__ == "__main__":
    verify_assets()
