"""PaperLab JITMem experiment.py — MUTABLE (agent may edit).

First run MUST use METHOD = "no_memory".
Also implements jitmem_base (BM25 k=3, raw traj bank, untrained read-time curator).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rank_bm25 import BM25Okapi

from prepare import (
    BM25_K,
    HISTORY_WINDOW,
    MAX_ENV_STEPS,
    AlfworldTaskEnv,
    TaskResult,
    evaluate_success_rate,
    llm_chat,
    parse_action,
    task_description,
)

# ---------------------------------------------------------------------------
# Method switch — baseline first
# ---------------------------------------------------------------------------

METHOD = "no_memory"  # "no_memory" | "jitmem_base"

# ---------------------------------------------------------------------------
# Memory bank (raw trajectories)
# ---------------------------------------------------------------------------


@dataclass
class MemoryEntry:
    task_desc: str
    trajectory: str  # textual (obs/action) transcript


@dataclass
class MemoryBank:
    entries: list[MemoryEntry] = field(default_factory=list)

    def add_if_success(self, task_desc: str, trajectory: str, success: bool) -> None:
        # v0 bank gate: env ground-truth success (noted; paper uses executor-judge at deploy)
        if success and trajectory.strip():
            self.entries.append(MemoryEntry(task_desc=task_desc, trajectory=trajectory))

    def retrieve(self, query: str, k: int = BM25_K) -> list[MemoryEntry]:
        if not self.entries or k <= 0:
            return []
        corpus = [e.task_desc.lower().split() for e in self.entries]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(self.entries)), key=lambda i: scores[i], reverse=True)
        return [self.entries[i] for i in ranked[:k]]


BANK = MemoryBank()


# ---------------------------------------------------------------------------
# Prompts (Appendix A — ALFWorld)
# ---------------------------------------------------------------------------

CURATOR_SYSTEM = """You are a Memory Curator. You will be given a task that an AI agent needs to solve in a household (ALFWorld) environment, along with retrieved past experiences from similar successful tasks.

Your job: synthesize these raw memories into a concise, actionable briefing that will help the agent solve the current task.

Each memory contains:
- A past task and the trajectory that solved it

Your output should:
1. Identify which past experiences are most relevant
2. Extract strategies that worked on similar tasks (e.g., where to find objects, useful action orders)
3. Give specific guidance for THIS task

Be concise - the agent has limited context."""


def build_curator_user(query: str, memories: list[MemoryEntry]) -> str:
    parts = [f"Question: {query}", "", "### Retrieved Memories:"]
    if not memories:
        parts.append("(none — cold start, no past successes yet)")
    for i, m in enumerate(memories, 1):
        parts.append(f"Memory {i}:")
        parts.append(f"Question: {m.task_desc}")
        parts.append("Trajectory:")
        parts.append(m.trajectory)
        parts.append("")
    return "\n".join(parts)


def curator_payload(task_desc: str, memories: list[MemoryEntry]) -> str:
    user = build_curator_user(task_desc, memories)
    return llm_chat(
        [
            {"role": "system", "content": CURATOR_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.6,
        max_tokens=512,
    )


def build_executor_prompt(
    task_description_text: str,
    retrieved_context: str,
    step_count: int,
    history: list[tuple[str, str]],
    current_observation: str,
    admissible_actions: list[str],
) -> str:
    # history: list of (observation, action) — most recent HISTORY_WINDOW
    hist = history[-HISTORY_WINDOW:]
    if hist:
        hist_lines = []
        for obs, act in hist:
            hist_lines.append(f"Observation: {obs}\nAction: {act}")
        action_history = "\n".join(hist_lines)
    else:
        action_history = "(none)"

    ctx = retrieved_context.strip() if retrieved_context.strip() else "(none)"
    adm = ", ".join(admissible_actions)
    current_step = step_count + 1
    return f"""You are an expert agent operating in the ALFRED Embodied Environment. Your task is to:
{task_description_text}

Here are past experiences and trajectories that might be helpful for your decision:

{ctx}

## Current Progress

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {len(hist)} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{adm}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation with the help of past relevant experiences.
Once you've finished your reasoning, you should choose an admissible action for current step and MUST present it within <action> </action> tags."""


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(task_id: str, method: str) -> TaskResult:
    env = AlfworldTaskEnv(task_id, max_steps=MAX_ENV_STEPS)
    try:
        step = env.reset()
        goal = env.goal or task_description(task_id)

        # Read-time curation
        retrieved_context = ""
        if method == "jitmem_base":
            memories = BANK.retrieve(goal, k=BM25_K)
            retrieved_context = curator_payload(goal, memories)
        elif method == "no_memory":
            retrieved_context = ""
        else:
            raise ValueError(f"Unknown METHOD={method}")

        history: list[tuple[str, str]] = []
        traj_lines: list[str] = [f"Task: {goal}", f"Init: {step.observation}"]
        steps_taken = 0
        won = False

        while not step.done and steps_taken < MAX_ENV_STEPS:
            prompt = build_executor_prompt(
                task_description_text=goal,
                retrieved_context=retrieved_context,
                step_count=steps_taken,
                history=history,
                current_observation=step.observation,
                admissible_actions=step.admissible_actions,
            )
            raw = llm_chat(
                [{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=512,
            )
            action = parse_action(raw, step.admissible_actions)
            traj_lines.append(f"Obs: {step.observation}")
            traj_lines.append(f"Act: {action}")
            history.append((step.observation, action))
            step = env.step(action)
            steps_taken += 1
            if step.won:
                won = True
                break

        trajectory = "\n".join(traj_lines)
        # Bank update (only meaningful for jitmem; harmless for no_memory)
        BANK.add_if_success(goal, trajectory, won)
        return TaskResult(task_id=task_id, success=won, steps=steps_taken, goal=goal)
    finally:
        env.close()


def main() -> None:
    global BANK
    BANK = MemoryBank()  # cold-start bank each experiment

    def run_one(task_id: str) -> TaskResult:
        return run_episode(task_id, METHOD)

    evaluate_success_rate(run_one, method=METHOD)


if __name__ == "__main__":
    main()
