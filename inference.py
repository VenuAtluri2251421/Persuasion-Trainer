"""
Inference Script — Persuasion Trainer Environment
===================================================
MANDATORY env vars:
    API_BASE_URL        The API endpoint for the LLM (default: HF router)
    MODEL_NAME          The model identifier (default: Qwen2.5-72B-Instruct)
    HF_TOKEN            Your Hugging Face / API key
    LOCAL_IMAGE_NAME    Docker image name (used with from_docker_image())
    GROQ_API_KEY        Your Groq API key (used by the environment server)
    ENV_BASE_URL        Optional: connect to an already-running server instead of Docker

STDOUT FORMAT (must not be changed):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import textwrap
from typing import List, Optional

from openai import OpenAI

try:
    from persuasion_trainer import PersuasionTrainerAction, PersuasionTrainerEnv
except ImportError:
    # Running from repo root without package installed
    from client import PersuasionTrainerEnv  # type: ignore[no-redef]
    from models import PersuasionTrainerAction  # type: ignore[no-redef]

# ── Mandatory env vars (checklist) ───────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")          # No default — must be set by caller
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # Optional — used with from_docker_image()

# ── Environment server connection ───────────────────────────────────────────
# Defaults to the live HF Space so the validator can connect without Docker.
# Override with LOCAL_IMAGE_NAME to use a local Docker container instead.
ENV_BASE_URL: Optional[str] = os.getenv(
    "ENV_BASE_URL",
    "https://venuatluri936-persuasion-trainer.hf.space",
)

# ── Episode config ────────────────────────────────────────────────────────────
BENCHMARK = "persuasion_trainer"
MAX_TURNS = 6          # environment ends at turn 6 (done=True)
TEMPERATURE = 0.7
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.5   # mean reward ≥ 0.5 → success

# ── Tasks (easy → medium → hard) ─────────────────────────────────────────────
TASKS = [
    {"task_type": "easy",   "task_name": "refund-negotiation"},
    {"task_type": "medium", "task_name": "salary-raise-negotiation"},
    {"task_type": "hard",   "task_name": "board-pivot-negotiation"},
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are a skilled negotiator in a high-stakes conversation.
    Your goal is to persuade the opponent clearly and confidently.
    Use logical arguments, concrete evidence, and confident language.
    Avoid filler words like 'um', 'uh', 'like', or 'I guess'.
    Reply with EXACTLY ONE persuasive message — no preamble, no quotes.
    Keep your reply under 3 sentences.
""").strip()


# ── Logging helpers (exact required format) ───────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Sanitise action: collapse newlines, strip quotes
    clean = action.replace("\n", " ").replace("\r", "").strip()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={clean!r} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── Agent message generation ──────────────────────────────────────────────────

def build_user_prompt(step: int, opponent_reply: str, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Step {step}. The opponent just said:
        "{opponent_reply}"

        Previous exchange:
        {history_block}

        Your persuasive reply:
    """).strip()


def get_agent_reply(
    client: OpenAI,
    step: int,
    opponent_reply: str,
    history: List[str],
) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(step, opponent_reply, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "I believe this proposal is in everyone's best interest."
    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        return "I believe this proposal is in everyone's best interest."


# ── Single episode runner ─────────────────────────────────────────────────────

def run_episode(env: PersuasionTrainerEnv, client: OpenAI, task: dict) -> dict:
    """Run one negotiation episode. Returns result dict."""
    task_type = task["task_type"]
    task_name = task["task_name"]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        obs = env.reset(task_type=task_type)
        opponent_reply: str = obs.reply_text
        history: List[str] = []

        for step in range(1, MAX_TURNS + 1):
            if getattr(obs, "done", False):
                break

            agent_msg = get_agent_reply(client, step, opponent_reply, history)
            action = PersuasionTrainerAction(message=agent_msg)

            try:
                result = env.step(action)
                obs = result.observation
                reward = float(result.reward or 0.0)
                done = result.done
                error = None
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)
                print(f"[DEBUG] env.step() error: {exc}", flush=True)

            steps_taken = step
            rewards.append(reward)

            log_step(step=step, action=agent_msg, reward=reward, done=done, error=error)

            opponent_reply = obs.reply_text
            history.append(f"Step {step} | You: {agent_msg!r} | Opponent ({obs.strategy_used}): {opponent_reply!r}")

            if done or error:
                break

        score = (sum(rewards) / len(rewards)) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "score": score, "success": success, "steps": steps_taken}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    all_results = []

    for task in TASKS:
        # Build a fresh env connection per episode
        if LOCAL_IMAGE_NAME:
            env = PersuasionTrainerEnv.from_docker_image(LOCAL_IMAGE_NAME)
        elif ENV_BASE_URL:
            env = PersuasionTrainerEnv(base_url=ENV_BASE_URL)
        else:
            raise EnvironmentError(
                "Set LOCAL_IMAGE_NAME (docker) or ENV_BASE_URL (running server) "
                "before running inference."
            )

        try:
            result = run_episode(env, llm_client, task)
            all_results.append(result)
        finally:
            try:
                env.close()
            except Exception as exc:
                print(f"[DEBUG] env.close() error: {exc}", flush=True)

    # Summary across all tasks
    print("\n[SUMMARY]", flush=True)
    for r in all_results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['task']}: score={r['score']:.2f} steps={r['steps']}", flush=True)

    overall = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0
    print(f"\n  Overall mean score: {overall:.2f}", flush=True)


if __name__ == "__main__":
    main()
