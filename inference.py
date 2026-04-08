"""
Inference Script — Persuasion Trainer Environment
===================================================
MANDATORY env vars:
    API_BASE_URL        LLM API endpoint (default: HF router)
    MODEL_NAME          Model identifier  (default: Qwen2.5-72B-Instruct)
    HF_TOKEN            Your Hugging Face / API key  (no default)
    LOCAL_IMAGE_NAME    Docker image name — accepted for spec compliance
                        but we connect to ENV_BASE_URL for reliability.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

try:
    from persuasion_trainer import PersuasionTrainerAction, PersuasionTrainerEnv
except ImportError:
    from client import PersuasionTrainerEnv  # type: ignore[no-redef]
    from models import PersuasionTrainerAction  # type: ignore[no-redef]

# ── Mandatory env vars (checklist) ────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")           # No default — set by caller
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")   # Accepted; see note above

# ── Environment server ────────────────────────────────────────────────────────
# We always connect via HTTP to the already-running HF Space.
# This guarantees GROQ_API_KEY is available (set as a Space secret) and
# avoids container-injection issues in the validator environment.
ENV_BASE_URL: str = os.getenv(
    "ENV_BASE_URL",
    "https://venuatluri936-persuasion-trainer.hf.space",
)

# ── Episode config ────────────────────────────────────────────────────────────
BENCHMARK               = "persuasion_trainer"
MAX_TURNS               = 6
TEMPERATURE             = 0.7
MAX_TOKENS              = 200
SUCCESS_SCORE_THRESHOLD = 0.5

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

_FALLBACK_MSG = "I believe this proposal is clearly in everyone's best interest."


# ── Logging helpers ───────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    clean     = action.replace("\n", " ").replace("\r", "").strip()
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


# ── LLM agent ─────────────────────────────────────────────────────────────────

def build_prompt(step: int, opponent_reply: str, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""
        Step {step}. The opponent just said:
        "{opponent_reply}"

        Previous exchange:
        {history_block}

        Your persuasive reply:
    """).strip()


def get_agent_reply(client: OpenAI, step: int, opponent: str, history: List[str]) -> str:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(step, opponent, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or _FALLBACK_MSG
    except Exception as exc:
        print(f"[DEBUG] LLM error step {step}: {exc}", flush=True)
        return _FALLBACK_MSG


# ── Episode runner ────────────────────────────────────────────────────────────

async def run_episode(env: PersuasionTrainerEnv, client: OpenAI, task: dict) -> dict:
    task_type = task["task_type"]
    task_name = task["task_name"]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards:     List[float] = []
    steps_taken: int         = 0
    success:     bool        = False
    score:       float       = 0.0

    try:
        # ── Reset ────────────────────────────────────────────────────────
        try:
            obs = await env.reset(task_type=task_type)
        except TypeError:
            # Some openenv versions don't forward kwargs in reset
            obs = await env.reset()

        opponent_reply: str       = getattr(obs, "reply_text", "Let's begin.")
        history:        List[str] = []

        # ── Steps ─────────────────────────────────────────────────────────
        for step in range(1, MAX_TURNS + 1):
            if getattr(obs, "done", False):
                break

            agent_msg = get_agent_reply(client, step, opponent_reply, history)
            action    = PersuasionTrainerAction(message=agent_msg)

            try:
                result        = await env.step(action)
                obs           = result.observation
                reward: float = float(result.reward or 0.0)
                done:   bool  = bool(result.done)
                err_msg       = None
            except Exception as exc:
                reward  = 0.0
                done    = True
                err_msg = str(exc)
                print(f"[DEBUG] env.step error: {exc}", flush=True)

            steps_taken = step
            rewards.append(reward)
            log_step(step=step, action=agent_msg, reward=reward, done=done, error=err_msg)

            opponent_reply = getattr(obs, "reply_text", "")
            strategy       = getattr(obs, "strategy_used", "")
            history.append(
                f"Step {step} | You: {agent_msg!r} | Opponent ({strategy}): {opponent_reply!r}"
            )

            if done or err_msg:
                break

        score   = min(max((sum(rewards) / len(rewards)) if rewards else 0.0, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        # Catch-all so [END] is always emitted
        print(f"[DEBUG] run_episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "score": score, "success": success, "steps": steps_taken}


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    llm_client  = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    all_results = []

    for task in TASKS:
        # Always connect via HTTP to the live HF Space.
        # LOCAL_IMAGE_NAME is kept in env for spec compliance but not used here
        # because the docker container wouldn't have GROQ_API_KEY injected.
        env = PersuasionTrainerEnv(base_url=ENV_BASE_URL)

        try:
            result = await run_episode(env, llm_client, task)
            all_results.append(result)
        except Exception as exc:
            print(f"[DEBUG] task={task['task_name']} top-level error: {exc}", flush=True)
        finally:
            try:
                await env.close()
            except Exception as exc:
                print(f"[DEBUG] env.close() error: {exc}", flush=True)

    # Summary
    if all_results:
        print("\n[SUMMARY]", flush=True)
        for r in all_results:
            mark = "✓" if r["success"] else "✗"
            print(f"  {mark} {r['task']}: score={r['score']:.2f} steps={r['steps']}", flush=True)
        overall = sum(r["score"] for r in all_results) / len(all_results)
        print(f"\n  Overall mean score: {overall:.2f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
