import os
import sys

from openai import OpenAI

try:
    from openenv.core.env_client import create_local_env
except ImportError:
    raise ImportError("openenv-core is required. Run: uv sync")

try:
    from persuasion_trainer import PersuasionTrainerAction
except ImportError:
    from models import PersuasionTrainerAction  # type: ignore[no-redef]

def run_baseline(task_type: str, max_turns: int = 6, persona: str = None, objective: str = None):
    title = task_type.upper() if not persona else f"CUSTOM: {persona.upper()}"
    print(f"\n==========================================")
    print(f"BASELINE RUN: {title} TASK")
    print(f"==========================================")
    
    # Initialize OpenEnv locally
    env = create_local_env("server.persuasion_trainer_environment:PersuasionTrainerEnvironment")
    
    # Check OpenAI Key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is missing. Baseline run aborted.")
        return
        
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # Start the task
    kwargs = {"task_type": task_type}
    if persona: kwargs["persona"] = persona
    if objective: kwargs["objective"] = objective
    
    obs = env.reset(**kwargs)
    print(f"\n[Opponent]: {obs.reply_text}\n")
    
    history = [
        {"role": "system", "content": f"You are an AI negotiating a task. Be firm but logical. MAXIMUM 3 SENTENCES per reply."}
    ]
    history.append({"role": "user", "content": obs.reply_text})
    
    for _ in range(max_turns):
        if obs.done:
            break
            
        # Agent think
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            temperature=0.7
        )
        agent_reply = completion.choices[0].message.content
        history.append({"role": "assistant", "content": agent_reply})
        
        print(f"[Agent]: {agent_reply}")
        
        # Step env
        action = PersuasionTrainerAction(message=agent_reply)
        obs = env.step(action)
        
        print(f"\n[Opponent ({obs.strategy_used})]: {obs.reply_text}")
        history.append({"role": "user", "content": obs.reply_text})
        
        if obs.done:
            print(f"\n*** EPISODE FINISHED ***")
            print(f"Final Internal Env Loss: {obs.metadata.get('loss', 0.0):.4f}")
            print(f"Hesitations detected: {obs.metadata.get('hesitations_detected', 0)}")
            print(f"Final Overall Reward Score: {obs.reward:.2f}/1.0")
            print(f"Grading Breakdown: {obs.grades}")
            break

if __name__ == "__main__":
    print("Starting OpenEnv OpenAI Baseline Inference...")
    try:
        run_baseline("easy")
        run_baseline("medium")
        run_baseline("hard")
        
        # Demonstrating a completely custom dynamic persona & objective!
        run_baseline(
            task_type="custom", 
            persona="Skeptical Coding Instructor", 
            objective="The user wants a 2-day extension on their final project. You demand strict proof that they have actually been working."
        )
    except Exception as e:
        print(f"Error executing baseline: {e}")
        sys.exit(1)
        
