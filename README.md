---
title: Multimodal Persuasion Trainer
emoji: 🎯
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 8000
tags:
  - openenv
  - negotiation
  - multimodal
  - reinforcement-learning
  - llama-4
  - pytorch
pinned: true
---

# 🎯 Multimodal Persuasion Trainer

> **An OpenEnv-compliant, multimodal negotiation training environment powered by Llama 4 Scout, Groq Whisper, Llama Vision, and PyTorch Online Policy Gradients.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://openenv.dev)
[![Llama 4](https://img.shields.io/badge/Llama%204%20Scout-17B-blueviolet)](https://groq.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-RL-orange)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/VenuAtluri936/Persuasion-Trainer)

---

## 🌟 What It Does

The Persuasion Trainer puts an AI agent in high-stakes real-world negotiation scenarios. The environment:

1. **Listens to the agent** — text message, voice recording (Whisper STT), or webcam frame (Llama Vision)
2. **Grades every turn** — clarity, logic, persuasion strength, confidence (0–10 each)
3. **Penalises hesitation** — detects filler words (`um`, `uh`, `like`, `I guess`) semantically
4. **Modulates scores via facial signals** — eye contact, nervousness, confidence from the webcam frame
5. **Adapts the opponent** — a PyTorch policy gradient network picks the next opponent strategy based on the agent's cumulative weaknesses
6. **Generates a counter-argument** — Llama 4 Scout plays a dynamic, realistic opponent

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Agent Turn (multimodal input)                 │
│                                                                 │
│   message (text) ──────────────────────────────────┐           │
│   audio_path  ──→ [Groq Whisper STT] ──→ transcript │           │
│   image_path  ──→ [Llama 4 Vision]  ──→ face signals│           │
└───────────────────────────────────────────┬─────────┘           │
                                            │ fused input
                    ┌───────────────────────▼──────────────────┐
                    │         Llama 4 Scout Grader              │
                    │  clarity / logic / persuasion / confidence│
                    │  + filler-word penalty (regex + semantic)  │
                    │  + facial expression modulation            │
                    └───────────────────┬──────────────────────┘
                                        │ reward signal (0–1)
                    ┌───────────────────▼──────────────────────┐
                    │     PyTorch Policy Gradient (online RL)   │
                    │  StrategyPolicy(4 → 16 → 5 → softmax)    │
                    │  Learns agent weaknesses across turns      │
                    │  Outputs next opponent strategy            │
                    └───────────────────┬──────────────────────┘
                                        │ strategy token
                    ┌───────────────────▼──────────────────────┐
                    │     Llama 4 Scout Opponent Generator       │
                    │  Dynamic persona + selected strategy       │
                    │  Generates realistic counter-argument      │
                    └──────────────────────────────────────────┘
```

---

## 📋 Tasks

| Task | Difficulty | Scenario |
|------|-----------|----------|
| `refund-negotiation` | 🟢 Easy | Convince a Customer Support Agent to issue a full refund |
| `job-offer-negotiation` | 🟢 Easy | Negotiate a better compensation package with a recruiter |
| `salary-raise-negotiation` | 🟡 Medium | Negotiate a 15% raise with a budget-constrained HR Manager |
| `landlord-rent-dispute` | 🟡 Medium | Persuade a landlord to reduce rent or cover repairs |
| `board-pivot-negotiation` | 🔴 Hard | Convince a hostile Board Member to approve a company pivot |
| `merger-acquisition-negotiation` | 🔴 Hard | Persuade a rival CEO to accept your M&A terms |

---

## 🧠 RL Policy

The `StrategyPolicy` is a 3-layer PyTorch MLP:

```python
StrategyPolicy(
  Linear(4, 16),   # input: [clarity, logic, persuasion, confidence]
  ReLU(),
  Linear(16, 5),   # output: [logical, emotional, examples, data, pressure]
  Softmax(dim=-1)
)
```

**Online REINFORCE** updates happen after every step using the normalised reward as the policy gradient signal. The opponent gets harder as the agent gets better — and exploits weaknesses when the agent underperforms.

---

## 📊 Reward Formula

```
base_score    = (clarity + logic + persuasion + confidence) / 40.0
filler_penalty = count(filler_words) * 0.05
vision_boost  = confidence_signal * 0.1 + eye_contact_signal * 0.05
nervousness_penalty = nervousness_signal * 0.05

reward = clip(base_score - filler_penalty + vision_boost - nervousness_penalty, 0, 1)
```

---

## 🔌 API Reference

### `POST /reset`
Start a new negotiation episode.

```json
// Request
{"task_type": "medium"}

// Response
{
  "observation": {
    "reply_text": "A 15% raise is not feasible at this time...",
    "strategy_used": "opening",
    "reward": 0.0,
    "grades": {},
    "done": false,
    "metadata": {}
  },
  "reward": 0.0,
  "done": false
}
```

### `POST /step`
Submit one turn of the negotiation.

```json
// Request
{
  "message": "My track record shows 40% revenue growth — I deserve this raise.",
  "audio_path": "/tmp/voice.wav",   // optional
  "image_path": "/tmp/webcam.jpg"   // optional
}

// Response
{
  "observation": {
    "reply_text": "Impressive metrics, but budget allocation for Q2...",
    "strategy_used": "data",
    "reward": 0.72,
    "grades": {"clarity": 8, "logic": 7, "persuasion": 8, "confidence": 7},
    "done": false,
    "metadata": {"turn": 2, "policy_loss": 0.031}
  },
  "reward": 0.72,
  "done": false
}
```

### `GET /health`
Returns `{"status": "ok"}` when server is running.

### `GET /docs`
Full interactive Swagger UI.

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | Powers Llama 4 Scout (text + vision) and Groq Whisper |
| `API_BASE_URL` | For inference | LLM endpoint for the agent (default: HF router) |
| `MODEL_NAME` | For inference | Agent model ID (default: Qwen/Qwen2.5-72B-Instruct) |
| `HF_TOKEN` | For inference | HuggingFace API token |
| `LOCAL_IMAGE_NAME` | Optional | Docker image name if using `from_docker_image()` |

---

## 🚀 Quick Start

### Run with Docker (local)

```bash
docker build -t persuasion-trainer . 
GROQ_API_KEY=your_key docker run -p 8000:8000 -e GROQ_API_KEY persuasion-trainer
```

### Run inference script

```bash
pip install openenv-core openai
export HF_TOKEN=your_hf_token
# ENV_BASE_URL defaults to the live HF Space automatically
python inference.py
```

### Expected stdout

```
[START] task=refund-negotiation env=persuasion_trainer model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action='I purchased this item...' reward=0.62 done=false error=null
[STEP] step=2 action='My consumer rights clearly state...' reward=0.74 done=false error=null
...
[END] success=true steps=4 score=0.71 rewards=0.62,0.74,0.75,0.73
```

---

## 🏆 What Makes This Stand Out

| Feature | Implementation |
|---------|---------------|
| **True Multimodal** | Text + audio (Whisper) + image (Llama Vision) — not just text |
| **Online RL** | PyTorch policy gradient updates after every step, not pre-trained |
| **Adaptive Difficulty** | RL policy exploits agent weaknesses dynamically |
| **Llama 4 Scout** | Latest multimodal model for both grading and opponent generation |
| **6 Task Scenarios** | Diverse real-world negotiation coverage across 2 difficulty tiers |
| **Fully OpenEnv Spec** | 5/5 pre-submission checklist, correct stdout format |

---

## 📁 Project Structure

```
persuasion_trainer/
├── inference.py                    # Mandatory hackathon inference script
├── baseline.py                     # Baseline agent for quick testing
├── client.py                       # OpenEnv HTTP client
├── models.py                       # Pydantic action/observation models
├── openenv.yaml                    # OpenEnv spec metadata
├── Dockerfile                      # HuggingFace Space build
├── pyproject.toml                  # Package config
└── server/
    ├── app.py                      # FastAPI application
    └── persuasion_trainer_environment.py  # Core RL environment
```

---

## 🤝 Hackathon Submission

- **Team**: APEX SYNDICATE
- **GitHub**: [VenuAtluri2251421/Persuasion-Trainer](https://github.com/VenuAtluri2251421/Persuasion-Trainer)
- **HF Space**: [VenuAtluri936/Persuasion-Trainer](https://huggingface.co/spaces/VenuAtluri936/Persuasion-Trainer)
- **Event**: Meta PyTorch Hackathon × Scaler School of Technology
