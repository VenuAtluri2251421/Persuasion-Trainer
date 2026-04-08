"""
Persuasion Trainer Environment Implementation.

A highly advanced multimodal environment powered by Llama 4 Scout,
Groq Whisper (voice STT), Llama Vision (facial expression analysis),
and PyTorch Online Policy Gradients.
"""

import base64
import json
import os
import random
import re
import warnings
from typing import Any, Optional
from uuid import uuid4

import torch
import torch.nn as nn
import torch.optim as optim
from groq import Groq

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import PersuasionTrainerAction, PersuasionTrainerObservation
except ImportError:
    from models import PersuasionTrainerAction, PersuasionTrainerObservation

warnings.filterwarnings("ignore", category=DeprecationWarning)

_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
_client: Optional[Groq] = None


def _get_client() -> Groq:
    """Lazy Groq client — raises a clear error only when first LLM call is made."""
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Set it before starting the server: export GROQ_API_KEY=your_key_here"
            )
        _client = Groq(api_key=key)
    return _client

STRATEGIES = ["logical", "emotional", "examples", "data", "pressure"]


class StrategyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, len(STRATEGIES)),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def call_llama(prompt: str, system: str = "", json_mode: bool = False) -> str:
    """Invokes Llama-4-Scout via Groq."""
    client = _get_client()
    system += " STRICT RULE: Keep all responses to 3 sentences max. Be punchy. Do not repeat the user's words back to them."
    response_format = {"type": "json_object"} if json_mode else {"type": "text"}
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format=response_format,
        temperature=0.7,
    )
    return completion.choices[0].message.content


def process_audio(audio_path: str) -> str:
    """Uses Groq Whisper to process voice inputs."""
    if not audio_path:
        return ""
    client = _get_client()
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3-turbo",
        )
        return transcription.text


def analyze_image(image_path: str) -> dict:
    """Uses Llama Vision (llama-4-scout multimodal) to analyze a webcam frame.

    Returns a dict with keys:
        emotion       (str)  - dominant facial expression, e.g. 'confident'
        confidence    (float) - 0-10 visual confidence score
        eye_contact   (float) - 0-10 estimated eye-contact score
        nervousness   (float) - 0-10 nervousness indicator (inverted: lower=better)
        vision_notes  (str)  - brief free-text observation
    """
    if not image_path:
        return {}
    try:
        with open(image_path, "rb") as f:
            raw = f.read()
        ext = image_path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        b64 = base64.b64encode(raw).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"

        completion = _get_client().chat.completions.create(
            model="llama-4-scout",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Analyze this person's facial expression in the context of a high-stakes negotiation. "
                                "Return ONLY valid JSON with these fields: "
                                '{"emotion": "string", "confidence": 0-10, "eye_contact": 0-10, "nervousness": 0-10, "vision_notes": "string"}. '
                                "Score confidence and eye_contact higher when the person looks assertive and composed. "
                                "Score nervousness higher when they look anxious or avoidant."
                            ),
                        },
                    ],
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        result = json.loads(completion.choices[0].message.content)
        # Normalise keys to floats where expected
        for k in ("confidence", "eye_contact", "nervousness"):
            result[k] = float(result.get(k, 5))
        result.setdefault("emotion", "neutral")
        result.setdefault("vision_notes", "")
        return result
    except Exception as exc:  # pragma: no cover
        # Non-fatal: vision analysis failure should not crash the episode
        return {"error": str(exc)}


class PersuasionTrainerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the persuasion_trainer environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.policy = StrategyPolicy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.05)
        self.current_scores = torch.tensor([5.0, 5.0, 5.0, 5.0])
        self.strategy_idx = 0
        self.turn = 0
        self.persona = ""
        self.objective = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> PersuasionTrainerObservation:
        """Reset the environment into one of 3 tasks / dynamic personas."""
        self.turn = 0
        self.current_scores = torch.tensor([5.0, 5.0, 5.0, 5.0])
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)

        task_type = kwargs.get("task_type", random.choice(["easy", "medium", "hard"]))

        if task_type == "easy":
            self.persona = "Customer Support Agent"
            self.objective = "The user wants a full refund for a slightly delayed delivery. You are very forgiving if they are just polite."
        elif task_type == "medium":
            self.persona = "Strict HR Manager"
            self.objective = "The user wants a 15% salary raise. You are budget-constrained."
        else:  # hard
            self.persona = "Hostile Board Member"
            self.objective = "The user is pitching to completely pivot the company. You hate the idea and will fight it."

        # Allow dynamic override
        self.persona = kwargs.get("persona", self.persona)
        self.objective = kwargs.get("objective", self.objective)

        opening = call_llama(
            f"Start the interaction. Obey your objective: {self.objective}",
            system=f"You are a {self.persona}. You are reacting to a user.",
        )

        return PersuasionTrainerObservation(
            reply_text=opening,
            strategy_used="opening",
            reward=0.0,
            grades={},
            done=False,
            metadata={"persona": self.persona, "task_type": task_type},
        )

    def step(
        self,
        action: PersuasionTrainerAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> PersuasionTrainerObservation:
        """Execute a step in the environment via RL policy and Llama grading."""
        self.turn += 1
        self._state.step_count += 1

        user_message = action.message or ""

        # ── Multimodal: Voice (Groq Whisper) ──────────────────────────────────
        if action.audio_path:
            user_message += "\n" + process_audio(action.audio_path)

        # ── Multimodal: Facial Expression (Llama Vision) ──────────────────────
        vision_data: dict = {}
        if action.image_path:
            vision_data = analyze_image(action.image_path)

        # 1. GRADE the user response using Llama
        grading_prompt = (
            f"Grade this negotiation response: \"{user_message}\". "
            "Return JSON with integer scores 0-10: "
            '{"clarity": 0, "logic": 0, "persuasion": 0, "confidence": 0}'
        )
        raw_grades = call_llama(
            grading_prompt,
            system="You are an expert debate judge. Return only valid JSON.",
            json_mode=True,
        )
        
        try:
            grades = json.loads(raw_grades)
        except json.JSONDecodeError:
            grades = {"clarity": 5, "logic": 5, "persuasion": 5, "confidence": 5}

        # 2. Semantic Penalty (Hesitations constraint)
        hesitations = len(re.findall(r"\b(um|uh|hm|like|i guess)\b", user_message.lower()))
        penalty = hesitations * 0.5
        grades["persuasion"] = max(0.0, float(grades.get("persuasion", 5)) - penalty)

        # 3. Vision Modulation — apply facial expression signals to grades
        if vision_data and "error" not in vision_data:
            # Confident, composed expression boosts confidence score
            face_conf_boost = (vision_data.get("confidence", 5) - 5) * 0.3
            # Strong eye-contact signals engagement → minor persuasion boost
            face_eye_boost = (vision_data.get("eye_contact", 5) - 5) * 0.2
            # Nervousness penalises confidence
            face_nerv_penalty = vision_data.get("nervousness", 5) * 0.1

            grades["confidence"] = max(0.0, min(10.0, float(grades.get("confidence", 5)) + face_conf_boost - face_nerv_penalty))
            grades["persuasion"] = max(0.0, min(10.0, float(grades.get("persuasion", 5)) + face_eye_boost))

        self.current_scores = torch.tensor(
            [
                float(grades.get("clarity", 5)),
                float(grades.get("logic", 5)),
                float(grades.get("persuasion", 5)),
                float(grades.get("confidence", 5)),
            ]
        )

        # 3. RL POLICY UPDATE (Online Policy Gradient)
        probs = self.policy(self.current_scores.unsqueeze(0))
        m = torch.distributions.Categorical(probs)

        # Epsilon-greedy exploration
        if random.random() < 0.2:
            self.strategy_idx = random.randint(0, len(STRATEGIES) - 1)
            action_tensor = torch.tensor([self.strategy_idx])
        else:
            action_tensor = m.sample()
            self.strategy_idx = action_tensor.item()

        current_strategy = STRATEGIES[self.strategy_idx]
        log_prob = m.log_prob(action_tensor)

        # Reward Calculation
        user_reward = (float(grades.get("logic", 0)) + float(grades.get("persuasion", 0))) / 20.0
        env_reward = 1.0 - user_reward  # Env wins if user performs poorly

        # Backpropagation Mid-Episode
        loss = -log_prob * env_reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 4. GENERATE COUNTER-ARGUMENT
        opponent_reply = call_llama(
            f"The user said: \"{user_message}\". Respond using a {current_strategy} strategy.",
            system=f"You are a {self.persona}. {self.objective}",
        )

        done = self.turn >= 6

        vision_meta = {
            "emotion": vision_data.get("emotion", "n/a"),
            "eye_contact": vision_data.get("eye_contact", None),
            "nervousness": vision_data.get("nervousness", None),
            "vision_notes": vision_data.get("vision_notes", ""),
        } if vision_data else {}

        return PersuasionTrainerObservation(
            reply_text=opponent_reply,
            strategy_used=current_strategy,
            reward=user_reward,
            grades=grades,
            done=done,
            metadata={
                "turn": self.turn,
                "loss": loss.item(),
                "hesitations_detected": hesitations,
                "vision": vision_meta,
            },
        )

    @property
    def state(self) -> State:
        return self._state
