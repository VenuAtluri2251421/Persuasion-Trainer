# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Multimodal Persuasion Trainer Environment.

Supporting Dynamic Personas, Whisper Audio, Llama Vision, and PyTorch RL tracking.
"""

from typing import Any, Dict, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class PersuasionTrainerAction(Action):
    """Action for the Multimodal Persuasion Environment."""

    message: Optional[str] = Field(default=None, description="Text message to the opponent")
    audio_path: Optional[str] = Field(default=None, description="Path to voice recording (for Whisper STT)")
    image_path: Optional[str] = Field(default=None, description="Path to webcam frame (for Llama Vision)")


class PersuasionTrainerObservation(Observation):
    """Observation returned after the opponent replies and grades the interaction."""

    # The AI Opponent's response
    reply_text: str = Field(..., description="The opponent's text response")
    strategy_used: str = Field(default="", description="The tactic (e.g. 'pressure', 'logic') the opponent deployed")
    
    # Grading & Rewards
    reward: float = Field(default=0.0, description="The user's normalized reward/score for the turn (0.0 to 1.0)")
    grades: Dict[str, Any] = Field(default_factory=dict, description="Detailed breakdown (clarity, logic, persuasion, confidence)")
    
    # System info
    done: bool = Field(default=False, description="True if the negotiation/interaction has ended")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Debug info (turn count, internal loss)")
