# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Persuasion Trainer Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import PersuasionTrainerAction, PersuasionTrainerObservation
except ImportError:
    from models import PersuasionTrainerAction, PersuasionTrainerObservation  # type: ignore[no-redef]



class PersuasionTrainerEnv(
    EnvClient[PersuasionTrainerAction, PersuasionTrainerObservation, State]
):
    """
    Client for the Persuasion Trainer Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with PersuasionTrainerEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(PersuasionTrainerAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = PersuasionTrainerEnv.from_docker_image("persuasion_trainer-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(PersuasionTrainerAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: PersuasionTrainerAction) -> Dict:
        """
        Convert PersuasionTrainerAction to JSON payload for step message.

        Args:
            action: PersuasionTrainerAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        payload = {"message": action.message}
        if action.audio_path:
            payload["audio_path"] = action.audio_path
        if action.image_path:
            payload["image_path"] = action.image_path
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[PersuasionTrainerObservation]:
        """
        Parse server response into StepResult[PersuasionTrainerObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with PersuasionTrainerObservation
        """
        obs_data = payload.get("observation", {})
        observation = PersuasionTrainerObservation(
            reply_text=obs_data.get("reply_text", ""),
            strategy_used=obs_data.get("strategy_used", ""),
            reward=float(obs_data.get("reward", 0.0)),
            grades=obs_data.get("grades", {}),
            done=obs_data.get("done", payload.get("done", False)),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=float(payload.get("reward", 0.0)),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
