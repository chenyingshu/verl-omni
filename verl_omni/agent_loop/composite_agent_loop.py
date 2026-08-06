# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

Composite agent framework for multi-stage visual generation by AR (LLM/MLLM) + DiT composite architecture, as well as for agentic RL.

- CompositeAgentLo

"""

import asyncio
import random
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pydantic import ConfigDict
from tensordict import TensorDict
from verl.base_config import BaseConfig
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    DictConfigWrap,
    _agent_loop_registry,
)
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import get_dataset_class
from verl.utils.profiler import simple_timer
from verl.workers.rollout.llm_server import LLMServerClient

from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopOutput, DiffusionAgentLoopWorker
from verl_omni.agent_loop.utils import maybe_per_rollout_seeds
from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig


def _config_to_sampling_dict(config: Optional[BaseConfig]) -> dict:
    if config is None:
        return {}
    return {k: v for k, v in config.items() if not k.startswith("_")}


class CompositeAgentLoopOutput(DiffusionAgentLoopOutput):
    """Agent loop output. Supplement additional fields for AR part."""

    llm_response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    llm_response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    llm_response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    llm_refined_prompts: Optional[str] = None
    """Refined prompts for the LLM."""
    llm_reward_score: Optional[float] = None
    """Reward score for the ar trajectory."""


class _InternalCompositeAgentLoopOutput(CompositeAgentLoopOutput):
    """Internal agent loop output with padded sequences.
    Supplement additional fields for AR part.
    """

    llm_response_ids: torch.Tensor
    """Padded response token ids."""
    llm_input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    llm_position_ids: torch.Tensor
    """Padded position ids."""
    llm_response_mask: torch.Tensor
    """Padded response mask."""
    llm_attention_mask: torch.Tensor
    """Padded attention mask."""
    llm_response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""


class CompositeAgentLoopWorker(DiffusionAgentLoopWorker):
    """Composite Agent loop worker takes a batch of messages and run each message in an agent loop.
    Two-stage rollout: LLM CoT+rewrite → prompt embeds → DiT sampling.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server replicas, produced by
            ``LLMServerManager.get_client()`` in the trainer.
        dit_client (LLMServerClient): Client for the LLM server replicas, produced by
            ``LLMServerManager.get_client()`` in the trainer. Optinoal if llm_client returns visual responses directly.
        teacher_client (dict[str, LLMServerClient]): Not used by diffusion training; accepted to
            keep the constructor signature compatible with verl's ``AgentLoopManager.create()``,
            which positionally forwards a teacher client argument to each worker.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming
            reward computation.
        ar_reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming
            reward computation. Optional if reward_loop_worker_handles provides reward computation for both LLM and DiT.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        dit_client: LLMServerClient | None = None,
        teacher_client: dict[str, LLMServerClient] | None = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
        ar_reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles)

        # LLM client is used for CoT+re-prompting plus diffusion sampling,
        # DiT client is used for diffusion sampling only. OR
        if not hasattr(self, "dit_server_manager"):
            self.dit_server_manager = dit_client

        self.ar_reward_loop_worker_handles = ar_reward_loop_worker_handles

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch with the following fields.

            - ``prompts``: ``[bsz, prompt_length]`` original prompt token ids from dataset.
            - ``responses``: diffusion output, typically ``[bsz, C, H, W]`` (image)
              or ``[bsz, T, C, H, W]`` (video).
            - ``llm_responses``: [bsz, response_length], output token ids include full response tokens of CoT and refined prompt.
            - ``llm_response_mask``: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - ``llm_input_ids``: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - ``llm_attention_mask``: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - ``llm_position_ids``: [bsz, prompt_length + response_length], incremental position ids
            - ``rm_scores`` (optional): ``[bsz, 2]`` reward model scores for ar and dit part.
            - ``meta_info``:
              - ``refined_prompts``: ``List[str]``, refined prompts after CoT. Optional.
              - ``llm_full_responses``: ``List[str]``, full LLM responses. Optional.
              - ``metrics``: ``List[dict]``, per-sample agent loop metrics.
              - ``reward_extra_keys`` (optional): ``List[str]``, keys for reward
                extra info for logging/validation.
        """

        # TODO: (susan) see if can simplify the code
        config = self.rollout_config

        # composite sampling configs
        sampling_params = {
            **_config_to_sampling_dict(config.pipeline),
            **_config_to_sampling_dict(config.algo),
            "logprobs": config.calculate_log_probs,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": 1.0,
            "llm_logprobs": config.llm_calculate_log_probs,
        }

        is_validate = batch.meta_info.get("validate", False)
        per_rollout_seeds: Optional[list[int]] = None

        if is_validate:
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.pipeline))
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.algo))
            sampling_params["seed"] = config.val_kwargs.seed
            sampling_params["logprobs"] = False

            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
        else:
            sampling_params["global_steps"] = batch.meta_info["global_steps"]
            # Prefer trainer-assigned global indices so chunked workers derive the
            # same per-row seed regardless of local batch position / pack order.
            global_indices = batch.non_tensor_batch.get("_rollout_seed_global_idx")
            if global_indices is not None:
                global_indices = np.asarray(global_indices, dtype=np.int64).reshape(-1)
            per_rollout_seeds = maybe_per_rollout_seeds(batch.meta_info, len(batch), global_indices)

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            task_sampling_params = sampling_params.copy()
            if per_rollout_seeds is not None:
                task_sampling_params["seed"] = per_rollout_seeds[i]
            tasks.append(asyncio.create_task(self._run_agent_loop(task_sampling_params, **kwargs)))
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs, input_non_tensor_batch=batch.non_tensor_batch)

        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> _InternalCompositeAgentLoopOutput:
        assert agent_name in _agent_loop_registry, (
            f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
        )

        agent_loop_config = _agent_loop_registry[agent_name]
        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.server_manager,
            dit_server_manager=self.dit_server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
            extra_tokenizer_map=self.model_config.extra_tokenizer_map,
        )
        output: CompositeAgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
        return await self._agent_loop_postprocess(output, **kwargs)

    async def _agent_loop_postprocess(self, output, **kwargs) -> _InternalCompositeAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        # Pad extra tensor outputs from vllm-omni (e.g. prompt embeddings).
        extra_fields = {}
        for k, v in output.extra_fields.items():
            if isinstance(v, torch.Tensor):
                if k in ["prompt_embeds", "negative_prompt_embeds"]:
                    pad_tuple = (0, 0, 0, self.max_prompt_embed_length - v.shape[0])
                    v = F.pad(v, pad_tuple, value=0)
                elif k in ["prompt_embeds_mask", "negative_prompt_embeds_mask"]:
                    pad_tuple = (0, self.max_prompt_embed_length - v.shape[0])
                    v = F.pad(v, pad_tuple, value=0)
                extra_fields[k] = v.unsqueeze(0)
            else:
                extra_fields[k] = v

        extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        # （optional) debugging use, extract refined prompt from llm_response_ids
        llm_response_ids = output.llm_response_ids.unsqueeze(0)
        extra_fields["refined_prompt"] = self.tokenizer.decode(llm_response_ids, skip_special_tokens=True)

        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        response_diffusion_output = output.response_diffusion_output.unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            response_logprobs = output.response_logprobs.unsqueeze(0)

        prompt_ids = prompt_output["input_ids"]
        extra_fields["attention_mask"] = prompt_output["attention_mask"]

        await self._compute_score(
            output,
            prompts=prompt_ids,
            responses=response_diffusion_output,
            kwargs=kwargs,
        )

        if "reward_extra_info" in output.extra_fields:
            extra_fields["reward_extra_info"] = output.extra_fields["reward_extra_info"]

        return _InternalCompositeAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_diffusion_output=response_diffusion_output,
            response_logprobs=response_logprobs,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=extra_fields,
        )

    # async def _compute_score(self, output, prompts, responses, kwargs):
    #     """Compute reward score for single sample."""
    #     enable_async_reward = self.reward_loop_worker_handles is not None

    #     if output.reward_score is None and enable_async_reward:
    #         timing = {}
    #         with simple_timer("compute_score", timing):
    #             batch = TensorDict(
    #                 {
    #                     "prompts": prompts,  # [1, prompt_length]
    #                     "responses": responses,  # [1, C, H, W] or [1, T, C, H, W]
    #                 },
    #                 batch_size=1,
    #             )
    #             non_tensor_batch = {
    #                 **{k: np.array([v]) for k, v in kwargs.items()},
    #                 "__num_turns__": np.array([output.num_turns]),
    #                 "tool_extra_fields": np.array([output.extra_fields], dtype=object),
    #             }

    #             data = DataProto(
    #                 batch=batch,
    #                 non_tensor_batch=non_tensor_batch,
    #             )
    #             selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
    #             result = await selected_reward_loop_worker_handle.compute_score.remote(data)
    #             output.reward_score = result["reward_score"]
    #             output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
    #         output.metrics.compute_score = timing["compute_score"]

    # def _postprocess(
    #     self,
    #     inputs: list[_InternalCompositeAgentLoopOutput],
    #     input_non_tensor_batch: dict | None = None,
    # ) -> DataProto:
    #     """Process the padded outputs from _run_agent_loop and combine them into a batch."""
    #     # Convert lists back to tensors and stack them to create a batch.

    #     prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
    #     response_diffusion_output = torch.cat([input.response_diffusion_output for input in inputs], dim=0)
    #     optional_outputs = {}
    #     if inputs[0].response_logprobs is not None:
    #         optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

    #     # Handle extra fields that are tensors
    #     extra_keys = [k for k, v in inputs[0].extra_fields.items() if isinstance(v, torch.Tensor)]
    #     for key in extra_keys:
    #         optional_outputs[key] = torch.cat([input.extra_fields[key] for input in inputs], dim=0)
    #         for input in inputs:
    #             del input.extra_fields[key]

    #     batch = TensorDict(
    #         {
    #             "prompts": prompt_ids,  # [bsz, prompt_length]
    #             "responses": response_diffusion_output,  # [bsz, C, H, W] or [bsz, T, C, H, W]
    #             **optional_outputs,
    #         },
    #         batch_size=len(inputs),
    #     )

    #     scores = [input.reward_score for input in inputs]
    #     if all(score is not None for score in scores):
    #         rm_scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
    #         batch["rm_scores"] = rm_scores

    #     non_tensor_batch = {
    #         "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
    #     }
    #     if input_non_tensor_batch:
    #         non_tensor_batch.update(input_non_tensor_batch)

    #     # add reward_extra_info to non_tensor_batch
    #     reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
    #     reward_extra_keys = list(reward_extra_infos[0].keys())
    #     for key in reward_extra_keys:
    #         non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

    #     metrics = [input.metrics.model_dump() for input in inputs]
    #     # Collect extra fields from all inputs and convert them to np.ndarray
    #     extra_fields = {}
    #     all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
    #     for key in all_keys:
    #         temp_arr = np.empty(len(inputs), dtype=object)
    #         temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
    #         extra_fields[key] = temp_arr

    #     non_tensor_batch.update(extra_fields)

    #     # Only include reward_extra_keys in meta_info if rm_scores is in batch
    #     # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
    #     if "rm_scores" in batch.keys():
    #         meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
    #     else:
    #         meta_info = {"metrics": metrics}

    #     return DataProto(
    #         batch=batch,
    #         non_tensor_batch=non_tensor_batch,
    #         meta_info=meta_info,
    #     )
