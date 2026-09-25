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
"""CPU tests for composite AR padding and AR reward extraction from DiT extras."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics

from verl_omni.agent_loop.composite_agent_loop import (
    AR_REWARD_KEY,
    ARAgentLoopOutput,
    CompositeAgentLoopWorker,
    _InternalARAgentLoopOutput,
)
from verl_omni.agent_loop.diffusion_agent_loop import _InternalDiffusionAgentLoopOutput


class _StubTokenizer:
    def __init__(self, pad_token_id=0, return_1d=True):
        self.pad_token_id = pad_token_id
        self.padding_side = "right"
        self.return_1d = return_1d

    def pad(self, encoded, padding, max_length, return_tensors, return_attention_mask):
        ids = list(encoded["input_ids"])
        pad_id = self.pad_token_id if self.pad_token_id is not None else 0
        pad_len = max_length - len(ids)
        if self.padding_side == "left":
            padded = [pad_id] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)
        else:
            padded = ids + [pad_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
        input_ids = torch.tensor(padded, dtype=torch.long)
        result = {"input_ids": input_ids if self.return_1d else input_ids.unsqueeze(0)}
        if return_attention_mask:
            attn = torch.tensor(mask, dtype=torch.long)
            result["attention_mask"] = attn if self.return_1d else attn.unsqueeze(0)
        return result


def _make_worker(*, reward_handles=object(), tokenizer=None):
    worker = object.__new__(CompositeAgentLoopWorker)
    worker.tokenizer = tokenizer or _StubTokenizer()
    worker.rollout_config = SimpleNamespace(
        prompt_length=4,
        ar=SimpleNamespace(response_length=6),
    )
    worker.reward_loop_worker_handles = reward_handles
    return worker


def _ar_output(**kwargs):
    fields = dict(
        prompt_ids=[1, 2],
        response_ids=[10, 11],
        response_mask=[1, 1],
        refined_prompt="refined",
        ar_reward_score=None,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )
    fields.update(kwargs)
    return ARAgentLoopOutput(**fields)


def _ar_internal(*, ar_reward_score=None, extra_fields=None):
    return _InternalARAgentLoopOutput(
        prompt_ids=torch.zeros(1, 4, dtype=torch.long),
        response_ids=torch.zeros(1, 6, dtype=torch.long),
        response_mask=torch.ones(1, 6, dtype=torch.long),
        refined_prompt="refined",
        input_ids=torch.zeros(1, 10, dtype=torch.long),
        position_ids=torch.zeros(1, 4, 10, dtype=torch.long),
        attention_mask=torch.ones(1, 10, dtype=torch.long),
        ar_reward_score=ar_reward_score,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields=extra_fields or {},
    )


def _diffusion_internal(reward_extra_info, *, reward_score=1.5):
    return _InternalDiffusionAgentLoopOutput(
        prompt_ids=torch.zeros(1, 4, dtype=torch.long),
        response_diffusion_output=torch.zeros(1, 3, 8, 8),
        reward_score=reward_score,
        num_turns=2,
        metrics=AgentLoopMetrics(),
        extra_fields={"reward_extra_info": dict(reward_extra_info)},
    )


def test_pad_token_ids_empty_sequence_uses_pad_id():
    worker = _make_worker(tokenizer=_StubTokenizer(pad_token_id=9))
    padded = worker._pad_token_ids([], max_length=4, padding_side="right", return_attention_mask=True)
    torch.testing.assert_close(padded["input_ids"], torch.full((1, 4), 9, dtype=torch.long))
    torch.testing.assert_close(padded["attention_mask"], torch.zeros((1, 4), dtype=torch.long))


def test_pad_token_ids_empty_sequence_defaults_pad_id_when_tokenizer_pad_is_none():
    worker = _make_worker(tokenizer=_StubTokenizer(pad_token_id=None))
    padded = worker._pad_token_ids([], max_length=3, padding_side="left", return_attention_mask=False)
    torch.testing.assert_close(padded["input_ids"], torch.zeros((1, 3), dtype=torch.long))
    assert "attention_mask" not in padded


def test_pad_token_ids_left_and_right_and_unsqueezes_1d_tokenizer_output():
    worker = _make_worker(tokenizer=_StubTokenizer(pad_token_id=0, return_1d=True))
    left = worker._pad_token_ids([7, 8], max_length=4, padding_side="left", return_attention_mask=True)
    right = worker._pad_token_ids([7, 8], max_length=4, padding_side="right", return_attention_mask=True)
    torch.testing.assert_close(left["input_ids"], torch.tensor([[0, 0, 7, 8]]))
    torch.testing.assert_close(left["attention_mask"], torch.tensor([[0, 0, 1, 1]]))
    torch.testing.assert_close(right["input_ids"], torch.tensor([[7, 8, 0, 0]]))
    torch.testing.assert_close(right["attention_mask"], torch.tensor([[1, 1, 0, 0]]))


def test_ar_postprocess_pads_ragged_responses_so_they_batch():
    worker = _make_worker()
    early = _ar_output(
        prompt_ids=[1, 2],
        response_ids=[10, 11],
        response_mask=[1, 1],
        ar_response_logprobs=[-0.1, -0.2],
    )
    full = _ar_output(
        prompt_ids=[1, 2, 3],
        response_ids=[10, 11, 12, 13, 14, 15],
        response_mask=[1, 1, 1, 1, 1, 1],
        ar_response_logprobs=[-0.3, -0.4, -0.5, -0.6, -0.7, -0.8],
    )
    raw_prompt = [{"role": "user", "content": "hi"}]
    early_out = asyncio.run(worker._agent_loop_ar_postprocess(early, raw_prompt=raw_prompt))
    full_out = asyncio.run(worker._agent_loop_ar_postprocess(full, raw_prompt=raw_prompt))

    assert early_out.response_ids.shape == (1, 6)
    assert full_out.response_ids.shape == (1, 6)
    torch.testing.assert_close(early_out.response_ids[0, :2], torch.tensor([10, 11]))
    torch.testing.assert_close(early_out.response_ids[0, 2:], torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(early_out.response_mask[0, :2], torch.tensor([1, 1]))
    torch.testing.assert_close(early_out.response_mask[0, 2:], torch.zeros(4, dtype=torch.long))
    torch.testing.assert_close(full_out.response_mask[0], torch.ones(6, dtype=torch.long))
    torch.testing.assert_close(early_out.ar_response_logprobs[0, 0, :2], torch.tensor([-0.1, -0.2]))
    torch.testing.assert_close(early_out.ar_response_logprobs[0, 0, 2:], torch.zeros(4))

    batched = worker._postprocess_ar([early_out, full_out])
    assert batched.batch["responses"].shape == (2, 6)
    assert batched.batch["rollout_ar_log_probs"].shape[:2] == (2, 1)


def test_apply_ar_reward_averages_reward_ar_not_combined():
    worker = _make_worker()
    output = _ar_output()
    ar_internal = _ar_internal()
    extras = [
        {
            "reward/combined": 1.5,
            "reward/dit": 1.5,
            "reward/dit/dit_msg": "dit",
            AR_REWARD_KEY: 1.0,
            "reward/ar/ar_msg": "first",
            "reward/ar/sub": 0.2,
        },
        {
            "reward/combined": 1.5,
            "reward/dit": 1.5,
            "reward/dit/dit_msg": "dit",
            AR_REWARD_KEY: 0.5,
            "reward/ar/ar_msg": "second",
            "reward/ar/sub": 0.4,
        },
    ]
    diffusion_internals = [_diffusion_internal(extra) for extra in extras]

    worker._apply_ar_reward_from_diffusion_internals(output, ar_internal, diffusion_internals)

    assert output.ar_reward_score == pytest.approx(0.75)
    assert ar_internal.ar_reward_score == pytest.approx(0.75)
    assert ar_internal.extra_fields["reward_extra_info"][AR_REWARD_KEY] == pytest.approx(0.75)
    assert ar_internal.extra_fields["reward_extra_info"]["reward/ar/sub"] == pytest.approx(0.3)
    assert ar_internal.extra_fields["reward_extra_info"]["reward/ar/ar_msg"] == "first"
    assert "reward/combined" not in ar_internal.extra_fields["reward_extra_info"]
    assert "reward/dit" not in ar_internal.extra_fields["reward_extra_info"]
    for item in diffusion_internals:
        leftover = item.extra_fields["reward_extra_info"]
        assert leftover["reward/combined"] == 1.5
        assert leftover["reward/dit"] == 1.5
        assert AR_REWARD_KEY not in leftover
        assert "reward/ar/ar_msg" not in leftover
        assert "reward/ar/sub" not in leftover


def test_apply_ar_reward_skips_when_score_already_set_or_handles_missing():
    output = _ar_output(ar_reward_score=9.0)
    ar_internal = _ar_internal(ar_reward_score=9.0)
    extras = {AR_REWARD_KEY: 1.0, "reward/combined": 1.5}
    diffusion_internals = [_diffusion_internal(extras)]

    worker = _make_worker()
    worker._apply_ar_reward_from_diffusion_internals(output, ar_internal, diffusion_internals)
    assert output.ar_reward_score == pytest.approx(9.0)
    assert AR_REWARD_KEY in diffusion_internals[0].extra_fields["reward_extra_info"]

    worker_no_handles = _make_worker(reward_handles=None)
    output_empty = _ar_output()
    ar_empty = _ar_internal()
    worker_no_handles._apply_ar_reward_from_diffusion_internals(output_empty, ar_empty, diffusion_internals)
    assert output_empty.ar_reward_score is None

    worker._apply_ar_reward_from_diffusion_internals(output_empty, ar_empty, [])
    assert output_empty.ar_reward_score is None


def test_apply_ar_reward_requires_reward_ar_key():
    worker = _make_worker()
    with pytest.raises(AssertionError, match="`ar` must be used as reward function name"):
        worker._apply_ar_reward_from_diffusion_internals(
            _ar_output(),
            _ar_internal(),
            [_diffusion_internal({"reward/combined": 1.5})],
        )
