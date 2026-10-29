# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Llama4 model."""

import unittest

from transformers import is_torch_available
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_large_accelerator,
    slow,
    torch_device,
)

from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch

    from transformers import (
        Llama4ForCausalLM,
        Llama4ForConditionalGeneration,
        Llama4Processor,
        Llama4TextConfig,
    )


@slow
@require_torch_large_accelerator
class Llama4IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "meta-llama/Llama-4-Scout-17B-16E"

    @classmethod
    def setUpClass(cls):
        cls.model = Llama4ForConditionalGeneration.from_pretrained(
            "meta-llama/Llama-4-Scout-17B-16E",
            device_map="auto",
            dtype=torch.float32,
            attn_implementation="eager",
        )

    def setUp(self):
        super().setUp()
        self.processor = Llama4Processor.from_pretrained("meta-llama/Llama-4-Scout-17B-16E", padding_side="left")

        url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        self.messages_1 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": url},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

        self.messages_2 = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
                        ),
                    },
                    {
                        "type": "image",
                        "url": url_to_local_path(
                            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/australia.jpg"
                        ),
                    },
                    {"type": "text", "text": "Are these images identical?"},
                ],
            },
        ]

    def test_model_17b_16e_fp32(self):
        EXPECTED_TEXTS = Expectations(
            {
                ("xpu", 3): ['system\n\nYou are a helpful assistant.user\n\nWhat is shown in this image?assistant\n\nThe image shows a cow standing on a beach with a blue sky and a body of water in the background. The cow is brown with a white face'],
                ("cuda", None): ['system\n\nYou are a helpful assistant.user\n\nWhat is shown in this image?assistant\n\nThe image shows a cow standing on a beach, with a blue sky and a body of water in the background. The cow is brown with a white'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()

        inputs = self.processor.apply_chat_template(
            self.messages_1, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device=torch_device, dtype=self.model.dtype)
        output = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        print(output_text)
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_model_17b_16e_batch(self):
        inputs = self.processor.apply_chat_template(
            [self.messages_1, self.messages_2],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            add_generation_prompt=True,
        ).to(device=torch_device, dtype=torch.float32)

        output = self.model.generate(**inputs, max_new_tokens=30, do_sample=False)
        output_text = self.processor.batch_decode(output, skip_special_tokens=True)

        EXPECTED_TEXTS = [
            'system\n\nYou are a helpful assistant.user\n\nWhat is shown in this image?assistant\n\nThe image shows a cow standing on a beach, with a blue sky and a body of water in the background. The cow is brown with a white',
            'system\n\nYou are a helpful assistant.user\n\nAre these images identical?assistant\n\nNo, these images are not identical. The first image shows a cow standing on a beach with a blue sky and a white cloud in the background.'
        ]  # fmt: skip
        self.assertEqual(output_text, EXPECTED_TEXTS)


@require_torch
class Llama4TextTemperatureTuningTest(MemoryCleanupMixin, unittest.TestCase):
    def _model(self):
        torch.manual_seed(0)
        # floor_scale=4 so that short padded / packed rows cross a temperature step; layer 3 is the NoPE layer
        config = Llama4TextConfig(
            vocab_size=99, hidden_size=32, intermediate_size=37, intermediate_size_mlp=37, num_hidden_layers=4,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, moe_layers=[], floor_scale=4,
            attn_scale=0.1, pad_token_id=0, initializer_range=0.2,
        )  # fmt: skip
        return Llama4ForCausalLM(config).to(torch_device).eval()

    def test_nope_temperature_left_padding(self):
        model = self._model()
        long = torch.randint(1, 99, (1, 10), device=torch_device)
        short = torch.randint(1, 99, (1, 6), device=torch_device)
        input_ids = torch.cat([long, torch.cat([torch.zeros_like(short[:, :4]), short], dim=1)])
        attention_mask = (input_ids != 0).long()
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            padded = model(input_ids, attention_mask=attention_mask, position_ids=position_ids).logits[1, 4:]
            alone = model(short).logits[0]
        torch.testing.assert_close(padded, alone, rtol=1e-4, atol=1e-4)

    def test_nope_temperature_packed(self):
        model = self._model()
        long = torch.randint(1, 99, (1, 10), device=torch_device)
        short = torch.randint(1, 99, (1, 6), device=torch_device)
        position_ids = torch.cat([torch.arange(10), torch.arange(6)]).to(torch_device)[None]
        with torch.no_grad():
            packed = model(torch.cat([long, short], 1), position_ids=position_ids, use_cache=False).logits[0, 10:]
            alone = model(short).logits[0]
        torch.testing.assert_close(packed, alone, rtol=1e-4, atol=1e-4)
