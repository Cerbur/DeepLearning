"""普通文本推理不再套用五绝格式，字节回退生成保持 UTF-8 完整。"""
import unittest
from unittest import mock

import torch

from sft_gpt import HybridTokenizer, MiniGPT
from sft_gpt.inference import generate
from sft_gpt.config import MAX_SEQ_LEN


class TextInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_prompt_whitespace_emoji_and_long_context_are_preserved(self):
        tokenizer = HybridTokenizer.build([])
        model = MiniGPT(len(tokenizer.tokens))
        lengths = []

        def eos_logits(inputs):
            lengths.append(inputs.shape[-1])
            logits = torch.full((1, inputs.shape[-1], len(tokenizer.tokens)), -30.0)
            logits[:, -1, tokenizer.eos] = 20
            return logits

        unit = " user:  hello\nassistant: 你好🦊\n"
        prompt = unit * (MAX_SEQ_LEN // len(unit) + 1)
        with mock.patch.object(model, "forward", side_effect=eos_logits):
            text = generate(model, tokenizer, prompt, top_k=1)
        self.assertEqual(text, prompt)
        self.assertEqual(lengths, [MAX_SEQ_LEN])

    def test_byte_generation_masks_invalid_utf8_and_finishes_within_budget(self):
        tokenizer = HybridTokenizer.build([])
        model = MiniGPT(len(tokenizer.tokens))

        def byte_logits(inputs):
            logits = torch.full((1, inputs.shape[-1], len(tokenizer.tokens)), -20.0)
            logits[:, -1, 4 + 0xC0] = 100  # illegal leading byte must never win
            logits[:, -1, tokenizer.pad] = 90
            logits[:, -1, 4 + 0xF0] = 20  # prefers starting a 4-byte character
            logits[:, -1, 4 + 0x9F] = 19
            logits[:, -1, 4 + ord("a")] = 18
            return logits

        for budget in range(1, 9):
            with self.subTest(budget=budget), mock.patch.object(model, "forward", side_effect=byte_logits):
                text = generate(model, tokenizer, "你好\n", max_new_tokens=budget, top_k=1)
                self.assertTrue(text.startswith("你好\n"))
                self.assertNotIn("�", text)
                self.assertEqual(text.encode("utf-8").decode("utf-8"), text)

    def test_poetry_is_opt_in_and_not_allowed_for_hybrid(self):
        tokenizer = HybridTokenizer.build(["你好"], min_freq=1)
        model = MiniGPT(len(tokenizer.tokens))
        with self.assertRaisesRegex(ValueError, "五绝格式"):
            generate(model, tokenizer, structured=True)
