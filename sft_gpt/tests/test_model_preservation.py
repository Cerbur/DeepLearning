"""锁定模型结构/初始化；允许 SDPA、精度边界和运行时梯度检查点。"""
import ast
import importlib.util
import inspect
import unittest
from pathlib import Path

import torch

from sft_gpt import config, models

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / "mini_gpt" / "mini_gpt.py"


def load_original():
    spec = importlib.util.spec_from_file_location("original_mini_gpt", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Compare unchanged definitions at the active model size, not the legacy
    # checkout's historical 96/4/128 defaults. Keep production modules untouched.
    for name, value in config.MODEL_CONFIG.items():
        setattr(module, name, value)
    module.MODEL_CONFIG = dict(config.MODEL_CONFIG)
    return module


class ModelPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original = load_original()

    def test_definitions_preserved_except_sdpa_and_precision_boundaries(self):
        source = ORIGINAL.read_text(encoding="utf-8")
        definitions = {node.name: node for node in ast.parse(source).body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        for name in models.__all__:
            with self.subTest(definition=name):
                migrated = inspect.getsource(getattr(models, name))
                node = definitions[name]
                original_lines = source.splitlines()[node.lineno - 1:node.end_lineno]
                if name == "attention":
                    # 原算法逐行保留在注释中；SDPA 的数值/梯度由专项测试覆盖。
                    module_source = Path(inspect.getfile(models.attention)).read_text(encoding="utf-8")
                    commented = "\n".join(("# " + line).rstrip() if line else "#"
                                          for line in original_lines)
                    self.assertIn(commented, module_source)
                    continue
                if name in {"RMSNorm", "RotaryPositionalEmbedding", "GPTBlock", "SwiGLU", "GPTDecoder"}:
                    # 允许 forward 的精度边界/梯度检查点；参数/缓冲初始化必须不变。
                    migrated_node = ast.parse(migrated).body[0]
                    original_init = next(n for n in node.body if isinstance(n, ast.FunctionDef)
                                         and n.name == "__init__")
                    migrated_init = next(n for n in migrated_node.body if isinstance(n, ast.FunctionDef)
                                         and n.name == "__init__")
                    self.assertEqual(ast.dump(migrated_init), ast.dump(original_init))
                    self.assertEqual([n.name for n in migrated_node.body if isinstance(n, ast.FunctionDef)],
                                     [n.name for n in node.body if isinstance(n, ast.FunctionDef)])
                    continue
                if name == "MultiHeadAttention":
                    # 唯一计算变更：RoPE 后将 Q/K 转回 V 的 dtype。
                    migrated = migrated.replace("self.rope(q).to(dtype=v.dtype)", "self.rope(q)")
                    migrated = migrated.replace("self.rope(k).to(dtype=v.dtype)", "self.rope(k)")
                else:
                    self.assertEqual(migrated.rstrip("\n"), "\n".join(original_lines))
                self.assertEqual(ast.dump(ast.parse(migrated).body[0]),
                                 ast.dump(definitions[name]))
                module_source = Path(inspect.getfile(getattr(models, name))).read_text(encoding="utf-8")
                self.assertLessEqual(sum(isinstance(n, ast.ClassDef) for n in ast.parse(module_source).body), 1)
        self.assertEqual(config.MODEL_CONFIG, self.original.MODEL_CONFIG)

    def test_initialization_exact_and_outputs_gradients_close(self):
        for training in (False, True):
            with self.subTest(training=training):
                torch.manual_seed(123)
                old = self.original.MiniGPT(32).train(training)
                torch.manual_seed(123)
                new = models.MiniGPT(32).train(training)
                self.assertEqual(list(old.state_dict()), list(new.state_dict()))
                for key, value in old.state_dict().items():
                    torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)
                self.assertIs(new.output_proj.weight, new.embedding.embedding.weight)
                inputs = torch.arange(14).reshape(2, 7)
                torch.manual_seed(456)
                expected = old(inputs)
                expected.square().mean().backward()
                torch.manual_seed(456)
                actual = new(inputs)
                actual.square().mean().backward()
                # SDPA 改变运算顺序；参数初始化仍要求上方的逐位一致。
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                for (name, parameter), (new_name, new_parameter) in zip(old.named_parameters(), new.named_parameters()):
                    self.assertEqual(name, new_name)
                    torch.testing.assert_close(new_parameter.grad, parameter.grad, rtol=1e-5, atol=1e-7)

    def test_causality_and_unbatched_input(self):
        model = models.MiniGPT(32).eval()
        inputs = torch.arange(14).reshape(2, 7)
        changed = inputs.clone()
        changed[:, 4:] = 31
        with torch.no_grad():
            torch.testing.assert_close(model(inputs)[:, :4], model(changed)[:, :4], rtol=0, atol=0)
            torch.testing.assert_close(model(inputs[0]), model(inputs[:1])[0], rtol=0, atol=0)
