"""锁定原始模型源码、参数、前向结果和梯度，防止拆包改变实现。"""
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
    return module


class ModelPreservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original = load_original()

    def test_all_nine_definitions_are_verbatim_and_ast_identical(self):
        source = ORIGINAL.read_text(encoding="utf-8")
        definitions = {node.name: node for node in ast.parse(source).body
                       if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
        for name in models.__all__:
            with self.subTest(definition=name):
                migrated = inspect.getsource(getattr(models, name))
                node = definitions[name]
                original_lines = source.splitlines()[node.lineno - 1:node.end_lineno]
                self.assertEqual(migrated.rstrip("\n"), "\n".join(original_lines))
                self.assertEqual(ast.dump(ast.parse(migrated).body[0]),
                                 ast.dump(definitions[name]))
                module_source = Path(inspect.getfile(getattr(models, name))).read_text(encoding="utf-8")
                self.assertLessEqual(sum(isinstance(n, ast.ClassDef) for n in ast.parse(module_source).body), 1)
        self.assertEqual(config.MODEL_CONFIG, self.original.MODEL_CONFIG)

    def test_initialization_outputs_and_gradients_are_exact(self):
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
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for (name, parameter), (new_name, new_parameter) in zip(old.named_parameters(), new.named_parameters()):
                    self.assertEqual(name, new_name)
                    torch.testing.assert_close(new_parameter.grad, parameter.grad, rtol=0, atol=0)

    def test_causality_and_unbatched_input(self):
        model = models.MiniGPT(32).eval()
        inputs = torch.arange(14).reshape(2, 7)
        changed = inputs.clone()
        changed[:, 4:] = 31
        with torch.no_grad():
            torch.testing.assert_close(model(inputs)[:, :4], model(changed)[:, :4], rtol=0, atol=0)
            torch.testing.assert_close(model(inputs[0]), model(inputs[:1])[0], rtol=0, atol=0)
