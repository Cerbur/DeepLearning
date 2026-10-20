"""预训练命令、默认路径和错误提示的回归测试。"""
import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

import torch

from sft_gpt.cli import build_parser, main
from sft_gpt.config import BASE_DIR, DEFAULT_OUTPUT
from sft_gpt.pre_training import load_checkpoint, pre_train


class PreTrainingCliTests(unittest.TestCase):
    def test_pretrain_command_and_checkpoint_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["preTrain"])
        self.assertEqual(args.command, "preTrain")
        self.assertIsNone(args.output_dir)  # Resolved after resume is known.
        self.assertEqual(DEFAULT_OUTPUT, BASE_DIR / "checkpoints" / "pre_train" / "lccc")
        self.assertEqual(parser.parse_args(["infer"]).checkpoint, DEFAULT_OUTPUT / "best.pt")
        self.assertIn("preTrain", parser.format_help())
        self.assertIn("预训练", parser.format_help())
        self.assertTrue(callable(pre_train))

    def test_pretrain_dispatch_and_default_command(self):
        for arguments in (["preTrain", "--device", "cpu"], []):
            with self.subTest(arguments=arguments), mock.patch("sft_gpt.cli.pre_train") as run, \
                    mock.patch("builtins.print"):
                main(arguments)
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0].command, "preTrain")
                self.assertEqual(run.call_args.args[0].output_dir, DEFAULT_OUTPUT)

    def test_resume_defaults_to_checkpoint_directory_not_new_lccc_output(self):
        for checkpoint in ("checkpoints/pre_train/last.pt", "old_wiki/last.pt", "last.pt"):
            with self.subTest(checkpoint=checkpoint), mock.patch("sft_gpt.cli.pre_train") as run, \
                    mock.patch("builtins.print"):
                main(["preTrain", "--resume", checkpoint, "--device", "cpu"])
                self.assertEqual(run.call_args.args[0].output_dir, Path(checkpoint).parent)

    def test_resume_honors_explicit_output_directory(self):
        with mock.patch("sft_gpt.cli.pre_train") as run, mock.patch("builtins.print"):
            main(["preTrain", "--resume", "old_wiki/last.pt",
                  "--output-dir", "copied_experiment", "--device", "cpu"])
            self.assertEqual(run.call_args.args[0].output_dir, Path("copied_experiment"))

    def test_old_train_command_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            build_parser().parse_args(["train"])
        self.assertEqual(error.exception.code, 2)

    def test_missing_checkpoint_points_to_pretrain(self):
        with mock.patch.object(Path, "is_file", return_value=False), \
                self.assertRaisesRegex(FileNotFoundError, "preTrain 预训练"):
            load_checkpoint("missing.pt", torch.device("cpu"))
