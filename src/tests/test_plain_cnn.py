"""Baseline integration: network, checkpoints, inference and train/validation."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train
from projector_distortion import cli
from projector_distortion.models import PlainCNNConfig, build_restorer, restorer_names
from projector_distortion.models.plain_cnn import PlainCNN
from projector_distortion.models.restoration import (
    ARCH_NAME, LEGACY_ARCH_NAME, RestorationConfig, architecture_name, build_network,
    config_from_settings, count_parameters, load_checkpoint, save_checkpoint,
)
from test_training_validation import make_triplet


class PlainCNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_default_topology_capacity_and_odd_size_gradient(self):
        net = build_network(PlainCNNConfig())
        self.assertIsInstance(net, PlainCNN)
        self.assertEqual(count_parameters(net), 30371)
        self.assertEqual(sum(isinstance(m, nn.Conv2d) for m in net.modules()), 5)
        inputs = torch.randn(2, 6, 31, 45, requires_grad=True)
        output = net(inputs)
        self.assertEqual(tuple(output.shape), (2, 3, 31, 45))
        self.assertLessEqual(output.abs().max().item(), 1.0)
        output.square().mean().backward()
        self.assertGreater(inputs.grad[:, :3].abs().sum().item(), 0)
        self.assertGreater(inputs.grad[:, 3:].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in net.parameters()))

    def test_config_selection_and_validation(self):
        settings = {"model": {"backend": "plain_cnn"},
                    "plain_cnn": {"width": 12, "depth": 3},
                    "ablation": {"use_ca": False}}
        cfg = config_from_settings(settings)
        self.assertEqual(cfg, PlainCNNConfig(width=12, depth=3))
        self.assertEqual(PlainCNNConfig.from_dict(cfg.to_dict()), cfg)
        for values in ({"width": 0}, {"depth": 1}, {"depth": 2.5},
                       {"width": True}, {"typo": 3}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PlainCNNConfig.from_dict(values)
        with self.assertRaisesRegex(ValueError, "unsupported trainable"):
            config_from_settings({"model": {"backend": "missing"}})
        self.assertEqual(architecture_name(config_from_settings({})), ARCH_NAME)

    def test_checkpoint_and_inference_round_trip(self):
        cfg = PlainCNNConfig(width=12, depth=3)
        net = build_network(cfg).eval()
        inputs = torch.randn(1, 6, 25, 33)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cnn.pt"
            save_checkpoint(path, net, cfg, epoch=4)
            loaded, back, meta = load_checkpoint(path)
            self.assertEqual(back, cfg)
            self.assertEqual(meta["arch"], "plain_cnn")
            self.assertEqual(meta["epoch"], 4)
            torch.testing.assert_close(net(inputs), loaded(inputs), rtol=0, atol=0)
            self.assertIn("plain_cnn", restorer_names())
            restorer = build_restorer("plain_cnn", path, input_size=(33, 25))
            image = np.full((27, 35, 3), 128, np.uint8)
            restored, residual, strength = restorer.restore_full(image, image)
            self.assertEqual(restored.shape, (25, 33, 3))
            self.assertEqual(residual.shape, restored.shape)
            self.assertEqual(restored.dtype, np.uint8)
            self.assertEqual(residual.dtype, np.uint8)
            self.assertTrue(0 <= strength <= 1)
            self.assertFalse(restorer.net.training)
            json.dumps(restorer.info())
            from projector_distortion.models import NullDetector
            from projector_distortion.pipeline import process_pair
            result = process_pair(image, image, restorer, NullDetector())
            self.assertEqual(result.restored.shape, restored.shape)
            self.assertEqual(result.det_restored, [])
            self.assertTrue(0 <= result.residual_mean <= 1)
            with self.assertRaisesRegex(RuntimeError, "does not match model.backend"):
                build_restorer("naf_se_unet", path)
            with self.assertRaisesRegex(RuntimeError, "supplied config"):
                load_checkpoint(path, cfg=RestorationConfig())

    def test_raw_cnn_checkpoint_uses_explicit_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.pt"
            net = build_network(PlainCNNConfig())
            torch.save(net.state_dict(), path)
            restorer = build_restorer("plain_cnn", path, input_size=(32, 24))
            self.assertEqual(restorer.cfg, PlainCNNConfig())
            self.assertEqual(restorer.meta["cfg_source"], "legacy-raw")

    def test_naf_and_legacy_checkpoint_compatibility(self):
        self.assertEqual(count_parameters(build_network()), 4184259)
        cfg = RestorationConfig(base_dim=16, enc_depth=(1, 1, 1),
                                dec_depth=(1, 1, 1), bottleneck_depth=1)
        net = build_network(cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            for arch in (ARCH_NAME, LEGACY_ARCH_NAME, None):
                blob = {"cfg": cfg.to_dict(), "state_dict": {
                    "module." + key: value for key, value in net.state_dict().items()}}
                if arch:
                    blob["arch"] = arch
                torch.save(blob, path)
                restored = build_restorer(ARCH_NAME, path)
                self.assertEqual(restored.cfg, cfg)
                for key, value in net.state_dict().items():
                    self.assertTrue(torch.equal(value, restored.net.state_dict()[key]))
                with self.assertRaisesRegex(RuntimeError, "does not match model.backend"):
                    build_restorer("plain_cnn", path)

    def test_cli_builds_cnn_from_its_own_config(self):
        cfg = PlainCNNConfig(width=12, depth=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cnn.pt"
            save_checkpoint(path, build_network(cfg), cfg)
            # Default YAML preserves the nondefault embedded checkpoint capacity.
            settings = {"model": {"backend": "plain_cnn", "weights": str(path),
                                   "input_size": [33, 25]},
                        "plain_cnn": PlainCNNConfig().to_dict(),
                        "ablation": {"use_ca": False}}
            with mock.patch.object(cli, "load_config", side_effect=[settings, {}]), \
                    contextlib.redirect_stdout(io.StringIO()):
                restorer, detector, _ = cli.build_models("cpu", need_detector=False)
            self.assertIsNone(detector)
            self.assertEqual(restorer.cfg, cfg)

    def test_training_validation_save_resume_and_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_triplet(root / "train", "training", 128, 120, size=(32, 32))
            make_triplet(root / "eval", "validation", 140, 120, size=(32, 32))
            settings = {"model": {"backend": "plain_cnn", "input_size": [32, 32]},
                        "plain_cnn": {"width": 8, "depth": 3}, "train": {
                            "epochs": 1, "batch_size": 1, "accum_steps": 1,
                            "patch_size": [16, 16], "resize_to": [32, 32],
                            "loss": {"l1": 1.0}, "validation": {
                                "root": str(root / "eval"), "save_images": 1}}}

            def serial_loader(dataset, **kwargs):
                kwargs["num_workers"] = 0
                return DataLoader(dataset, **kwargs)

            def run_training(out, extra=()):
                with mock.patch.object(train, "load_config", return_value=settings), \
                        mock.patch.object(train, "PerceptualLoss", return_value=nn.L1Loss()), \
                        mock.patch.object(train, "DataLoader", side_effect=serial_loader), \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(train.main([
                        "--data-root", str(root / "train"), "--out", str(out),
                        "--device", "cpu", *extra]), 0)
                run = next(out.iterdir())
                checkpoint = run / "restorer_CNN-W8-D3_best.pt"
                self.assertTrue(checkpoint.exists())
                with open(run / "loss_log.csv", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertTrue(np.isfinite(float(rows[0]["val_psnr"])))
                blob = torch.load(checkpoint, weights_only=True)
                self.assertEqual(blob["arch"], "plain_cnn")
                self.assertEqual(blob["selection_metric"], "val_psnr")
                self.assertEqual(blob["validation_triplets"], 1)
                return checkpoint

            first = run_training(root / "first")
            baseline = build_restorer("plain_cnn", first, input_size=(32, 32))
            image = np.full((32, 32, 3), 128, np.uint8)
            self.assertEqual(baseline.restore(image, image)[0].shape, image.shape)
            # Resume uses the checkpoint family even if YAML currently selects NAF.
            settings["model"]["backend"] = "naf_se_unet"
            resumed = run_training(root / "resumed", ("--resume", str(first)))
            self.assertEqual(load_checkpoint(resumed)[1], PlainCNNConfig(width=8, depth=3))


if __name__ == "__main__":
    unittest.main()
