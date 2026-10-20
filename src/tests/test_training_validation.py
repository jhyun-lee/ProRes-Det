"""Validation must not train the network, and best must follow held-out PSNR.

Run without optional pytest: python -m unittest discover -s src/tests
                          -p test_training_validation.py -v
"""

import contextlib
import csv
import io
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train


def make_triplet(root, sid, distorted_value=255, surface_value=0, size=(32, 40)):
    root = Path(root)
    paths = [root / "distorted" / f"distorted_{sid}_clip_001.jpg",
             root / "surface" / f"surface_{sid}.jpg",
             root / "light" / "light_clip_001.jpg"]
    for path, value in zip(paths, (distorted_value, surface_value, 64)):
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), np.full((*size, 3), value, np.uint8))
    return [str(p) for p in paths]


class ZeroResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(3)
        self.seen = []

    def forward(self, image):
        self.seen.append((self.training, torch.is_grad_enabled(), tuple(image.shape)))
        return self.norm(image[:, :3]) * 0


class ScalarResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, image):
        return torch.ones_like(image[:, :3]) * self.bias


class TrainingValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_validation_is_deterministic_full_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            triplet = make_triplet(directory, "one")
            dataset = train.ValidationDataset([triplet], (40, 32))
            first, second = dataset[0], dataset[0]
            self.assertEqual(tuple(first[0].shape), (3, 32, 40))
            for a, b in zip(first[:3], second[:3]):
                self.assertTrue(torch.equal(a, b))
            self.assertEqual(first[3], "distorted_one_clip_001")

    def test_scene_guard_rejects_prefix_renaming(self):
        train_set = [["projected_same_clip_a.jpg", "OriSame.jpg", "a.jpg"]]
        val_set = [["distorted_same_clip_b.jpg", "surface_same.jpg", "b.jpg"]]
        with self.assertRaisesRegex(SystemExit, "share surfaceId"):
            train._check_scene_separation(train_set, val_set)
        train._check_scene_separation(train_set,
                                     [["distorted_other_clip.jpg", "x.jpg", "b.jpg"]])

    def test_preview_limit_prefers_distinct_scenes(self):
        triplets = [[f"distorted_{sid}_{light}.jpg", "s", "l"]
                    for sid in ("one", "two") for light in ("a", "b")]
        names = train._preview_names(triplets, 2)
        self.assertEqual({train.surface_id(n) for n in names}, {"one", "two"})
        self.assertEqual(train._preview_names(triplets, 0), set())
        self.assertEqual(len(train._preview_names(triplets, -1)), 4)
        with self.assertRaises(ValueError):
            train._preview_names(triplets, -2)

    def test_validation_preserves_gradients_weights_and_batchnorm(self):
        with tempfile.TemporaryDirectory() as directory:
            triplets = [make_triplet(directory, sid) for sid in ("one", "two", "three")]
            loader = DataLoader(train.ValidationDataset(triplets, (40, 32)), batch_size=1)
            net = ZeroResidual().train()
            for p in net.parameters():
                p.grad = torch.ones_like(p)
            states = {k: v.clone() for k, v in net.state_dict().items()}
            gradients = [p.grad.clone() for p in net.parameters()]
            result = train.validate_epoch(
                net, loader, "cpu", {"l1": 1.0}, nn.L1Loss(), nn.L1Loss(),
                train.HighFrequencyWaveletLoss(), train._ssim_module("cpu"),
                str(Path(directory) / "output"), train._preview_names(triplets, 2))
            self.assertEqual(len(net.seen), 3)
            self.assertTrue(all(not training and not grads for training, grads, _ in net.seen))
            self.assertTrue(net.training)
            for key, expected in states.items():
                self.assertTrue(torch.equal(net.state_dict()[key], expected))
            for p, expected in zip(net.parameters(), gradients):
                self.assertTrue(torch.equal(p.grad, expected))
            self.assertAlmostEqual(result["val_loss"], 2.0)
            self.assertAlmostEqual(result["val_psnr"], 0.0)
            self.assertAlmostEqual(result["val_ssim"], 0.0001 / 1.0001, places=7)
            with open(Path(directory) / "output" / "metrics.csv", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)
            restored = list((Path(directory) / "output" / "restored").glob("*.png"))
            self.assertEqual(len(restored), 2)
            self.assertEqual(cv2.imread(str(restored[0])).shape, (32, 40, 3))

    def test_validation_error_restores_training_mode(self):
        class Invalid(nn.Module):
            def forward(self, image):
                return image[:, :3] * float("nan")

        with tempfile.TemporaryDirectory() as directory:
            loader = DataLoader(train.ValidationDataset(
                [make_triplet(directory, "one")], (40, 32)), batch_size=1)
            net = Invalid().train()
            with self.assertRaisesRegex(RuntimeError, "non-finite validation prediction"):
                train.validate_epoch(net, loader, "cpu", {"l1": 1.0}, nn.L1Loss(),
                                     nn.L1Loss(), train.HighFrequencyWaveletLoss(),
                                     train._ssim_module("cpu"), directory, set())
            self.assertTrue(net.training)

    def test_validation_metrics_use_original_uint8_target(self):
        with tempfile.TemporaryDirectory() as directory:
            triplet = make_triplet(directory, "gradient")
            reference = (np.arange(32 * 40 * 3).reshape(32, 40, 3) % 256).astype(np.uint8)
            # Lossless fixtures isolate the metric's target conversion.
            for index in (0, 1):
                path = Path(triplet[index]).with_suffix(".png")
                cv2.imwrite(str(path), reference)
                triplet[index] = str(path)
            dataset = train.ValidationDataset([triplet], (40, 32))
            expected_output = train.tensor_to_bgr(dataset[0][0].unsqueeze(0))
            result = train.validate_epoch(
                ZeroResidual(), DataLoader(dataset, batch_size=1), "cpu",
                {"l1": 1.0}, nn.L1Loss(), nn.L1Loss(),
                train.HighFrequencyWaveletLoss(), train._ssim_module("cpu"),
                str(Path(directory) / "output"), set())
            self.assertEqual(result["val_psnr"], train.psnr(reference, expected_output))
            self.assertEqual(result["val_ssim"], train.ssim(reference, expected_output))

    def test_best_checkpoint_uses_validation_not_training_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_triplet(root / "train", "training", size=(32, 32))
            # Improving the training target makes this held-out target worse.
            make_triplet(root / "eval", "validation", 128, 255, size=(32, 32))
            config = {"model": {"input_size": [32, 32]}, "train": {
                "epochs": 2, "batch_size": 1, "accum_steps": 1, "lr": 0.1,
                "patch_size": [16, 16], "resize_to": [32, 32],
                "loss": {"l1": 1.0},
                "validation": {"root": str(root / "eval"), "save_images": 0}}}

            def serial_loader(dataset, **kwargs):
                kwargs["num_workers"] = 0
                kwargs["pin_memory"] = False
                return DataLoader(dataset, **kwargs)

            with mock.patch.object(train, "load_config", return_value=config), \
                    mock.patch.object(train, "build_network", return_value=ScalarResidual()), \
                    mock.patch.object(train, "PerceptualLoss", return_value=nn.L1Loss()), \
                    mock.patch.object(train, "DataLoader", side_effect=serial_loader), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(train.main([
                    "--data-root", str(root / "train"), "--out", str(root / "runs"),
                    "--device", "cpu"]), 0)
            run = next((root / "runs").iterdir())
            with open(run / "loss_log.csv", newline="") as handle:
                history = list(csv.DictReader(handle))
            self.assertLess(float(history[1]["total"]), float(history[0]["total"]))
            self.assertLess(float(history[1]["val_psnr"]), float(history[0]["val_psnr"]))
            best = torch.load(run / "restorer_FULL_best.pt", map_location="cpu",
                              weights_only=True)
            last = torch.load(run / "epoch_2.pt", map_location="cpu", weights_only=True)
            self.assertEqual(best["epoch"], 1)
            self.assertEqual(last["epoch"], 2)
            self.assertEqual(last["best_epoch"], 1)
            self.assertEqual(best["selection_metric"], "val_psnr")
            self.assertEqual(best["validation_triplets"], 1)
            self.assertTrue(math.isfinite(best["val_psnr"]))
            self.assertTrue((run / "validation" / "epoch_001" / "metrics.csv").exists())
            self.assertTrue((run / "validation" / "epoch_002" / "metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()
