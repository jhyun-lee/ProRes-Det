"""Restoration: network shapes, ablation, checkpoint round-trip, wrapper contract."""

import os
import tempfile

import numpy as np
import pytest
import torch

from conftest import RESTORER_W, needs_restorer
from projector_distortion.models.restoration import (
    ARCH_NAME, LEGACY_ARCH_NAME, TOGGLES, NAFSEUNetRestorer, RestorationConfig,
    build_network, count_parameters, load_checkpoint, save_checkpoint,
)

REFERENCE_PARAMS = 4_184_259     # base_dim 48, all toggles on
REFERENCE_KEYS = 354


# --- config -------------------------------------------------------------------

def test_default_config_is_the_reference_shape():
    net = build_network(RestorationConfig())
    assert count_parameters(net) == REFERENCE_PARAMS
    assert len(net.state_dict()) == REFERENCE_KEYS


def test_config_round_trips_through_dict():
    cfg = RestorationConfig(use_ca=False, base_dim=32, enc_depth=[1, 2, 2])
    again = RestorationConfig.from_dict(cfg.to_dict())
    assert again == cfg
    assert again.enc_depth == (1, 2, 2), "lists must be coerced back to tuples"


def test_from_dict_ignores_unknown_keys():
    cfg = RestorationConfig.from_dict({"use_ca": False, "nonsense": 1})
    assert cfg.use_ca is False


def test_tag_names_the_ablated_parts():
    assert RestorationConfig().tag() == "FULL"
    assert RestorationConfig(use_ca=False).tag() == "NoCA"
    assert RestorationConfig(use_ca=False, use_skip3=False).tag() == "NoCA-NoSkip3"


# --- forward ------------------------------------------------------------------

@pytest.mark.parametrize("h,w", [(360, 640), (180, 320), (270, 478), (64, 64)])
def test_forward_preserves_spatial_size(h, w):
    """Fully convolutional, and the decoder realigns odd sizes."""
    net = build_network(RestorationConfig(base_dim=16, enc_depth=(1, 1, 1),
                                          dec_depth=(1, 1, 1), bottleneck_depth=1)).eval()
    with torch.no_grad():
        out = net(torch.randn(1, 6, h, w))
    assert out.shape == (1, 3, h, w)


def test_tanh_bounds_the_residual():
    net = build_network(RestorationConfig(base_dim=16, enc_depth=(1, 1, 1),
                                          dec_depth=(1, 1, 1), bottleneck_depth=1)).eval()
    with torch.no_grad():
        out = net(torch.randn(1, 6, 64, 64) * 10)
    assert out.min() >= -1.0 and out.max() <= 1.0


@pytest.mark.parametrize("attr,tag", TOGGLES)
def test_every_toggle_builds_trains_one_step(attr, tag):
    cfg = RestorationConfig(**{attr: False}, base_dim=16, enc_depth=(1, 1, 1),
                            dec_depth=(1, 1, 1), bottleneck_depth=1)
    net = build_network(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    distorted = torch.rand(1, 3, 64, 64) * 2 - 1
    light = torch.rand(1, 3, 64, 64) * 2 - 1
    target = torch.rand(1, 3, 64, 64) * 2 - 1

    residual = net(torch.cat([distorted, light], 1))
    assert residual.shape == (1, 3, 64, 64), tag
    loss = torch.nn.functional.l1_loss((distorted - residual).clamp(-1, 1), target)
    loss.backward()

    grads = [p.grad for p in net.parameters()]
    assert all(g is not None for g in grads), f"{tag}: some parameter got no gradient"
    assert all(torch.isfinite(g).all() for g in grads), f"{tag}: non-finite gradient"
    opt.step()


def test_disabling_ca_and_prenorm_removes_those_parameters():
    full = count_parameters(build_network(RestorationConfig()))
    no_ca = count_parameters(build_network(RestorationConfig(use_ca=False)))
    assert no_ca < full


def test_disabling_simple_gate_widens_the_projections():
    """GELU keeps the channel count, so conv3/conv5 grow rather than shrink."""
    full = count_parameters(build_network(RestorationConfig()))
    gelu = count_parameters(build_network(RestorationConfig(use_simple_gate=False)))
    assert gelu > full


def test_skips_change_the_join_conv_width():
    with_skip = build_network(RestorationConfig())
    without = build_network(RestorationConfig(use_skip1=False))
    assert with_skip.j1.in_channels == 2 * without.j1.in_channels


# --- checkpoints --------------------------------------------------------------

def test_checkpoint_carries_its_config():
    cfg = RestorationConfig(use_ca=False, base_dim=16, enc_depth=(1, 1, 1),
                            dec_depth=(1, 1, 1), bottleneck_depth=1)
    net = build_network(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        save_checkpoint(path, net, cfg, epoch=3, loss=0.5)

        loaded, back, meta = load_checkpoint(path, device="cpu")   # no cfg passed in
        assert back == cfg
        assert meta["cfg_source"] == "embedded"
        assert meta["epoch"] == 3
        for k, v in net.state_dict().items():
            assert torch.equal(v, loaded.state_dict()[k])


def test_legacy_bare_state_dict_defaults_to_the_reference():
    net = build_network(RestorationConfig())
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "raw.pt")
        torch.save(net.state_dict(), path)
        loaded, cfg, meta = load_checkpoint(path, device="cpu")
        assert meta["cfg_source"] == "legacy-raw"
        assert cfg.is_default()
        assert count_parameters(loaded) == REFERENCE_PARAMS


def test_wrong_config_fails_loudly_not_silently():
    net = build_network(RestorationConfig())
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "raw.pt")
        torch.save(net.state_dict(), path)
        with pytest.raises(RuntimeError):
            load_checkpoint(path, device="cpu",
                            cfg=RestorationConfig(use_simple_gate=False))


def test_dataparallel_prefixes_are_stripped():
    net = build_network(RestorationConfig(base_dim=16, enc_depth=(1, 1, 1),
                                          dec_depth=(1, 1, 1), bottleneck_depth=1))
    cfg = net.cfg
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pref.pt")
        torch.save({"format": 2, "arch": ARCH_NAME, "cfg": cfg.to_dict(),
                    "state_dict": {"module." + k: v
                                   for k, v in net.state_dict().items()}}, path)
        load_checkpoint(path, device="cpu")


def test_a_pre_rename_arch_tag_still_loads():
    """
    The architecture comes from the embedded `cfg`, never from `arch`.

    Checkpoints written before the naf_se_unet rename carry arch "restormer_like";
    reading that field to rebuild the network would have stranded every one of them.
    """
    cfg = RestorationConfig(base_dim=16, enc_depth=(1, 1, 1), dec_depth=(1, 1, 1),
                            bottleneck_depth=1)
    net = build_network(cfg)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "legacy.pt")
        torch.save({"format": 2, "arch": LEGACY_ARCH_NAME, "cfg": cfg.to_dict(),
                    "state_dict": net.state_dict()}, path)
        _, back, meta = load_checkpoint(path, device="cpu")
        assert back == cfg
        assert meta["arch"] == LEGACY_ARCH_NAME


def test_missing_checkpoint_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        load_checkpoint("does/not/exist.pt")


# --- wrapper ------------------------------------------------------------------

@needs_restorer
def test_shipped_weights_load_at_the_reference_size():
    r = NAFSEUNetRestorer(RESTORER_W, device="cpu")
    assert r.params == REFERENCE_PARAMS
    assert r.cfg.is_default(), "the shipped checkpoint should be the FULL variant"


@needs_restorer
def test_wrapper_returns_uint8_images_at_input_size(distorted_light):
    distorted, light = distorted_light
    r = NAFSEUNetRestorer(RESTORER_W, device="cpu", input_size=(320, 180))
    restored, residual = r.restore(distorted, light)
    for img in (restored, residual):
        assert img.dtype == np.uint8
        assert img.shape == (180, 320, 3)


@needs_restorer
def test_restore_full_matches_restore_and_adds_the_scalar(distorted_light):
    distorted, light = distorted_light
    r = NAFSEUNetRestorer(RESTORER_W, device="cpu", input_size=(320, 180))
    a, b = r.restore(distorted, light)
    a2, b2, mean = r.restore_full(distorted, light)
    assert np.array_equal(a, a2) and np.array_equal(b, b2)
    assert 0.0 <= mean <= 1.0, mean


@needs_restorer
def test_restored_differs_from_the_input(distorted_light):
    """A no-op restorer would silently pass every other test."""
    distorted, light = distorted_light
    r = NAFSEUNetRestorer(RESTORER_W, device="cpu", input_size=(320, 180))
    restored, _, mean = r.restore_full(distorted, light)
    from projector_distortion.utils.image import resize
    assert not np.array_equal(restored, resize(distorted, (320, 180)))
    assert mean > 0.0


@needs_restorer
def test_info_is_json_serialisable():
    import json
    r = NAFSEUNetRestorer(RESTORER_W, device="cpu")
    json.dumps(r.info())


# --- backend registry ---------------------------------------------------------

def test_the_shipped_backend_is_registered_under_its_arch_name():
    from projector_distortion.models import restorer_names
    from projector_distortion.models.restoration import ARCH_NAME
    assert ARCH_NAME in restorer_names()


def test_a_third_party_restorer_is_selected_by_name():
    """
    Registering a class is the whole extension step - the mirror of @register_detector.

    A backend that ignores `light` and emits an image directly (no residual) still
    satisfies the interface, which is what makes non-residual models comparable.
    """
    from projector_distortion.models import (
        BaseRestorer, build_restorer, register_restorer, restorer_names,
    )

    @register_restorer("dummy-test-restorer")
    class _Passthrough(BaseRestorer):
        name = "passthrough"

        def __init__(self, weights, device="cpu", input_size=(64, 32), **_):
            self.input_size = tuple(input_size)

        def restore(self, distorted_bgr, light_bgr):
            from projector_distortion.utils.image import resize
            out = resize(distorted_bgr, self.input_size)
            return out, np.zeros_like(out)

    try:
        assert "dummy-test-restorer" in restorer_names()
        r = build_restorer("dummy-test-restorer", weights="ignored", input_size=(64, 32))
        restored, residual = r.restore(np.zeros((90, 160, 3), np.uint8),
                                       np.zeros((90, 160, 3), np.uint8))
        assert restored.shape == (32, 64, 3) and residual.shape == restored.shape
        # restore_full's default fills in the scalar the pipeline reads per frame
        assert r.restore_full(restored, restored)[2] == 0.0
    finally:
        from projector_distortion.models.base import _RESTORERS
        _RESTORERS.pop("dummy-test-restorer", None)


def test_registering_a_non_restorer_is_refused():
    from projector_distortion.models import register_restorer
    with pytest.raises(TypeError):
        register_restorer("bad")(object)


def test_an_unknown_restorer_is_named_as_such():
    from projector_distortion.models import build_restorer
    with pytest.raises(ValueError, match="unknown restorer"):
        build_restorer("no-such-backend", weights="ignored")
