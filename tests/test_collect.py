"""data/collect.py - the capture-side dataset builder, and the live warp evidence."""

import importlib.util
import json
import os
import tempfile

import cv2
import numpy as np
import pytest

from conftest import ROOT
from projector_distortion.data import find_samples
from projector_distortion.utils.image import psnr
from projector_distortion.utils.recording import RunRecorder
from projector_distortion.utils.visualize import warp_before_after

_spec = importlib.util.spec_from_file_location(
    "collect", os.path.join(ROOT, "data", "collect.py"))
collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collect)

# A screen seen well off-axis: the case a bilinear interior model gets wrong.
CORNERS = np.float32([[150, 120], [1130, 60], [1180, 860], [95, 800]])
SIZE = (1280, 720)


def _quad_contour(corners=CORNERS, per_edge=200):
    """The quad outline as a dense contour, like findContours would return."""
    edges = [collect.resample(np.vstack([corners[i], corners[(i + 1) % 4]]),
                              per_edge + 1)[:-1] for i in range(4)]
    return np.vstack(edges).reshape(-1, 1, 2).astype(np.int32)


def _capture(img, corners=CORNERS, canvas=(1280, 960)):
    """A flat image seen off-axis, the way the webcam sees the projected screen."""
    src = np.float32([[0, 0], [img.shape[1], 0],
                      [img.shape[1], img.shape[0]], [0, img.shape[0]]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, corners), canvas)


# --- boundary sampling --------------------------------------------------------

def test_resample_keeps_both_ends_and_spaces_evenly():
    pts = collect.resample(np.float32([[0, 0], [10, 0], [100, 0]]), 11)
    assert len(pts) == 11
    assert pts[0].tolist() == [0, 0] and pts[-1].tolist() == [100, 0]
    gaps = np.diff(pts[:, 0])
    assert np.allclose(gaps, gaps[0], atol=1e-3), "arc length, not index, spacing"


def test_boundary_points_are_matched_pairs_without_duplicate_corners():
    contour = _quad_contour()
    corners = collect.quad_corners(contour)
    src, dst = collect.boundary_points(collect.edge_paths(contour, corners), 20, SIZE)
    assert len(src) == len(dst) == 20
    assert len(np.unique(dst, axis=0)) == 20, "TPS rejects duplicated correspondences"


def test_corners_come_back_ordered_tl_tr_br_bl():
    corners = collect.quad_corners(_quad_contour())
    assert corners is not None
    assert np.allclose(corners, CORNERS, atol=2)


# --- the warp -----------------------------------------------------------------

def test_boundary_warp_matches_the_homography_on_a_flat_screen():
    """
    Straight edges must leave the corner homography alone.

    A plain Coons patch over the four edges makes the interior bilinear, which on an
    off-axis rig misplaces the middle of the image by tens of pixels - the reason the
    correction is computed in rectified space instead.
    """
    contour = _quad_contour()
    corners = collect.quad_corners(contour)
    map_x, map_y = collect.boundary_map(collect.edge_paths(contour, corners), corners,
                                        SIZE)

    w, h = SIZE
    rect = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    grid = np.stack(np.meshgrid(np.arange(w, dtype=np.float32),
                                np.arange(h, dtype=np.float32)), axis=-1)
    expected = collect.project(grid, cv2.getPerspectiveTransform(rect, corners))

    assert np.abs(map_x - expected[..., 0]).max() < 2.0
    assert np.abs(map_y - expected[..., 1]).max() < 2.0


@pytest.mark.parametrize("mode", ["boundary", "homography"])
def test_rectifier_recovers_the_projected_image(bgr_image, mode):
    source = cv2.resize(bgr_image, SIZE)
    capture = _capture(source)
    contour = collect.screen_contour(capture, inset=0)
    assert contour is not None, "no screen boundary found in the synthetic capture"

    out = collect.Rectifier(contour, SIZE, mode=mode)(capture)
    assert out.shape[:2] == (SIZE[1], SIZE[0])
    assert psnr(source, out) > 18, "rectification did not undo the perspective"


def test_rectifier_rejects_a_boundary_that_is_not_quad_like():
    circle = np.array([[[int(640 + 300 * np.cos(t)), int(480 + 300 * np.sin(t))]]
                       for t in np.linspace(0, 2 * np.pi, 200, endpoint=False)],
                      np.int32)
    with pytest.raises(ValueError, match="4-corner"):
        collect.Rectifier(circle, SIZE)


def test_tps_mode_explains_itself_when_opencv_dropped_the_shape_module():
    if hasattr(cv2, "createThinPlateSplineShapeTransformer"):
        pytest.skip("this OpenCV still ships the shape module")
    with pytest.raises(SystemExit, match="shape module"):
        collect.build_tps(np.zeros((4, 2), np.float32), np.zeros((4, 2), np.float32))


# --- the session on disk ------------------------------------------------------

def test_warp_stage_writes_a_dataset_the_loaders_accept(bgr_image):
    """The whole point of the naming: a collected session needs no conversion."""
    source = cv2.resize(bgr_image, SIZE)
    capture = _capture(source)

    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("raw/surface", "raw/distorted", "light"):
            os.makedirs(os.path.join(tmp, sub), exist_ok=True)
        cv2.imwrite(os.path.join(tmp, "raw", "surface", "surface_0409001429.jpg"),
                    capture)
        cv2.imwrite(os.path.join(tmp, "raw", "distorted",
                                 "distorted_0409001429_0803120000_1000_0.jpg"), capture)
        cv2.imwrite(os.path.join(tmp, "light", "light_0803120000_1000_0.jpg"), source)

        assert collect.main(["warp", "--root", tmp, "--inset", "0"]) == 0

        samples = find_samples(tmp, gt_root=tmp)
        assert len(samples) == 1
        assert samples[0].surface_id == "0409001429"
        assert samples[0].surface and os.path.basename(samples[0].surface) == \
            "surface_0409001429.jpg"
        for path in (samples[0].distorted, samples[0].surface):
            img = cv2.imread(path)
            assert (img.shape[1], img.shape[0]) == (640, 360)
        assert os.listdir(os.path.join(tmp, "debug")) == ["0409001429_warp.jpg"]

        meta = os.path.join(tmp, "collect_meta.json")
        assert os.path.exists(meta)


def test_limit_does_not_mistake_skipped_scenes_for_orphans(bgr_image, capsys):
    """--limit leaves scenes unprocessed; their surface shots still exist."""
    capture = _capture(cv2.resize(bgr_image, SIZE))

    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("raw/surface", "raw/distorted"):
            os.makedirs(os.path.join(tmp, sub), exist_ok=True)
        for sid in ("0409001429", "0409232547"):
            cv2.imwrite(os.path.join(tmp, "raw", "surface", f"surface_{sid}.jpg"),
                        capture)
            cv2.imwrite(os.path.join(tmp, "raw", "distorted",
                                     f"distorted_{sid}_0803120000_1000_0.jpg"), capture)

        assert collect.main(["warp", "--root", tmp, "--inset", "0",
                             "--limit", "1", "--no-debug"]) == 0
        assert "no surface shot" not in capsys.readouterr().out

        with open(os.path.join(tmp, "collect_meta.json"), encoding="utf-8") as f:
            assert json.load(f)["warp"]["orphan_surface_ids"] == []


def test_captures_without_a_surface_shot_are_still_reported(bgr_image, capsys):
    capture = _capture(cv2.resize(bgr_image, SIZE))

    with tempfile.TemporaryDirectory() as tmp:
        for sub in ("raw/surface", "raw/distorted"):
            os.makedirs(os.path.join(tmp, sub), exist_ok=True)
        cv2.imwrite(os.path.join(tmp, "raw", "surface", "surface_0409001429.jpg"),
                    capture)
        cv2.imwrite(os.path.join(tmp, "raw", "distorted",
                                 "distorted_9999999999_0803120000_1000_0.jpg"), capture)

        assert collect.main(["warp", "--root", tmp, "--inset", "0", "--no-debug"]) == 0
        assert "9999999999" in capsys.readouterr().out


def test_light_stage_ids_survive_the_filename_convention():
    """`lightId` is parsed off the first '_', so the timestamp tag must not carry one."""
    from projector_distortion.data import light_id, surface_id

    name = "light_0803120000_1000_600.jpg"
    assert light_id(name) == "0803120000_1000_600"
    assert surface_id(f"distorted_0409001429_{light_id(name)}.jpg") == "0409001429"


# --- live warp evidence -------------------------------------------------------

def test_warp_figure_holds_both_views_side_by_side():
    pre = np.full((960, 1280, 3), 64, np.uint8)
    post = np.full((360, 640, 3), 200, np.uint8)
    fig = warp_before_after(pre, post, [(10, 10), (600, 20), (620, 500), (20, 480)])
    assert fig.shape[1] > fig.shape[0], "two tiles wide, one tall"
    assert fig.shape[1] >= 2 * 640


def test_recorder_writes_the_first_frame_warp_pair():
    pre = np.full((960, 1280, 3), 64, np.uint8)
    post = np.full((360, 640, 3), 200, np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        with RunRecorder(tmp, save_every=1) as rec:
            written, fig = rec.save_warp_pair(pre, post, [(10, 10), (600, 20),
                                                          (620, 500), (20, 480)])
        assert fig is not None
        assert len(written) == 3
        assert sorted(os.listdir(os.path.join(tmp, "calib"))) == [
            "frame_compare.jpg", "frame_post.jpg", "frame_pre.jpg"]
