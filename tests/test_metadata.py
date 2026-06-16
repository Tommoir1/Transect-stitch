from datetime import datetime, timezone
from pathlib import Path

from transect_stitch.metadata import (
    ORDER_AUTO,
    ORDER_FILENAME,
    ORDER_TIME,
    FrameInfo,
    discover_images,
    gps_gaps_m,
    order_frames,
)


def _frame(name, ts=None, lat=None, lon=None, src="none"):
    return FrameInfo(path=Path(name), timestamp=ts, lat=lat, lon=lon, timestamp_source=src)


def test_natural_filename_order():
    frames = [_frame("frame_10.jpg"), _frame("frame_2.jpg"), _frame("frame_1.jpg")]
    ordered = order_frames(frames, ORDER_FILENAME)
    assert [f.path.name for f in ordered] == ["frame_1.jpg", "frame_2.jpg", "frame_10.jpg"]


def test_timestamp_order():
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 1, 12, 0, 5, tzinfo=timezone.utc)
    frames = [_frame("b.jpg", ts=t1, src="gps"), _frame("a.jpg", ts=t0, src="gps")]
    ordered = order_frames(frames, ORDER_TIME)
    assert [f.path.name for f in ordered] == ["a.jpg", "b.jpg"]


def test_auto_falls_back_to_filename_without_timestamps():
    frames = [_frame("img_3.jpg"), _frame("img_1.jpg")]
    ordered = order_frames(frames, ORDER_AUTO)
    assert [f.path.name for f in ordered] == ["img_1.jpg", "img_3.jpg"]


def test_gps_gaps():
    frames = [
        _frame("a.jpg", lat=0.0, lon=0.0),
        _frame("b.jpg", lat=0.0, lon=0.0),
        _frame("c.jpg"),  # missing GPS
        _frame("d.jpg", lat=1.0, lon=0.0),
    ]
    gaps = gps_gaps_m(frames)
    assert gaps[0] is None
    assert gaps[1] == 0.0
    assert gaps[2] is None  # c has no GPS
    assert gaps[3] is None  # previous (c) has no GPS


def test_discover_images_filters_extensions(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("ignore me")
    found = {p.name for p in discover_images([str(tmp_path)])}
    assert found == {"a.jpg", "b.png"}


def test_empty_inputs():
    assert order_frames([], ORDER_AUTO) == []
