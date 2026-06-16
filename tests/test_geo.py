import math

from transect_stitch.geo import bearing_deg, haversine_m


def test_haversine_zero():
    assert haversine_m(10.0, 20.0, 10.0, 20.0) == 0.0


def test_haversine_one_degree_lat():
    # One degree of latitude is ~111 km anywhere on the globe.
    d = haversine_m(0.0, 0.0, 1.0, 0.0)
    assert math.isclose(d, 111_195, rel_tol=0.01)


def test_haversine_symmetric():
    a = haversine_m(51.5, -0.12, 48.85, 2.35)
    b = haversine_m(48.85, 2.35, 51.5, -0.12)
    assert math.isclose(a, b, rel_tol=1e-9)


def test_bearing_due_north():
    assert math.isclose(bearing_deg(0.0, 0.0, 1.0, 0.0), 0.0, abs_tol=1e-6)


def test_bearing_due_east():
    assert math.isclose(bearing_deg(0.0, 0.0, 0.0, 1.0), 90.0, abs_tol=1e-6)
