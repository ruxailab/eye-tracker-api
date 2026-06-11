from app.models.session import Session


def _make_session(**overrides):
    defaults = dict(
        id=1, title="Test Session", description="A test session",
        user_id=42, created_date="2025-01-01",
        website_url="https://example.com",
        screen_record_url="https://example.com/screen.webm",
        webcam_record_url="https://example.com/webcam.webm",
        heatmap_url="https://example.com/heatmap.png",
        calib_points=[[100, 200], [300, 400]],
        iris_points=[[0.3, 0.4], [0.5, 0.6]],
    )
    defaults.update(overrides)
    return Session(**defaults)


def test_attributes_are_stored():
    s = _make_session()
    assert s.id == 1
    assert s.title == "Test Session"
    assert s.user_id == 42
    assert s.website_url == "https://example.com"


def test_to_dict():
    s = _make_session(id=5, title="X", calib_points=[[1, 2]], iris_points=[[0.1, 0.2]])
    d = s.to_dict()

    assert d["id"] == 5
    assert d["title"] == "X"
    assert d["callib_points"] == [[1, 2]]
    assert d["iris_points"] == [[0.1, 0.2]]


def test_to_dict_has_all_expected_keys():
    d = _make_session().to_dict()
    assert set(d.keys()) == {
        "id", "title", "description", "user_id", "created_date",
        "website_url", "screen_record_url", "webcam_record_url",
        "heatmap_url", "callib_points", "iris_points",
    }


def test_none_values_are_preserved():
    s = Session(
        id=None, title=None, description=None, user_id=None,
        created_date=None, website_url=None, screen_record_url=None,
        webcam_record_url=None, heatmap_url=None, calib_points=None,
        iris_points=None,
    )
    d = s.to_dict()
    assert d["id"] is None
    assert d["callib_points"] is None
