"""API smoke tests via FastAPI's in-process test client."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def base_payload(**overrides):
    payload = {
        "peaks": [
            {"frequency": 200.4, "tolerance": 1.5},
            {"frequency": 299.6, "tolerance": 1.5},
            {"frequency": 401.1, "tolerance": 1.5},
            {"frequency": 555.2, "tolerance": 1.5},
            {"frequency": 600.3, "tolerance": 1.5},
            {"frequency": 799.7, "tolerance": 1.5},
            {"frequency": 900.5, "tolerance": 1.5},
        ],
        "f0_min": 80,
        "f0_max": 130,
        "max_harmonic": 12,
        "max_rejected": 1,
    }
    payload.update(overrides)
    return payload


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "联合基频分析" in r.text


def test_solvable_case():
    r = client.post("/api/analyze", json=base_payload())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["feasible"] is True
    assert data["f0"] is not None
    assert 99.0 <= data["f0"] <= 101.0
    assert data["adopted_count"] == 6
    assert data["rejected_count"] == 1
    rejected = [p["index"] for p in data["peaks"] if not p["adopted"]]
    assert rejected == [3]
    hs = [p["harmonic"] for p in data["peaks"] if p["adopted"]]
    assert hs == sorted(set(hs))
    # Every adopted peak lies inside its tolerance of n*f0.
    for p in data["peaks"]:
        if p["adopted"]:
            assert abs(p["frequency"] - p["harmonic"] * data["f0"]) <= p["tolerance"] + 1e-9
            assert p["rel_error"] is not None and p["rel_error"] >= 0


def test_unsolvable_case_preserves_peaks_and_reports_evidence():
    payload = {
        "peaks": [
            {"frequency": 200.0, "tolerance": 2.0},
            {"frequency": 256.0, "tolerance": 2.0},
            {"frequency": 300.0, "tolerance": 2.0},
            {"frequency": 384.0, "tolerance": 2.0},
            {"frequency": 512.0, "tolerance": 2.0},
            {"frequency": 600.0, "tolerance": 2.0},
        ],
        "f0_min": 90,
        "f0_max": 130,
        "max_harmonic": 12,
        "max_rejected": 1,
    }
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["feasible"] is False
    assert data["f0"] is None
    assert data["first_unexplainable_index"] == 3
    # Original peak values are preserved, none marked adopted.
    freqs = [p["frequency"] for p in data["peaks"]]
    assert freqs == [200.0, 256.0, 300.0, 384.0, 512.0, 600.0]
    assert all(not p["adopted"] and p["harmonic"] is None for p in data["peaks"])
    assert data["reason"] == "no_common_fundamental"


def test_validation_rejects_non_sorted_peaks():
    p = base_payload()
    p["peaks"][3], p["peaks"][4] = p["peaks"][4], p["peaks"][3]
    r = client.post("/api/analyze", json=p)
    assert r.status_code == 422


def test_validation_rejects_wrong_count():
    p = base_payload()
    p["peaks"] = p["peaks"][:4]
    r = client.post("/api/analyze", json=p)
    assert r.status_code == 422


def test_validation_rejects_rejecting_all_peaks():
    # Rejecting every peak leaves nothing to define a fundamental.
    payload = {
        "peaks": [
            {"frequency": 100.0 + 100 * i, "tolerance": 0.5} for i in range(6)
        ],
        "f0_min": 1,
        "f0_max": 1000,
        "max_harmonic": 12,
        "max_rejected": 6,
    }
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == 422


def test_allow_rejecting_all_but_one():
    payload = {
        "peaks": [
            {"frequency": 100.0 + 100 * i, "tolerance": 0.5} for i in range(6)
        ],
        "f0_min": 1,
        "f0_max": 1000,
        "max_harmonic": 12,
        "max_rejected": 5,
    }
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == 200, r.text
