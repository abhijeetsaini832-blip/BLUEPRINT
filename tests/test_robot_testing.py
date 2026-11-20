import pytest
from fastapi.testclient import TestClient

from app import app
from physics_engine import simulate_robot


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def robot(**overrides):
    config = {
        "total_mass_kg": 2,
        "motor_torque_nm": 0.5,
        "motor_count": 2,
        "wheel_radius_m": 0.05,
        "wheel_count": 4,
        "gravity_m_s2": 9.81,
    }
    config.update(overrides)
    return config


def test_physics_engine_uses_supplied_gravity():
    result = simulate_robot(
        total_mass_kg=2,
        motor_torque_nm=0.5,
        motor_count=2,
        wheel_radius_m=0.05,
        slope_degrees=30,
        gravity_m_s2=1.62,
    )
    assert result["gravity_resistance_n"] == pytest.approx(1.62, abs=1e-3)
    assert result["gravity_m_s2"] == pytest.approx(1.62)


def test_terrain_coefficients_change_traction_and_report_obstacle_limit(client):
    base = {
        "robot": robot(),
        "terrain_type": "flat_concrete",
        "slope_degrees": 0,
        "obstacle_type": "step",
        "obstacle_height_m": 0.1,
        "obstacle_width_m": 0.2,
        "duration_s": 1,
    }
    low_grip = client.post(
        "/api/tests/terrain",
        json={**base, "friction_coefficient": 0.1},
    ).json()
    high_grip = client.post(
        "/api/tests/terrain",
        json={**base, "friction_coefficient": 0.8},
    ).json()
    assert low_grip["results"]["available_traction_n"] < high_grip["results"]["available_traction_n"]
    assert low_grip["results"]["obstacle_traversal_status"] == "NOT_EVALUATED"
    assert high_grip["status"] == "WARNING"


def test_payload_sweep_reports_current_conditions_and_unknown_stability(client):
    response = client.post(
        "/api/tests/payload",
        json={
            "robot": robot(),
            "payload_mass_kg": 1,
            "payload_position_x_m": 0.1,
            "payload_distribution": "front",
            "terrain_type": "sand",
            "slope_degrees": 5,
            "sweep_step_kg": 0.5,
        },
    )
    assert response.status_code == 200
    result = response.json()
    assert result["results"]["total_loaded_mass_kg"] == 3
    assert result["results"]["recommended_payload_kg"] == pytest.approx(
        result["results"]["maximum_modeled_payload_kg"] * 0.8
    )
    assert result["results"]["stability_status"] == "NOT_EVALUATED"
    assert result["results"]["payload_position_m"]["x"] == 0.1
    assert result["results"]["payload_distribution"] == "front"


def test_payload_motor_overload_uses_required_steady_torque(client):
    result = client.post(
        "/api/tests/payload",
        json={
            "robot": robot(motor_torque_nm=0.01, motor_count=1),
            "payload_mass_kg": 0,
            "sweep_step_kg": 1,
        },
    ).json()
    assert result["status"] == "FAIL"
    assert result["results"]["motor_overload"] is True
    assert result["results"]["motor_load_ratio"] > 1


def test_structural_bending_reference_case_and_missing_geometry(client):
    complete = client.post(
        "/api/tests/structural",
        json={
            "test_type": "bending",
            "load_force_n": 100,
            "span_m": 1,
            "width_m": 0.1,
            "thickness_m": 0.02,
            "youngs_modulus_pa": 200e9,
            "allowable_strength_pa": 250e6,
            "support_condition": "simply_supported_center_load",
        },
    ).json()
    assert complete["status"] == "PASS"
    assert complete["results"]["stress_pa"] == pytest.approx(15e6)
    assert complete["results"]["deformation_mm"] == pytest.approx(0.15625)

    missing = client.post(
        "/api/tests/structural",
        json={"test_type": "compression", "load_force_n": 100},
    ).json()
    assert missing["status"] == "NOT_EVALUATED"


def test_structural_uses_supplied_yield_strength_if_allowable_is_omitted(client):
    result = client.post(
        "/api/tests/structural",
        json={
            "test_type": "tension",
            "load_force_n": 100,
            "cross_section_area_m2": 0.01,
            "youngs_modulus_pa": 200e9,
            "yield_strength_pa": 250e6,
        },
    ).json()
    assert result["status"] == "PASS"
    assert result["results"]["allowable_strength_source"] == "yield_strength_pa"


def test_wheel_pressure_is_not_reported_as_internal_chassis_stress(client):
    result = client.post(
        "/api/tests/structural",
        json={
            "test_type": "wheel_pressure",
            "normal_force_n": 100,
            "contact_area_m2": 0.01,
        },
    ).json()
    assert result["results"]["contact_pressure_pa"] == pytest.approx(10000)
    assert result["results"]["internal_stress_status"] == "NOT_EVALUATED"
    assert result["status"] == "NOT_EVALUATED"


def test_wheel_pressure_exceeding_user_limit_fails(client):
    result = client.post(
        "/api/tests/structural",
        json={
            "test_type": "wheel_pressure",
            "normal_force_n": 100,
            "contact_area_m2": 0.01,
            "allowable_contact_pressure_pa": 5000,
        },
    ).json()
    assert result["status"] == "FAIL"


def test_environment_and_endurance_disclose_missing_model_inputs(client):
    environment = client.post(
        "/api/tests/environment",
        json={"scenario": "vacuum"},
    ).json()
    assert environment["status"] == "NOT_EVALUATED"

    endurance = client.post(
        "/api/tests/endurance",
        json={
            "duration_s": 3600,
            "voltage_v": 12,
            "current_a": 2,
            "battery_capacity_ah": 1,
            "speed_ms": 0.5,
        },
    ).json()
    assert endurance["status"] == "FAIL"
    assert endurance["results"]["energy_consumed_wh"] == 24
    assert endurance["results"]["remaining_energy_wh"] == -12


def test_testing_api_rejects_boolean_and_out_of_range_values(client):
    response = client.post(
        "/api/tests/terrain",
        json={
            "robot": robot(),
            "terrain_type": "sand",
            "slope_degrees": True,
        },
    )
    assert response.status_code == 422


def test_structural_api_returns_controlled_error_for_overflowing_inputs(client):
    response = client.post(
        "/api/tests/structural",
        json={
            "test_type": "tension",
            "load_force_n": 1e308,
            "cross_section_area_m2": 1e-308,
            "youngs_modulus_pa": 1e3,
            "allowable_strength_pa": 1e6,
        },
    )
    assert response.status_code == 422
    assert "detail" in response.json()
