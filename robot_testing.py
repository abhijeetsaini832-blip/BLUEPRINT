"""Engineering test calculations built on BLUEPRINT's existing physics model."""

import math
import uuid
from datetime import datetime, timezone

from physics_engine import simulate_robot, simulate_robot_motion


TERRAIN_PRESETS = {
    "flat_concrete": {"friction_coefficient": 0.8, "rolling_resistance_coefficient": 0.015},
    "rough_concrete": {"friction_coefficient": 0.65, "rolling_resistance_coefficient": 0.03},
    "sand": {"friction_coefficient": 0.45, "rolling_resistance_coefficient": 0.12},
    "loose_soil": {"friction_coefficient": 0.5, "rolling_resistance_coefficient": 0.1},
    "gravel": {"friction_coefficient": 0.55, "rolling_resistance_coefficient": 0.08},
    "rocky_terrain": {"friction_coefficient": 0.6, "rolling_resistance_coefficient": 0.1},
    "mud": {"friction_coefficient": 0.35, "rolling_resistance_coefficient": 0.16},
    "lunar_regolith": {"friction_coefficient": 0.4, "rolling_resistance_coefficient": 0.12},
    "martian_terrain": {"friction_coefficient": 0.45, "rolling_resistance_coefficient": 0.1},
}

TERRAIN_APPROXIMATION_NOTE = (
    "Terrain coefficients are configurable first-order approximations, not measured soil properties."
)


def _report(test_type, inputs, status, results, assumptions, warnings=None):
    _ensure_finite_values({"inputs": inputs, "results": results})
    return {
        "test_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_type": test_type,
        "status": status,
        "inputs": inputs,
        "results": results,
        "assumptions": assumptions,
        "warnings": warnings or [],
    }


def _ensure_finite_values(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("An input or calculation produced a non-finite value; check the input magnitudes.")
    if isinstance(value, dict):
        for item in value.values():
            _ensure_finite_values(item)
    elif isinstance(value, list):
        for item in value:
            _ensure_finite_values(item)


def _physics_inputs(robot, mass_kg, slope_degrees, friction, rolling_resistance):
    values = robot.copy()
    values.pop("wheel_count", None)
    values.pop("spec_notes", None)
    values.pop("installed_component_ids", None)
    values["total_mass_kg"] = mass_kg
    values["slope_degrees"] = slope_degrees
    values["traction_coefficient"] = friction
    values["rolling_resistance_coefficient"] = rolling_resistance
    return values


def _can_move(robot, mass_kg, slope_degrees, friction, rolling_resistance):
    return simulate_robot(
        **_physics_inputs(
            robot, mass_kg, slope_degrees, friction, rolling_resistance
        )
    )["can_move_under_model"]


def run_terrain_test(request):
    profile = TERRAIN_PRESETS[request.terrain_type]
    friction = (
        request.friction_coefficient
        if request.friction_coefficient is not None
        else profile["friction_coefficient"]
    )
    rolling_resistance = (
        request.rolling_resistance_coefficient
        if request.rolling_resistance_coefficient is not None
        else profile["rolling_resistance_coefficient"]
    )
    robot = request.robot.model_dump()
    physics = _physics_inputs(
        robot,
        robot["total_mass_kg"],
        request.slope_degrees,
        friction,
        rolling_resistance,
    )
    simulation = simulate_robot(**physics)
    motion = simulate_robot_motion(
        **physics,
        duration_s=request.duration_s,
        time_step_s=max(0.1, request.duration_s / 1999.0),
    )

    max_climbable_slope = 0.0
    for step in range(1, 357):
        candidate = step * 0.25
        if not _can_move(robot, robot["total_mass_kg"], candidate, friction, rolling_resistance):
            break
        max_climbable_slope = candidate

    warnings = list(simulation["engineering_warnings"])
    warnings.extend(request.robot.spec_notes)
    warnings.append(TERRAIN_APPROXIMATION_NOTE)
    required_tractive_force = max(0.0, simulation["total_resistance_n"])
    required_motor_torque = (
        required_tractive_force
        * robot["wheel_radius_m"]
        / (
            robot["motor_count"]
            * robot["gear_ratio"]
            * robot["gearbox_efficiency"]
            * robot["drivetrain_efficiency"]
        )
    )
    available_motor_torque = simulation["estimated_motor_torque_nm"]
    motor_load_ratio = (
        required_motor_torque / available_motor_torque
        if available_motor_torque > 0
        else None
    )
    if motor_load_ratio is not None and motor_load_ratio > 1:
        warnings.append("Required steady force exceeds the modeled motor torque at the supplied speed.")
    return _report(
        "terrain",
        request.model_dump(),
        (
            "FAIL"
            if not simulation["can_move_under_model"]
            else "WARNING"
            if request.obstacle_type != "none"
            else "PASS"
        ),
        {
            "terrain_type": request.terrain_type,
            "friction_coefficient": friction,
            "rolling_resistance_coefficient": rolling_resistance,
            "maximum_climbable_slope_degrees": max_climbable_slope,
            "requested_slope_degrees": request.slope_degrees,
            "available_traction_n": simulation["traction_limit_n"],
            "driving_force_n": simulation["driving_force_n"],
            "net_force_n": simulation["net_force_n"],
            "acceleration_ms2": simulation["theoretical_acceleration_ms2"],
            "wheel_torque_nm": simulation["effective_wheel_torque_nm"],
            "motor_power_w": simulation["total_motor_power_w"],
            "required_motor_torque_per_motor_nm": required_motor_torque,
            "available_motor_torque_per_motor_nm": available_motor_torque,
            "motor_load_ratio": motor_load_ratio,
            "traction_limited": simulation["traction_limited"],
            "wheel_slip_status": "NOT_EVALUATED",
            "obstacle_traversal_status": "NOT_EVALUATED",
            "obstacle_reason": (
                "Obstacle dimensions are recorded, but wheel geometry, suspension, "
                "and contact dynamics are not modeled."
            ),
            "motion": motion,
        },
        [
            "Existing BLUEPRINT force, torque, slope, and Coulomb traction calculations are reused.",
            "Maximum slope is the highest passing 0.25-degree static force-balance sample.",
            "Dynamic load transfer, deformable terrain, slip dynamics, and obstacle contact are not modeled.",
            TERRAIN_APPROXIMATION_NOTE,
        ],
        warnings,
    )


def _payload_capacity(robot, slope, friction, rolling_resistance, step_size, max_payload):
    base_mass = robot["total_mass_kg"]
    tested = 0.0
    while tested < max_payload - 1e-9:
        candidate = min(tested + step_size, max_payload)
        if base_mass + candidate > 100.0:
            break
        if not _can_move(
            robot,
            base_mass + candidate,
            slope,
            friction,
            rolling_resistance,
        ):
            break
        tested = candidate
    return round(tested, 3), tested >= max_payload - 1e-9


def run_payload_test(request):
    profile = TERRAIN_PRESETS[request.terrain_type]
    friction = (
        request.friction_coefficient
        if request.friction_coefficient is not None
        else profile["friction_coefficient"]
    )
    rolling_resistance = (
        request.rolling_resistance_coefficient
        if request.rolling_resistance_coefficient is not None
        else profile["rolling_resistance_coefficient"]
    )
    robot = request.robot.model_dump()
    total_mass = robot["total_mass_kg"] + request.payload_mass_kg
    sim = simulate_robot(
        **_physics_inputs(
            robot, total_mass, request.slope_degrees, friction, rolling_resistance
        )
    )
    max_payload = min(50.0, 100.0 - robot["total_mass_kg"])
    resolution = request.sweep_step_kg
    current_capacity, current_capped = _payload_capacity(
        robot,
        request.slope_degrees,
        friction,
        rolling_resistance,
        resolution,
        max_payload,
    )
    theoretical_capacity, theoretical_capped = _payload_capacity(
        robot, 0.0, 2.0, 0.0, resolution, max_payload
    )
    wheel_load_n = total_mass * robot["gravity_m_s2"] / robot["wheel_count"]
    required_tractive_force = max(0.0, sim["total_resistance_n"])
    required_motor_torque = (
        required_tractive_force
        * robot["wheel_radius_m"]
        / (
            robot["motor_count"]
            * robot["gear_ratio"]
            * robot["gearbox_efficiency"]
            * robot["drivetrain_efficiency"]
        )
    )
    available_motor_torque = sim["estimated_motor_torque_nm"]
    motor_load_ratio = (
        required_motor_torque / available_motor_torque
        if available_motor_torque > 0
        else None
    )
    warnings = list(sim["engineering_warnings"]) + list(request.robot.spec_notes)
    if current_capped:
        warnings.append(
            "Current-condition payload sweep reached its 50 kg payload or 100 kg total-mass search limit."
        )
    if theoretical_capped:
        warnings.append(
            "Flat-ground model payload sweep reached its 50 kg payload or 100 kg total-mass search limit."
        )

    return _report(
        "payload",
        request.model_dump(),
        "WARNING" if sim["can_move_under_model"] else "FAIL",
        {
            "base_mass_kg": robot["total_mass_kg"],
            "payload_mass_kg": request.payload_mass_kg,
            "total_loaded_mass_kg": total_mass,
            "wheel_load_estimate_n": round(wheel_load_n, 3),
            "wheel_load_status": "ESTIMATED_EQUAL_STATIC_DISTRIBUTION",
            "payload_position_m": {
                "x": request.payload_position_x_m,
                "y": request.payload_position_y_m,
                "z": request.payload_position_z_m,
            },
            "payload_distribution": request.payload_distribution,
            "required_tractive_force_n": required_tractive_force,
            "required_motor_torque_per_motor_nm": required_motor_torque,
            "available_motor_torque_per_motor_nm": available_motor_torque,
            "motor_load_ratio": motor_load_ratio,
            "motor_overload": (
                motor_load_ratio is not None and motor_load_ratio > 1
            ),
            "available_tractive_force_n": sim["driving_force_n"],
            "acceleration_ms2": sim["theoretical_acceleration_ms2"],
            "maximum_modeled_payload_kg": current_capacity,
            "maximum_modeled_payload_resolution_kg": resolution,
            "maximum_modeled_payload_search_capped": current_capped,
            "maximum_flat_ground_model_payload_kg": theoretical_capacity,
            "maximum_flat_ground_search_capped": theoretical_capped,
            "recommended_payload_kg": round(
                current_capacity * (1.0 - request.safety_margin_fraction), 3
            ),
            "safety_margin_fraction": request.safety_margin_fraction,
            "stability_status": "NOT_EVALUATED",
            "structural_limit_status": "NOT_EVALUATED",
        },
        [
            "Capacity sweep uses existing motor-torque, traction, slope, and rolling-resistance limits.",
            "Sweep resolution is the configured payload step; capacities are not continuous guarantees.",
            "Wheel load assumes equal static distribution across all configured wheels.",
            "Payload position and distribution are recorded but do not alter wheel loads without chassis geometry and a center-of-gravity model.",
            "Center of gravity, tipping geometry, chassis strength, and dynamic load transfer are not modeled.",
            "Motor torque is the supplied builder estimate; continuous thermal rating is unavailable.",
        ],
        warnings,
    )


def run_structural_test(request):
    inputs = request.model_dump()
    allowable_strength = (
        request.allowable_strength_pa
        or request.yield_strength_pa
    )
    allowable_strength_source = (
        "allowable_strength_pa"
        if request.allowable_strength_pa is not None
        else "yield_strength_pa"
        if request.yield_strength_pa is not None
        else None
    )
    if request.test_type == "wheel_pressure":
        if request.normal_force_n is None or request.contact_area_m2 is None:
            return _report(
                "structural",
                inputs,
                "NOT_EVALUATED",
                {"contact_pressure_pa": None, "internal_stress_status": "NOT_EVALUATED"},
                ["Normal force and contact area are both required for wheel-ground pressure."],
            )
        pressure = request.normal_force_n / request.contact_area_m2
        status = (
            "PASS"
            if request.allowable_contact_pressure_pa is not None
            and pressure <= request.allowable_contact_pressure_pa
            else "FAIL"
            if request.allowable_contact_pressure_pa is not None
            else "NOT_EVALUATED"
        )
        return _report(
            "structural",
            inputs,
            status,
            {
                "applied_force_n": request.normal_force_n,
                "contact_pressure_pa": pressure,
                "contact_pressure_mpa": pressure / 1e6,
                "allowable_contact_pressure_pa": request.allowable_contact_pressure_pa,
                "internal_stress_status": "NOT_EVALUATED",
            },
            ["Wheel-ground contact pressure is force divided by supplied contact area."],
            [] if status == "PASS" else ["No contact-pressure limit was supplied; pressure is not a chassis stress result."]
            if status == "NOT_EVALUATED" else ["Contact pressure exceeds the user-supplied allowable limit."],
        )

    if request.test_type in ("compression", "tension"):
        required = (
            request.load_force_n,
            request.cross_section_area_m2,
            request.youngs_modulus_pa,
            allowable_strength,
        )
        if any(value is None for value in required):
            return _not_evaluated_structural(inputs, "Load, area, modulus, and allowable strength are required.")
        stress = request.load_force_n / request.cross_section_area_m2
        strain = stress / request.youngs_modulus_pa
        deformation = (
            strain * request.member_length_m
            if request.member_length_m is not None
            else None
        )
    else:
        required = (
            request.load_force_n,
            request.span_m,
            request.width_m,
            request.thickness_m,
            request.youngs_modulus_pa,
            allowable_strength,
        )
        if any(value is None for value in required) or request.support_condition != "simply_supported_center_load":
            return _not_evaluated_structural(
                inputs,
                "Bending requires a simply-supported center load, rectangular section, span, width, thickness, modulus, and allowable strength.",
            )
        inertia = request.width_m * request.thickness_m**3 / 12.0
        stress = 6.0 * request.load_force_n * request.span_m / (
            request.width_m * request.thickness_m**2
        )
        strain = stress / request.youngs_modulus_pa
        deformation = (
            request.load_force_n * request.span_m**3
            / (48.0 * request.youngs_modulus_pa * inertia)
        )

    safety_factor = allowable_strength / stress if stress else None
    return _report(
        "structural",
        inputs,
        "PASS" if safety_factor is not None and safety_factor >= 1.0 else "FAIL",
        {
            "applied_force_n": request.load_force_n,
            "stress_pa": stress,
            "stress_mpa": stress / 1e6,
            "strain": strain,
            "deformation_m": deformation,
            "deformation_mm": deformation * 1000.0 if deformation is not None else None,
            "allowable_strength_pa": allowable_strength,
            "allowable_strength_source": allowable_strength_source,
            "material": {
                "name": request.material_name,
                "density_kg_m3": request.density_kg_m3,
                "yield_strength_pa": request.yield_strength_pa,
                "ultimate_strength_pa": request.ultimate_strength_pa,
                "user_supplied_allowable_strength_pa": request.allowable_strength_pa,
            },
            "safety_factor": safety_factor,
            "failure_threshold_force_n": (
                allowable_strength / stress * request.load_force_n
            ),
            "model": (
                "axial_uniform_stress"
                if request.test_type in ("compression", "tension")
                else "simply_supported_rectangular_beam_center_load"
            ),
        },
        [
            "Uses user-supplied material properties and idealized geometry; no material database is assumed.",
            "Safety factor uses the supplied allowable strength, or supplied yield strength if allowable strength is omitted; ultimate strength alone is not treated as an allowable limit.",
            "Density is recorded but not included in the static load equations.",
            "Linear-elastic deformation is reported only when member length (axial) or beam dimensions (bending) are supplied.",
            "Stress concentrations, joints, fatigue, buckling, and finite-element effects are not modeled.",
        ],
    )


def _not_evaluated_structural(inputs, reason):
    return _report(
        "structural",
        inputs,
        "NOT_EVALUATED",
        {"stress_pa": None, "safety_factor": None},
        [reason, "No material or geometry values are invented."],
        [reason],
    )


def run_environment_test(request):
    return _report(
        "environment",
        request.model_dump(),
        "NOT_EVALUATED",
        {"scenario": request.scenario, "component_survival": "NOT_EVALUATED"},
        [
            "No component temperature, dust-ingress, vibration, impact, or pressure qualification limits are present in the catalog.",
            "Exposure inputs are recorded for a future qualified component model; they do not prove survival.",
        ],
        ["Environmental survivability cannot be determined from current BLUEPRINT component data."],
    )


def run_endurance_test(request):
    energy_wh = None
    remaining_wh = None
    runtime_h = None
    status = "NOT_EVALUATED"
    warnings = []
    if request.current_a is not None:
        energy_wh = request.voltage_v * request.current_a * request.duration_s / 3600.0
    if request.battery_capacity_ah is not None:
        available_wh = request.voltage_v * request.battery_capacity_ah
        if energy_wh is not None:
            remaining_wh = available_wh - energy_wh
            status = "PASS" if remaining_wh >= 0 else "FAIL"
        if request.current_a is not None and request.current_a > 0:
            runtime_h = request.battery_capacity_ah / request.current_a
    if energy_wh is None:
        warnings.append("Supply measured or explicitly assumed current to calculate electrical energy.")
    if remaining_wh is None:
        warnings.append("Battery capacity is missing; remaining charge and depletion status are not evaluated.")

    return _report(
        "endurance",
        request.model_dump(),
        status,
        {
            "energy_consumed_wh": energy_wh,
            "remaining_energy_wh": remaining_wh,
            "estimated_runtime_h": runtime_h,
            "travel_distance_m": request.speed_ms * request.duration_s,
            "battery_status": status,
        },
        [
            "Energy uses E = V × I × t with user-supplied voltage and current.",
            "Runtime assumes constant current and nominal battery capacity; voltage sag, efficiency, temperature, and battery aging are not modeled.",
            "No wear-life or component-temperature prediction is supported.",
        ],
        warnings,
    )


def test_campaigns():
    return [
        {"id": "earth_rover", "name": "Earth Rover", "terrain_type": "gravel", "gravity_m_s2": 9.81},
        {"id": "mars_rover", "name": "Mars Rover", "terrain_type": "martian_terrain", "gravity_m_s2": 3.71},
        {"id": "lunar_rover", "name": "Lunar Rover", "terrain_type": "lunar_regolith", "gravity_m_s2": 1.62},
        {"id": "heavy_payload", "name": "Heavy Payload", "terrain_type": "flat_concrete", "gravity_m_s2": 9.81},
        {"id": "extreme_environment", "name": "Extreme Environment", "terrain_type": "rocky_terrain", "gravity_m_s2": 9.81},
    ]
