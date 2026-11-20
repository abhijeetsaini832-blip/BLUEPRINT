const API = location.protocol === "file:" ? "http://127.0.0.1:8000" : location.origin;
const $ = id => document.getElementById(id);
const HISTORY_KEY = "blueprint_testing_history";
let robots = [];
let currentMotion = [];
let animationFrame = null;
let animationStartedAt = 0;
let pausedAt = 0;
let currentTrajectoryTime = 0;

function readJson(key, fallback) {
  const raw = localStorage.getItem(key);
  if (raw === null || raw === "") return fallback;
  try {
    const parsed = JSON.parse(raw);
    return parsed ?? fallback;
  } catch (error) {
    console.error(`Could not read ${key}:`, error);
    return fallback;
  }
}

function numericValue(id) {
  const raw = $(id).value.trim();
  return raw === "" ? null : Number(raw);
}

function setStatus(type, status) {
  const chip = $(`${type}-status`);
  chip.textContent = status.replaceAll("_", " ");
  chip.dataset.status = status;
}

function renderMetrics(type, result) {
  const container = $(`${type}-metrics`);
  container.replaceChildren();
  const values = result?.results || {};
  const summaryKeys = {
    terrain: ["terrain_type", "maximum_climbable_slope_degrees", "available_traction_n", "motor_load_ratio", "net_force_n", "acceleration_ms2", "obstacle_traversal_status", "wheel_slip_status"],
    payload: ["total_loaded_mass_kg", "wheel_load_estimate_n", "required_motor_torque_per_motor_nm", "motor_load_ratio", "maximum_modeled_payload_kg", "maximum_modeled_payload_search_capped", "maximum_flat_ground_model_payload_kg", "maximum_flat_ground_search_capped", "recommended_payload_kg", "stability_status", "structural_limit_status"],
    structural: ["applied_force_n", "stress_mpa", "strain", "deformation_mm", "safety_factor", "contact_pressure_mpa"],
    environment: ["scenario", "component_survival"],
    endurance: ["energy_consumed_wh", "remaining_energy_wh", "estimated_runtime_h", "travel_distance_m"]
  };
  for (const key of summaryKeys[type] || []) {
    const value = values[key];
    if (value === undefined || value === null) continue;
    const card = document.createElement("div");
    card.className = "metric";
    const name = document.createElement("span");
    name.textContent = key.replaceAll("_", " ");
    const output = document.createElement("b");
    output.textContent = typeof value === "number"
      ? Number(value.toPrecision(4)).toString()
      : String(value);
    card.append(name, output);
    container.append(card);
  }
  if (result.warnings?.length) {
    const warning = document.createElement("p");
    warning.className = "hint";
    warning.textContent = result.warnings.join(" ");
    container.append(warning);
  }
}

function selectedRobot() {
  const robot = robots.find(item => item.id === $("robot-select").value);
  return robot?.configuration || null;
}

function renderRobotOptions() {
  const select = $("robot-select");
  select.replaceChildren();
  robots.forEach(robot => select.add(new Option(robot.name, robot.id)));
  if (!robots.length) {
    select.add(new Option("No saved builder assembly", ""));
    $("robot-summary").textContent =
      "No saved builder assembly found. Open Build and use the Testing Lab link to save the current robot.";
    return;
  }
  $("robot-summary").textContent = "";
  updateRobotSummary();
}

function updateRobotSummary() {
  const robot = robots.find(item => item.id === $("robot-select").value);
  if (!robot) return;
  const config = robot.configuration;
  $("robot-summary").textContent =
    `${config.total_mass_kg.toFixed(2)} kg · ${config.motor_count} powered motor(s) · ` +
    `${config.wheel_count} wheel(s) · ${config.wheel_radius_m} m wheel radius · ` +
    `${robot.component_ids.join(", ")}` +
    (config.spec_notes?.length ? ` · Assumptions: ${config.spec_notes.join(" ")}` : "");
  $("endurance-voltage").value = config.battery_voltage_v || 12;
}

async function postTest(type, payload) {
  const response = await fetch(`${API}/api/tests/${type}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`The server returned an invalid response (HTTP ${response.status}).`);
  }
  if (!response.ok) {
    const detail = typeof data.detail === "string"
      ? data.detail
      : Array.isArray(data.detail)
        ? data.detail.map(item => item.msg || JSON.stringify(item)).join("; ")
        : JSON.stringify(data.detail || data);
    throw new Error(detail);
  }
  return data;
}

function addToHistory(result) {
  const savedResult = {
    ...result,
    results: { ...result.results }
  };
  if (savedResult.results.motion?.trajectory) {
    const sampleCount = savedResult.results.motion.trajectory.length;
    savedResult.results.motion = { ...savedResult.results.motion };
    delete savedResult.results.motion.trajectory;
    savedResult.results.motion.trajectory_sample_count = sampleCount;
  }
  const history = readJson(HISTORY_KEY, []);
  const next = [savedResult, ...(Array.isArray(history) ? history : [])].slice(0, 30);
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
  } catch (error) {
    console.error("Could not persist the Testing Lab report:", error);
    $("comparison").textContent =
      `Test completed, but browser history could not be saved: ${error.message}`;
  }
  renderHistory(next);
  $("report-json").textContent = JSON.stringify(savedResult, null, 2);
}

function renderHistory(history = readJson(HISTORY_KEY, [])) {
  const host = $("history-list");
  host.replaceChildren();
  if (!Array.isArray(history) || !history.length) {
    host.textContent = "No tests saved in this browser yet.";
    return;
  }
  for (const test of history.slice(0, 10)) {
    const item = document.createElement("div");
    item.className = "history-item";
    const name = document.createElement("b");
    name.textContent = `${test.test_type} · ${test.status}`;
    const id = document.createElement("span");
    id.textContent = ` · ${test.test_id}`;
    item.append(name, id);
    item.title = new Date(test.created_at || Date.now()).toLocaleString();
    item.addEventListener("click", () => {
      $("report-json").textContent = JSON.stringify(test, null, 2);
    });
    host.append(item);
  }
}

function showTestError(type, error) {
  setStatus(type, "ERROR");
  const host = $(`${type}-metrics`);
  host.replaceChildren();
  const message = document.createElement("p");
  message.setAttribute("role", "alert");
  message.textContent = error.message;
  host.append(message);
}

async function runAndDisplay(type, payload) {
  if (!selectedRobot() && ["terrain", "payload"].includes(type)) {
    throw new Error("Select an assembly saved from the BLUEPRINT Builder first.");
  }
  const result = await postTest(type, payload);
  setStatus(type, result.status);
  renderMetrics(type, result);
  addToHistory(result);
  return result;
}

function optionalNumber(id) {
  const value = numericValue(id);
  return value === null || Number.isFinite(value) ? value : null;
}

function renderTerrain() {
  const canvas = $("terrain-canvas");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0a1220";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#354861";
  context.lineWidth = 3;
  context.beginPath();
  context.moveTo(20, height - 34);
  context.lineTo(width - 20, height - 34);
  context.stroke();
  context.fillStyle = "#45d6e8";
  context.font = "12px Segoe UI";
  context.fillText("Calculated 1D trajectory", 20, 22);

  const t = currentTrajectoryTime;
  const sample = currentMotion.findLast
    ? currentMotion.findLast(point => point.time_s <= t)
    : currentMotion.filter(point => point.time_s <= t).pop();
  const totalDistance = currentMotion.length
    ? Math.max(currentMotion[currentMotion.length - 1].position_m, 0.01)
    : 0;
  const progress = sample ? Math.min(1, sample.position_m / totalDistance) : 0;
  const x = 35 + progress * (width - 100);
  const y = height - 68;
  context.fillStyle = "#4c9aff";
  context.fillRect(x - 25, y - 18, 50, 20);
  context.fillStyle = "#45d6e8";
  context.fillRect(x - 8, y - 30, 20, 12);
  context.fillStyle = "#202a3b";
  for (const wheelX of [x - 16, x + 16]) {
    context.beginPath();
    context.arc(wheelX, y + 4, 8, 0, Math.PI * 2);
    context.fill();
    context.strokeStyle = "#91a2bb";
    context.stroke();
  }
  context.fillStyle = "#e8f0fc";
  context.fillText(
    sample ? `t ${sample.time_s.toFixed(1)} s · v ${sample.velocity_ms.toFixed(2)} m/s` : "Ready",
    20,
    height - 8
  );
}

function animate(timestamp) {
  if (!currentMotion.length) return;
  if (!animationStartedAt) animationStartedAt = timestamp - pausedAt;
  currentTrajectoryTime = (timestamp - animationStartedAt) / 1000;
  const last = currentMotion[currentMotion.length - 1];
  if (currentTrajectoryTime >= last.time_s) {
    currentTrajectoryTime = last.time_s;
    $("playback-status").textContent = `Playback complete · ${last.position_m.toFixed(2)} m traveled.`;
    $("pause-terrain").disabled = true;
    $("resume-terrain").disabled = true;
    renderTerrain();
    animationFrame = null;
    return;
  }
  renderTerrain();
  animationFrame = requestAnimationFrame(animate);
}

function resetPlayback() {
  if (animationFrame !== null) cancelAnimationFrame(animationFrame);
  animationFrame = null;
  currentMotion = [];
  animationStartedAt = 0;
  pausedAt = 0;
  currentTrajectoryTime = 0;
  $("pause-terrain").disabled = true;
  $("resume-terrain").disabled = true;
  $("playback-status").textContent =
    "1D playback uses the calculated trajectory. Obstacle contact and suspension are not modeled.";
  renderTerrain();
}

function terrainPayload() {
  const robot = selectedRobot();
  if (!robot) throw new Error("Select an assembly saved from the BLUEPRINT Builder first.");
  robot.gravity_m_s2 = Number($("gravity-body").value);
  const friction = optionalNumber("friction");
  const rolling = optionalNumber("rolling-resistance");
  return {
    robot,
    campaign_id: $("campaign-select").value || null,
    terrain_type: $("terrain-type").value,
    slope_degrees: numericValue("slope-degrees"),
    friction_coefficient: friction,
    rolling_resistance_coefficient: rolling,
    obstacle_type: $("obstacle-type").value,
    obstacle_height_m: numericValue("obstacle-height"),
    obstacle_width_m: numericValue("obstacle-width"),
    duration_s: numericValue("terrain-duration")
  };
}

async function runTerrain() {
  resetPlayback();
  $("start-terrain").disabled = true;
  try {
    const payload = terrainPayload();
    const result = await runAndDisplay("terrain", payload);
    currentMotion = result.results.motion.trajectory || [];
    $("playback-status").textContent =
      `Physics trajectory ready · ${result.results.motion.distance_traveled_m.toFixed(2)} m planned.`;
    $("pause-terrain").disabled = false;
    animationStartedAt = 0;
    animationFrame = requestAnimationFrame(animate);
  } catch (error) {
    showTestError("terrain", error);
  } finally {
    $("start-terrain").disabled = false;
  }
}

function setCampaign(campaign) {
  if (!campaign) return;
  $("terrain-type").value = campaign.terrain_type;
  $("gravity-body").value = String(campaign.gravity_m_s2);
}

async function loadCampaigns() {
  try {
    const response = await fetch(`${API}/api/tests/campaigns`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const body = await response.json();
    for (const campaign of body.campaigns || []) {
      $("campaign-select").add(new Option(campaign.name, campaign.id));
      $("campaign-select").lastChild.dataset.terrain = campaign.terrain_type;
      $("campaign-select").lastChild.dataset.gravity = campaign.gravity_m_s2;
    }
    $("api-status").textContent = "API connected";
  } catch (error) {
    $("api-status").textContent = "API unavailable";
    console.error("Could not load Testing Lab campaigns:", error);
  }
}

function exportCsv() {
  const history = readJson(HISTORY_KEY, []);
  const rows = [["test_id", "test_type", "status", "terrain", "total_mass_kg", "payload_kg", "gravity_m_s2"]];
  for (const item of Array.isArray(history) ? history : []) {
    const results = item.results || {};
    const config = item.inputs?.robot || {};
    rows.push([
      item.test_id, item.test_type, item.status,
      results.terrain_type || item.inputs?.terrain_type || "",
      results.total_loaded_mass_kg ?? config.total_mass_kg ?? "",
      results.payload_mass_kg ?? item.inputs?.payload_mass_kg ?? "",
      config.gravity_m_s2 ?? ""
    ]);
  }
  const csv = rows.map(row => row.map(value =>
    `"${String(value).replaceAll('"', '""')}"`
  ).join(",")).join("\r\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "blueprint-test-history.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function compareLatest() {
  const history = readJson(HISTORY_KEY, []);
  if (!Array.isArray(history) || history.length < 2) {
    $("comparison").textContent = "Run at least two tests to compare saved results.";
    return;
  }
  const latest = history[0];
  const previous = history.find((item, index) => index > 0 && item.test_type === latest.test_type);
  if (!previous) {
    $("comparison").textContent = `No earlier ${latest.test_type} report is available to compare.`;
    return;
  }
  const deltas = [];
  for (const [key, latestValue] of Object.entries(latest.results || {})) {
    const previousValue = previous.results?.[key];
    if (typeof latestValue === "number" && typeof previousValue === "number") {
      deltas.push(`${key.replaceAll("_", " ")} ${latestValue >= previousValue ? "+" : ""}${(latestValue - previousValue).toFixed(3)}`);
    }
  }
  $("comparison").textContent =
    `${latest.test_type}: latest ${latest.status}, previous ${previous.status}. ` +
    (deltas.length ? `Metric deltas: ${deltas.slice(0, 4).join("; ")}.` : "No shared numeric metrics to compare.");
}

function bindActions() {
  $("robot-select").addEventListener("change", updateRobotSummary);
  $("campaign-select").addEventListener("change", () => {
    const option = $("campaign-select").selectedOptions[0];
    if (!option?.value) return;
    setCampaign({ terrain_type: option.dataset.terrain, gravity_m_s2: option.dataset.gravity });
  });
  $("payload-preset").addEventListener("change", () => {
    const value = $("payload-preset").value;
    if (value !== "custom") $("payload-mass").value = value;
  });
  $("start-terrain").addEventListener("click", runTerrain);
  $("pause-terrain").addEventListener("click", () => {
    if (animationFrame !== null) cancelAnimationFrame(animationFrame);
    animationFrame = null;
    pausedAt = currentTrajectoryTime * 1000;
    $("pause-terrain").disabled = true;
    $("resume-terrain").disabled = false;
    $("playback-status").textContent = `Paused at ${currentTrajectoryTime.toFixed(1)} s.`;
  });
  $("resume-terrain").addEventListener("click", () => {
    animationStartedAt = 0;
    $("pause-terrain").disabled = false;
    $("resume-terrain").disabled = true;
    animationFrame = requestAnimationFrame(animate);
  });
  $("reset-terrain").addEventListener("click", resetPlayback);
  $("run-payload").addEventListener("click", async () => {
    try {
      const base = selectedRobot();
      if (!base) throw new Error("Select an assembly saved from the BLUEPRINT Builder first.");
      const robot = { ...base, gravity_m_s2: Number($("gravity-body").value) };
      const result = await runAndDisplay("payload", {
        robot,
        payload_mass_kg: numericValue("payload-mass"),
        payload_position_x_m: optionalNumber("payload-x"),
        payload_position_y_m: optionalNumber("payload-y"),
        payload_position_z_m: optionalNumber("payload-z"),
        payload_distribution: $("payload-distribution").value,
        terrain_type: $("terrain-type").value,
        slope_degrees: numericValue("slope-degrees"),
        friction_coefficient: optionalNumber("friction"),
        rolling_resistance_coefficient: optionalNumber("rolling-resistance"),
        safety_margin_fraction: numericValue("safety-margin") / 100,
        sweep_step_kg: numericValue("payload-step")
      });
    } catch (error) {
      showTestError("payload", error);
    }
  });
  $("run-structural").addEventListener("click", async () => {
    try {
      const ids = [
        "material-density", "yield-strength", "ultimate-strength", "structural-load",
        "structural-area", "member-length", "youngs-modulus", "allowable-strength",
        "beam-span", "beam-width", "beam-thickness", "normal-force", "contact-area",
        "allowable-pressure"
      ];
      const payload = {
        test_type: $("structural-type").value,
        robot: selectedRobot()
      };
      const materialName = $("material-name").value.trim();
      if (materialName) payload.material_name = materialName;
      const keys = [
        "density_kg_m3", "yield_strength_pa", "ultimate_strength_pa", "load_force_n",
        "cross_section_area_m2", "member_length_m", "youngs_modulus_pa",
        "allowable_strength_pa", "span_m", "width_m", "thickness_m",
        "normal_force_n", "contact_area_m2", "allowable_contact_pressure_pa"
      ];
      ids.forEach((id, index) => {
        const value = optionalNumber(id);
        if (value !== null) payload[keys[index]] = value;
      });
      payload.support_condition = $("support-condition").value;
      await runAndDisplay("structural", payload);
    } catch (error) {
      showTestError("structural", error);
    }
  });
  $("run-environment").addEventListener("click", async () => {
    try {
      const payload = {
        scenario: $("environment-scenario").value,
        duration_s: numericValue("environment-duration"),
        robot: selectedRobot()
      };
      [
        ["temperature_c", "environment-temperature"],
        ["vibration_acceleration_ms2", "vibration-acceleration"],
        ["impact_energy_j", "impact-energy"],
        ["pressure_pa", "environment-pressure"]
      ].forEach(([key, id]) => {
        const value = optionalNumber(id);
        if (value !== null) payload[key] = value;
      });
      await runAndDisplay("environment", payload);
    } catch (error) {
      showTestError("environment", error);
    }
  });
  $("run-endurance").addEventListener("click", async () => {
    try {
      const payload = {
        duration_s: numericValue("endurance-duration"),
        voltage_v: numericValue("endurance-voltage"),
        speed_ms: numericValue("endurance-speed"),
        robot: selectedRobot()
      };
      const current = optionalNumber("endurance-current");
      const capacity = optionalNumber("battery-capacity");
      if (current !== null) payload.current_a = current;
      if (capacity !== null) payload.battery_capacity_ah = capacity;
      await runAndDisplay("endurance", payload);
    } catch (error) {
      showTestError("endurance", error);
    }
  });
  $("export-csv").addEventListener("click", exportCsv);
  $("compare-tests").addEventListener("click", compareLatest);
}

robots = readJson("blueprint_testing_robots", []);
if (!Array.isArray(robots)) robots = [];
renderRobotOptions();
renderHistory();
renderTerrain();
bindActions();
loadCampaigns();
