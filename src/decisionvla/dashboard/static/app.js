"use strict";
const $ = (id) => document.getElementById(id);
const labels = ["ΔX", "ΔY", "ΔZ", "ΔRX", "ΔRY", "ΔRZ", "GRIP"];
const svgNS = "http://www.w3.org/2000/svg";
let session,
  rows = [],
  visible = [],
  frame,
  index = 0,
  timestep = 0,
  dimension = 0;
let initialized = false,
  playing = false,
  following = true,
  requestNumber = 0,
  episode = "";

function svgElement(type, attributes, text) {
  const element = document.createElementNS(svgNS, type);
  Object.entries(attributes).forEach(([key, value]) =>
    element.setAttribute(key, value),
  );
  if (text !== undefined) element.textContent = text;
  return element;
}
function add(svg, type, attributes, text) {
  svg.appendChild(svgElement(type, attributes, text));
}
function fixed(value, digits = 3) {
  return Number(value).toFixed(digits);
}
function signed(value) {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(3)}`;
}
function reportError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}
async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`Request failed (${response.status})`);
  return response.json();
}
function renderGrid() {
  const grid = $("action-grid");
  grid.replaceChildren(document.createElement("span"));
  labels.forEach((label) => {
    const cell = document.createElement("span");
    cell.className = "column-label";
    cell.textContent = label;
    grid.appendChild(cell);
  });
  for (let t = 0; t < 8; t++) {
    const name = document.createElement("span");
    name.className = "row-label";
    name.textContent = `T+${t + 1}`;
    grid.appendChild(name);
    for (let d = 0; d < 7; d++) {
      const confidence = frame.diagnostics.metrics.ordered_confidence[t][d];
      const button = document.createElement("button");
      button.type = "button";
      button.className =
        "action-cell" + (t === timestep && d === dimension ? " selected" : "");
      button.style.backgroundColor = `hsl(86 43% ${96 - confidence * 22}%)`;
      button.textContent = signed(frame.actions[t][d]);
      button.setAttribute(
        "aria-label",
        `T+${t + 1} ${labels[d]}, action ${signed(frame.actions[t][d])}, confidence ${fixed(confidence)}`,
      );
      button.setAttribute(
        "aria-pressed",
        String(t === timestep && d === dimension),
      );
      button.title = `Confidence ${fixed(confidence)} · selected probability ${fixed(frame.diagnostics.metrics.selected_probability[t][d])}`;
      button.onclick = () => {
        timestep = t;
        dimension = d;
        renderGrid();
        renderDistribution();
      };
      grid.appendChild(button);
    }
  }
}
function renderDistribution() {
  const metrics = frame.diagnostics.metrics,
    probabilities = frame.diagnostics.probabilities[timestep][dimension];
  const selected = frame.diagnostics.action_bins[timestep][dimension],
    gripper = dimension === 6;
  $("selection").textContent = `T+${timestep + 1} / ${labels[dimension]}`;
  $("selected-action").textContent = signed(frame.actions[timestep][dimension]);
  $("selected-probability").textContent =
    `${fixed(metrics.selected_probability[timestep][dimension] * 100, 1)}%`;
  $("component-confidence").textContent = fixed(
    metrics.ordered_confidence[timestep][dimension],
  );
  $("entropy").textContent = fixed(
    metrics.normalized_entropy[timestep][dimension],
  );
  $("distribution-note").textContent = gripper
    ? "Two allowed gripper values · probability mass at −1 and +1"
    : "256 discrete action bins · final denoising readout";
  const svg = $("distribution");
  svg.replaceChildren();
  const left = 47,
    right = 744,
    top = 12,
    bottom = 138,
    width = right - left;
  const peak = Math.max(...probabilities),
    ymax = Math.min(1, Math.max(peak * 1.15, 0.01));
  for (let i = 0; i <= 3; i++) {
    const y = bottom - ((bottom - top) * i) / 3;
    add(svg, "line", { x1: left, y1: y, x2: right, y2: y, class: "svg-grid" });
    add(
      svg,
      "text",
      { x: left - 9, y: y + 3, "text-anchor": "end" },
      `${fixed(((ymax * i) / 3) * 100, 1)}%`,
    );
  }
  const bins = gripper ? [0, 255] : Array.from({ length: 256 }, (_, i) => i);
  bins.forEach((bin) => {
    const x = gripper
      ? left + width * (bin === 0 ? 0.24 : 0.74)
      : left + (bin * width) / 256;
    const height = (probabilities[bin] / ymax) * (bottom - top);
    const barWidth = gripper ? 32 : Math.max(1, width / 256 - 0.25);
    add(svg, "rect", {
      x: x,
      y: bottom - height,
      width: barWidth,
      height: height,
      fill: bin === selected ? "#465f2b" : "#aeca85",
      rx: gripper ? 2 : 0,
    });
  });
  const xSelected = gripper
    ? left + width * (selected === 0 ? 0.24 : 0.74) + 16
    : left + ((selected + 0.5) * width) / 256;
  add(svg, "line", {
    x1: xSelected,
    y1: top,
    x2: xSelected,
    y2: bottom,
    stroke: "#6c8550",
    "stroke-dasharray": "3 4",
    opacity: 0.6,
  });
  add(svg, "line", {
    x1: left,
    y1: bottom,
    x2: right,
    y2: bottom,
    class: "svg-axis",
  });
  (gripper
    ? [
        [0.24 + 16 / width, "−1"],
        [0.74 + 16 / width, "+1"],
      ]
    : [
        [0, "−1"],
        [0.25, "−0.5"],
        [0.5, "0"],
        [0.75, "+0.5"],
        [1, "+1"],
      ]
  ).forEach(([x, label]) => {
    add(
      svg,
      "text",
      { x: left + width * x, y: 159, "text-anchor": "middle" },
      label,
    );
  });
}
function renderHistory() {
  const svg = $("history");
  svg.replaceChildren();
  const left = 30,
    right = 374,
    top = 14,
    bottom = 142;
  const x = (i) =>
    visible.length === 1
      ? (left + right) / 2
      : left + (i / (visible.length - 1)) * (right - left);
  const y = (c) => bottom - c * (bottom - top);
  [0, 0.5, 1].forEach((c) => {
    add(svg, "line", {
      x1: left,
      y1: y(c),
      x2: right,
      y2: y(c),
      class: "svg-grid",
    });
    add(svg, "text", { x: 21, y: y(c) + 3, "text-anchor": "end" }, String(c));
  });
  let points = [];
  function draw() {
    if (points.length > 1)
      add(svg, "polyline", {
        points: points.join(" "),
        fill: "none",
        stroke: "#89a662",
        "stroke-width": 1.8,
      });
    points = [];
  }
  visible.forEach((row, i) => {
    if (i && row.context.episode !== visible[i - 1].context.episode) draw();
    points.push(`${x(i)},${y(row.confidence)}`);
  });
  draw();
  visible.forEach((row, i) =>
    add(svg, "circle", {
      cx: x(i),
      cy: y(row.confidence),
      r: i === index ? 4 : 2,
      fill: i === index ? "#3d5927" : "#b7cb9b",
    }),
  );
  if (visible[index])
    add(svg, "line", {
      x1: x(index),
      y1: top,
      x2: x(index),
      y2: bottom,
      stroke: "#788d5f",
      "stroke-dasharray": "3 4",
      opacity: 0.5,
    });
  add(svg, "text", { x: left, y: 162 }, "FIRST QUERY");
  add(svg, "text", { x: right, y: 162, "text-anchor": "end" }, "LATEST");
  $("scrubber").max = Math.max(0, visible.length - 1);
  $("scrubber").value = index;
  $("position").textContent = `Query ${index + 1} / ${visible.length}`;
  $("previous").disabled = index === 0;
  $("next").disabled = index >= visible.length - 1;
}
function renderOutcome() {
  if (!frame) return;
  const success =
    frame.context.recorded_success ??
    session.outcomes?.[frame.context.episode]?.success;
  $("outcome").textContent =
    typeof success !== "boolean"
      ? ""
      : `RECORDED EPISODE · ${success ? "SUCCESS" : "FAILURE"}`;
  $("outcome").classList.toggle("failed", success === false);
}
function renderFrame() {
  $("observation").src = frame.image;
  $("observation").alt = `Policy observation: ${frame.instruction}`;
  $("instruction").textContent = frame.instruction;
  $("query-label").textContent =
    `QUERY ${String((frame.context.query ?? frame.id) + 1).padStart(3, "0")}`;
  $("control-step").textContent =
    frame.context.control_step !== undefined
      ? `CONTROL ${String(frame.context.control_step).padStart(3, "0")}`
      : `SEED ${frame.seed}`;
  renderOutcome();
  $("latency").textContent = fixed(frame.policy_seconds * 1000, 1);
  $("latency-note").textContent =
    session.mode === "live"
      ? "Latest observed policy call"
      : "Recorded policy call";
  $("chunk-confidence").textContent = fixed(frame.diagnostics.chunk_confidence);
  $("denoising").textContent = `${frame.denoising_steps} denoising passes`;
  $("empty").hidden = true;
  $("content").hidden = false;
  renderGrid();
  renderDistribution();
  renderHistory();
}
async function selectFrame(nextIndex, manual = false) {
  if (!visible.length) return;
  index = Math.max(0, Math.min(nextIndex, visible.length - 1));
  if (manual) {
    playing = false;
    if (session.mode === "live") following = false;
    updateButtons();
  }
  const number = ++requestNumber,
    id = visible[index].id;
  try {
    const result = await getJSON(`/api/frame?id=${id}`);
    if (number !== requestNumber) return;
    frame = result;
    renderFrame();
    reportError("");
  } catch (error) {
    reportError(`Could not load this query: ${error.message}`);
  }
}
function updateButtons() {
  $("play").textContent = playing ? "Ⅱ Pause" : "▶ Play";
  $("play").setAttribute(
    "aria-label",
    playing ? "Pause recorded queries" : "Play recorded queries",
  );
  $("follow").textContent = following ? "Following latest" : "Follow latest";
  $("follow").classList.toggle("active", following);
}
function filterRows() {
  visible = rows.filter((row) => !episode || row.context.episode === episode);
}
function updateEpisodes() {
  const episodes = [
    ...new Set(rows.map((row) => row.context.episode).filter(Boolean)),
  ];
  const signature = episodes.join("|");
  if ($("episode").dataset.signature === signature) return;
  $("episode").dataset.signature = signature;
  $("episode").replaceChildren(new Option("All episodes", ""));
  episodes.forEach((ep) =>
    $("episode").add(new Option(ep.replaceAll("-", " "), ep)),
  );
  if (!initialized && session.mode === "replay" && episodes.length)
    episode = episodes[0];
  $("episode").value = episode;
}
function renderConnection(connected) {
  if (!connected) {
    $("connection").textContent = "Disconnected";
    $("status-dot").className = "dot offline";
    return;
  }
  const last = rows.at(-1),
    age = last ? (Date.now() - Date.parse(last.recorded_at)) / 1000 : Infinity;
  const live = session.mode === "live";
  $("connection").textContent = !live
    ? "Recorded session"
    : session.recorder?.closed
      ? "Recorder closed"
      : age < 5
        ? "Live · receiving"
        : last
          ? "Live · waiting for next query"
          : "Live · awaiting inference";
  $("status-dot").className =
    "dot" + (live && age < 5 && !session.recorder?.closed ? " live" : "");
}
async function poll() {
  try {
    const next = await getJSON("/api/state"),
      oldId = rows.at(-1)?.id;
    session = next;
    rows = next.frames;
    $("mode").textContent = session.mode.toUpperCase();
    $("mode").classList.toggle("live", session.mode === "live");
    $("source-label").textContent =
      session.mode === "live"
        ? "Observed inference stream"
        : "Recorded model outputs · no inference running";
    $("device").textContent = session.metadata.device || "Device not recorded";
    $("precision").textContent =
      session.metadata.precision || "Precision not recorded";
    $("checkpoint").textContent = session.metadata.checkpoint_step
      ? `STEP ${Number(session.metadata.checkpoint_step).toLocaleString()}`
      : "Checkpoint not recorded";
    $("record-count").textContent = `${session.total.toLocaleString()} queries`;
    $("runtime-note").textContent =
      session.metadata.execution || "Execution metadata not recorded";
    const dropped = session.recorder?.dropped ?? session.dropped,
      errors = session.recorder?.errors ?? session.errors;
    $("telemetry-note").textContent =
      dropped || errors
        ? `Telemetry: ${dropped} dropped · ${errors} errors`
        : "Observation only";
    $("telemetry-note").title = session.recorder?.last_error || "";
    $("follow").hidden = session.mode !== "live";
    $("play").hidden = session.mode === "live";
    $("playback-rate").textContent =
      session.mode === "live" ? "Refresh: 250ms" : "Playback: 0.75s / query";
    updateEpisodes();
    filterRows();
    renderConnection(true);
    if (!rows.length) {
      $("empty").hidden = false;
      $("content").hidden = true;
    } else if (
      !initialized ||
      (!frame && visible.length) ||
      (session.mode === "live" && following && rows.at(-1).id !== oldId)
    ) {
      if (session.mode === "live" && following) {
        episode = "";
        $("episode").value = "";
        filterRows();
      }
      await selectFrame(session.mode === "live" ? visible.length - 1 : 0);
    } else if (frame && session.mode === "live") {
      index = visible.findIndex((row) => row.id === frame.id);
      if (index < 0) await selectFrame(0);
      else renderHistory();
    }
    initialized = true;
    renderOutcome();
    updateButtons();
  } catch (error) {
    renderConnection(false);
    reportError(`Dashboard connection interrupted. Retrying… ${error.message}`);
  }
  setTimeout(poll, session?.mode === "live" ? 250 : 2000);
}
$("episode").onchange = () => {
  episode = $("episode").value;
  filterRows();
  selectFrame(0, true);
};
$("previous").onclick = () => selectFrame(index - 1, true);
$("next").onclick = () => selectFrame(index + 1, true);
$("scrubber").oninput = () => selectFrame(Number($("scrubber").value), true);
$("play").onclick = () => {
  if (index >= visible.length - 1) selectFrame(0);
  playing = !playing;
  updateButtons();
};
$("follow").onclick = () => {
  following = true;
  episode = "";
  $("episode").value = "";
  filterRows();
  selectFrame(visible.length - 1);
  updateButtons();
};
$("history").onclick = (event) => {
  const rect = $("history").getBoundingClientRect();
  const ratio = (((event.clientX - rect.left) / rect.width) * 390 - 30) / 344;
  selectFrame(Math.round(ratio * (visible.length - 1)), true);
};
$("observation").onload = () => {
  $("image-size").textContent =
    `${$("observation").naturalWidth} × ${$("observation").naturalHeight}`;
};
$("download").onclick = () => {
  if (!frame) return;
  const blob = new Blob(
    [JSON.stringify({ metadata: session.metadata, frame }, null, 2)],
    { type: "application/json" },
  );
  const url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = `decisionvla-query-${frame.id}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
setInterval(() => {
  if (!playing) return;
  if (index >= visible.length - 1) {
    playing = false;
    updateButtons();
  } else selectFrame(index + 1);
}, 750);
poll();
