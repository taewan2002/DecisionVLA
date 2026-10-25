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
let evaluation,
  lastEvaluationPoll = 0,
  distributionView = false,
  evaluationRenderKey = "";
let activeVideo = null,
  snapshotMode = false,
  manualSelection = null;
const queryCache = new Map(),
  pendingQueries = new Map();
let timingMode = "recorded",
  timedPlan = null,
  timedElapsed = 0,
  timedAnchor = 0,
  timedAnimation = null,
  timedSegment = "";

function measuredPlayback() {
  return showingVideo() && timingMode === "recorded" && timedPlan;
}
function pauseTimed() {
  if (timedAnimation !== null) cancelAnimationFrame(timedAnimation);
  timedAnimation = null;
  playing = false;
  $("rollout-video").pause();
  updateButtons();
}
function paintTimed(forceSeek = false) {
  if (!measuredPlayback()) return;
  const state = ReplayClock.sample(timedPlan, timedElapsed),
    video = $("rollout-video");
  const segment = `${state.phase}:${state.query}`,
    changed = segment !== timedSegment;
  timedSegment = segment;
  const waiting = state.phase === "inference" && !state.finished;
  $("content").classList.toggle(
    "first-query-pending",
    waiting && state.readyQuery === null,
  );
  $("media-time").textContent =
    `${fixed(state.time, 2)} / ${fixed(timedPlan.duration, 2)}s · recorded wait replay`;
  $("replay-phase").textContent = state.finished
    ? "COMPLETE · recorded episode"
    : waiting
      ? `INFERENCE ${String(state.query + 1).padStart(2, "0")} · ${fixed((state.end - state.time) * 1000, 0)} ms remaining${state.readyQuery === null ? " · output pending" : " · previous output shown"}`
      : `EXECUTING · query ${state.query + 1} ready · 20Hz controls`;
  $("replay-phase").classList.toggle("pending", waiting);
  $("timed-scrubber").max = timedPlan.duration;
  $("timed-scrubber").value = state.time;
  if (waiting || !playing || state.finished) {
    video.pause();
    if (video.readyState && (changed || forceSeek || state.finished))
      video.currentTime = state.mediaTime;
  } else {
    if (
      video.readyState &&
      (changed ||
        forceSeek ||
        Math.abs(video.currentTime - state.mediaTime) > 0.12)
    )
      video.currentTime = state.mediaTime;
    if (video.paused)
      video.play().catch((error) => {
        pauseTimed();
        reportError(error.message);
      });
  }
  const readyIndex =
    state.readyQuery === null
      ? -1
      : visible.findIndex(
          (row) =>
            row.context.episode === activeVideo.episode &&
            row.context.query === state.readyQuery,
        );
  if (
    readyIndex >= 0 &&
    visible[readyIndex].id !== frame?.id &&
    readyIndex !== index
  )
    selectFrame(readyIndex, false, true);
  if (state.finished && playing) pauseTimed();
}
function tickTimed(now) {
  timedAnimation = null;
  if (!playing || !measuredPlayback()) return;
  timedElapsed = Math.min(timedPlan.duration, (now - timedAnchor) / 1000);
  paintTimed();
  if (playing) timedAnimation = requestAnimationFrame(tickTimed);
}
function toggleTimed() {
  if (playing) {
    pauseTimed();
    return;
  }
  if ($("rollout-video").readyState < 2) {
    reportError("Video is loading; press Play when ready.");
    return;
  }
  if (timedElapsed >= timedPlan.duration) {
    timedElapsed = 0;
    timedSegment = "";
  }
  timedAnchor = performance.now() - timedElapsed * 1000;
  playing = true;
  paintTimed(true);
  timedAnimation = requestAnimationFrame(tickTimed);
  updateButtons();
}

async function loadQuery(id) {
  if (queryCache.has(id)) return queryCache.get(id);
  if (!pendingQueries.has(id)) {
    const request = getJSON(`/api/frame?id=${id}`)
      .then((value) => {
        queryCache.set(id, value);
        while (queryCache.size > 64)
          queryCache.delete(queryCache.keys().next().value);
        return value;
      })
      .finally(() => pendingQueries.delete(id));
    pendingQueries.set(id, request);
  }
  return pendingQueries.get(id);
}
function showingVideo() {
  return activeVideo && !snapshotMode && session?.mode === "replay";
}
function configureVideo() {
  const next =
    session.mode === "replay"
      ? session.videos?.find((v) => v.episode === frame.context.episode)
      : null;
  const video = $("rollout-video");
  if (next?.id !== activeVideo?.id) {
    pauseTimed();
    video.pause();
    activeVideo = next || null;
    timedPlan = next?.query_timing?.length ? ReplayClock.plan(next) : null;
    timedElapsed = 0;
    timedSegment = "";
    if (next) {
      video.src = next.url;
      video.load();
    } else {
      video.removeAttribute("src");
      video.load();
    }
  }
  const show = Boolean(showingVideo());
  const measured = Boolean(measuredPlayback());
  video.controls = !measured;
  $("timing-controls").hidden = !show || !timedPlan;
  $("timed-scrubber").hidden = !measured;
  $("replay-phase").hidden = !measured;
  $("timing-note").textContent = measured
    ? "Recorded call waits + nominal 20Hz controls. Simulator processing time excluded; not a live run."
    : "Simulation time only; inference waits omitted.";
  if (!measured) $("content").classList.remove("first-query-pending");
  video.hidden = !show;
  $("observation").hidden = show;
  $("media-toggle").hidden = !activeVideo;
  $("media-toggle").textContent = show
    ? "Inspect policy input"
    : "Watch rollout";
  $("media-kind").textContent = show
    ? measured
      ? "RECORDED INFERENCE TIMING"
      : "RECORDED ROLLOUT · 20 FPS"
    : "POLICY OBSERVATION";
  if (!measured)
    $("media-time").textContent = show
      ? `${fixed(video.currentTime, 2)}s · simulation time, inference waits omitted`
      : "Exact policy input · updated once per query";
  updatePlaybackLabel();
}
function updatePlaybackLabel() {
  $("playback-rate").textContent = measuredPlayback()
    ? "Recorded call waits + 20Hz controls"
    : showingVideo()
      ? "Video: 20fps · panels follow recorded queries"
      : session?.mode === "live"
        ? "Query updates · refresh 250ms"
        : "Snapshots: 0.75s / query";
}
function seekToQuery() {
  if (!showingVideo() || !frame) return;
  if (measuredPlayback()) {
    const segment = timedPlan.segments.find(
      (s) => s.phase === "action" && s.query === frame.context.query,
    );
    timedElapsed = frame.context.query === 0 ? 0 : segment?.start || 0;
    timedSegment = "";
    paintTimed(true);
    return;
  }
  const video = $("rollout-video"),
    target = frame.context.control_step / activeVideo.fps;
  if (video.readyState) video.currentTime = target;
  // loadedmetadata uses the current frame when an episode source has just changed.
}
function syncVideo(seconds) {
  if (!showingVideo() || measuredPlayback() || manualSelection !== null) return;
  $("media-time").textContent =
    `${fixed(seconds, 2)}s · simulation time, inference waits omitted`;
  const next = ReplayClock.queryIndex(
    visible,
    activeVideo.episode,
    seconds,
    activeVideo.fps,
  );
  if (next >= 0 && (next !== index || visible[next].id !== frame?.id))
    selectFrame(next, false, true);
}
function watchVideoFrame(_now, metadata) {
  syncVideo(metadata.mediaTime);
  $("rollout-video").requestVideoFrameCallback(watchVideoFrame);
}

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
      if (distributionView) renderMiniDistribution(button, t, d);
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
  add(svg, "text", { x: left, y: 162 }, "EARLIEST SHOWN");
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
    session.outcomes?.[frame.context.episode]?.success ??
    evaluation?.episodes.find((row) => row.episode === frame.context.episode)
      ?.success;
  $("outcome").textContent =
    typeof success !== "boolean"
      ? ""
      : `${session.mode === "live" ? "COMPLETED" : "RECORDED"} EPISODE · ${success ? "SUCCESS" : "FAILURE"}`;
  $("outcome").classList.toggle("failed", success === false);
}
function renderFrame() {
  $("observation").src = frame.image;
  $("observation").alt = `Policy observation: ${frame.instruction}`;
  configureVideo();
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
  renderEvaluation();
}
async function selectFrame(nextIndex, manual = false, fromVideo = false) {
  if (!visible.length) return;
  const number = ++requestNumber;
  index = Math.max(0, Math.min(nextIndex, visible.length - 1));
  if (manual) {
    manualSelection = number;
    pauseTimed();
    $("rollout-video").pause();
    playing = false;
    if (session.mode === "live") following = false;
    updateButtons();
  }
  const id = visible[index].id;
  try {
    const result = await loadQuery(id);
    if (number !== requestNumber) return;
    frame = result;
    renderFrame();
    if (!fromVideo) seekToQuery();
    visible
      .slice(index + 1, index + 5)
      .forEach((row) => loadQuery(row.id).catch(() => {}));
    reportError("");
  } catch (error) {
    reportError(`Could not load this query: ${error.message}`);
  } finally {
    if (manualSelection === number) manualSelection = null;
  }
}
function updateButtons() {
  $("media-play").textContent = playing ? "Ⅱ Pause" : "▶ Play";
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
    if (Date.now() - lastEvaluationPoll >= 2000) {
      evaluation = await getJSON("/api/evaluation");
      lastEvaluationPoll = Date.now();
    }
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
    updatePlaybackLabel();
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
    renderEvaluation();
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
  if (measuredPlayback()) {
    toggleTimed();
    return;
  }
  if (showingVideo()) {
    const video = $("rollout-video");
    if (!video.paused) video.pause();
    else {
      if (video.ended) video.currentTime = 0;
      video
        .play()
        .catch((error) =>
          reportError(`Video could not play: ${error.message}`),
        );
    }
    return;
  }
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
  if (!playing || showingVideo()) return;
  if (index >= visible.length - 1) {
    playing = false;
    updateButtons();
  } else selectFrame(index + 1);
}, 750);
$("media-toggle").onclick = () => {
  pauseTimed();
  $("rollout-video").pause();
  playing = false;
  snapshotMode = !snapshotMode;
  configureVideo();
  seekToQuery();
  updateButtons();
};
$("media-play").onclick = () => $("play").click();
$("timing-mode").onchange = () => {
  pauseTimed();
  timingMode = $("timing-mode").value;
  configureVideo();
  seekToQuery();
};
$("timed-scrubber").oninput = () => {
  pauseTimed();
  timedElapsed = Number($("timed-scrubber").value);
  paintTimed(true);
};
$("rollout-video").onloadedmetadata = () =>
  measuredPlayback() ? paintTimed(true) : seekToQuery();
$("rollout-video").ontimeupdate = () =>
  syncVideo($("rollout-video").currentTime);
$("rollout-video").onplay = () => {
  if (measuredPlayback()) return;
  playing = true;
  updateButtons();
};
$("rollout-video").onpause = $("rollout-video").onended = () => {
  if (measuredPlayback()) return;
  playing = false;
  updateButtons();
};
$("rollout-video").onerror = () =>
  reportError(
    "Recorded video unavailable; use Inspect policy input to view query snapshots.",
  );
if ($("rollout-video").requestVideoFrameCallback)
  $("rollout-video").requestVideoFrameCallback(watchVideoFrame);
function renderMiniDistribution(button, t, d) {
  button.classList.add("mini-distribution");
  button.replaceChildren();
  const svg = svgElement("svg", {
    viewBox: "0 0 256 32",
    "aria-hidden": "true",
    preserveAspectRatio: "none",
  });
  const p = frame.diagnostics.probabilities[t][d],
    peak = Math.max(...p);
  if (d === 6) {
    [0, 255].forEach((bin, i) =>
      add(svg, "rect", {
        x: 38 + i * 128,
        y: 30 - (p[bin] / peak) * 28,
        width: 52,
        height: (p[bin] / peak) * 28,
        fill: "#638b42",
      }),
    );
  } else {
    const points = p.map((v, i) => `${i},${30 - (v / peak) * 28}`).join(" ");
    add(svg, "polygon", {
      points: `0,31 ${points} 255,31`,
      fill: "#6f934f",
      opacity: 0.85,
    });
  }
  button.appendChild(svg);
  const value = document.createElement("span");
  value.textContent = signed(frame.actions[t][d]);
  button.appendChild(value);
}
function setDistributionView(enabled) {
  distributionView = enabled;
  $("view-values").classList.toggle("active", !enabled);
  $("view-distributions").classList.toggle("active", enabled);
  $("view-values").setAttribute("aria-pressed", String(!enabled));
  $("view-distributions").setAttribute("aria-pressed", String(enabled));
  $("grid-caption").textContent = enabled
    ? "All 56 distributions · each chart scaled to its own peak · select one to inspect exact probabilities."
    : "Select a control to inspect its distribution. Values are normalized to [−1, 1].";
  if (frame) renderGrid();
}
function inspectEpisode(key) {
  if (!rows.some((row) => row.context.episode === key)) return;
  episode = key;
  $("episode").value = key;
  filterRows();
  selectFrame(0, true);
}
function percentile(values, fraction) {
  if (!values.length) return null;
  const a = [...values].sort((x, y) => x - y),
    k = (a.length - 1) * fraction,
    lo = Math.floor(k);
  return a[lo] + (a[Math.ceil(k)] - a[lo]) * (k - lo);
}
function renderEvaluation() {
  if (!evaluation || !session) return;
  const key = JSON.stringify([
    evaluation,
    rows[0]?.id,
    rows.at(-1)?.id,
    rows.length,
    frame?.id,
  ]);
  if (key === evaluationRenderKey) return;
  evaluationRenderKey = key;
  const e = evaluation,
    subset = e.scope === "recorded_subset";
  $("benchmark-label").textContent =
    session.metadata.benchmark === "plus"
      ? "LIBERO PLUS EVALUATION"
      : "LIBERO EVALUATION";
  $("evaluation-status").textContent = subset
    ? "SELECTED REPLAY"
    : e.status.toUpperCase();
  $("progress-label").textContent = subset
    ? "Recorded episodes in this view"
    : "Completed episodes · this worker";
  $("eval-completed").textContent = e.expected
    ? `${e.completed} / ${e.expected}`
    : String(e.completed);
  $("eval-progress").max = e.expected || Math.max(1, e.total_episodes);
  $("eval-progress").value = e.completed;
  $("eval-success").textContent = String(e.successes);
  $("eval-failure").textContent = String(e.failures);
  $("eval-rate").textContent = subset
    ? "Selected episodes only"
    : e.completed
      ? `${fixed((e.successes / e.completed) * 100, 1)}% of completed episodes`
      : "No completed outcomes yet";
  $("eval-errors").textContent =
    `${e.errors} errors · excluded from success/failure counts`;
  const timings = rows
    .map((r) => r.policy_seconds)
    .filter((v) => Number.isFinite(v) && v > 0);
  const p50 = percentile(timings, 0.5),
    p95 = percentile(timings, 0.95);
  $("eval-latency").textContent =
    p50 === null ? "—" : `${fixed(p50 * 1000, 0)} / ${fixed(p95 * 1000, 0)} ms`;
  $("eval-hz").textContent =
    p50 === null
      ? "Awaiting policy calls"
      : `≈${fixed(1 / p50, 1)} calls/s at p50 · ${timings.length} available queries`;
  $("suite-summary").replaceChildren();
  e.suites.forEach((s) => {
    const chip = document.createElement("span");
    chip.textContent = `${s.suite.replace("libero_", "").toUpperCase()} · ${s.successes}/${s.completed} observed successes`;
    $("suite-summary").appendChild(chip);
  });
  $("selection-note").textContent = [
    subset
      ? "Replay subset, not a benchmark success rate."
      : "Live observations; formal results still require the artifact audit.",
    e.selection || "",
    session.metadata.latency_scope ||
      "Policy timing excludes display and telemetry overhead.",
  ]
    .filter(Boolean)
    .join(" ");
  $("outcome-scope").textContent =
    e.total_episodes > e.listed_episodes
      ? `Latest ${e.listed_episodes} / ${e.total_episodes} episodes`
      : `${e.completed} completed episodes`;
  const svg = $("outcome-plot");
  svg.replaceChildren();
  const left = 86,
    right = 652,
    x = (c) => left + c * (right - left);
  [0, 0.25, 0.5, 0.75, 1].forEach((c) => {
    add(svg, "line", {
      x1: x(c),
      x2: x(c),
      y1: 22,
      y2: 124,
      class: "svg-grid",
    });
    add(svg, "text", { x: x(c), y: 147, "text-anchor": "middle" }, fixed(c, 2));
  });
  add(svg, "text", { x: left, y: 166 }, "FIRST-QUERY CONFIDENCE");
  add(svg, "text", { x: 12, y: 51 }, "Success");
  add(svg, "text", { x: 12, y: 108 }, "Failure");
  const eligible = e.episodes.filter(
    (r) =>
      r.status === "completed" &&
      typeof r.success === "boolean" &&
      Number.isFinite(r.first_confidence),
  );
  eligible.forEach((r, i) => {
    const available = rows.some((row) => row.context.episode === r.episode);
    const circle = svgElement("circle", {
      cx: x(r.first_confidence),
      cy: (r.success ? 47 : 104) + ((i % 5) - 2) * 4,
      r: 5,
      fill: r.success ? "#759653" : "#bc8a68",
      stroke: frame?.context.episode === r.episode ? "#25391c" : "white",
      "stroke-width": frame?.context.episode === r.episode ? 2 : 1,
      class: "outcome-point",
      tabindex: available ? 0 : -1,
      role: "button",
      "aria-label": `${r.episode}: ${r.success ? "success" : "failure"}, first confidence ${fixed(r.first_confidence)}`,
    });
    circle.appendChild(
      svgElement(
        "title",
        {},
        `${r.episode} · ${fixed(r.first_confidence)} · ${r.success ? "success" : "failure"}`,
      ),
    );
    circle.onclick = () => inspectEpisode(r.episode);
    circle.onkeydown = (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        inspectEpisode(r.episode);
      }
    };
    svg.appendChild(circle);
  });
  if (!eligible.length)
    add(
      svg,
      "text",
      { x: 370, y: 80, "text-anchor": "middle" },
      "Waiting for completed outcomes with a first-query score",
    );
  $("outcome-note").textContent =
    `${eligible.length} first-query scores shown. Descriptive comparison; no calibrated failure prediction or confidence threshold is applied.`;
  $("episode-list").replaceChildren();
  [...e.episodes]
    .reverse()
    .slice(0, 20)
    .forEach((r) => {
      const button = document.createElement("button");
      button.className =
        "episode-row" +
        (frame?.context.episode === r.episode ? " selected" : "");
      button.type = "button";
      button.disabled = !rows.some((row) => row.context.episode === r.episode);
      const title = document.createElement("div");
      title.textContent = r.episode
        .replaceAll("libero_", "")
        .replaceAll("-", " ");
      const sub = document.createElement("span");
      sub.textContent = `First confidence ${r.first_confidence === null ? "—" : fixed(r.first_confidence)} · ${r.policy_queries} queries`;
      title.appendChild(sub);
      button.appendChild(title);
      const result = document.createElement("strong");
      const outcome =
        r.status === "completed"
          ? r.success
            ? "success"
            : "failure"
          : r.status;
      result.className = "episode-result " + outcome;
      result.textContent = outcome.toUpperCase();
      button.appendChild(result);
      button.onclick = () => inspectEpisode(r.episode);
      $("episode-list").appendChild(button);
    });
}
$("view-values").onclick = () => setDistributionView(false);
$("view-distributions").onclick = () => setDistributionView(true);
$("inspect-weakest").onclick = () => {
  if (!frame) return;
  let weakest = Infinity;
  frame.diagnostics.metrics.ordered_confidence.forEach((row, t) =>
    row.forEach((c, d) => {
      if (c < weakest) {
        weakest = c;
        timestep = t;
        dimension = d;
      }
    }),
  );
  renderGrid();
  renderDistribution();
};
poll();
