// Run with node tests/dashboard_playback.cjs; no browser or model required.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const source = fs.readFileSync(
  path.join(__dirname, "../src/decisionvla/dashboard/static/playback.js"),
  "utf8",
);
const clock = vm.runInNewContext(source + "\nReplayClock;");
const rows = [
  { context: { episode: "a", control_step: 0 } },
  { context: { episode: "a", control_step: 8 } },
  { context: { episode: "a", control_step: 16 } },
  { context: { episode: "b", control_step: 0 } },
];
assert.equal(clock.queryIndex(rows, "a", 0, 20), 0);
assert.equal(clock.queryIndex(rows, "a", 0.399, 20), 0);
assert.equal(clock.queryIndex(rows, "a", 0.4, 20), 1);
assert.equal(clock.queryIndex(rows, "a", 0.8, 20), 2);
assert.equal(clock.queryIndex(rows, "a", 1.1, 20), 2); // Last partial chunk.
assert.equal(clock.queryIndex(rows, "a", 0.1, 20), 0); // Backward seek.
assert.equal(clock.queryIndex(rows, "b", 0.5, 20), 3); // Episode isolation.
assert.equal(clock.queryIndex(rows.slice(1), "a", 0.1, 20), -1); // Missing first query.
const plan = clock.plan({
  fps: 20,
  frames: 18,
  query_timing: [
    { query: 0, control_step: 0, call_seconds: 0.3 },
    { query: 1, control_step: 8, call_seconds: 0.4 },
    { query: 2, control_step: 16, call_seconds: 0.2 },
  ],
});
assert.ok(Math.abs(plan.duration - 1.75) < 1e-9); // .85s controls + .9s waits.
assert.equal(clock.sample(plan, 0.1).phase, "inference");
assert.equal(clock.sample(plan, 0.1).readyQuery, null);
assert.equal(clock.sample(plan, 0.1).mediaTime, 0);
assert.equal(clock.sample(plan, 0.3).phase, "action");
assert.equal(clock.sample(plan, 0.3).readyQuery, 0);
assert.ok(Math.abs(clock.sample(plan, 0.5).mediaTime - 0.2) < 1e-9);
assert.equal(clock.sample(plan, 0.8).phase, "inference");
assert.equal(clock.sample(plan, 0.8).readyQuery, 0); // No future output during query 1 wait.
assert.equal(clock.sample(plan, 0.8).mediaTime, 0.4);
assert.equal(clock.sample(plan, 1.2).readyQuery, 1);
assert.equal(clock.sample(plan, 2).finished, true);
assert.ok(Math.abs(clock.sample(plan, 2).mediaTime - 0.85) < 1e-9);
console.log("21 video/query/timing synchronization checks passed");
