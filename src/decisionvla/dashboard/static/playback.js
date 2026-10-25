"use strict";
// Video frame zero is the post-settle observation; frame N follows control N.
// Retain the latest recorded query. Never interpolate model probabilities.
const ReplayClock = {
  plan(video) {
    let elapsed = 0;
    const segments = [];
    video.query_timing.forEach((row, i) => {
      const mediaStart = row.control_step / video.fps;
      const next = video.query_timing[i + 1];
      const mediaEnd =
        (next ? next.control_step : video.frames - 1) / video.fps;
      segments.push({
        start: elapsed,
        end: elapsed + row.call_seconds,
        mediaStart,
        mediaEnd: mediaStart,
        phase: "inference",
        query: row.query,
        readyQuery: i ? video.query_timing[i - 1].query : null,
      });
      elapsed += row.call_seconds;
      segments.push({
        start: elapsed,
        end: elapsed + mediaEnd - mediaStart,
        mediaStart,
        mediaEnd,
        phase: "action",
        query: row.query,
        readyQuery: row.query,
      });
      elapsed += mediaEnd - mediaStart;
    });
    return { segments, duration: elapsed };
  },
  sample(plan, elapsed) {
    const time = Math.max(0, Math.min(elapsed, plan.duration));
    const segment =
      plan.segments.find((s) => time < s.end) || plan.segments.at(-1);
    return {
      ...segment,
      time,
      mediaTime:
        segment.mediaStart +
        (segment.phase === "action"
          ? Math.min(
              time - segment.start,
              segment.mediaEnd - segment.mediaStart,
            )
          : 0),
      finished: time >= plan.duration,
    };
  },
  queryIndex(rows, episode, seconds, fps) {
    const step = Math.max(0, Math.floor(seconds * fps + 1e-6));
    let found = -1;
    rows.forEach((row, i) => {
      if (
        row.context.episode === episode &&
        Number.isInteger(row.context.control_step) &&
        row.context.control_step <= step &&
        (found < 0 ||
          row.context.control_step > rows[found].context.control_step)
      )
        found = i;
    });
    return found;
  },
};
