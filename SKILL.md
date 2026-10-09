---
name: dance-pose-coach
description: Analyze a locally stored dance video alone or against a reference using lightweight pose, rhythm, and movement-continuity metrics. Use when a user wants to run, configure, interpret, or extend the Dance Pose Coach project; do not present its scores as medical or professional biomechanical diagnosis.
---

# Dance Pose Coach

Use the bundled application to analyze a user's dance video alone or compare it with a reference while keeping uploaded media on the local machine.

## Workflow

1. Install the Python package from this skill directory. FFmpeg is bundled through
   `imageio-ffmpeg`, and the default 33-landmark OpenCV CPU model is bundled in the
   package; use `dance-pose-doctor` to confirm both are available.
2. When no reference is supplied, run coaching-first single-video mode. Lead with concrete problems and drills in ordinary language; do not open with BPM, timing offsets, ratios, or confidence numbers. Analyze vertical groove/bounce, chest and torso pulse, arm frame and extension, hip/leg use, rhythm placement, movement-energy/force proxies, and transitions. Every highlighted issue must include its visible effect, evidence, useful time spans, and step-by-step practice instructions. Put technical numbers in a secondary evidence section. Do not invent a pose-correctness or choreography-reproduction score.
3. Prefer an explicit reference URL or file when the user supplies one. Otherwise use automatic song recognition and web search only when the user explicitly enables online services, and ask the user to confirm the retrieved reference before treating it as authoritative.
4. Run `dance-pose-coach` for the Gradio UI or `dance-pose-analyze` for the CLI. Both automatically select single-video or comparison mode.
5. Interpret scores as coaching signals. Highlight low-confidence pose segments and recognition uncertainty instead of overstating precision.
   Treat “power/force” as a speed-and-acceleration proxy only; never claim to measure muscle force, joint load, injury risk, or artistic quality.
6. Keep source videos, extracted audio, and reports inside the configured local workspace. Do not upload them to another service unless the user explicitly approves it.
7. Prefer the default `opencv-cpu` pose backend. Use the optional legacy MediaPipe backend only for compatibility; it may require a macOS graphical session and can fail with `kGpuService` or `NSOpenGLPixelFormat` in an AI sandbox.

## Important constraints

- The pose model is reliable only when the whole body is visible, lighting is adequate, and one main dancer dominates the frame.
- Mirror orientation can materially change results; use automatic mirror selection or lock the setting when choreography requires a fixed left/right interpretation.
- Reference search results may be covers, tutorials, mirrored versions, or copyrighted media. Preserve source attribution and comply with the source platform's terms.
- Rhythm analysis compares audio beats and motion accents; it does not infer artistic quality or injury risk.
- In solo mode, describe patterns as coaching observations rather than choreography errors. Chest rise/fall, bounce, and “power” are 2D motion proxies affected by camera movement, clothing, and occlusion.

Read [references/metrics.md](references/metrics.md) when explaining score definitions or changing thresholds.
