from __future__ import annotations

import json
from pathlib import Path

import cv2

from app.capture.sources import VideoFrameSource
from app.diagnostics.overlay import draw_developer_overlay
from app.diagnostics.session import DiagnosticSession
from app.pipeline import MacroPipeline


def replay_video(
    video: Path,
    profile: dict,
    output_dir: Path,
    render_overlay: bool = True,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> dict:
    source = VideoFrameSource(video, start_s=start_s, duration_s=duration_s)
    pipeline = MacroPipeline(profile)
    session = DiagnosticSession(
        output_dir,
        {
            "mode": "offline_replay",
            "video": str(video),
            "profile": profile.get("_profile_name"),
            "start_s": start_s,
            "duration_s": duration_s,
            "input_enabled": False,
            "shared_pipeline": "app.pipeline.MacroPipeline",
        },
    )
    writer = None
    try:
        while True:
            packet = source.read()
            if packet is None:
                break
            result = pipeline.process(packet)
            session.record(result)
            if render_overlay:
                if writer is None:
                    height, width = packet.frame_bgr.shape[:2]
                    fps = source.capture.get(cv2.CAP_PROP_FPS) or 60.0
                    writer = cv2.VideoWriter(
                        str(output_dir / "overlay.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                writer.write(draw_developer_overlay(result))
    finally:
        source.close()
        if writer is not None:
            writer.release()
        session.close()
    summary = session.summary.serializable()
    print(json.dumps(summary, indent=2))
    return summary
