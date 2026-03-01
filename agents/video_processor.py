"""Video frame extraction and annotated feed generation for Sovereign Sentinel.

Extracts key frames at configurable FPS, saves annotated frames with overlays,
and provides video metadata for the dashboard.
"""

import time
from pathlib import Path

import cv2
import numpy as np


class VideoProcessor:
    """Extract and manage key frames from warehouse video feeds."""

    SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}

    def __init__(self, sample_fps: float = 2.0, max_frames: int = 60):
        self.sample_fps = sample_fps
        self.max_frames = max_frames

    @staticmethod
    def is_video(path: str) -> bool:
        return Path(path).suffix.lower() in VideoProcessor.SUPPORTED_EXTS

    def extract_frames(self, video_path: str) -> list[dict]:
        """Extract key frames at target FPS. Returns list of frame metadata + arrays."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(video_fps / self.sample_fps))

        results: list[dict] = []
        frame_idx = 0

        while cap.isOpened() and len(results) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                timestamp_s = round(frame_idx / video_fps, 2)
                results.append({
                    "frame": frame,
                    "timestamp_s": timestamp_s,
                    "index": len(results),
                    "frame_idx": frame_idx,
                })

            frame_idx += 1

        cap.release()
        return results

    def extract_and_save(self, video_path: str, output_dir: str) -> list[dict]:
        """Extract frames and save as JPEGs. Returns metadata with file paths."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        frames = self.extract_frames(video_path)
        saved: list[dict] = []

        stem = Path(video_path).stem
        for f in frames:
            fname = f"{stem}_frame_{f['index']:04d}.jpg"
            fpath = out / fname
            cv2.imwrite(str(fpath), f["frame"])
            saved.append({
                "path": str(fpath),
                "timestamp_s": f["timestamp_s"],
                "index": f["index"],
                "shape": list(f["frame"].shape),
            })

        return saved

    def save_annotated_frame(
        self, frame: np.ndarray, observation: dict, output_dir: str, prefix: str = "annotated"
    ) -> str:
        """Save a frame with bounding box overlays drawn from observation data."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        annotated = frame.copy()

        for box in observation.get("face_boxes", []):
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, "HUMAN", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for box in observation.get("hand_boxes", []):
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 255), 2)

        anomaly = observation.get("anomaly", "none")
        if anomaly != "none":
            severity = observation.get("severity", "low")
            color = (0, 0, 255) if severity == "critical" else (0, 165, 255)
            label = f"{anomaly.upper()} [{severity}]"
            cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ts = observation.get("timestamp", time.time())
        fname = f"{prefix}_{int(ts * 1000)}.jpg"
        fpath = out / fname
        cv2.imwrite(str(fpath), annotated)
        return str(fpath)

    def get_video_info(self, video_path: str) -> dict:
        """Return basic video metadata."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        return {
            "fps": round(fps, 2),
            "total_frames": total,
            "duration_s": round(total / fps, 2) if fps else 0,
            "resolution": [w, h],
            "sample_fps": self.sample_fps,
            "estimated_analysis_frames": min(
                self.max_frames,
                int((total / fps) * self.sample_fps) if fps else 0,
            ),
        }
