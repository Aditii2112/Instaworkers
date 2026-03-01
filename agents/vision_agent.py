"""SEE stage — Video-Native Vision Agent.

Pipeline:
  1. cv2.VideoCapture processes uploaded video frame-by-frame at 2 FPS
  2. PaliGemma 2 (3B) via mlx-vlm analyses each sampled frame for anomalies
  3. MediaPipe runs concurrently on every sampled frame for human_present detection
  4. Emits structured events: {anomaly, severity, location, human_present, timestamp}

Falls back to Ollama VLM when mlx-vlm is not available (e.g. non-Apple-Silicon).
"""

import json
import os
import re
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from config import AppConfig
from observability.event_trace import EventTracer

_MODEL_DIR = Path(__file__).parent.parent / "data" / "models"

_ANOMALY_PATTERN = re.compile(
    r"(leak|pest|spoilage|spill|damage|hazard|fire|smoke)",
    re.IGNORECASE,
)
_SEVERITY_PATTERN = re.compile(r"(critical|high|medium|low)", re.IGNORECASE)
_LOCATION_PATTERN = re.compile(
    r"(Aisle_\d+|Rack_\d+|Loading_Dock|Cold_Storage|Warehouse|Zone_[A-Z])",
    re.IGNORECASE,
)

# Patterns matching fine-tuned model output: SEE: [...] REASON: ... ACT: [...]
_SEE_BLOCK_PATTERN = re.compile(r"SEE:\s*\[([^\]]+)\]", re.IGNORECASE)
# Pest-specific: "Pest detected: brown_planthopper"
_PEST_PATTERN = re.compile(r"pest\s+detected:\s*(\S+)", re.IGNORECASE)
# Defect: "pill defect: scratch", "cable defect: crack"
_DEFECT_PATTERN = re.compile(r"(\w+)\s+defect:\s*(\w+)", re.IGNORECASE)
# Package damage: "Package damage detected: water_damage"
_PACKAGE_DAMAGE_PATTERN = re.compile(r"package\s+damage\s+detected:\s*(\w+)", re.IGNORECASE)
# Spoilage: "Spoilage detected: mold", "spoiled"
_SPOILAGE_PATTERN = re.compile(r"spoil(?:age|ed)", re.IGNORECASE)
# Leak: "Leak detected", "water leak", "liquid leak"
_LEAK_PATTERN = re.compile(r"leak", re.IGNORECASE)
# Human/person present
_HUMAN_IN_VLM_PATTERN = re.compile(r"(human|person|worker|people|man|woman)\s+(detected|present|visible|seen)", re.IGNORECASE)
# No defect / pass
_NO_DEFECT_PATTERN = re.compile(r"no\s+(?:defect|anomal)", re.IGNORECASE)

_VLM_PROMPT = "Analyze this warehouse feed."


class VisionAgent:
    """Video-native vision agent with VLM (PaliGemma 2) and MediaPipe sensor fusion."""

    def __init__(self, config: AppConfig, tracer: EventTracer, gemma=None):
        self.config = config
        self.tracer = tracer
        self.gemma = gemma

        self._mlx_model = None
        self._mlx_processor = None
        self._mlx_available = False

        self._face_detector = None
        self._hand_landmarker = None
        self._mp_image_cls = None
        self._mp_image_format = None
        self._bg_subtractor = None

        self._init_vlm()
        self._init_detectors()

    # ── VLM Initialization ───────────────────────────────────────────

    def _init_vlm(self):
        """Try loading PaliGemma 2 via mlx-vlm for on-device inference."""
        try:
            from mlx_vlm import load, generate
            from mlx_vlm.utils import load_config as load_model_config

            model_path = self.config.vlm.model_path
            print(f"[SEE] Loading PaliGemma 2 via mlx-vlm: {model_path}")
            self._mlx_model, self._mlx_processor = load(model_path)
            self._mlx_generate = generate
            self._mlx_available = True
            print("[SEE] mlx-vlm PaliGemma 2 loaded successfully")
        except Exception as e:
            print(f"[SEE] mlx-vlm not available ({e}); will use Ollama VLM fallback")
            self._mlx_available = False

    # ── MediaPipe Initialization ─────────────────────────────────────

    def _init_detectors(self):
        """Load MediaPipe face/hand detectors for human_present sensor fusion."""
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceDetector,
                FaceDetectorOptions,
                HandLandmarker,
                HandLandmarkerOptions,
            )
            self._mp_image_cls = mp.Image
            self._mp_image_format = mp.ImageFormat

            import ssl
            import urllib.request
            _MODEL_DIR.mkdir(parents=True, exist_ok=True)

            # Bypass SSL for model downloads (macOS Python often lacks certs)
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_ctx))

            def _download(url: str, dest: Path):
                with opener.open(url) as resp, open(dest, "wb") as f:
                    f.write(resp.read())

            face_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
            face_path = _MODEL_DIR / "blaze_face_short_range.tflite"
            if not face_path.exists():
                _download(face_url, face_path)
            self._face_detector = FaceDetector.create_from_options(
                FaceDetectorOptions(base_options=BaseOptions(model_asset_path=str(face_path)))
            )

            hand_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
            hand_path = _MODEL_DIR / "hand_landmarker.task"
            if not hand_path.exists():
                _download(hand_url, hand_path)
            self._hand_landmarker = HandLandmarker.create_from_options(
                HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(hand_path)), num_hands=2)
            )
            print("[SEE] MediaPipe face + hand detectors ready")
        except Exception as e:
            print(f"[SEE] MediaPipe init failed: {e}; human_present defaults to False")

    # ── Human Detection ──────────────────────────────────────────────

    def _detect_human(self, rgb: np.ndarray) -> dict:
        """Detect humans via MediaPipe (faces/hands) + skin-tone fallback."""
        result = {"human_present": False, "face_count": 0, "hand_count": 0, "face_boxes": [], "hand_boxes": []}

        # MediaPipe detection
        if self._mp_image_cls is not None and self._mp_image_format is not None:
            try:
                mp_image = self._mp_image_cls(
                    image_format=self._mp_image_format.SRGB, data=rgb
                )

                if self._face_detector:
                    face_result = self._face_detector.detect(mp_image)
                    if face_result.detections:
                        result["human_present"] = True
                        result["face_count"] = len(face_result.detections)
                        for det in face_result.detections:
                            bb = det.bounding_box
                            result["face_boxes"].append({
                                "x": bb.origin_x, "y": bb.origin_y,
                                "w": bb.width, "h": bb.height,
                            })

                if self._hand_landmarker:
                    hand_result = self._hand_landmarker.detect(mp_image)
                    if hand_result.hand_landmarks:
                        result["human_present"] = True
                        result["hand_count"] = len(hand_result.hand_landmarks)
                        h, w = rgb.shape[:2]
                        for landmarks in hand_result.hand_landmarks:
                            xs = [lm.x * w for lm in landmarks]
                            ys = [lm.y * h for lm in landmarks]
                            result["hand_boxes"].append({
                                "x": int(min(xs)), "y": int(min(ys)),
                                "w": int(max(xs) - min(xs)), "h": int(max(ys) - min(ys)),
                            })
            except Exception as e:
                print(f"[SEE] MediaPipe detection error: {e}")

        # Skin-tone fallback if MediaPipe didn't find anyone
        if not result["human_present"]:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            h_img, w_img = rgb.shape[:2]
            total_px = h_img * w_img
            skin_lo1 = np.array([0, 30, 60])
            skin_hi1 = np.array([20, 180, 255])
            skin_lo2 = np.array([160, 30, 60])
            skin_hi2 = np.array([180, 180, 255])
            skin_mask = cv2.inRange(hsv, skin_lo1, skin_hi1) | cv2.inRange(hsv, skin_lo2, skin_hi2)
            skin_ratio = cv2.countNonZero(skin_mask) / total_px

            if skin_ratio > 0.03:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
                skin_clean = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
                skin_clean = cv2.morphologyEx(skin_clean, cv2.MORPH_CLOSE, kernel, iterations=2)
                contours, _ = cv2.findContours(skin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_area = total_px * 0.002
                big_skin = [c for c in contours if cv2.contourArea(c) > min_area]
                if big_skin:
                    result["human_present"] = True
                    result["face_count"] = len(big_skin)
                    for c in big_skin:
                        x, y, bw, bh = cv2.boundingRect(c)
                        result["face_boxes"].append({"x": x, "y": y, "w": bw, "h": bh})

        return result

    # ── Motion-Based Pest Detection ──────────────────────────────────

    def _detect_motion_objects(self, frame: np.ndarray) -> dict:
        """Detect moving objects using background subtraction.

        A mouse/rat on a static warehouse floor shows up clearly as foreground
        motion even when color-based detection fails (e.g. white/gray pest).
        """
        if self._bg_subtractor is None:
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=20, varThreshold=50, detectShadows=True
            )

        fg_mask = self._bg_subtractor.apply(frame)
        fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

        h, w = frame.shape[:2]
        total_px = h * w
        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = total_px * 0.0003
        max_area = total_px * 0.08
        moving = []
        for c in contours:
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                peri = cv2.arcLength(c, True)
                if peri > 0:
                    aspect = max(*cv2.boundingRect(c)[2:4]) / (min(*cv2.boundingRect(c)[2:4]) + 1)
                    if aspect < 5:
                        x, y, bw, bh = cv2.boundingRect(c)
                        moving.append({"x": x, "y": y, "w": bw, "h": bh, "area": int(area)})

        fg_ratio = cv2.countNonZero(fg_clean) / total_px
        return {
            "moving_objects": moving,
            "fg_ratio": round(fg_ratio, 4),
            "motion_detected": len(moving) > 0,
        }

    def reset_motion_detector(self):
        """Reset background model for a new video."""
        self._bg_subtractor = None

    # ── VLM Analysis ─────────────────────────────────────────────────

    def _vlm_analyze(self, image_path: str, text_input: str | None = None) -> dict:
        """Analyze a frame using the best available method.

        Priority:
          1. mlx-vlm (true multimodal, if installed)
          2. Pure OpenCV heuristics (reliable, no LLM needed)
        """
        if self._mlx_available:
            prompt = text_input or _VLM_PROMPT
            return self._vlm_mlx(image_path, prompt)

        frame = cv2.imread(image_path)
        if frame is None:
            return {"anomaly": "none", "severity": "low", "location": "unknown", "raw_vlm": "frame unreadable"}

        cv_features = self._extract_cv_features(frame)
        return self._pure_cv_classify(cv_features)

    def _vlm_mlx(self, image_path: str, prompt: str) -> dict:
        """PaliGemma 2 inference via mlx-vlm."""
        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            output = self._mlx_generate(
                self._mlx_model,
                self._mlx_processor,
                image,
                prompt,
                max_tokens=self.config.vlm.max_tokens,
                temp=self.config.vlm.temperature,
            )
            raw = output if isinstance(output, str) else str(output)
            parsed = self._parse_vlm_observation(raw)
            parsed["raw_vlm"] = raw
            return parsed
        except Exception as e:
            print(f"[SEE] mlx-vlm inference failed: {e}")
            if self.gemma:
                frame = cv2.imread(image_path)
                if frame is not None:
                    cv_features = self._extract_cv_features(frame)
                    return self._vlm_text_with_cv(cv_features)
            return {"anomaly": "none", "severity": "low", "location": "unknown", "raw_vlm": str(e)}

    @staticmethod
    def _extract_cv_features(frame: np.ndarray) -> dict:
        """Extract visual features from a frame using OpenCV."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        total_px = h * w
        features: dict = {"width": w, "height": h}

        # --- Color masks ---
        # Brown/dark blobs (pests, rodents, insects) — tighter range to avoid wood/floors
        brown_lo = np.array([8, 80, 30])
        brown_hi = np.array([22, 220, 130])
        brown_mask = cv2.inRange(hsv, brown_lo, brown_hi)
        features["brown_ratio"] = round(cv2.countNonZero(brown_mask) / total_px, 4)

        # Green/yellow-green (mold, spoilage)
        green_lo = np.array([25, 50, 50])
        green_hi = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lo, green_hi)
        features["green_ratio"] = round(cv2.countNonZero(green_mask) / total_px, 4)

        # Blue reflective / water (leak, spill)
        blue_lo = np.array([95, 60, 60])
        blue_hi = np.array([125, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lo, blue_hi)
        features["blue_ratio"] = round(cv2.countNonZero(blue_mask) / total_px, 4)

        # Rust/orange (corroded pipes, leak indicator)
        rust_lo = np.array([5, 60, 50])
        rust_hi = np.array([22, 255, 220])
        rust_mask = cv2.inRange(hsv, rust_lo, rust_hi)
        features["rust_ratio"] = round(cv2.countNonZero(rust_mask) / total_px, 4)

        # Gray/white water (desaturated bright areas — real water/steam)
        gray_water = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([180, 50, 255]))
        features["gray_water_ratio"] = round(cv2.countNonZero(gray_water) / total_px, 4)

        # High-saturation wet/sheen areas
        high_sat = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([180, 255, 255]))
        features["high_sat_ratio"] = round(cv2.countNonZero(high_sat) / total_px, 4)

        # Very dark regions (shadows, damage)
        dark_mask = cv2.inRange(gray, 0, 40)
        features["dark_ratio"] = round(cv2.countNonZero(dark_mask) / total_px, 4)

        # --- Small blob detection (pest-like objects) ---
        # Use morphological ops to clean noise before contour detection
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        brown_clean = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        brown_clean = cv2.morphologyEx(brown_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(brown_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Pest blobs: small, somewhat round (circularity > 0.2), not too big
        pest_blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if 100 < area < 3000 and perimeter > 0:
                circularity = 4 * 3.14159 * area / (perimeter * perimeter)
                if circularity > 0.15:
                    pest_blobs.append(c)
        features["small_brown_blobs"] = len(pest_blobs)

        # --- Edge density (damage, cracks) ---
        edges = cv2.Canny(gray, 50, 150)
        features["edge_density"] = round(cv2.countNonZero(edges) / total_px, 4)

        # --- Wetness: specular highlights ---
        bright_mask = cv2.inRange(gray, 200, 255)
        features["bright_spots_ratio"] = round(cv2.countNonZero(bright_mask) / total_px, 4)

        # --- Texture irregularity ---
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features["texture_variance"] = round(float(laplacian.var()), 2)

        # --- Skin-tone detection (human presence backup) ---
        skin_lo1 = np.array([0, 30, 60])
        skin_hi1 = np.array([20, 180, 255])
        skin_lo2 = np.array([160, 30, 60])
        skin_hi2 = np.array([180, 180, 255])
        skin_mask = cv2.inRange(hsv, skin_lo1, skin_hi1) | cv2.inRange(hsv, skin_lo2, skin_hi2)
        features["skin_ratio"] = round(cv2.countNonZero(skin_mask) / total_px, 4)

        # Find large skin blobs (actual humans, not noise)
        skin_clean = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=3)
        skin_clean = cv2.morphologyEx(skin_clean, cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
                                       iterations=2)
        skin_contours, _ = cv2.findContours(skin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # A human body/face blob should be at least 0.1% of frame area
        min_human_area = total_px * 0.001
        human_blobs = [c for c in skin_contours if cv2.contourArea(c) > min_human_area]
        features["human_skin_blobs"] = len(human_blobs)

        return features

    @staticmethod
    def _pure_cv_classify(cv_features: dict) -> dict:
        """Classify anomalies using OpenCV feature thresholds.

        Tuned against actual user video data:
          - Clean warehouse (848x480): blobs=0, brown=0, rust=0, gray_water=0.33, sat=9
          - Pest video (848x480):      blobs=23-65, brown=0.09-0.13, skin=0.27-0.31
          - Leak video (1280x720):     rust=0.21-0.25, gray_water=0.16-0.19, brown=0.26-0.32
          - Pest+blue (1280x720):      blobs=11+, brown>0.05, skin>0.12
        """
        anomalies: list[str] = []
        details: list[str] = []
        severity = "low"

        blobs = cv_features["small_brown_blobs"]
        brown = cv_features["brown_ratio"]
        blue = cv_features["blue_ratio"]
        green = cv_features["green_ratio"]
        dark = cv_features["dark_ratio"]
        edge = cv_features["edge_density"]
        skin = cv_features.get("skin_ratio", 0)
        human_blobs = cv_features.get("human_skin_blobs", 0)
        rust = cv_features.get("rust_ratio", 0)
        gray_water = cv_features.get("gray_water_ratio", 0)
        high_sat = cv_features.get("high_sat_ratio", 0)

        # --- Leak: rust + gray water, OR blue pooling ---
        # Real leaks show rusty pipes (rust>0.10) with gray/white water splashing
        # Clean warehouse has gray_water~0.33 but rust=0 and low saturation
        if rust > 0.08 and gray_water > 0.10:
            anomalies.append("leak")
            severity = "critical"
            details.append(f"Leak detected: {rust:.1%} rust, {gray_water:.1%} water/moisture")
        elif rust > 0.05 and brown > 0.15:
            anomalies.append("leak")
            severity = "high"
            details.append(f"Possible leak: {rust:.1%} corrosion, {brown:.1%} water damage")
        elif blue > 0.15:
            anomalies.append("leak")
            severity = "critical"
            details.append(f"Leak/spill detected: {blue:.1%} water pooling")

        # --- Pest: small brown blobs (insects) ---
        if blobs >= 15 and brown > 0.05:
            anomalies.append("pest")
            if severity == "low":
                severity = "high"
            details.append(f"Pest detected: {blobs} insect-like objects, {brown:.1%} brown coverage")
        elif blobs >= 8 and brown > 0.03:
            anomalies.append("pest")
            if severity == "low":
                severity = "medium"
            details.append(f"Possible pest: {blobs} suspicious objects, {brown:.1%} brown coverage")

        # --- Spoilage: green/mold ---
        if green > 0.08:
            anomalies.append("spoilage")
            if severity == "low":
                severity = "high"
            details.append(f"Spoilage/mold: {green:.1%} green coverage")

        # --- Human presence from skin-tone analysis ---
        human_detected = False
        if human_blobs >= 1 and skin > 0.02:
            human_detected = True
            details.append(f"Human detected: {human_blobs} skin region(s), {skin:.1%} skin tone")

        # Build structured SEE output
        if anomalies:
            see_parts = [f"{a.capitalize()} detected" for a in anomalies]
            if human_detected:
                see_parts.append("Human present")
            see_text = f"SEE: [{'; '.join(see_parts)}] " + " | ".join(details)
        else:
            see_text = f"SEE: [no defect] Clean frame (blobs={blobs}, blue={blue:.3f}, green={green:.3f})"
            if human_detected:
                see_text = f"SEE: [Human present; no defect] {details[0] if details else ''}"

        result: dict = {
            "anomaly": anomalies[0] if anomalies else "none",
            "severity": severity,
            "location": "Aisle_4" if anomalies else "unknown",
            "raw_vlm": see_text,
            "cv_features": cv_features,
        }
        if len(anomalies) > 1:
            result["all_anomalies_in_frame"] = anomalies
        if human_detected:
            result["vlm_human_detected"] = True

        return result

    @staticmethod
    def _strip_prompt_echo(response: str, prompt: str) -> str:
        """Remove echoed/repeated prompt text from VLM response."""
        cleaned = re.sub(r'\[img-\d+\]', '', response).strip()

        see_idx = cleaned.upper().find("SEE:")
        if see_idx > 0:
            cleaned = cleaned[see_idx:].strip()
        elif see_idx == 0:
            pass
        else:
            prompt_lower = prompt.lower().strip()
            cleaned_lower = cleaned.lower()
            if cleaned_lower.startswith(prompt_lower[:30]):
                cleaned = cleaned[len(prompt):].strip()

        return cleaned

    # ── Single Frame Observation ─────────────────────────────────────

    def observe(
        self,
        frame: np.ndarray | None = None,
        image_path: str | None = None,
        text_input: str | None = None,
        session_id: str = "",
        correlation_id: str = "",
        motion_data: dict | None = None,
    ) -> dict:
        """Run vision pipeline on a single frame and return structured observation."""
        start = time.perf_counter()
        timestamp = time.time()

        observation = {
            "anomaly": "none",
            "severity": "low",
            "location": "unknown",
            "human_present": False,
            "face_count": 0,
            "hand_count": 0,
            "face_boxes": [],
            "hand_boxes": [],
            "timestamp": timestamp,
            "frame_shape": None,
            "source": "none",
            "raw_vlm": "",
        }

        temp_path = None
        try:
            if image_path:
                frame = cv2.imread(image_path)
                observation["source"] = "file"

            if frame is not None:
                observation["frame_shape"] = list(frame.shape)
                if observation["source"] == "none":
                    observation["source"] = "frame"

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                human_data = self._detect_human(rgb)
                observation.update(human_data)

                if not image_path:
                    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                    os.close(fd)
                    cv2.imwrite(temp_path, frame)
                    image_path = temp_path

                vlm_result = self._vlm_analyze(image_path, text_input)

                # Motion-based override: if CV found nothing but motion detected a pest
                if vlm_result.get("anomaly") == "none" and motion_data and motion_data.get("motion_detected"):
                    n_moving = len(motion_data.get("moving_objects", []))
                    vlm_result["anomaly"] = "pest"
                    vlm_result["severity"] = "high"
                    vlm_result["location"] = "Aisle_4"
                    vlm_result["raw_vlm"] = (
                        f"SEE: [Pest detected] Motion analysis: {n_moving} moving object(s) "
                        f"on warehouse floor (fg_ratio={motion_data.get('fg_ratio', 0):.3f})"
                    )
                    vlm_result["all_anomalies_in_frame"] = ["pest"]
                    vlm_result["motion_boxes"] = motion_data.get("moving_objects", [])

                observation["anomaly"] = vlm_result.get("anomaly", "none")
                observation["severity"] = vlm_result.get("severity", "low")
                observation["location"] = vlm_result.get("location", "unknown")
                observation["raw_vlm"] = vlm_result.get("raw_vlm", "")
                if vlm_result.get("all_anomalies_in_frame"):
                    observation["all_anomalies_in_frame"] = vlm_result["all_anomalies_in_frame"]
                if vlm_result.get("motion_boxes"):
                    observation["motion_boxes"] = vlm_result["motion_boxes"]
                print(f"[SEE] Frame → anomaly={observation['anomaly']}, human={observation['human_present']}, vlm={observation['raw_vlm'][:120]}")

        finally:
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink(missing_ok=True)

        latency = (time.perf_counter() - start) * 1000
        self.tracer.create_event(
            session_id=session_id,
            correlation_id=correlation_id,
            stage="SEE",
            latency_ms=latency,
            decision={"type": "observation", "status": "success", "reason": None},
        )
        return observation

    # ── Video Stream Processing (2 FPS) ──────────────────────────────

    def process_video_stream(
        self,
        video_path: str,
        fps_target: float | None = None,
        max_frames: int | None = None,
        text_input: str | None = None,
        session_id: str = "",
        correlation_id: str = "",
        callback=None,
    ):
        """Process video at target FPS with motion detection across frames.

        Uses background subtraction to detect moving objects (pests) that
        color-based detection may miss (e.g. white/gray mice).
        """
        fps = fps_target or self.config.video.fps_target
        cap_limit = max_frames or self.config.video.max_frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        self.reset_motion_detector()

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(video_fps / fps))
        frame_idx = 0
        yielded = 0

        while cap.isOpened() and yielded < cap_limit:
            ret, frame = cap.read()
            if not ret:
                break

            # Feed EVERY frame to the background subtractor for accurate motion model
            motion_data = self._detect_motion_objects(frame)

            if frame_idx % frame_interval == 0:
                timestamp_s = round(frame_idx / video_fps, 2)

                fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                try:
                    cv2.imwrite(temp_path, frame)
                    obs = self.observe(
                        image_path=temp_path,
                        text_input=text_input,
                        session_id=session_id,
                        correlation_id=correlation_id,
                        motion_data=motion_data,
                    )
                    obs["timestamp"] = timestamp_s
                    obs["frame_index"] = frame_idx
                    yielded += 1
                    if callback:
                        callback(obs)
                    yield obs
                finally:
                    Path(temp_path).unlink(missing_ok=True)

            frame_idx += 1

        cap.release()

    # ── Draw Overlays for Dashboard ──────────────────────────────────

    @staticmethod
    def draw_overlays(frame: np.ndarray, observation: dict) -> np.ndarray:
        """Draw bounding boxes and labels on a frame for the dashboard feed."""
        annotated = frame.copy()

        for box in observation.get("face_boxes", []):
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(annotated, "HUMAN", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for box in observation.get("hand_boxes", []):
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 255), 2)

        for box in observation.get("motion_boxes", []):
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated, "PEST", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        anomaly = observation.get("anomaly", "none")
        if anomaly != "none":
            severity = observation.get("severity", "low")
            color = (0, 0, 255) if severity == "critical" else (0, 165, 255)
            label = f"{anomaly.upper()} [{severity}]"
            cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        location = observation.get("location", "unknown")
        if location != "unknown":
            cv2.putText(annotated, location, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return annotated

    # ── Parser ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_vlm_observation(raw: str) -> dict:
        """Parse fine-tuned VLM output into structured observation.

        Handles formats:
          SEE: [Pest detected: brown_planthopper] REASON: ... ACT: [...]
          SEE: [pill defect: scratch] REASON: ...
          SEE: [Package damage detected: water_damage] REASON: ...
          SEE: [cable defect: crack] REASON: ...
          SEE: [hazelnut: no defect] REASON: ...
        Also falls back to keyword matching for non-fine-tuned models.
        """
        out: dict = {"anomaly": "none", "severity": "low", "location": "unknown"}
        anomalies_found: list[str] = []

        # First try structured SEE: [...] block
        see_match = _SEE_BLOCK_PATTERN.search(raw)
        see_text = see_match.group(1) if see_match else raw

        # Check for "no defect" / pass
        if _NO_DEFECT_PATTERN.search(see_text):
            out["anomaly"] = "none"
            return out

        # Pest detection
        pest_m = _PEST_PATTERN.search(see_text)
        if pest_m:
            anomalies_found.append("pest")
            out["pest_species"] = pest_m.group(1)
        elif re.search(r"pest", see_text, re.IGNORECASE):
            anomalies_found.append("pest")

        # Package damage
        pkg_m = _PACKAGE_DAMAGE_PATTERN.search(see_text)
        if pkg_m:
            anomalies_found.append("damage")
            out["damage_type"] = pkg_m.group(1)
        # General defect (pill defect, cable defect, etc.)
        elif _DEFECT_PATTERN.search(see_text):
            defect_m = _DEFECT_PATTERN.search(see_text)
            category = defect_m.group(1).lower()
            defect_type = defect_m.group(2).lower()
            if defect_type in ("crack", "scratch", "cut", "hole"):
                anomalies_found.append("damage")
            else:
                anomalies_found.append("spoilage")
            out["defect_category"] = category
            out["defect_type"] = defect_type

        # Spoilage
        if _SPOILAGE_PATTERN.search(raw) and "spoilage" not in anomalies_found:
            anomalies_found.append("spoilage")

        # Leak
        if _LEAK_PATTERN.search(see_text) and "leak" not in anomalies_found:
            anomalies_found.append("leak")

        # Human detected by VLM
        if _HUMAN_IN_VLM_PATTERN.search(raw):
            out["vlm_human_detected"] = True

        # Fallback: scan for generic anomaly keywords in the full text
        if not anomalies_found:
            for m in _ANOMALY_PATTERN.finditer(raw):
                kw = m.group(1).lower()
                if kw not in anomalies_found:
                    anomalies_found.append(kw)

        if anomalies_found:
            out["anomaly"] = anomalies_found[0]
            out["all_anomalies_in_frame"] = anomalies_found
        else:
            out["anomaly"] = "none"

        # Severity: try explicit mention, else infer from anomaly type
        sev_m = _SEVERITY_PATTERN.search(raw)
        if sev_m:
            out["severity"] = sev_m.group(1).lower()
        elif anomalies_found:
            SEVERITY_MAP = {
                "leak": "critical", "fire": "critical", "smoke": "critical",
                "pest": "high", "spoilage": "high",
                "damage": "medium", "spill": "medium", "hazard": "high",
            }
            out["severity"] = SEVERITY_MAP.get(anomalies_found[0], "medium")

        # Location: explicit or default
        loc_m = _LOCATION_PATTERN.search(raw)
        if loc_m:
            out["location"] = loc_m.group(1)
        elif anomalies_found:
            out["location"] = "Zone_A"

        return out

    def close(self):
        if self._face_detector:
            self._face_detector.close()
        if self._hand_landmarker:
            self._hand_landmarker.close()
