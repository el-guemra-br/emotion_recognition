from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import cv2
from fer import FER


LOGGER = logging.getLogger("emotion_recognition")


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    use_mtcnn: bool = False
    mirror: bool = True
    show_fps: bool = True
    confidence_threshold: float = 0.0
    window_name: str = "Emotion Recognition"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emotion-recognition",
        description="Run real-time facial emotion recognition on a webcam feed.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera device index to open.",
    )
    parser.add_argument(
        "--use-mtcnn",
        action="store_true",
        help="Enable MTCNN-based face detection if the optional dependency is installed.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable the mirrored selfie view.",
    )
    parser.add_argument(
        "--no-fps",
        action="store_true",
        help="Hide the FPS overlay.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Skip labels below this confidence threshold.",
    )
    return parser


def create_detector(use_mtcnn: bool) -> FER:
    if use_mtcnn:
        try:
            __import__("mtcnn")
        except ImportError as exc:
            raise RuntimeError(
                "MTCNN support is not installed. Install the optional dependency with 'pip install .[mtcnn]' "
                "or run without '--use-mtcnn'."
            ) from exc

    return FER(mtcnn=use_mtcnn)


def clamp_box(box: list[int] | tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x, y, box_width, box_height = box
    x = max(0, x)
    y = max(0, y)
    right = min(width, x + max(0, box_width))
    bottom = min(height, y + max(0, box_height))

    if right <= x or bottom <= y:
        return None

    return x, y, right - x, bottom - y


def annotate_frame(frame, detections, confidence_threshold: float) -> None:
    frame_height, frame_width = frame.shape[:2]

    for detection in detections:
        box = clamp_box(detection.get("box", (0, 0, 0, 0)), frame_width, frame_height)
        if box is None:
            continue

        emotions = detection.get("emotions") or {}
        if not emotions:
            continue

        emotion, score = max(emotions.items(), key=lambda item: item[1])
        if score < confidence_threshold:
            continue

        x, y, box_width, box_height = box
        label = f"{emotion} ({score:.2f})"
        label_y = max(25, y - 10)

        cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), (60, 220, 120), 2)
        cv2.putText(
            frame,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )


def overlay_status(frame, text: str) -> None:
    cv2.putText(
        frame,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )


def run(config: AppConfig) -> None:
    detector = create_detector(config.use_mtcnn)
    capture = cv2.VideoCapture(config.camera_index)

    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera index {config.camera_index}.")

    frame_count = 0
    fps = 0.0
    last_fps_update = time.monotonic()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                LOGGER.warning("Camera frame could not be read; stopping the session.")
                break

            if config.mirror:
                frame = cv2.flip(frame, 1)

            detections = detector.detect_emotions(frame)
            annotate_frame(frame, detections, config.confidence_threshold)

            frame_count += 1
            now = time.monotonic()
            elapsed = now - last_fps_update
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                last_fps_update = now

            if config.show_fps:
                overlay_status(frame, f"FPS: {fps:.1f} | Press Q to quit")
            else:
                overlay_status(frame, "Press Q to quit")

            cv2.imshow(config.window_name, frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args() -> AppConfig:
    parser = build_parser()
    args = parser.parse_args()
    return AppConfig(
        camera_index=args.camera_index,
        use_mtcnn=args.use_mtcnn,
        mirror=not args.no_mirror,
        show_fps=not args.no_fps,
        confidence_threshold=max(0.0, args.confidence_threshold),
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        run(parse_args())
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user.")
        return 0

    return 0
