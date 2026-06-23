#!/usr/bin/env python3
"""Detect yellow-coloured objects from a webcam feed and mark their region of interest.

The algorithm:
  1. Grab a frame from the webcam (BGR colour space).
  2. Convert it to HSV, where colour (hue) is separated from brightness/shading,
     making a single colour far easier to threshold than in RGB.
  3. Build a binary mask of every pixel whose hue/saturation/value falls inside the
     yellow range.
  4. Clean the mask with morphological open/close to drop speckle noise and fill holes.
  5. Find contours in the mask, keep the ones above a minimum area, and draw an
     axis-aligned bounding box (the region of interest) around each.

Usage:
    python3 scripts/yellow_detection.py                  # default webcam (index 0)
    python3 scripts/yellow_detection.py --camera 1       # pick another camera
    python3 scripts/yellow_detection.py --min-area 1500  # ignore smaller blobs

Controls (while the window is focused):
    q / Esc   quit
    m         toggle the binary mask view on/off
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

# Yellow occupies roughly hue 20-35 on OpenCV's 0-179 hue scale. We also require a
# reasonably high saturation and value so dull/dark yellows (and white) are excluded.
LOWER_YELLOW = np.array([20, 100, 100], dtype=np.uint8)
UPPER_YELLOW = np.array([35, 255, 255], dtype=np.uint8)


def detect_yellow(frame: np.ndarray, min_area: float) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Return the cleaned yellow mask and a list of (x, y, w, h) ROIs for the frame."""
    # Optional blur smooths sensor noise so the mask has cleaner edges.
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Pixels inside the yellow band become white (255), everything else black (0).
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    # Morphology: OPEN removes tiny noise specks, CLOSE fills small holes in blobs.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Contours trace the outline of each connected yellow region.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois: list[tuple[int, int, int, int]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue  # skip blobs too small to be the object of interest
        rois.append(cv2.boundingRect(contour))  # (x, y, w, h)

    return mask, rois


def annotate(frame: np.ndarray, rois: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Draw bounding boxes + centroids for each ROI onto a copy of the frame."""
    annotated = frame.copy()
    for index, (x, y, w, h) in enumerate(rois):
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy = x + w // 2, y + h // 2
        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
        label = f"yellow #{index + 1}  ({w}x{h})"
        cv2.putText(
            annotated, label, (x, max(y - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )

    status = f"objects: {len(rois)}"
    cv2.putText(annotated, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="Yellow-object detector using a webcam.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument("--min-area", type=float, default=800.0, help="Minimum contour area in px to count (default: 800).")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam at index {args.camera}.")

    print("Press 'q' or Esc to quit, 'm' to toggle the mask view.")
    show_mask = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera; stopping.")
                break

            mask, rois = detect_yellow(frame, args.min_area)
            annotated = annotate(frame, rois)

            cv2.imshow("Yellow detection", annotated)
            if show_mask:
                cv2.imshow("Mask", mask)
            elif cv2.getWindowProperty("Mask", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Mask")

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            if key == ord("m"):
                show_mask = not show_mask
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
