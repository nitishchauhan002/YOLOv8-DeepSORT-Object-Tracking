import argparse
import cv2
import os
from collections import defaultdict

from detect         import YOLOv8Detector
from deep_sort      import DeepSORT


# ── Colour palette for unique track IDs ──────────────────────────────────────
def get_color(track_id: int):
    palette = [
        (255, 56,  56),  (255, 157, 151), (255, 112,  31),
        (255, 178, 29),  (207, 210,  49), (72,  249, 10),
        (146, 204,  23), (61,  219, 134), (26,  147, 52),
        (0,  212, 187),  (44,  153, 168), (0,  194, 255),
        (52,  69, 147),  (100,  115, 255),(0,   24, 236),
        (132,  56, 255), (82,   0, 133),  (203,  56, 255),
        (255, 149, 200), (255,  55, 199),
    ]
    return palette[track_id % len(palette)]


# ── Drawing helper ────────────────────────────────────────────────────────────
def draw_tracks(frame, tracks, trails):
    for track in tracks:
        x1, y1, x2, y2 = track.to_xyxy()
        tid   = track.track_id
        label = f"ID:{tid} {track.cls_name} {track.conf:.2f}"
        color = get_color(tid)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Motion trail (centroid history)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        trails[tid].append((cx, cy))
        if len(trails[tid]) > 40:
            trails[tid].pop(0)
        for i in range(1, len(trails[tid])):
            if trails[tid][i - 1] and trails[tid][i]:
                thickness = max(1, int(3 * (i / len(trails[tid]))))
                cv2.line(frame, trails[tid][i - 1], trails[tid][i], color, thickness)

    return frame


# ── Stats overlay ─────────────────────────────────────────────────────────────
def draw_stats(frame, n_tracks, frame_idx, fps):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, f"Active Tracks : {n_tracks}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
    cv2.putText(frame, f"Frame         : {frame_idx}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
    cv2.putText(frame, f"FPS           : {fps:.1f}", (10, 79),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 180), 2)
    return frame


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run(args):
    detector = YOLOv8Detector(
        model_path  = args.model,
        conf_thresh = args.conf,
        device      = args.device,
    )
    tracker = DeepSORT(max_age=30, n_init=3)

    # ── Open source ───────────────────────────────────────────────────────
    source = int(args.source) if args.source.isdigit() else args.source
    cap    = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {args.source}")
        return

    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # ── Output video writer ───────────────────────────────────────────────
    writer = None
    if args.save:
        os.makedirs("output", exist_ok=True)
        out_path = os.path.join("output", "tracked_output.mp4")
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
        print(f"[INFO] Saving output to: {out_path}")

    trails    = defaultdict(list)
    frame_idx = 0

    print("[INFO] Starting tracker — press 'q' to quit.\n")

    import time
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ── Detect ────────────────────────────────────────────────────────
        detections = detector.detect(frame)

        # ── Track ─────────────────────────────────────────────────────────
        active_tracks = tracker.update(detections)

        # ── FPS ───────────────────────────────────────────────────────────
        curr_time = time.time()
        disp_fps  = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time

        # ── Visualise ─────────────────────────────────────────────────────
        frame = draw_tracks(frame, active_tracks, trails)
        frame = draw_stats(frame, len(active_tracks), frame_idx, disp_fps)

        if writer:
            writer.write(frame)

        cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[INFO] Quit signal received.")
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Tracking complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 + DeepSORT Object Tracker")

    parser.add_argument("--source", type=str,  default="0",
                        help="Video file path or '0' for webcam")
    parser.add_argument("--model",  type=str,  default="weights/yolov8n.pt",
                        help="Path to YOLOv8 .pt weights")
    parser.add_argument("--conf",   type=float, default=0.4,
                        help="Detection confidence threshold")
    parser.add_argument("--device", type=str,  default="cpu",
                        help="Inference device: 'cpu' or 'cuda'")
    parser.add_argument("--save",   action="store_true",
                        help="Save annotated output video to output/")

    args = parser.parse_args()
    run(args)
