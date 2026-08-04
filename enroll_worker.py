"""
Enroll authorized lab workers by face -- run this once (or whenever you
add/remove someone) BEFORE running the main world model.

Usage:
  python enroll_worker.py "Dr. Ali Murtaza"
  python enroll_worker.py "Shariq"

Looks at your RealSense color stream, waits for a clear face, captures
5 samples for robustness, averages the embedding, saves it to
known_workers.yaml under that name. Press 'c' to capture a sample once
your face is clearly framed, 'q' to quit early.
"""

import sys
import cv2
import numpy as np
import yaml
import pyrealsense2 as rs
from insightface.app import FaceAnalysis

KNOWN_WORKERS_YAML = "known_workers.yaml"
SAMPLES_NEEDED = 5


def load_registry():
    try:
        with open(KNOWN_WORKERS_YAML) as f:
            data = yaml.safe_load(f) or {}
        return {k: np.array(v) for k, v in data.items()}
    except FileNotFoundError:
        return {}


def save_registry(registry):
    with open(KNOWN_WORKERS_YAML, "w") as f:
        yaml.dump({k: v.tolist() for k, v in registry.items()}, f)


def main():
    if len(sys.argv) < 2:
        print('Usage: python enroll_worker.py "Person Name"')
        sys.exit(1)
    name = sys.argv[1]

    print("Loading InsightFace...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(320, 320))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    samples = []
    print(f"Enrolling '{name}'. Face the camera. Press 'c' to capture ({SAMPLES_NEEDED} needed), 'q' to quit.")

    try:
        while len(samples) < SAMPLES_NEEDED:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())

            faces = app.get(image)
            display = image.copy()
            if faces:
                f = faces[0]
                box = f.bbox.astype(int)
                cv2.rectangle(display, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(display, "Face detected - press 'c'", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No face detected", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(display, f"Samples: {len(samples)}/{SAMPLES_NEEDED}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Enroll Worker", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and faces:
                samples.append(faces[0].normed_embedding)
                print(f"Captured sample {len(samples)}/{SAMPLES_NEEDED}")
            elif key == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if len(samples) == 0:
        print("No samples captured. Nothing saved.")
        return

    avg_embedding = np.mean(samples, axis=0)
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

    registry = load_registry()
    registry[name] = avg_embedding
    save_registry(registry)
    print(f"Saved '{name}' to {KNOWN_WORKERS_YAML} with {len(samples)} samples averaged.")


if __name__ == "__main__":
    main()
