# RealSense World Model → YAML Pipeline (SAFER Extension)

Real-time RGB-D object detection that builds a live 3D world model and
writes it to `world.yaml`, ready to feed into the SAFER Task/Safety
Planning LLM prompts.

---

## PART 1 — Camera Setup on Windows

### Step 1: Identify your camera
The camera in your photo is a small white USB stick module. Before
anything else, confirm the exact model:
- Plug it into a **USB 3.0 port** (blue port, or labeled "SS"/lightning
  bolt icon). RealSense cameras need USB 3.0 for full depth+color
  streaming — USB 2.0 ports will fail to start both streams at once.
- Once you install the SDK (Step 3), open **Intel RealSense Viewer** —
  it will show the exact model name (e.g., D415, D435, SR300) at the top.
  This matters because different models support different
  resolutions/frame rates.

### Step 2: Install Windows drivers
1. Go to: https://github.com/IntelRealSense/librealsense/releases
2. Download the latest **Intel.RealSense.SDK.exe** (Windows installer)
3. Run it — this installs:
   - USB drivers for the camera
   - **Intel RealSense Viewer** (GUI tool)
   - `librealsense` SDK (the C++/Python backend)
   - Example tools

### Step 3: Verify the camera works (no coding yet)
1. Plug in the camera
2. Open **Intel RealSense Viewer** from the Start Menu
3. Turn on the **Depth** and **RGB** stream toggles on the left panel
4. You should see a live color feed and a colorized depth feed
5. If nothing shows up:
   - Try a different USB 3.0 port
   - Check Windows Device Manager → look for "Intel(R) RealSense(TM)"
     under Cameras/Imaging devices — if it has a yellow warning icon,
     reinstall the driver from Step 2

Don't move on until this step works — it confirms the camera itself is
fine before we add any code.

### Step 4: Install Python
- Install **Python 3.10** (recommended — `pyrealsense2` wheels are most
  reliable on 3.9–3.11; avoid 3.12+ for now)
- During install, check "Add Python to PATH"
- Verify in Command Prompt:
  ```
  python --version
  ```

### Step 5: Set up a virtual environment and install packages
Open Command Prompt in the folder where you saved these files:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
This installs:
- `pyrealsense2` — Python bindings for the camera
- `ultralytics` — YOLOv8 for object detection
- `opencv-python` — image display/processing
- `numpy`, `pyyaml`

> If `pip install pyrealsense2` fails, it usually means your Python
> version doesn't have a matching wheel. Switch to Python 3.10 and
> retry.

### Step 6: Run the camera test script
```bash
python test_camera.py
```
You should see a window with the color feed side-by-side with a
colorized depth feed. Press `q` to quit. If this works, your camera +
Python setup is confirmed working end to end.

---

## PART 2 — Building the Real-Time World Model

### Step 7: Enroll authorized lab workers (once, before running the main pipeline)
The robot needs to know who it's allowed to work close to. Run this once
per authorized person (Dr. Ali Murtaza, lab assistants, etc.), in the
same lighting you'll demo in:
```bash
python enroll_worker.py "Dr. Ali Murtaza"
```
Face the camera, press `c` to capture (5 samples), `q` to quit early.
This saves an averaged face embedding to `known_workers.yaml`. Re-run
any time to add more people.

> Anyone NOT enrolled here is automatically treated as unauthorized by
> the pipeline — this is intentional (fail-safe default), not a bug.

### Step 8: Run the world model script
```bash
python world_model_v2.py
```

What happens:
1. Loads YOLO-World-L (open-vocabulary detector — detects anything in
   the `VOCAB` list by text description, no training needed)
2. Opens the camera, aligns depth to the color image
3. For every detected object:
   - Gets its pixel bounding box
   - Uses the depth at the box's center to compute its **3D position**
     in meters relative to the camera
   - Estimates rough width/height/depth from the bounding box size and
     depth (similar-triangles projection)
4. **Non-person objects**: tracked by class + spatial proximity only
   (no persistent per-instance identity beyond "this class near this
   position"). Gives you ids like `bottle_1`, `box_1`, etc.
5. **People**: tracked spatially frame-to-frame (so a person's id
   doesn't churn while they just stand or walk around normally), and
   periodically face-checked against `known_workers.yaml`.
   - Confident match → `authorized: true`, `name: "..."`
   - No face visible / no match → `authorized: false` (fail-safe
     default — always the case for anyone not enrolled)
6. Computes velocity from frame-to-frame position change, and marks
   objects `static: true/false` with hysteresis to avoid flicker on
   truly static objects.
7. Every 0.5 seconds, writes the current snapshot to `world.yaml` in
   compact flow-style YAML (fewer tokens for the LLM prompt)
8. A live window shows detections with IDs — people are boxed green
   (`authorized`) or red (`unauthorized`), with their name if known.

### Step 9: Inspect world.yaml
While the script is running, open `world.yaml` in a text editor — it
updates live. Example output (compact flow-style, to save LLM tokens):

```yaml
{timestamp: '2026-07-21 14:32:10', objects: [{id: bottle_1, class: bottle, position: [0.12, -0.08, 0.65], dimensions: [0.06, 0.18, 0.06], velocity: [0.0, 0.0, 0.0], static: true, visible: true, confidence: 0.87, last_seen: 0.1}, {id: person_1, class: person, position: [0.9, 0.1, 1.4], dimensions: [0.45, 1.6, 0.3], velocity: [0.12, 0.0, -0.05], static: false, visible: true, confidence: 0.91, last_seen: 0.1, authorized: true, name: 'Dr. Ali Murtaza'}]}
```

`authorized` / `name` only appear on `person` objects — use these to
key the robot's proximity behavior in the SAFER planning prompt.

---

## PART 3 — Notes and Next Steps

- **Coordinate frame**: positions are relative to the camera, in
  meters (X = right, Y = down, Z = forward, standard RealSense
  convention). If your robot's coordinate frame is different, you'll
  need a fixed transform (camera-to-robot-base) to convert — this is a
  one-time calibration step, worth doing once the camera is mounted in
  its final position.
- **Class labels**: YOLOv8-nano only knows COCO's 80 classes (person,
  bottle, chair, cup, etc.) — it won't recognize "table A" as a named
  object out of the box. For your specific lab objects (robot arms,
  specific props), you'll likely want to either:
  - Fine-tune YOLO on a small custom dataset of your lab's objects, or
  - Swap in an open-vocabulary detector (e.g., Grounding DINO) so you
    can detect by text prompt like `"spray can"` without retraining —
    slower per-frame but no training needed. I can help set this up
    once the basic pipeline above is working.
- **Robots and humans**: currently everything detected (including
  people) goes into the same `objects` list. For the safety-parameter
  extraction step, you'll likely want to split this into `humans`,
  `robots`, and `objects` sections in the YAML, and add role tags
  (e.g., `role: non_technical`) — that logic can be layered on top of
  this once detection is stable.
- **Performance**: YOLOv8-nano on CPU should give you several frames
  per second, which is enough for `UPDATE_INTERVAL = 0.5s` YAML
  updates. If you have an NVIDIA GPU, installing the CUDA build of
  PyTorch will make detection much faster.

Once this pipeline is producing clean YAML files, the next step is
writing the prompt template that feeds `world.yaml` into SAFER's Task
Planning and Safety Planning LLMs (following the Fig. 5 prompt
structure from the paper) — happy to help with that next.
