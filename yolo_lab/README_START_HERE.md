# HazardScout-AI / YOLO-only starter pack

> Standalone training/testing files. **Do not overwrite** code/YOLO_detector.py,
> GUIServer_custom.py, GUIServer_mobile.py, robot_gui.py, OLED_eyes.py or any team code.

## 1. Installation on Windows laptop (PowerShell / VS Code terminal)

Unzip this package into your existing `HazardScout-AI/` root so it becomes `HazardScout-AI/yolo_lab/`.

```powershell
cd yolo_lab
python -m pip install -r requirements-laptop.txt
python prepare_models.py
```

- Internet is required the first time to download the pretrained weights and possibly prompt encoder dependencies.
- `models/yolov8n.pt`: existing 80-class baseline model.
- `models/world_hazard.pt`: YOLO-World zero-shot model with custom **text prompts** baked into weights. **This is not custom training.**
- If YOLO-World setup fails, baseline can still be tested; keep the error message for debugging.
- Optional: `python prepare_models.py --with-yoloe` downloads YOLOE prompt-free; test only after confirming sufficient resources.

## 2. Test with laptop camera or with images/videos

```powershell
python test_yolo.py --mode baseline --source 0
python test_yolo.py --mode world --source 0
python test_yolo.py --mode world --source path/to/test.jpg
python test_yolo.py --mode world --source path/to/test.mp4 --save
```

- In a preview press **q** to quit and **s** for a screenshot.
- Outputs: `results/<run>/detections.csv`, optional screenshots and annotated video.
- For no display, add `--no-show --max-frames 100` for webcam/video.
- For accurate recognition speed, use `--every 1`. Pi defaults to `--every 3` for performance.
- A skipped inference frame is deliberately **not** drawn with stale object boxes.

## 3. Fine-tune a custom model (REQUIRES LABELED IMAGES)

Arrange an Ultralytics YOLO detection dataset:

```
dataset/
  data.yaml
  images/train/    (*.jpg, *.png)
  images/val/
  labels/train/    (one same-stem .txt per labeled image)
  labels/val/
```

Each label line: `class_id x_center y_center width height`, normalized to 0..1.
Ensure class IDs exactly match `dataset/data.yaml`. The YAML is a **template** only. Add/remove
classes to match your actual labeled dataset. Train/val must be distinct scenes where feasible.

```powershell
python train_yolo.py --model models/yolov8n.pt --epochs 50 --batch 8 --imgsz 640 --name hazard_v1
python test_yolo.py --mode custom --source 0
```

Train automatically exports `runs/hazard_v1/weights/best.pt` to `models/hazard_v1.pt`.
If training on CPU is extremely slow or runs out of memory, use fewer batch samples or Colab GPU.

**Warning:** Training on only helmet, fire extinguisher, cone, etc. will NOT preserve existing COCO 80-class detection automatically. To ship a SINGLE unified model, prepare a properly relabeled joint dataset including the general classes you wish to retain, and validate on both class families. Until then, keep baseline and zero-shot separately.

## 4. Tomorrow on Raspberry Pi 4 (optional, without the team's server)

Copy entire `yolo_lab/` folder, especially **models/*.pt**, into the Raspberry Pi,
then use the environment where **Ultralytics, OpenCV, Picamera2** are already available.

**Stop other camera processes/server first.** Only one application should operate the Pi camera at a time. This test does NOT control servos/motors/GPIO.

```bash
cd /path/to/HazardScout-AI/yolo_lab
python3 test_pi_camera.py --mode baseline --max-frames 120
python3 test_pi_camera.py --mode world --max-frames 120
# Only after a real labeled training run:
python3 test_pi_camera.py --mode custom --max-frames 120
```

If there is no Pi desktop (SSH only), add `--no-show`.
World may be too slow on Pi 4. Start with baseline. Stop the process with Ctrl+C.
Save `results/pi_.../` images and timing CSV for team review.

## 5. Later integration with team code

Current servers already construct `YOLODetector(model_path=YOLO_MODEL_PATH, ...)`,
so the team member owning the server can change **their** YOLO_MODEL_PATH to an absolute path
of `models/hazard_v1.pt` or `models/world_hazard.pt` **after** verifying compatibility. The YOLO detector expects an Ultralytics compatible model; model prompts for world are saved in `world_hazard.pt`.

No team server, OLED, motor or GUI changes are included in this package.

## 6. Checklist before leaving home

- [ ] `models/yolov8n.pt` exists
- [ ] `models/world_hazard.pt` exists
- [ ] `python test_yolo.py --mode world --source path/to/image.jpg` works if you have an image
- [ ] `models/hazard_v1.pt` exists **only if a labeled dataset was actually trained**
- [ ] Zip/copy `yolo_lab/` with pretrained models so tomorrow does not depend on school Wi-Fi
- [ ] Server-owning teammate knows a standalone Pi camera test cannot run simultaneously with their GUI server
