"""데이터 누출·누락 라벨·장치 반환을 검증합니다. 실제 모델 시험은 별도로 실행."""
import argparse
import contextlib
import csv
import io
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_utils import ROOT, output_path, model_path, inference_args
from prepare_dataset import read_boxes, build, check_dataset
from test_pi_camera import pi_frames


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "results/tests").mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "results/tests")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def make_image(self):
        from PIL import Image
        image = self.folder / "image.png"
        image_data = random.Random(42).randbytes(32*32*3)
        Image.frombytes("RGB", (32,32), image_data).save(image)
        return image

    def manifest(self, image, label, groups, splits):
        path = self.folder / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["image", "label", "source", "group", "split", "reviewed"])
            writer.writeheader()
            for group, split in zip(groups, splits):
                writer.writerow(dict(image=str(image),label=str(label),source="local",group=group,split=split,reviewed="yes"))
        return path

    def test_invalid_and_missing_boxes_stop_build_before_writing(self):
        image = self.make_image()
        label = self.folder / "label.txt"
        for text in ["7 .5 .5 .2 .2", "0 nan .5 .2 .2", "0 .01 .5 .8 .2", "0 .5 .5 .2"]:
            label.write_text(text)
            with self.assertRaises(ValueError):
                build(self.manifest(image,label,["scene"],["train"]), self.folder / "output")
            self.assertFalse((self.folder / "output").exists())
        label.unlink()  # Temporary test file inside yolo_lab/results only
        with self.assertRaises(ValueError):
            build(self.manifest(image,label,["scene"],["train"]), self.folder / "output")

    def test_scene_group_cannot_cross_splits(self):
        image = self.make_image(); label = self.folder / "label.txt"
        label.write_text("0 .5 .5 .2 .2\n")
        manifest = self.manifest(image,label,["scene01","scene01"],["train","test"])
        with self.assertRaisesRegex(ValueError, "split"):
            build(manifest, self.folder / "output")
        self.assertFalse((self.folder / "output").exists())

    def test_different_images_in_one_scene_stay_in_same_split(self):
        from PIL import Image
        image = self.make_image(); other = self.folder / "other.png"
        Image.frombytes("RGB", (32,32), random.Random(7).randbytes(32*32*3)).save(other)
        label = self.folder / "label.txt"; label.write_text("0 .5 .5 .2 .2\n")
        manifest = self.manifest(image,label,["scene01"],["train"])
        with manifest.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow([str(other),str(label),"local","scene01","train","yes"])
        with contextlib.redirect_stdout(io.StringIO()):
            build(manifest,self.folder / "output")
        report=json.loads((self.folder / "output/build_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["images"],2)

    def test_exact_duplicate_is_not_copied_into_test(self):
        image = self.make_image(); label = self.folder / "label.txt"
        label.write_text("0 .5 .5 .2 .2\n")
        manifest = self.manifest(image,label,["scene01","scene02"],["train","test"])
        with contextlib.redirect_stdout(io.StringIO()):
            build(manifest, self.folder / "output")
        report = json.loads((self.folder / "output/build_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["images"], 1)
        self.assertFalse(any((self.folder / "output/images/test").iterdir()))
        with self.assertRaises(ValueError):
            check_dataset(self.folder / "output")

    def test_missing_classes_cannot_start_seven_class_training(self):
        from PIL import Image
        import yaml
        from lab_utils import NAMES
        folder=self.folder / "partial"
        for i,split in enumerate(["train","val","test"]):
            (folder / "images" / split).mkdir(parents=True)
            (folder / "labels" / split).mkdir(parents=True)
            Image.frombytes("RGB",(32,32),random.Random(10+i).randbytes(32*32*3)).save(folder / "images" / split / "sample.png")
            (folder / "labels" / split / "sample.txt").write_text("0 .5 .5 .2 .2\n")
        (folder / "data.yaml").write_text(yaml.safe_dump({"names":dict(enumerate(NAMES)),"train":"images/train","val":"images/val","test":"images/test"}))
        with self.assertRaisesRegex(ValueError,"safety_gloves"):
            check_dataset(folder)

    def test_current_dataset_has_all_seven_classes_in_every_split(self):
        counts=check_dataset(ROOT / "dataset/safety7")
        for cls in range(7):
            self.assertGreaterEqual(sum(c[cls] for c in counts.values()),30)
            self.assertTrue(all(c[cls]>0 for c in counts.values()))

    def test_no_external_output_and_no_missing_custom_fallback(self):
        with self.assertRaises(ValueError):
            output_path("../outside")
        with self.assertRaises(ValueError):
            model_path("custom", str(self.folder / "missing.pt"))
        parser=argparse.ArgumentParser(); inference_args(parser)
        args=parser.parse_args([])
        self.assertEqual((args.imgsz,args.conf,args.every),(320,0.35,3))

    def test_picamera_color_order_and_release_on_generator_close(self):
        import numpy as np
        frame=np.array([[[3,2,1]]], dtype=np.uint8)
        class Camera:
            def __init__(self): self.closed=False; self.stopped=False
            def create_video_configuration(self, **kw):
                self.config=kw; return kw
            def configure(self, config): pass
            def start(self): pass
            def capture_array(self, name): return frame
            def stop(self): self.stopped=True
            def close(self): self.closed=True
        camera=Camera()
        with patch.dict(sys.modules,{"picamera2":types.SimpleNamespace(Picamera2=lambda:camera)}), patch("test_pi_camera.time.sleep"):
            stream=pi_frames()
            self.assertIs(next(stream),frame)
            self.assertEqual(camera.config["main"]["format"],"RGB888")
            stream.close()
        self.assertTrue(camera.stopped and camera.closed)


if __name__ == "__main__":
    unittest.main()
