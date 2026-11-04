"""이미지/영상/USB 카메라 CPU 추론 또는 custom 검증·테스트 평가."""
import argparse
from pathlib import Path
from lab_utils import ROOT, inference_args, validate_inference_args


def opencv_frames(source):
    import cv2
    import numpy as np
    from PIL import Image, ImageOps, UnidentifiedImageError
    path = Path(source).expanduser()
    if path.is_file():
        try:
            with Image.open(path) as image:
                image.load()
                # Ultralytics numpy 입력은 BGR입니다. 확장자 없는 사용자 JPEG도 지원합니다.
                yield np.asarray(ImageOps.exif_transpose(image).convert("RGB"))[:, :, ::-1].copy()
            return
        except UnidentifiedImageError:
            pass  # 비디오 파일이면 VideoCapture로 처리
    cap = cv2.VideoCapture(int(source) if source.isdecimal() else str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"카메라/영상 열기 실패: {source}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inference_args(parser)
    parser.add_argument("--source", default="0", help="카메라 번호 또는 파일 경로; 상대 경로는 현재 디렉터리 기준")
    parser.add_argument("--evaluate", choices=["val", "test"], help="custom 7종 정량 평가")
    args = parser.parse_args()
    validate_inference_args(parser, args)
    if args.evaluate:
        if args.mode != "custom":
            parser.error("--evaluate는 --mode custom에서 사용하세요.")
        from prepare_dataset import check_dataset
        from lab_utils import load_model
        import yaml
        from datetime import datetime
        check_dataset(ROOT / "dataset/safety7")
        cfg = yaml.safe_load((ROOT / "dataset/safety7/data.yaml").read_text(encoding="utf-8"))
        cfg["path"] = str(ROOT / "dataset/safety7")
        folder = ROOT / "results" / ("eval_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        folder.mkdir(parents=True)
        data = folder / "data_runtime.yaml"
        data.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        model = load_model("custom", args.model)
        model.val(data=str(data), split=args.evaluate, imgsz=args.imgsz, device="cpu", workers=0,
                  project=str(folder), name=args.evaluate, plots=True)
        return
    from inference import run_frames
    run_frames(opencv_frames(args.source), args)


if __name__ == "__main__":
    main()
