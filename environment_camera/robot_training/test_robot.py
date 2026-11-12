"""Draw robot predictions on a saved photo; no camera required."""
import argparse
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parent

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('images',nargs='+',type=Path)
    p.add_argument('--model',type=Path,default=ROOT/'models'/'robot_v1.pt')
    p.add_argument('--conf',type=float,default=.30)
    a=p.parse_args()
    if not a.model.exists():
        raise SystemExit('No trained model! Run train_robot.py first (best.pt is not bundled).')
    from ultralytics import YOLO
    model=YOLO(str(a.model));out=ROOT/'results';out.mkdir(exist_ok=True)
    for path in a.images:
        image=cv2.imread(str(path))
        if image is None:
            print('SKIP unreadable:',path);continue
        res=model.predict(image,conf=a.conf,imgsz=640,verbose=False)[0]
        annotated=res.plot(); target=out/(path.stem+'_robot.jpg')
        cv2.imwrite(str(target),annotated)
        print(path.name, '=>',[(res.names[int(b.cls[0])],round(float(b.conf[0]),2)) for b in res.boxes], 'saved:',target)
