"""Train SEPARATE mobile_robot + robot_arm YOLO detector; never alters safety7.pt."""
import argparse
import csv
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent


def run(args):
    data=ROOT/'dataset'/'combined'/'data.yaml'
    manifest=ROOT/'dataset'/'combined'/'manifest.csv'
    if not data.exists() or not manifest.exists():
        raise SystemExit('First run: python robot_training/prepare_dataset.py --external PATH')
    with manifest.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    train_positive=sum(row['split']=='train' and int(row['num_boxes'])>0 for row in rows)
    val_positive=sum(row['split']=='val' and int(row['num_boxes'])>0 for row in rows)
    if train_positive < 20 and not args.allow_starter_only:
        raise SystemExit(f'Only {train_positive} positive training images. Download public data first, '
                         'or use --allow-starter-only ONLY for a pipeline smoke test (unreliable model).')
    if val_positive < 1:
        raise SystemExit('No positive validation images.')
    from ultralytics import YOLO
    local_baseline=ROOT.parent.parent/'yolo_lab'/'models'/'yolov8n.pt'
    base=str(local_baseline) if local_baseline.is_file() else 'yolov8n.pt'
    model=YOLO(base)
    model.train(data=str(data),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,
                device=args.device,workers=0,project=str(ROOT/'runs'),name=args.name,
                exist_ok=False,seed=42,patience=15,verbose=True)
    best=Path(model.trainer.best)
    if not best.exists():
        raise SystemExit('Training completed but best.pt was not found')
    target=ROOT/'models'/'robot_v1.pt'
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(best,target)
    print('\nRobot model:',target)
    print('Set in environment_camera/.env:')
    print('ROBOT_YOLO_MODEL=robot_training/models/robot_v1.pt')
    if train_positive < 20:
        print('WARNING: This is a smoke-test model trained on tiny data, not a validated detector.')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--device',default='cpu')
    p.add_argument('--name',default='robot_v1')
    p.add_argument('--allow-starter-only',action='store_true')
    run(p.parse_args())
