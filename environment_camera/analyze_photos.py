"""Run existing YOLO models on still photographs WITHOUT a connected camera.

Usage (PowerShell, from environment_camera):
    python analyze_photos.py C:\\Photos\\test_scene.jpg
    python analyze_photos.py C:\\Photos\\scene1.jpg C:\\Photos\\scene2.jpg

Photographs captured from DIFFERENT angles cannot be used as scene-change
reference/current pairs. This script runs known-label detection only.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2

from config import DATA_DIR
from vision import OverheadDetector, resize


def main():
    parser = argparse.ArgumentParser(description='안전용품 YOLO 사진 인식(카메라 연결 불필요)')
    parser.add_argument('images', nargs='+', help='시험할 사진 경로 여러 개')
    args = parser.parse_args()

    detector = OverheadDetector()
    if not detector.models:
        raise SystemExit(f'YOLO 모델을 불러오지 못했습니다. {detector.model_error}')
    output = DATA_DIR / 'photo_tests' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=True)
    report = {'models': [m['path'] for m in detector.models], 'results': []}

    for index, source in enumerate(args.images, 1):
        path = Path(source)
        # imdecode handles Korean / Unicode Windows paths better than imread.
        import numpy as np
        try:
            frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            frame = None
        if frame is None:
            print(f'건너뜀 - 읽을 수 없음: {path}')
            continue
        img = resize(frame)
        detector.frame_no = 0
        known = detector.known_objects(img)
        annotated = img.copy()
        for item in known:
            x1, y1, x2, y2 = item['box']
            color = (255, 165, 50) if item['role'] == 'robot' else (118, 210, 112)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{item['label']} {item['conf']:.2f}",
                        (x1, max(16, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
        output_name = f'{index:02d}_{path.stem[:48]}_detected.jpg'
        ok, encoded = cv2.imencode('.jpg', annotated)
        if not ok:
            print('프리뷰 인코딩 실패:', path)
            continue
        encoded.tofile(str(output / output_name))
        report['results'].append({'image': str(path), 'preview': output_name,
                                  'objects': known, 'model_error': detector.model_error})
        names = ', '.join(f"{i['label']}({i['conf']:.2f})" for i in known) or '탐지 없음'
        print(f'{path.name}: {names}')
    (output / 'results.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n결과 폴더: {output}')
    print('주의: 로봇 전용 학습 모델 없이 로봇 자동 구분은 불가능합니다.')
    print('서로 각도가 다른 사진을 UNKNOWN 배경차이 비교에 사용하지 마세요.')


if __name__ == '__main__':
    main()
