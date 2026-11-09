#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_barcodes.py

site_map.json 에 있는 바코드를 인쇄용 PNG 라벨로 만든다.
- Code128 바코드 + 아래에 번호 / 막힌 칸 / 회피 방향 표시
- 300dpi 기준 가로 20cm (프린터 설정에서 '실제 크기'로 인쇄)
- 막대 부분의 실제 폭을 출력한다. 로봇이 바코드까지 거리를 계산할 때 쓰는 값이라
  robot_config.json 의 barcode.bar_width_m 과 같아야 한다

설치:
  pip3 install "python-barcode[images]"

실행:
  python3 make_barcodes.py                # labels/ 폴더에 생성
  python3 make_barcodes.py site_map.json out_dir
"""

import json
import os
import sys

import barcode
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw, ImageFont

DPI = 300
LABEL_CM = 20            # 라벨 가로 길이. 클수록 멀리서 읽힘 (A4 가로 인쇄 시 최대 약 27cm)
LABEL_W = int(LABEL_CM / 2.54 * DPI)
BARCODE_H = LABEL_W // 3
TEXT_H = LABEL_W // 6
QUIET_MODULES = 10       # 바코드 양옆 여백 (막대 10개 폭)

ACTION_TEXT = {
    "left": "NO PATH  <-  GO LEFT",
    "right": "NO PATH  ->  GO RIGHT",
    "back": "NO PATH  U-TURN",
    "stop": "NO PATH  STOP",
    "none": "CHECKPOINT",
}


def load_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_label(code, marker):
    bc = barcode.get("code128", code, writer=ImageWriter())

    # 막대 폭이 정수 픽셀이 되도록 계산 (리사이즈하면 막대 폭이 들쭉날쭉해져 인식률이 떨어짐)
    modules = len(bc.build()[0]) + 2 * QUIET_MODULES
    module_px = max(2, LABEL_W // modules)
    module_mm = module_px * 25.4 / DPI
    bar_width_m = (modules - 2 * QUIET_MODULES) * module_px / DPI * 0.0254

    img = bc.render({
        "module_width": module_mm,
        "module_height": BARCODE_H * 25.4 / DPI,
        "quiet_zone": QUIET_MODULES * module_mm,
        "margin_top": 0,
        "margin_bottom": 0,
        "write_text": False,
        "dpi": DPI,
    })

    label = Image.new("RGB", (LABEL_W, BARCODE_H + TEXT_H), "white")
    label.paste(img, ((LABEL_W - img.width) // 2, 0))

    draw = ImageDraw.Draw(label)
    big = load_font(LABEL_W // 20)
    small = load_font(LABEL_W // 32)

    line1 = ACTION_TEXT.get(marker["action"], marker["action"])
    x, y = marker["cell"]
    line2 = f"{code}   cell ({x},{y})   facing {marker['robot_heading']}"

    top1 = BARCODE_H + LABEL_W // 60
    top2 = top1 + int(LABEL_W // 20 * 1.3)
    for text, font, top in ((line1, big, top1), (line2, small, top2)):
        w = draw.textlength(text, font=font)
        draw.text(((LABEL_W - w) / 2, top), text, fill="black", font=font)

    draw.rectangle([0, 0, LABEL_W - 1, label.height - 1], outline="black", width=4)
    return label, bar_width_m


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "site_map.json")
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "labels")

    with open(config_path, encoding="utf-8") as f:
        markers = json.load(f)["barcodes"]

    os.makedirs(out_dir, exist_ok=True)

    widths = set()
    for code, marker in markers.items():
        path = os.path.join(out_dir, f"{code}.png")
        label, bar_width_m = make_label(code, marker)
        label.save(path, dpi=(DPI, DPI))
        widths.add(round(bar_width_m, 4))
        print("생성:", path)

    print()
    print(f"라벨 가로 {LABEL_CM}cm / 막대 부분 폭: {', '.join(f'{w * 100:.2f}cm' for w in sorted(widths))}")

    robot_path = os.path.join(here, "robot_config.json")
    if os.path.exists(robot_path):
        with open(robot_path, encoding="utf-8") as f:
            configured = float(json.load(f)["barcode"]["bar_width_m"])
        if any(abs(w - configured) > 0.002 for w in widths):
            print(f"[주의] robot_config.json 의 barcode.bar_width_m ({configured}) 을 위 값으로 바꾸세요")
            print("       바코드 번호 자릿수가 서로 다르면 폭도 달라지니 같은 자릿수로 맞추세요")
        else:
            print("robot_config.json 의 bar_width_m 과 일치합니다")


if __name__ == "__main__":
    main()
