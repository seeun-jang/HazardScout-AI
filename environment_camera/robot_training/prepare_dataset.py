"""Merge private starter images and optional Roboflow YOLOv8 exports into 2-class YOLO dataset.

Private images and third-party datasets must not be uploaded to GitHub by default.
Images in an exported dataset with labels for unrelated categories are omitted,
not silently reinterpreted as robot negatives.
"""
import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter
from pathlib import Path

try:
    import yaml
except ImportError:
    raise SystemExit('PyYAML not installed. Run: python -m pip install pyyaml')

ROOT = Path(__file__).resolve().parent
CLASSES = ['mobile_robot', 'robot_arm']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def match_class(name):
    normalized = str(name).lower().replace('_', '').replace('-', '').replace(' ', '')
    if normalized in ('mobilerobot', 'rover', 'wheeledrobot'):
        return 0
    if normalized in ('robotarm', 'roboticarm'):
        return 1
    return None


def names_from_yaml(file):
    data = yaml.safe_load(file.read_text(encoding='utf-8')) or {}
    names = data.get('names', {})
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(i): str(n) for i, n in names.items()}
    raise ValueError('names not found in '+str(file))


def load_yolo_labels(path, mapping):
    output = []
    if not path.is_file():
        return output
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        vals = line.split()
        if len(vals) != 5:
            raise ValueError(f'Bad YOLO label: {path} -> {line}')
        old = int(vals[0]); new = mapping.get(old)
        if new is None:
            continue
        coords = [float(n) for n in vals[1:]]
        if any(not 0 <= v <= 1 for v in coords) or coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f'Invalid normalized bbox in {path}')
        output.append(f'{new} '+ ' '.join(f'{c:.6f}' for c in coords))
    return output


def source_records(root, include_empty=False):
    """Read one YOLOv8 export with train/valid/test/images or images/train layout."""
    data_files = list(root.rglob('data.yaml'))
    if not data_files:
        raise FileNotFoundError(f'data.yaml is missing in {root}: export as YOLOv8, unzip it')
    if len(data_files) > 1:
        raise ValueError(f'Multiple data.yaml found in {root}; supply the specific export folder')
    data_file = data_files[0]
    mapping = {id: match_class(name) for id, name in names_from_yaml(data_file).items()}
    mapping = {i: j for i, j in mapping.items() if j is not None}
    if not mapping:
        raise ValueError(f'No robot arm/mobile robot class in {data_file}; got {names_from_yaml(data_file)}')
    images = sorted(p for p in data_file.parent.rglob('*')
                    if p.is_file() and p.suffix.lower() in EXTENSIONS and p.parent.name == 'images')
    images.extend(sorted(p for p in (data_file.parent / 'images').glob('**/*')
                         if p.is_file() and p.suffix.lower() in EXTENSIONS and p not in images))
    records = []
    for im in images:
        # YOLOv8 layouts: train/images/a.jpg vs images/train/a.jpg
        if im.parent.name == 'images':
            label = im.parent.parent / 'labels' / (im.stem + '.txt')
        else:
            label = data_file.parent / 'labels' / im.parent.name / (im.stem + '.txt')
        anns = load_yolo_labels(label, mapping)
        # Other-class-only images are NOT valid negatives for robot: their unlabeled robot arms may exist.
        if not anns and not include_empty:
            continue
        records.append((im, anns, 'public:'+root.name))
    return records


def starter_records():
    out = []
    for im in sorted((ROOT / 'starter' / 'images').glob('*.jpg')):
        label = ROOT / 'starter' / 'labels' / (im.stem + '.txt')
        annotations = load_yolo_labels(label, {0: 0, 1: 1})
        if im.stem.startswith('own_') and not annotations:
            raise ValueError(f'Missing positive robot label: {im}')
        out.append((im, annotations, 'own:starter'))
    return out


def make_split(records, seed):
    """Keep source images intact in only one split; no augmentation before splitting."""
    rng = random.Random(seed)
    own_pos = [r for r in records if r[2] == 'own:starter' and r[1]]
    own_neg = [r for r in records if r[2] == 'own:starter' and not r[1]]
    public = [r for r in records if r[2] != 'own:starter']
    rng.shuffle(own_pos); rng.shuffle(own_neg); rng.shuffle(public)
    train, val = [], []
    # Underpowered demonstration split — metrics NOT suitable for claims of performance.
    if own_pos:
        val.append(own_pos.pop()); train.extend(own_pos)
    if own_neg:
        val.append(own_neg.pop()); train.extend(own_neg)
    if len(public) > 1:
        k = max(1, round(len(public) * .15))
        val.extend(public[:k]); train.extend(public[k:])
    else:
        train.extend(public)
    return train, val


def run(external, output, seed):
    records = starter_records()
    counts = {'starter': len(records)}
    for raw in external:
        source = Path(raw).resolve()
        if not source.is_dir():
            raise FileNotFoundError(f'Unzip the public dataset first: {source}')
        items = source_records(source)
        if not items:
            raise ValueError(f'No supported labeled images in {source}; check annotations')
        records.extend(items)
        counts[source.name] = len(items)
    # Drop byte-for-byte duplicates to avoid exact leakage across sources.
    unique = []
    sha_seen = set()
    for r in records:
        sha = hashlib.sha256(r[0].read_bytes()).hexdigest()
        if sha in sha_seen:
            continue
        sha_seen.add(sha)
        unique.append(r)
    tr, va = make_split(unique, seed)
    if not tr or not va:
        raise ValueError('Need labeled train and validation images')
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split, group in [('train', tr), ('val', va)]:
        for sub in ('images', 'labels'):
            shutil.rmtree(output/sub/split, ignore_errors=True)
            (output/sub/split).mkdir(parents=True, exist_ok=True)
        for idx, (img, annotations, src) in enumerate(group):
            # Prefix name and index avoid collisions. Keep original bytes; no data augmentation leakage.
            name = f'{src.split(":")[-1]}_{idx:05d}'
            target = output/'images'/split/(name + img.suffix.lower())
            shutil.copy2(img, target)
            (output/'labels'/split/(name+'.txt')).write_text('\n'.join(annotations) + ('\n' if annotations else ''),encoding='utf-8')
            manifest.append([split,src,str(img),str(target),len(annotations)])
    (output/'data.yaml').write_text(yaml.safe_dump({
        'path':str(output.resolve()),'train':'images/train','val':'images/val',
        'names':{0:CLASSES[0],1:CLASSES[1]}},allow_unicode=True,sort_keys=False),encoding='utf-8')
    with (output/'manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['split','source','original','output','num_boxes']);writer.writerows(manifest)
    summary = Counter(row[0] for row in manifest)
    boxes = Counter(int(a.split()[0]) for _,annotations,_ in unique for a in annotations)
    print('Data YAML:',output/'data.yaml')
    print('Sources:',counts,'Unique:',len(unique),'Train:',summary['train'],'Val:',summary['val'])
    print('Objects:',dict(zip(CLASSES,(boxes[0],boxes[1]))))
    if not external:
        print('WARNING: starter-only data (2 robot photos) is insufficient for trustworthy detection!')
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser(description='Build a robot YOLO dataset without changing safety7')
    p.add_argument('--external',action='append',default=[],help='Path to extracted Roboflow YOLOv8 dataset (repeatable)')
    p.add_argument('--output',type=Path,default=ROOT/'dataset'/'combined')
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();run(a.external,a.output.resolve(),a.seed)
