import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('robot_prepare',ROOT/'prepare_dataset.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_starter_and_normalized_boxes(tmp_path):
    result=m.run([],tmp_path/'out',42)
    assert result['train']==4 and result['val']==2
    data=__import__('yaml').safe_load((tmp_path/'out'/'data.yaml').read_text())
    assert data['names']=={0:'mobile_robot',1:'robot_arm'}
    files=list((tmp_path/'out'/'labels').rglob('*.txt'))
    assert len(files)==6
    assert sum(len(p.read_text().strip().splitlines()) for p in files)==4
    for f in files:
        for line in f.read_text().splitlines():
            assert all(0<=float(n)<=1 for n in line.split()[1:])


def test_import_roboflow_layout(tmp_path):
    """train/images and images/train exports both work; unrelated gripper only is skipped."""
    for layout in ('train/images','images/train'):
        src=tmp_path/layout.replace('/','_')
        (src/layout).mkdir(parents=True)
        labeldir=(src/'train'/'labels' if layout=='train/images' else src/'labels'/'train')
        labeldir.mkdir(parents=True)
        (src/'data.yaml').write_text('names: [robot arm, gripper]\n')
        (src/layout/'a.jpg').write_bytes(b'test_jpg_image')
        (labeldir/'a.txt').write_text('0 .5 .5 .5 .5\n1 .5 .5 .2 .2\n')
        assert len(m.source_records(src))==1
        assert m.source_records(src)[0][1]==['1 0.500000 0.500000 0.500000 0.500000']


def test_invalid_and_other_class(tmp_path):
    data=tmp_path/'bb.txt';data.write_text('0 0.5 0.5 1.5 0.1\n')
    with pytest.raises(ValueError):m.load_yolo_labels(data,{0:1})
    data.write_text('1 0.5 0.5 0.3 0.3\n')
    assert m.load_yolo_labels(data,{0:1})==[]
