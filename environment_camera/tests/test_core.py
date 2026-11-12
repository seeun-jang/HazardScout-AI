"""Run: python -m unittest discover -s tests -v"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ['ENABLE_YOLO'] = '0'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from storage import Storage
import vision


class StoreTests(unittest.TestCase):
    def test_patrol_and_review(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Storage(Path(folder) / 'monitor.db')
            pid = db.open_patrol()
            aid = db.create_alert(pid, 'A구역', details='new object')
            result = db.finish_patrol(pid, 'A구역에서 미확인 물체')
            self.assertEqual(result['unknown_total'], 1)
            self.assertEqual(result['status'], 'REVIEW_REQUIRED')
            self.assertEqual(db.snapshot()['pending_total'], 1)
            self.assertTrue(db.resolve_alert(aid, 'CONFIRMED'))
            self.assertEqual(db.snapshot()['pending_total'], 0)
            self.assertEqual(db.snapshot()['patrols'][0]['status'], 'REVIEW_DONE')

    def test_manual_report_is_an_alert(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Storage(Path(folder) / 'test.db')
            pid = db.open_patrol()
            out = db.finish_patrol(pid, 'A구역에서 확인되지 않은 물체 발견함', 'A구역')
            self.assertEqual(out['unknown_total'], 1)
            self.assertEqual(db.snapshot()['alerts'][0]['source'], 'MANUAL')


class VisionTests(unittest.TestCase):
    def setUp(self):
        self.detector = vision.OverheadDetector()
        self.reference = np.full((540, 960, 3), 150, np.uint8)
        self.detector.reference = self.reference.copy()
        self.with_object = self.reference.copy()
        cv2.rectangle(self.with_object, (50, 80), (170, 200), (8, 8, 8), -1)

    def test_unknown_is_red_and_emits_once(self):
        candidates = []
        for _ in range(vision.UNKNOWN_STABLE_FRAMES + 3):
            result, _ = self.detector.process(self.with_object, patrol_id=10,
                          on_alert=lambda candidate, image: candidates.append(candidate))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['zone'], 'A구역')
        x, y, *_ = candidates[0]['box']
        self.assertGreater(result[y, x, 2], 170)  # BGR red channel

    def test_moving_blob_does_not_immediately_trigger_unknown(self):
        emitted = []
        # A rover-shaped moving blob must not become a stable UNKNOWN simply
        # because it overlaps its previous position by a small amount.
        for i in range(vision.UNKNOWN_STABLE_FRAMES * 2):
            frame = self.reference.copy()
            x = 30 + 18 * i
            cv2.rectangle(frame, (x, 90), (x + 90, 195), (0, 0, 0), -1)
            self.detector.process(frame, patrol_id=5,
                                  on_alert=lambda c, f: emitted.append(c))
        self.assertEqual(emitted, [])

    def test_normalized_roi(self):
        poly = vision.parse_roi('0.10,0.10;0.90,0.10;0.90,0.90;0.10,0.90')
        self.assertEqual(len(poly), 4)
        with self.assertRaises(ValueError):
            vision.parse_roi('1.5,0;1,1;0,1')

    def test_yolo_known_overlap_suppresses_unknown(self):
        known = [{'box':[45, 75, 175, 205], 'label':'robot', 'conf':.8}]
        boxes = self.detector.scene_candidates(self.with_object, known)
        self.assertEqual(boxes, [])

    def test_no_unknown_when_not_patrolling(self):
        seen = []
        for _ in range(vision.UNKNOWN_STABLE_FRAMES + 3):
            self.detector.process(self.with_object, None,
                                  on_alert=lambda candidate, image: seen.append(candidate))
        self.assertEqual(seen, [])


class WorkflowTests(unittest.TestCase):
    def test_manual_unknown_blocks_new_patrol_until_review(self):
        import server
        with tempfile.TemporaryDirectory() as folder:
            db = Storage(Path(folder) / 'monitor.db')
            detector = vision.OverheadDetector()
            detector.reference = np.zeros((540,960,3), np.uint8)
            monitor = server.Monitor(storage=db, detector=detector)
            monitor.camera_ok = True
            pid = monitor.start_patrol()
            self.assertGreater(pid, 0)
            outcome = monitor.complete_patrol('A구역에서 확인되지 않은 물체 발견함')
            self.assertEqual(outcome['status'], 'REVIEW_REQUIRED')
            with self.assertRaises(ValueError):
                monitor.start_patrol()
            aid = db.snapshot()['alerts'][0]['id']
            self.assertTrue(db.resolve_alert(aid, 'CONFIRMED'))
            pid2 = monitor.start_patrol()
            self.assertGreater(pid2, pid)

    def test_auto_and_manual_same_zone_do_not_duplicate_alert(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Storage(Path(folder) / 'monitor.db')
            pid = db.open_patrol()
            db.create_alert(pid, 'A구역')
            outcome = db.finish_patrol(pid, 'A구역에서 미확인 물체 발견', 'A구역')
            self.assertEqual(outcome['unknown_total'], 1)



class DemoModeTests(unittest.TestCase):
    def test_demo_unknown_is_separate_and_reviewable(self):
        import server
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(server.config, 'DEMO_MODE', True), \
                 patch.object(server.config, 'DEMO_DB_PATH', root / 'demo.db'), \
                 patch.object(server.config, 'DEMO_SCENE_PATH', root / 'none.jpg'):
                m = server.Monitor()
                m.start_worker()
                state = m.state()
                self.assertTrue(state['demo_mode'])
                self.assertTrue(state['camera_ok'])
                self.assertEqual(state['model_names'], [])
                pid = m.start_patrol()
                alert_id = m.inject_demo_unknown()
                self.assertEqual(m.db.snapshot()['pending_total'], 1)
                self.assertEqual(m.db.snapshot()['alerts'][0]['source'], 'DEMO')
                self.assertEqual(m.complete_patrol('테스트 물체 확인')['unknown_total'], 1)
                self.assertTrue(m.db.resolve_alert(alert_id, 'FALSE_ALARM'))
                self.assertEqual(m.db.snapshot()['pending_total'], 0)
                self.assertGreater(m.start_patrol(), pid)

if __name__ == "__main__":
    unittest.main()
