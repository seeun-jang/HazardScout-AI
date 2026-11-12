"""SQLite persists patrol rounds and red UNKNOWN review events."""
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from config import DB_PATH


def now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')


class Storage:
    def __init__(self, path=DB_PATH):
        self.path = str(path)
        self.lock = threading.RLock()
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS patrols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    note TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'IN_PROGRESS',
                    unknown_total INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patrol_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    zone TEXT NOT NULL,
                    label TEXT NOT NULL DEFAULT 'UNKNOWN',
                    source TEXT NOT NULL DEFAULT 'AUTO',
                    image TEXT,
                    details TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    resolved_at TEXT,
                    FOREIGN KEY(patrol_id) REFERENCES patrols(id)
                );
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def set_setting(self, key, value):
        with self.lock, self.connection() as db:
            db.execute('INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)', (key, value))
            db.commit()

    def get_setting(self, key, default=''):
        with self.connection() as db:
            row = db.execute('SELECT value FROM settings WHERE key=?', (key,)).fetchone()
            return row['value'] if row else default

    def open_patrol(self):
        with self.lock, self.connection() as db:
            cursor = db.execute('INSERT INTO patrols(started_at) VALUES(?)', (now_iso(),))
            db.commit()
            return cursor.lastrowid

    def finish_patrol(self, patrol_id, note, manual_alert_zone=None):
        with self.lock, self.connection() as db:
            if manual_alert_zone:
                existing = db.execute('''SELECT COUNT(*) AS n FROM alerts WHERE patrol_id=?
                    AND zone=? AND status='PENDING' ''', (patrol_id, manual_alert_zone)).fetchone()['n']
                if not existing:
                    db.execute('''INSERT INTO alerts(patrol_id,created_at,zone,source,details)
                        VALUES(?,?,?,?,?)''', (patrol_id, now_iso(), manual_alert_zone, 'MANUAL', note))
            n = db.execute('SELECT COUNT(*) AS n FROM alerts WHERE patrol_id=?', (patrol_id,)).fetchone()['n']
            pending = db.execute('''SELECT COUNT(*) AS n FROM alerts
                 WHERE patrol_id=? AND status='PENDING' ''', (patrol_id,)).fetchone()['n']
            status = 'REVIEW_REQUIRED' if pending else 'NORMAL'
            db.execute('''UPDATE patrols SET completed_at=?,note=?,status=?,unknown_total=?
                  WHERE id=? AND completed_at IS NULL''', (now_iso(), note, status, n, patrol_id))
            db.commit()
            return {'status': status, 'unknown_total': n}

    def create_alert(self, patrol_id, zone, image='', details='', source='AUTO'):
        with self.lock, self.connection() as db:
            cur = db.execute('''INSERT INTO alerts(patrol_id,created_at,zone,image,details,source)
                VALUES(?,?,?,?,?,?)''', (patrol_id, now_iso(), zone, image, details, source))
            db.commit()
            return cur.lastrowid

    def resolve_alert(self, alert_id, status):
        with self.lock, self.connection() as db:
            cur = db.execute('''UPDATE alerts SET status=?,resolved_at=? WHERE id=? AND status='PENDING' ''',
                             (status, now_iso(), alert_id))
            if not cur.rowcount:
                return False
            row = db.execute('SELECT patrol_id FROM alerts WHERE id=?', (alert_id,)).fetchone()
            patrol_id = row['patrol_id']
            pending = db.execute('''SELECT COUNT(*) AS n FROM alerts WHERE patrol_id=? AND status='PENDING' ''',
                                 (patrol_id,)).fetchone()['n']
            db.execute('''UPDATE patrols SET status=? WHERE id=? AND completed_at IS NOT NULL''',
                       ('REVIEW_REQUIRED' if pending else 'REVIEW_DONE', patrol_id))
            db.commit()
            return True

    def snapshot(self, limit=24):
        with self.connection() as db:
            alerts = [dict(row) for row in db.execute('SELECT * FROM alerts ORDER BY id DESC LIMIT ?', (limit,))]
            patrols = [dict(row) for row in db.execute('SELECT * FROM patrols ORDER BY id DESC LIMIT 15')]
            count = db.execute("SELECT COUNT(*) AS n FROM alerts WHERE status='PENDING'").fetchone()['n']
            return {'alerts': alerts, 'patrols': patrols, 'pending_total': count}
