import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


class SQLiteStateStore:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS kv_store (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS event_store (
                    namespace TEXT NOT NULL,
                    stream_id TEXT NOT NULL,
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    value TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_store_stream ON event_store(namespace, stream_id, seq)"
            )
            self.conn.commit()

    def ping(self) -> bool:
        try:
            with self.lock:
                cur = self.conn.cursor()
                cur.execute("SELECT 1")
                cur.fetchone()
            return True
        except Exception:
            return False

    def get(self, namespace: str, key: str) -> Optional[Any]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT value FROM kv_store WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
            row = cur.fetchone()
        if not row:
            return None
        return json.loads(row["value"])

    def set(self, namespace: str, key: str, value: Any) -> None:
        now = time.time()
        payload = json.dumps(value)
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO kv_store(namespace, key, value, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at
                """,
                (namespace, key, payload, now, now),
            )
            self.conn.commit()

    def delete(self, namespace: str, key: str) -> None:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM kv_store WHERE namespace = ? AND key = ?", (namespace, key))
            self.conn.commit()

    def items(self, namespace: str) -> List[Tuple[str, Any]]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT key, value FROM kv_store WHERE namespace = ?", (namespace,))
            rows = cur.fetchall()
        return [(row["key"], json.loads(row["value"])) for row in rows]

    def append_event(self, namespace: str, stream_id: str, value: Any) -> int:
        now = time.time()
        payload = json.dumps(value)
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO event_store(namespace, stream_id, value, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (namespace, stream_id, payload, now),
            )
            seq = int(cur.lastrowid)
            self.conn.commit()
            return seq

    def list_events(self, namespace: str, stream_id: str, after_seq: int = 0) -> List[Tuple[int, Any]]:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                """
                SELECT seq, value
                FROM event_store
                WHERE namespace = ? AND stream_id = ? AND seq > ?
                ORDER BY seq ASC
                """,
                (namespace, stream_id, after_seq),
            )
            rows = cur.fetchall()
        return [(int(row["seq"]), json.loads(row["value"])) for row in rows]

    def delete_older_than(self, namespace: str, older_than_epoch: float) -> int:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "DELETE FROM kv_store WHERE namespace = ? AND updated_at < ?",
                (namespace, older_than_epoch),
            )
            count = cur.rowcount
            self.conn.commit()
            return max(0, count)

    def delete_events_older_than(self, namespace: str, older_than_epoch: float) -> int:
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "DELETE FROM event_store WHERE namespace = ? AND created_at < ?",
                (namespace, older_than_epoch),
            )
            count = cur.rowcount
            self.conn.commit()
            return max(0, count)


class DurableRecord(dict):
    def __init__(self, value: Dict[str, Any], persist) -> None:
        super().__init__(value)
        self._persist = persist

    def _sync(self) -> None:
        self._persist(dict(self))

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self._sync()

    def __delitem__(self, key) -> None:
        super().__delitem__(key)
        self._sync()

    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        self._sync()

    def clear(self) -> None:
        super().clear()
        self._sync()

    def pop(self, key, default=None):
        value = super().pop(key, default)
        self._sync()
        return value

    def popitem(self):
        item = super().popitem()
        self._sync()
        return item

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        value = super().setdefault(key, default)
        self._sync()
        return value


class DurableDict:
    def __init__(self, store: SQLiteStateStore, namespace: str) -> None:
        self.store = store
        self.namespace = namespace

    def _wrap(self, key: str, value: Any) -> Any:
        if isinstance(value, dict):
            return DurableRecord(value, lambda updated: self.store.set(self.namespace, key, updated))
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self.store.set(self.namespace, key, value)

    def __getitem__(self, key: str) -> Any:
        value = self.store.get(self.namespace, key)
        if value is None:
            raise KeyError(key)
        return self._wrap(key, value)

    def get(self, key: str, default: Any = None) -> Any:
        value = self.store.get(self.namespace, key)
        if value is None:
            return default
        return self._wrap(key, value)

    def pop(self, key: str, default: Any = None) -> Any:
        value = self.store.get(self.namespace, key)
        if value is None:
            return default
        self.store.delete(self.namespace, key)
        return value

    def items(self) -> List[Tuple[str, Any]]:
        rows = self.store.items(self.namespace)
        return [(key, self._wrap(key, value)) for key, value in rows]

    def setdefault(self, key: str, default: Any) -> Any:
        value = self.store.get(self.namespace, key)
        if value is not None:
            return self._wrap(key, value)
        self.store.set(self.namespace, key, default)
        return self._wrap(key, default)

    def __contains__(self, key: str) -> bool:
        return self.store.get(self.namespace, key) is not None


class DurableEventBuffer:
    def __init__(self, store: SQLiteStateStore, namespace: str) -> None:
        self.store = store
        self.namespace = namespace

    def append(self, stream_id: str, value: Dict[str, Any]) -> int:
        return self.store.append_event(self.namespace, stream_id, value)

    def list_after(self, stream_id: str, after_seq: int = 0) -> List[Tuple[int, Dict[str, Any]]]:
        rows = self.store.list_events(self.namespace, stream_id, after_seq)
        return [(seq, value) for seq, value in rows]
