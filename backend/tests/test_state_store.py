from pathlib import Path

from backend.state_store import DurableDict, DurableEventBuffer, SQLiteStateStore


def test_durable_dict_persists_across_instances(tmp_path):
    db_path = tmp_path / "state.db"
    store_a = SQLiteStateStore(str(db_path))
    runs_a = DurableDict(store_a, "runs")
    runs_a["run1"] = {"status": "completed", "score": 0.12}

    store_b = SQLiteStateStore(str(db_path))
    runs_b = DurableDict(store_b, "runs")
    run = runs_b.get("run1")
    assert run is not None
    assert run["status"] == "completed"
    assert run["score"] == 0.12


def test_durable_record_autosync(tmp_path):
    db_path = tmp_path / "state.db"
    store = SQLiteStateStore(str(db_path))
    schedules = DurableDict(store, "schedules")
    schedules["sched1"] = {"enabled": True, "cron": "* * * * *"}
    row = schedules.get("sched1")
    row["enabled"] = False
    refreshed = schedules.get("sched1")
    assert refreshed["enabled"] is False


def test_event_buffer_orders_by_sequence(tmp_path):
    db_path = tmp_path / "state.db"
    store = SQLiteStateStore(str(db_path))
    events = DurableEventBuffer(store, "event_stream")
    events.append("run-x", {"event": "start"})
    events.append("run-x", {"event": "middle"})
    rows = events.list_after("run-x", after_seq=0)
    assert len(rows) == 2
    assert rows[0][1]["event"] == "start"
    assert rows[1][1]["event"] == "middle"
