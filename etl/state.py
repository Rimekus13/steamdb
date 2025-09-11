
import json, os, pathlib

STATE_DIR = pathlib.Path(".state")
STATE_DIR.mkdir(exist_ok=True)

def _path(app_id: str) -> pathlib.Path:
    return STATE_DIR / f"state_{app_id}.json"

def load_state(app_id: str) -> dict:
    p = _path(app_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"max_timestamp_updated": 0, "last_cursor": "*"}

def save_state(app_id: str, state: dict):
    _path(app_id).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
