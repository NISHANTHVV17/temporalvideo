"""SQLite store for resumable indexing and structured temporal evidence."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS videos (
  video_id TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL,
  duration REAL NOT NULL DEFAULT 0, width INTEGER, height INTEGER,
  indexed_at TEXT DEFAULT CURRENT_TIMESTAMP, metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS tracks (
  track_id TEXT PRIMARY KEY, video_id TEXT NOT NULL, local_id TEXT NOT NULL,
  shot_id INTEGER NOT NULL, class TEXT NOT NULL, t_start REAL NOT NULL,
  t_end REAL NOT NULL, embedding_json TEXT, merged_into TEXT,
  FOREIGN KEY(video_id) REFERENCES videos(video_id)
);
CREATE TABLE IF NOT EXISTS track_points (
  point_id INTEGER PRIMARY KEY, track_id TEXT NOT NULL, pts REAL NOT NULL,
  x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
  confidence REAL NOT NULL, stabilized_json TEXT,
  FOREIGN KEY(track_id) REFERENCES tracks(track_id)
);
CREATE INDEX IF NOT EXISTS track_points_time ON track_points(track_id, pts);
CREATE UNIQUE INDEX IF NOT EXISTS track_points_unique ON track_points(track_id, pts);
CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY, video_id TEXT NOT NULL, track_id TEXT,
  class TEXT, event_type TEXT NOT NULL, t_start REAL NOT NULL, t_end REAL NOT NULL,
  zone TEXT, confidence REAL NOT NULL, meta_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_time ON events(video_id, t_start, t_end);
CREATE TABLE IF NOT EXISTS zones (
  zone_id TEXT PRIMARY KEY, video_id TEXT NOT NULL, name TEXT NOT NULL,
  polygon_json TEXT NOT NULL, shot_id INTEGER
);
CREATE TABLE IF NOT EXISTS semantic_frames (
  video_id TEXT NOT NULL, pts REAL NOT NULL, image_path TEXT NOT NULL,
  embedding_json TEXT, PRIMARY KEY(video_id, pts)
);
CREATE TABLE IF NOT EXISTS checkpoints (
  video_id TEXT PRIMARY KEY, last_pts REAL NOT NULL, frame_count INTEGER NOT NULL,
  shot_id INTEGER NOT NULL, updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS track_merges (
  merge_id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT NOT NULL,
  source_track_id TEXT NOT NULL, target_track_id TEXT NOT NULL,
  score REAL NOT NULL, crossed_cut INTEGER NOT NULL, reason TEXT NOT NULL,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class EvidenceDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def execute(self, sql: str, values: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        cursor = self.connection.execute(sql, values)
        self.connection.commit()
        return cursor

    def executemany(self, sql: str, rows: list[tuple[Any, ...]]) -> None:
        self.connection.executemany(sql, rows)
        self.connection.commit()

    def add_video(self, video_id: str, path: str, sha256: str, duration: float,
                  width: int | None, height: int | None, metadata: dict[str, Any] | None = None) -> None:
        self.execute("""INSERT OR REPLACE INTO videos
          (video_id,path,sha256,duration,width,height,metadata_json)
          VALUES(?,?,?,?,?,?,?)""", (video_id, path, sha256, duration, width, height,
                                         json.dumps(metadata or {})))

    def add_event(self, event: dict[str, Any]) -> None:
        self.execute("""INSERT OR REPLACE INTO events
          (event_id,video_id,track_id,class,event_type,t_start,t_end,zone,confidence,meta_json)
          VALUES(:event_id,:video_id,:track_id,:class,:event_type,:t_start,:t_end,:zone,:confidence,:meta_json)""", event)

    def events(self, video_id: str, start: float | None = None, end: float | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM events WHERE video_id=?"
        args: list[Any] = [video_id]
        if start is not None:
            sql += " AND t_end>=?"
            args.append(start)
        if end is not None:
            sql += " AND t_start<=?"
            args.append(end)
        return list(self.connection.execute(sql + " ORDER BY t_start", args))

    def checkpoint(self, video_id: str) -> sqlite3.Row | None:
        return self.connection.execute("SELECT * FROM checkpoints WHERE video_id=?", (video_id,)).fetchone()

    def set_checkpoint(self, video_id: str, last_pts: float, frame_count: int, shot_id: int) -> None:
        self.execute("""INSERT OR REPLACE INTO checkpoints(video_id,last_pts,frame_count,shot_id)
          VALUES(?,?,?,?)""", (video_id, last_pts, frame_count, shot_id))

    def rows(self, sql: str, values: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self.connection.execute(sql, values))

    def write_event_log(self, video_id: str) -> Path:
        rows = self.rows("SELECT * FROM events WHERE video_id=? ORDER BY t_start, t_end", (video_id,))
        log_path = self.path.with_suffix(".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"video_id={video_id}\n")
            handle.write(f"database={self.path}\n")
            handle.write(f"event_count={len(rows)}\n\n")
            if not rows:
                handle.write("No events recorded.\n")
                return log_path
            for row in rows:
                start = float(row["t_start"])
                end = float(row["t_end"])
                handle.write(
                    "event_id={event_id} | type={event_type} | class={class_name} | "
                    "track_id={track_id} | t_start={t_start:.2f}s | t_end={t_end:.2f}s | "
                    "duration={duration:.2f}s | zone={zone} | confidence={confidence:.2f} | meta={meta}\n".format(
                        event_id=row["event_id"],
                        event_type=row["event_type"],
                        class_name=row["class"],
                        track_id=row["track_id"] or "",
                        t_start=start,
                        t_end=end,
                        duration=max(0.0, end - start),
                        zone=row["zone"] or "",
                        confidence=float(row["confidence"]),
                        meta=row["meta_json"],
                    )
                )
        return log_path

    def write_summary_log(self, video_id: str, summary: list[dict[str, Any]],
                duration: float | None = None) -> Path:
        log_path = self.path.with_name(f"{video_id}.summary.txt")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write("Time\tEvent\n")
            if not summary:
                handle.write("No summary available.\n")
                return log_path
            for item in summary:
                handle.write(f"{item.get('time', '')}\t{item.get('event', '')}\n")
            data_path = log_path.with_suffix(".json")
            data_path.write_text(json.dumps({"video_duration": duration, "events": summary},
                    ensure_ascii=False, indent=2), encoding="utf-8")
        return log_path
