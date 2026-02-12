from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from brad.asr.base import TranscriptSegment
from brad.storage.fts import create_fts_schema, insert_segment_fts, search_fts
from brad.storage.models import ExportRecord, MeetingRecord, SearchHit, SegmentRecord, SummaryRecord


class BradDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @contextmanager
    def _session(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        with self._session() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meetings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    language TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    duration_seconds REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id INTEGER NOT NULL,
                    start_s REAL NOT NULL,
                    end_s REAL NOT NULL,
                    text TEXT NOT NULL,
                    FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_segments_meeting ON segments(meeting_id)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    template_name TEXT NOT NULL,
                    method TEXT NOT NULL,
                    llm_model TEXT,
                    text TEXT NOT NULL,
                    FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    meeting_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    export_format TEXT NOT NULL,
                    path TEXT NOT NULL,
                    FOREIGN KEY(meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
                )
                """
            )
            try:
                create_fts_schema(conn)
            except sqlite3.OperationalError as exc:
                raise RuntimeError(
                    "SQLite FTS5 is unavailable in this Python build. "
                    "Install a Python/SQLite build with FTS5 support."
                ) from exc

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()

    def create_meeting(
        self,
        *,
        source_path: str,
        language: str,
        model_name: str,
        duration_seconds: float,
    ) -> int:
        with self._session() as conn:
            cursor = conn.execute(
                """
                INSERT INTO meetings(created_at, source_path, language, model_name, duration_seconds)
                VALUES (?, ?, ?, ?, ?)
                """,
                (self._now_iso(), source_path, language, model_name, duration_seconds),
            )
            return int(cursor.lastrowid)

    def add_segments(self, meeting_id: int, segments: Iterable[TranscriptSegment]) -> None:
        with self._session() as conn:
            for segment in segments:
                cursor = conn.execute(
                    """
                    INSERT INTO segments(meeting_id, start_s, end_s, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (meeting_id, segment.start, segment.end, segment.text),
                )
                segment_id = int(cursor.lastrowid)
                insert_segment_fts(conn, segment_id=segment_id, meeting_id=meeting_id, text=segment.text)

    def get_meeting(self, meeting_id: int) -> MeetingRecord | None:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
        if row is None:
            return None
        return MeetingRecord(
            id=int(row["id"]),
            created_at=str(row["created_at"]),
            source_path=str(row["source_path"]),
            language=str(row["language"]),
            model_name=str(row["model_name"]),
            duration_seconds=float(row["duration_seconds"]),
        )

    def get_segments(self, meeting_id: int) -> list[SegmentRecord]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT * FROM segments WHERE meeting_id = ? ORDER BY start_s ASC",
                (meeting_id,),
            ).fetchall()
        return [
            SegmentRecord(
                id=int(row["id"]),
                meeting_id=int(row["meeting_id"]),
                start=float(row["start_s"]),
                end=float(row["end_s"]),
                text=str(row["text"]),
            )
            for row in rows
        ]

    def transcript_text(self, meeting_id: int) -> str:
        segments = self.get_segments(meeting_id)
        return "\n".join(segment.text for segment in segments)

    def add_summary(
        self,
        *,
        meeting_id: int,
        template_name: str,
        method: str,
        text: str,
        llm_model: str | None,
    ) -> int:
        with self._session() as conn:
            cursor = conn.execute(
                """
                INSERT INTO summaries(meeting_id, created_at, template_name, method, llm_model, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (meeting_id, self._now_iso(), template_name, method, llm_model, text),
            )
            return int(cursor.lastrowid)

    def get_latest_summary(self, meeting_id: int) -> SummaryRecord | None:
        with self._session() as conn:
            row = conn.execute(
                """
                SELECT * FROM summaries
                WHERE meeting_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (meeting_id,),
            ).fetchone()
        if row is None:
            return None
        return SummaryRecord(
            id=int(row["id"]),
            meeting_id=int(row["meeting_id"]),
            created_at=str(row["created_at"]),
            template_name=str(row["template_name"]),
            method=str(row["method"]),
            llm_model=str(row["llm_model"]) if row["llm_model"] is not None else None,
            text=str(row["text"]),
        )

    def add_export(self, *, meeting_id: int, export_format: str, path: str) -> int:
        with self._session() as conn:
            cursor = conn.execute(
                """
                INSERT INTO exports(meeting_id, created_at, export_format, path)
                VALUES (?, ?, ?, ?)
                """,
                (meeting_id, self._now_iso(), export_format, path),
            )
            return int(cursor.lastrowid)

    def get_exports(self, meeting_id: int) -> list[ExportRecord]:
        with self._session() as conn:
            rows = conn.execute(
                "SELECT * FROM exports WHERE meeting_id = ? ORDER BY id ASC",
                (meeting_id,),
            ).fetchall()
        return [
            ExportRecord(
                id=int(row["id"]),
                meeting_id=int(row["meeting_id"]),
                created_at=str(row["created_at"]),
                export_format=str(row["export_format"]),
                path=str(row["path"]),
            )
            for row in rows
        ]

    def search_segments(
        self,
        query: str,
        *,
        meeting_id: int | None = None,
        limit: int = 25,
    ) -> list[SearchHit]:
        with self._session() as conn:
            return search_fts(conn, query=query, meeting_id=meeting_id, limit=limit)
