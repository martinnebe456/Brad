from __future__ import annotations

import sqlite3

from brad.storage.models import SearchHit


def create_fts_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts
        USING fts5(
            segment_id UNINDEXED,
            meeting_id UNINDEXED,
            text
        )
        """
    )


def insert_segment_fts(conn: sqlite3.Connection, segment_id: int, meeting_id: int, text: str) -> None:
    conn.execute(
        "INSERT INTO segments_fts(segment_id, meeting_id, text) VALUES (?, ?, ?)",
        (segment_id, meeting_id, text),
    )


def search_fts(
    conn: sqlite3.Connection,
    query: str,
    meeting_id: int | None = None,
    limit: int = 25,
) -> list[SearchHit]:
    sql = """
        SELECT
            s.id AS segment_id,
            s.meeting_id,
            s.start_s,
            s.end_s,
            s.text,
            snippet(segments_fts, 2, '[', ']', ' ... ', 12) AS snippet
        FROM segments_fts
        JOIN segments s ON s.id = segments_fts.segment_id
        WHERE segments_fts MATCH ?
    """
    params: list[object] = [query]
    if meeting_id is not None:
        sql += " AND s.meeting_id = ?"
        params.append(meeting_id)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    rows = conn.execute(sql, tuple(params)).fetchall()
    return [
        SearchHit(
            segment_id=int(row["segment_id"]),
            meeting_id=int(row["meeting_id"]),
            start=float(row["start_s"]),
            end=float(row["end_s"]),
            text=str(row["text"]),
            snippet=str(row["snippet"]),
        )
        for row in rows
    ]
