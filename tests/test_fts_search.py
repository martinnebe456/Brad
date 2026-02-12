from brad.asr.base import TranscriptSegment
from brad.storage.db import BradDB


def test_fts_search_basic(tmp_path) -> None:
    db = BradDB(tmp_path / "brad.db")
    db.initialize()
    meeting_id = db.create_meeting(
        source_path="demo.wav",
        language="en",
        model_name="small",
        duration_seconds=12.0,
    )
    db.add_segments(
        meeting_id,
        [
            TranscriptSegment(start=0.0, end=2.0, text="We discussed timeline and budget risks."),
            TranscriptSegment(start=2.1, end=4.0, text="Action item: finalize proposal by Friday."),
        ],
    )

    hits = db.search_segments("budget")
    assert len(hits) == 1
    assert hits[0].meeting_id == meeting_id
    assert "budget" in hits[0].text.lower()
