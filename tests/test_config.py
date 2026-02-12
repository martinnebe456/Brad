from __future__ import annotations

import inspect

from brad.config import ASR_BACKENDS
from brad.services import BradService


def test_only_faster_whisper_backend_supported() -> None:
    assert ASR_BACKENDS == ("faster-whisper",)


def test_transcribe_api_has_no_backend_selector() -> None:
    signature = inspect.signature(BradService.transcribe_file)

    assert "backend_name" not in signature.parameters
