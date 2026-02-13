import pytest

from brad.cli import resolve_ui_mode


def test_resolve_ui_mode_desktop_aliases() -> None:
    assert resolve_ui_mode("desktop") == "desktop"
    assert resolve_ui_mode("native") == "desktop"
    assert resolve_ui_mode("  DESKTOP  ") == "desktop"


def test_resolve_ui_mode_web_aliases() -> None:
    assert resolve_ui_mode("web") == "web"
    assert resolve_ui_mode("gradio") == "web"


def test_resolve_ui_mode_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="desktop\\|web"):
        resolve_ui_mode("electron")
