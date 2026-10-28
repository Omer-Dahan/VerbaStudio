from pathlib import Path

import pytest

from hs.cli import resolve_output_path


def test_output_directory_uses_hebrew_suffixed_filename(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    destination = tmp_path / "rendered"
    destination.mkdir()

    assert resolve_output_path(video, str(destination)) == destination / "source_he.mp4"


def test_explicit_mp4_output_is_used_exactly(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    destination = tmp_path / "chosen.mp4"

    assert resolve_output_path(video, str(destination)) == destination


def test_invalid_output_path_is_rejected_with_usage_exit_code(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"

    with pytest.raises(ValueError, match="existing directory"):
        resolve_output_path(video, str(tmp_path / "not-a-video"))
