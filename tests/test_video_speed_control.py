import shutil
import subprocess
import sys
import inspect
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core._1_ytdlp import find_video_files
from core.st_utils.video_speed_control import (
    apply_video_speed,
    build_atempo_filter,
    has_processing_artifacts,
    restore_original_video,
)


def test_build_atempo_filter():
    assert build_atempo_filter(1.0) == "atempo=1"
    assert build_atempo_filter(2.0) == "atempo=2"
    assert build_atempo_filter(0.5) == "atempo=0.5"
    assert build_atempo_filter(4.0) == "atempo=2,atempo=2"
    assert build_atempo_filter(0.25) == "atempo=0.5,atempo=0.5"


def test_build_atempo_filter_rejects_invalid_speed():
    with pytest.raises(ValueError):
        build_atempo_filter(0)


def test_has_processing_artifacts(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    assert not has_processing_artifacts(str(output_dir))

    (output_dir / "original").mkdir()
    (output_dir / "original" / "source.mp4").write_text("placeholder", encoding="utf-8")
    assert not has_processing_artifacts(str(output_dir))

    artifact_cases = [
        ("log", "translation_results.xlsx"),
        ("audio", "raw.mp3"),
        (None, "trans.srt"),
        (None, "output_sub.mp4"),
    ]
    for dirname, filename in artifact_cases:
        case_dir = tmp_path / f"case_{filename.replace('.', '_')}" / "output"
        case_dir.mkdir(parents=True)
        target_dir = case_dir / dirname if dirname else case_dir
        target_dir.mkdir(exist_ok=True)
        (target_dir / filename).write_text("placeholder", encoding="utf-8")
        assert has_processing_artifacts(str(case_dir))


def test_apply_and_restore_video_speed_with_ffmpeg(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe not available")

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    source = output_dir / "sample.mp4"
    _create_test_video(source, duration=4)

    adjusted_one_point_five = Path(apply_video_speed(str(source), 1.5))
    assert adjusted_one_point_five.exists()
    assert find_video_files(save_path=str(output_dir)).replace("\\", "/") == str(adjusted_one_point_five).replace("\\", "/")
    assert _duration(adjusted_one_point_five) == pytest.approx(4.0 / 1.5, rel=0.2, abs=0.5)
    assert _top_level_videos(output_dir) == [adjusted_one_point_five]

    adjusted_fast = Path(apply_video_speed(str(adjusted_one_point_five), 2.0))
    assert adjusted_fast.exists()
    assert find_video_files(save_path=str(output_dir)).replace("\\", "/") == str(adjusted_fast).replace("\\", "/")
    assert _duration(adjusted_fast) == pytest.approx(2.0, rel=0.2, abs=0.4)
    assert _top_level_videos(output_dir) == [adjusted_fast]

    adjusted_slow = Path(apply_video_speed(str(adjusted_fast), 0.5))
    assert adjusted_slow.exists()
    assert find_video_files(save_path=str(output_dir)).replace("\\", "/") == str(adjusted_slow).replace("\\", "/")
    assert _duration(adjusted_slow) == pytest.approx(8.0, rel=0.2, abs=0.8)
    assert _top_level_videos(output_dir) == [adjusted_slow]

    restored = Path(restore_original_video(str(adjusted_slow)))
    assert restored.exists()
    assert find_video_files(save_path=str(output_dir)).replace("\\", "/") == str(restored).replace("\\", "/")
    assert _duration(restored) == pytest.approx(4.0, rel=0.2, abs=0.5)
    assert _top_level_videos(output_dir) == [restored]


def test_audio_upload_black_screen_video_can_be_speed_adjusted(tmp_path, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe not available")

    import core.st_utils.download_video_section as download_section

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    audio_file = output_dir / "uploaded.wav"
    _create_test_audio(audio_file, duration=4)

    monkeypatch.setattr(download_section, "OUTPUT_DIR", str(output_dir))
    black_screen = Path(download_section.convert_audio_to_video(str(audio_file)))
    assert black_screen.name == "black_screen.mp4"
    assert black_screen.exists()
    assert not audio_file.exists()
    assert find_video_files(save_path=str(output_dir)).replace("\\", "/") == str(black_screen).replace("\\", "/")

    adjusted = Path(apply_video_speed(str(black_screen), 2.0))
    assert adjusted.exists()
    assert find_video_files(save_path=str(output_dir)).replace("\\", "/") == str(adjusted).replace("\\", "/")
    assert _duration(adjusted) == pytest.approx(2.0, rel=0.25, abs=0.5)
    assert _top_level_videos(output_dir) == [adjusted]


def test_asr_and_video_merge_still_resolve_active_video_with_find_video_files():
    import core._2_asr as asr
    import core._7_sub_into_vid as sub_into_vid

    assert "find_video_files()" in inspect.getsource(asr.transcribe)
    assert "find_video_files()" in inspect.getsource(sub_into_vid.merge_subtitles_to_video)


def _create_test_video(path: Path, duration: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=320x240:rate=25:duration={duration}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=1000:duration={duration}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")


def _create_test_audio(path: Path, duration: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=1000:duration={duration}",
        "-c:a",
        "pcm_s16le",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8")


def _duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return float(result.stdout.strip())


def _top_level_videos(output_dir: Path) -> list[Path]:
    return sorted(path for path in output_dir.iterdir() if path.is_file() and path.suffix == ".mp4")
