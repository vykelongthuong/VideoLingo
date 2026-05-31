import json
import os
import shutil
import subprocess
from pathlib import Path

import streamlit as st

from core._1_ytdlp import sanitize_filename
from core.utils.config_utils import load_key
from translations.translations import translate as t

OUTPUT_DIR = Path("output")
ORIGINAL_DIR_NAME = "original"
STATE_FILE_NAME = "video_speed_state.json"
TEMP_DIR_NAME = ".speed_tmp"
DEFAULT_SPEED = 1.0


def build_atempo_filter(speed: float) -> str:
    """Build a chained FFmpeg atempo filter for any positive speed."""
    speed = float(speed)
    if speed <= 0:
        raise ValueError("Speed must be greater than 0.")

    factors = []
    remaining = speed
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6g}" for factor in factors)


def change_video_speed(input_path: str, output_path: str, speed: float) -> None:
    """Create a speed-adjusted video. The caller swaps it into place after success."""
    input_path = str(input_path)
    output_path = str(output_path)
    speed = float(speed)

    if speed <= 0:
        raise ValueError("Speed must be greater than 0.")
    if not _has_audio_stream(input_path):
        raise RuntimeError("Input video has no audio stream; subtitle generation requires audio.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    atempo_filter = build_atempo_filter(speed)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-filter_complex",
        f"[0:v]setpts=PTS/{speed:.8g}[v];[0:a]{atempo_filter}[a]",
        "-map",
        "[v]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        output_path,
    ]

    try:
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        tail = stderr[-1200:] if stderr else str(exc)
        raise RuntimeError(tail) from exc


def ensure_original_video(active_video: str) -> str:
    """Ensure the original source video is stored outside output's top level."""
    active_path = Path(active_video)
    output_dir = active_path.parent
    original_dir = output_dir / ORIGINAL_DIR_NAME
    original_dir.mkdir(parents=True, exist_ok=True)

    state = _read_state(output_dir)
    state_original = state.get("original_video")
    if state_original and Path(state_original).exists():
        return state_original

    safe_stem = sanitize_filename(active_path.stem).replace(" ", "_")
    original_path = original_dir / f"{safe_stem}{active_path.suffix.lower()}"
    if original_path.exists() and not _same_file(active_path, original_path):
        original_path = original_dir / f"{safe_stem}_original{active_path.suffix.lower()}"

    if not _same_file(active_path, original_path):
        shutil.copy2(active_path, original_path)

    _write_state(
        output_dir,
        {
            "original_video": str(original_path),
            "current_speed": DEFAULT_SPEED,
            "active_video": str(active_path),
        },
    )
    return str(original_path)


def apply_video_speed(active_video: str, speed: float) -> str:
    """Apply speed from the saved original video and make the result the active input."""
    active_path = Path(active_video)
    output_dir = active_path.parent
    original_path = Path(ensure_original_video(active_video))
    speed = float(speed)

    safe_stem = sanitize_filename(original_path.stem).replace(" ", "_")
    speed_token = f"{speed:.2f}".replace(".", "_")
    output_path = output_dir / f"speed_adjusted_{safe_stem}_{speed_token}x.mp4"
    temp_path = output_dir / TEMP_DIR_NAME / output_path.name

    if temp_path.exists():
        temp_path.unlink()

    change_video_speed(str(original_path), str(temp_path), speed)
    _replace_active_video(output_dir, temp_path, output_path)
    _write_state(
        output_dir,
        {
            "original_video": str(original_path),
            "current_speed": speed,
            "active_video": str(output_path),
        },
    )
    return str(output_path)


def restore_original_video(active_video: str) -> str:
    """Restore the saved original as the only active top-level input video."""
    active_path = Path(active_video)
    output_dir = active_path.parent
    original_path = Path(ensure_original_video(active_video))
    restored_path = output_dir / original_path.name
    temp_path = output_dir / TEMP_DIR_NAME / restored_path.name
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    if not _same_file(original_path, restored_path):
        shutil.copy2(original_path, temp_path)
        _replace_active_video(output_dir, temp_path, restored_path)

    _write_state(
        output_dir,
        {
            "original_video": str(original_path),
            "current_speed": DEFAULT_SPEED,
            "active_video": str(restored_path),
        },
    )
    return str(restored_path)


def has_processing_artifacts(output_dir: str = "output") -> bool:
    """Detect generated pipeline artifacts that would become stale after speed changes."""
    root = Path(output_dir)
    if not root.exists():
        return False

    artifact_dirs = [root / "log", root / "audio", root / "gpt_log"]
    if any(path.exists() and any(path.iterdir()) for path in artifact_dirs if path.is_dir()):
        return True

    artifact_patterns = ["*.srt", "output*.mp4", "output*.mov", "output*.mkv", "output*.avi"]
    return any(any(root.glob(pattern)) for pattern in artifact_patterns)


def render_video_speed_control(video_file: str) -> None:
    output_dir = Path(video_file).parent
    current_speed = _get_current_speed(output_dir)
    artifacts_exist = has_processing_artifacts(str(output_dir))

    st.markdown(f"**{t('Video speed for subtitle generation')}**")
    st.caption(f"{t('Current speed')}: {current_speed:.2f}x")
    st.caption(t("The adjusted video will be used for transcription and subtitle generation."))

    slider_value = min(2.0, max(0.5, current_speed))
    selected_speed = st.slider(
        t("Playback speed"),
        min_value=0.5,
        max_value=2.0,
        value=slider_value,
        step=0.05,
        format="%.2fx",
        disabled=artifacts_exist,
    )

    if artifacts_exist:
        st.warning(
            t(
                "Speed changes are disabled after subtitle processing has started. Please delete/reselect or archive before changing speed."
            )
        )
        return

    apply_disabled = (
        abs(selected_speed - DEFAULT_SPEED) < 0.001
        or abs(selected_speed - current_speed) < 0.001
    )
    reset_disabled = abs(current_speed - DEFAULT_SPEED) < 0.001

    col1, col2 = st.columns(2)
    with col1:
        if st.button(t("Apply speed"), key="apply_video_speed", disabled=apply_disabled):
            with st.spinner(t("Processing speed-adjusted video...")):
                try:
                    apply_video_speed(video_file, selected_speed)
                except Exception as exc:
                    st.error(f"{t('Failed to change video speed')}: {exc}")
                    return
            st.success(t("Video speed applied successfully."))
            st.rerun()

    with col2:
        if st.button(t("Reset to original"), key="reset_video_speed", disabled=reset_disabled):
            with st.spinner(t("Processing speed-adjusted video...")):
                try:
                    restore_original_video(video_file)
                except Exception as exc:
                    st.error(f"{t('Failed to change video speed')}: {exc}")
                    return
            st.success(t("Original video restored."))
            st.rerun()


def _has_audio_stream(input_path: str) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        input_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("Failed to inspect video audio stream with ffprobe.") from exc
    return "audio" in result.stdout.lower()


def _read_state(output_dir: Path) -> dict:
    state_path = output_dir / ORIGINAL_DIR_NAME / STATE_FILE_NAME
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_state(output_dir: Path, state: dict) -> None:
    state_dir = output_dir / ORIGINAL_DIR_NAME
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / STATE_FILE_NAME).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _get_current_speed(output_dir: Path) -> float:
    state = _read_state(output_dir)
    try:
        return float(state.get("current_speed", DEFAULT_SPEED))
    except (TypeError, ValueError):
        return DEFAULT_SPEED


def _replace_active_video(output_dir: Path, temp_path: Path, final_path: Path) -> None:
    top_level_videos = _top_level_video_paths(output_dir)
    for path in top_level_videos:
        if path.resolve() != temp_path.resolve():
            path.unlink()
    if final_path.exists():
        final_path.unlink()
    shutil.move(str(temp_path), str(final_path))


def _top_level_video_paths(output_dir: Path) -> list[Path]:
    allowed_suffixes = _allowed_video_suffixes()
    return [
        path
        for path in output_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in allowed_suffixes
        and not path.name.startswith("output")
    ]


def _allowed_video_suffixes() -> set[str]:
    try:
        formats = load_key("allowed_video_formats")
    except Exception:
        formats = ["mp4", "mov", "avi", "mkv", "flv", "wmv", "webm"]
    return {f".{fmt.lower().lstrip('.')}" for fmt in formats}


def _same_file(path_a: Path, path_b: Path) -> bool:
    try:
        return path_a.exists() and path_b.exists() and path_a.resolve() == path_b.resolve()
    except OSError:
        return False
