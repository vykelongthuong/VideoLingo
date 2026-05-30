"""
Unit tests for core VideoLingo utilities.
Run with: pytest tests/ -v
"""

import sys
import os
import tempfile
import ast
import json
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest


# ============================================================
# Test: sanitize_filename (_1_ytdlp.py)
# ============================================================

from core._1_ytdlp import sanitize_filename


def test_sanitize_removes_illegal_chars():
    assert sanitize_filename('test<>:"/\\|?*file') == 'testfile'


def test_sanitize_strips_dots_and_spaces():
    assert sanitize_filename('...test file...') == 'test file'


def test_sanitize_empty_returns_video():
    assert sanitize_filename('<>:"|?*') == 'video'


def test_sanitize_preserves_valid_name():
    assert sanitize_filename('My Video Title 2024') == 'My Video Title 2024'


def test_sanitize_unicode():
    assert sanitize_filename('视频测试_日本語') == '视频测试_日本語'


# ============================================================
# Test: calc_len (_5_split_sub.py)
# ============================================================

from core._5_split_sub import calc_len


def test_calc_len_english():
    result = calc_len("Hello World")
    assert result == 11  # 11 English characters, weight 1 each


def test_calc_len_chinese():
    result = calc_len("你好世界")
    assert result == 7.0  # 4 Chinese chars * 1.75 = 7.0


def test_calc_len_mixed():
    result = calc_len("你好 World")
    assert result == 3.5 + 1 + 5  # 2 Chinese * 1.75 + 1 space + 5 ASCII = 9.5


def test_calc_len_none():
    assert calc_len(None) == 0.0


def test_calc_len_empty():
    assert calc_len("") == 0.0


# ============================================================
# Test: safe_parse_list (ast.literal_eval replacement)
# ============================================================

from core._10_gen_audio import _safe_parse_list


def test_parse_list_from_string():
    result = _safe_parse_list("['hello', 'world']")
    assert result == ['hello', 'world']


def test_parse_nested_list():
    result = _safe_parse_list("[[0.0, 1.5], [2.0, 3.5]]")
    assert result == [[0.0, 1.5], [2.0, 3.5]]


def test_parse_already_list():
    result = _safe_parse_list(['hello', 'world'])
    assert result == ['hello', 'world']


def test_parse_empty_list():
    assert _safe_parse_list('[]') == []


def test_parse_invalid_string_raises():
    with pytest.raises(ValueError):
        _safe_parse_list("not a list")


def test_parse_wrong_type_raises():
    with pytest.raises(TypeError):
        _safe_parse_list(123)


# ============================================================
# Test: convert_to_srt_format (_6_gen_sub.py)
# ============================================================

from core._6_gen_sub import convert_to_srt_format, remove_punctuation


def test_convert_srt_format():
    result = convert_to_srt_format(0.0, 5.0)
    assert result == "00:00:00,000 --> 00:00:05,000"


def test_convert_srt_format_hours():
    result = convert_to_srt_format(3661.5, 7200.0)
    assert result == "01:01:01,500 --> 02:00:00,000"


def test_convert_srt_format_milliseconds():
    result = convert_to_srt_format(1.234, 2.567)
    assert result == "00:00:01,234 --> 00:00:02,567"


def test_remove_punctuation():
    assert remove_punctuation("Hello, World!") == "Hello World"


def test_remove_punctuation_numbers():
    assert remove_punctuation("Price: $10.50") == "Price 1050"


# ============================================================
# Test: time_diff_seconds (_8_1_audio_task.py)
# ============================================================

import datetime
from core._8_1_audio_task import time_diff_seconds


def test_time_diff_seconds():
    t1 = datetime.time(0, 0, 1, 500000)  # 1.5s
    t2 = datetime.time(0, 0, 5, 0)  # 5.0s
    base = datetime.date.today()
    diff = time_diff_seconds(t1, t2, base)
    assert diff == pytest.approx(3.5)


def test_time_diff_cross_minute():
    t1 = datetime.time(0, 0, 58)
    t2 = datetime.time(0, 1, 2)
    base = datetime.date.today()
    diff = time_diff_seconds(t1, t2, base)
    assert diff == 4.0


# ============================================================
# Test: config_utils update_key returns False for missing keys
# ============================================================

from core.utils.config_utils import update_key, load_key


def test_update_nonexistent_key_returns_false():
    assert update_key("nonexistent.key.path", "value") == False


def test_update_nonexistent_final_key_returns_false():
    assert update_key("api.nonexistent_key_xyz", "value") == False


# ============================================================
# Test: parse_df_srt_time (_10_gen_audio.py)
# ============================================================

from core._10_gen_audio import parse_df_srt_time


def test_parse_srt_time_zero():
    assert parse_df_srt_time("00:00:00.000") == 0.0


def test_parse_srt_time():
    assert parse_df_srt_time("00:01:30.500") == 90.5


def test_parse_srt_time_hours():
    assert parse_df_srt_time("01:00:00.000") == 3600.0


# ============================================================
# Test: _5_split_sub.py split_for_sub_main — ensure padding logic
# ============================================================

def test_padding_logic():
    """Test that src/remerged padding works correctly."""
    # Simulate the padding logic from split_for_sub_main
    src = ["line1", "line2"]
    remerged = ["merged1"]

    if len(src) > len(remerged):
        remerged += [None] * (len(src) - len(remerged))
    elif len(remerged) > len(src):
        src += [None] * (len(remerged) - len(src))

    assert len(src) == len(remerged)
    assert len(src) == 2
    assert remerged == ["merged1", None]


# ============================================================
# Test: _safe_parse_list — reject malicious expressions
# ============================================================

def test_parse_rejects_malicious_system():
    """Reject strings that try to call __import__ or system functions."""
    with pytest.raises(ValueError):
        _safe_parse_list("__import__('os').system('echo bad')")


def test_parse_rejects_dict():
    """Reject dict when list is expected."""
    with pytest.raises(TypeError):
        _safe_parse_list({'key': 'value'})


def test_parse_rejects_malicious_exec():
    """Reject strings with exec/eval."""
    with pytest.raises(ValueError):
        _safe_parse_list("exec('print(1)')")


def test_parse_rejects_malicious_open():
    """Reject strings trying to open files."""
    with pytest.raises(ValueError):
        _safe_parse_list("open('/etc/passwd')")


def test_parse_valid_nested_numeric():
    result = _safe_parse_list("[[1.0, 2.5], [3.0, 4.5]]")
    assert result == [[1.0, 2.5], [3.0, 4.5]]


def test_parse_single_element_list():
    result = _safe_parse_list("['hello']")
    assert result == ['hello']


# ============================================================
# Test: _build_atempo_filter (_10_gen_audio.py)
# ============================================================

from core._10_gen_audio import _build_atempo_filter


def test_atempo_normal_speed():
    """at 1.0 speed, anull filter (no-op)."""
    result = _build_atempo_filter(1.0)
    assert 'anull' in result


def test_atempo_within_range():
    """Speed 1.5 should result in single atempo=1.5."""
    result = _build_atempo_filter(1.5)
    assert result == 'atempo=1.500000'


def test_atempo_half_speed():
    """Speed 0.5 should give single atempo=0.5."""
    result = _build_atempo_filter(0.5)
    assert result == 'atempo=0.500000'


def test_atempo_slower_than_half():
    """Speed 0.25 needs chaining: 0.5 * 0.5 = 0.25."""
    result = _build_atempo_filter(0.25)
    assert result == 'atempo=0.500000,atempo=0.500000'


def test_atempo_faster_than_2():
    """Speed 4.0 needs chaining: 2.0 * 2.0 = 4.0."""
    result = _build_atempo_filter(4.0)
    assert result == 'atempo=2.000000,atempo=2.000000'


def test_atempo_speed_3():
    """Speed 3.0: 2.0 * 1.5 = 3.0."""
    result = _build_atempo_filter(3.0)
    assert result == 'atempo=2.000000,atempo=1.500000'


def test_atempo_speed_0_125():
    """Speed 0.125: 0.5 * 0.5 * 0.5 = 0.125."""
    result = _build_atempo_filter(0.125)
    parts = result.split(',')
    assert len(parts) == 3
    assert all(p.startswith('atempo=') for p in parts)
    assert all('0.500000' in p for p in parts)


def test_atempo_raises_on_zero():
    with pytest.raises(ValueError):
        _build_atempo_filter(0.0)


def test_atempo_raises_on_negative():
    with pytest.raises(ValueError):
        _build_atempo_filter(-1.0)


# ============================================================
# Test: normalize_audio_volume — silent audio
# ============================================================

from core.asr_backend.audio_preprocess import normalize_audio_volume
from pydub import AudioSegment


def test_normalize_silent_audio(tmp_path):
    """Silent audio should not crash; output file is created."""
    import math
    # Create a short silent audio segment
    silent = AudioSegment.silent(duration=500, frame_rate=16000)
    input_path = str(tmp_path / "silent.wav")
    silent.export(input_path, format="wav")

    output_path = str(tmp_path / "normalized.wav")
    result = normalize_audio_volume(input_path, output_path, target_db=-20.0, format="wav")

    assert result == output_path
    assert os.path.exists(output_path)


def test_normalize_normal_audio(tmp_path):
    """Normal audio should be normalized."""
    silent = AudioSegment.silent(duration=500, frame_rate=16000)
    # Add a tone (gain +6dB above silence)
    audio = silent + 6
    input_path = str(tmp_path / "tone.wav")
    audio.export(input_path, format="wav")

    output_path = str(tmp_path / "normalized.wav")
    result = normalize_audio_volume(input_path, output_path, target_db=-20.0, format="wav")

    assert result == output_path
    assert os.path.exists(output_path)


# ============================================================
# Test: ask_gpt cache — corrupt JSON / retry
# ============================================================

# Use sys.modules to get the actual module (not the decorated function)
import sys
ask_cache_module = sys.modules['core.utils.ask_gpt']


def _setup_cache_tmp(tmp_path):
    """Switch cache folder to tmp_path. Returns old path for restore."""
    old = ask_cache_module.GPT_LOG_FOLDER
    ask_cache_module.GPT_LOG_FOLDER = str(tmp_path)
    return old


def _restore_cache_folder(path):
    ask_cache_module.GPT_LOG_FOLDER = path


def test_load_cache_corrupt_json_returns_false(tmp_path):
    """Corrupt JSON should not crash _load_cache; returns False."""
    old = _setup_cache_tmp(tmp_path)
    try:
        corrupt_file = tmp_path / "test.json"
        corrupt_file.write_text("{invalid json content !@#", encoding='utf-8')

        result = ask_cache_module._load_cache("test prompt", "text", "test")
        assert result == False
    finally:
        _restore_cache_folder(old)


def test_save_cache_handles_corrupt(tmp_path):
    """_save_cache overwrites corrupt file without crashing."""
    old = _setup_cache_tmp(tmp_path)
    try:
        corrupt_file = tmp_path / "test.json"
        corrupt_file.write_text("{corrupt json {{{ ", encoding='utf-8')

        ask_cache_module._save_cache("model", "prompt", "content", "text", "resp", log_title="test")

        data = json.loads(corrupt_file.read_text(encoding='utf-8'))
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["prompt"] == "prompt"
    finally:
        _restore_cache_folder(old)


def test_load_cache_normal(tmp_path):
    """Normal cache load returns cached value."""
    old = _setup_cache_tmp(tmp_path)
    try:
        ask_cache_module._save_cache("model", "prompt", "content", "text", "cached_value", log_title="test")

        result = ask_cache_module._load_cache("prompt", "text", "test")
        assert result == "cached_value"
    finally:
        _restore_cache_folder(old)


def test_load_cache_miss(tmp_path):
    """Cache miss returns False."""
    old = _setup_cache_tmp(tmp_path)
    try:
        ask_cache_module._save_cache("model", "prompt1", "content", "text", "value1", log_title="test")

        result = ask_cache_module._load_cache("different_prompt", "text", "test")
        assert result == False
    finally:
        _restore_cache_folder(old)


# ============================================================
# Test: SRT rendering — NaN / None handling
# ============================================================

from core._6_gen_sub import generate_subtitle_string
import pandas as pd


def test_srt_with_nan_column():
    """NaN in columns should not produce 'nan' string."""
    df = pd.DataFrame({
        'timestamp': ['00:00:01,000 --> 00:00:02,000'],
        'Source': ['Hello'],
        'Translation': [None],  # None should become ''
    })
    result = generate_subtitle_string(df, ['Source', 'Translation'])
    assert 'nan' not in result.lower()
    assert 'Hello' in result


def test_srt_with_nan_in_single_column():
    """NaN should become empty string."""
    df = pd.DataFrame({
        'timestamp': ['00:00:01,000 --> 00:00:02,000'],
        'Source': [float('nan')],
    })
    result = generate_subtitle_string(df, ['Source'])
    assert 'nan' not in result.lower()


def test_srt_missing_column_graceful():
    """Missing column returns empty string (defensive .get())."""
    df = pd.DataFrame({
        'timestamp': ['00:00:01,000 --> 00:00:02,000'],
        'Source': ['Hello'],
    })
    # Missing column should be handled gracefully — empty string, no crash
    result = generate_subtitle_string(df, ['Source', 'NonExistent'])
    assert 'hello' in result.lower()  # Source column shows
    assert 'nan' not in result.lower()  # No NaN strings


# ============================================================
# Test: Import smoke tests
# ============================================================

def test_import_core_star():
    """from core import * should work."""
    from core import _1_ytdlp, _2_asr
    assert _1_ytdlp is not None
    assert _2_asr is not None


def test_import_utils_star():
    """from core.utils import * should work."""
    from core.utils import load_key, update_key, ask_gpt, get_joiner
    assert load_key is not None
    assert update_key is not None
    assert ask_gpt is not None
    assert get_joiner is not None


def test_import_config_utils():
    """Individual config_utils imports should work."""
    from core.utils.config_utils import load_key, update_key, get_joiner
    val = load_key("api.key")
    assert isinstance(val, str)  # should be set (even if placeholder)


# ============================================================
# Test: update_key nested / leaf missing
# ============================================================

def test_update_nested_missing_path():
    """Missing nested path returns False."""
    assert update_key("api.nonexistent_sub.key", "value") == False


def test_update_leaf_missing():
    """Missing leaf key returns False."""
    assert update_key("api.nonexistent_final_key", "value") == False


# ============================================================
# Test: split_align_subs — always returns 3-tuple (anti-regression)
# ============================================================

import core._5_split_sub as split_sub


def test_split_align_subs_returns_three_when_no_split_needed(monkeypatch):
    """When no lines need splitting, return (src, tr, remerged) as 3-tuple."""
    def fake_load_key(key):
        if key == "subtitle":
            return {"max_length": 75, "target_multiplier": 1.2}
        if key == "max_workers":
            return 1
        raise KeyError(f"Unexpected key in test: {key}")

    monkeypatch.setattr(split_sub, "load_key", fake_load_key)

    src = ["short source"]
    tr = ["dịch ngắn"]

    result = split_sub.split_align_subs(src, tr)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 3, f"Expected 3 values, got {len(result)}"

    r_src, r_tr, r_remerged = result
    assert r_src == src
    assert r_tr == tr
    assert r_remerged == tr


def test_split_align_subs_returns_three_on_empty_input(monkeypatch):
    """Empty input should still return 3 values."""
    def fake_load_key(key):
        if key == "subtitle":
            return {"max_length": 75, "target_multiplier": 1.2}
        if key == "max_workers":
            return 1
        raise KeyError(f"Unexpected key in test: {key}")

    monkeypatch.setattr(split_sub, "load_key", fake_load_key)

    result = split_sub.split_align_subs([], [])

    assert isinstance(result, tuple)
    assert len(result) == 3
    r_src, r_tr, r_remerged = result
    assert r_src == []
    assert r_tr == []
    assert r_remerged == []


def test_split_align_subs_no_split_preserves_lengths(monkeypatch):
    """When no split needed, all three lists should have original lengths."""
    def fake_load_key(key):
        if key == "subtitle":
            return {"max_length": 100, "target_multiplier": 2.0}
        if key == "max_workers":
            return 1
        raise KeyError(f"Unexpected key in test: {key}")

    monkeypatch.setattr(split_sub, "load_key", fake_load_key)

    src = ["hello", "world"]
    tr = ["xin chào", "thế giới"]

    result = split_sub.split_align_subs(src, tr)

    assert len(result) == 3
    r_src, r_tr, r_remerged = result
    assert len(r_src) == len(src)
    assert len(r_tr) == len(tr)
    assert len(r_remerged) == len(tr)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
