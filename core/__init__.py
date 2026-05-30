import logging

_logger = logging.getLogger(__name__)

try:
    from . import (
        _1_ytdlp,
        _2_asr,
        _3_1_split_nlp,
        _3_2_split_meaning,
        _4_1_summarize,
        _4_2_translate,
        _5_split_sub,
        _6_gen_sub,
        _7_sub_into_vid,
    )
    from .utils import *
    from .utils.onekeycleanup import cleanup
except ImportError as e:
    _logger.warning("core/__init__ import failed in lightweight mode: %s", e)

__all__ = [
    'ask_gpt',
    'load_key',
    'update_key',
    'cleanup',
    '_1_ytdlp',
    '_2_asr',
    '_3_1_split_nlp',
    '_3_2_split_meaning',
    '_4_1_summarize',
    '_4_2_translate',
    '_5_split_sub',
    '_6_gen_sub',
    '_7_sub_into_vid',
]
