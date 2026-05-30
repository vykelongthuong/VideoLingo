import pandas as pd
from typing import List, Tuple
import concurrent.futures

from core._3_2_split_meaning import split_sentence
from core.prompts import get_align_prompt
from rich.panel import Panel
from rich.console import Console
from rich.table import Table
from core.utils import *
from core.utils.models import *
console = Console()

# ! You can modify your own weights here
# Chinese and Japanese 2.5 characters, Korean 2 characters, Thai 1.5 characters, full-width symbols 2 characters, other English-based and half-width symbols 1 character
def calc_len(text: str) -> float:
    if text is None:
        return 0.0
    text = str(text) # force convert
    def char_weight(char):
        code = ord(char)
        if 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF:  # Chinese and Japanese
            return 1.75
        elif 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF:  # Korean
            return 1.5
        elif 0x0E00 <= code <= 0x0E7F:  # Thai
            return 1
        elif 0xFF01 <= code <= 0xFF5E:  # full-width symbols
            return 1.75
        else:  # other characters (e.g. English and half-width symbols)
            return 1

    return sum(char_weight(char) for char in text)

def align_subs(src_sub: str, tr_sub: str, src_part: str) -> Tuple[List[str], List[str], str]:
    align_prompt = get_align_prompt(src_sub, tr_sub, src_part)
    
    def valid_align(response_data):
        if 'align' not in response_data:
            return {"status": "error", "message": "Missing required key: `align`"}
        if len(response_data['align']) < 2:
            return {"status": "error", "message": "Align does not contain more than 1 part as expected!"}
        return {"status": "success", "message": "Align completed"}
    parsed = ask_gpt(align_prompt, resp_type='json', valid_def=valid_align, log_title='align_subs')
    align_data = parsed['align']
    src_parts = src_part.split('\n')
    tr_parts = [item[f'target_part_{i+1}'].strip() for i, item in enumerate(align_data)]
    
    whisper_language = load_key("whisper.language")
    language = load_key("whisper.detected_language") if whisper_language == 'auto' else whisper_language
    joiner = get_joiner(language)
    tr_remerged = joiner.join(tr_parts)
    
    table = Table(title="🔗 Aligned parts")
    table.add_column("Language", style="cyan")
    table.add_column("Parts", style="magenta")
    table.add_row("SRC_LANG", "\n".join(src_parts))
    table.add_row("TARGET_LANG", "\n".join(tr_parts))
    console.print(table)
    
    return src_parts, tr_parts, tr_remerged

def split_align_subs(src_lines: List[str], tr_lines: List[str]):
    """Split long subtitle lines using LLM alignment.

    Always returns a 3-tuple: (src_lines, tr_lines, remerged_tr_lines).
    """
    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]
    remerged_tr_lines = tr_lines.copy()

    to_split = []
    for i, (src, tr) in enumerate(zip(src_lines, tr_lines)):
        src_str, tr_str = str(src) if src is not None else '', str(tr) if tr is not None else ''
        if len(src_str) > MAX_SUB_LENGTH or calc_len(tr_str) * TARGET_SUB_MULTIPLIER > MAX_SUB_LENGTH:
            to_split.append(i)
            table = Table(title=f"📏 Line {i} needs to be split")
            table.add_column("Type", style="cyan")
            table.add_column("Content", style="magenta")
            table.add_row("Source Line", src_str)
            table.add_row("Target Line", tr_str)
            console.print(table)

    if not to_split:
        console.print("[green]No subtitle lines need splitting.[/green]")
        return src_lines, tr_lines, remerged_tr_lines

    console.print(f"[cyan]✂️ {len(to_split)} line(s) need splitting...[/cyan]")

    @except_handler("Error in split_align_subs")
    def process(i):
        split_src = split_sentence(src_lines[i], num_parts=2).strip()
        src_parts, tr_parts, tr_remerged = align_subs(src_lines[i], tr_lines[i], split_src)
        src_lines[i] = src_parts
        tr_lines[i] = tr_parts
        remerged_tr_lines[i] = tr_remerged

    with concurrent.futures.ThreadPoolExecutor(max_workers=load_key("max_workers")) as executor:
        # Consume the iterator to force execution and raise any exceptions
        list(executor.map(process, to_split))

    # Flatten `src_lines` and `tr_lines`
    src_lines = [item for sublist in src_lines for item in (sublist if isinstance(sublist, list) else [sublist])]
    tr_lines = [item for sublist in tr_lines for item in (sublist if isinstance(sublist, list) else [sublist])]

    console.print(f"[cyan]📊 After split: src={len(src_lines)} lines, trans={len(tr_lines)} lines[/cyan]")
    return src_lines, tr_lines, remerged_tr_lines

def split_for_sub_main():
    console.print("[bold green]🚀 Start splitting subtitles...[/bold green]")

    df = pd.read_excel(_4_2_TRANSLATION)
    src = df['Source'].tolist()
    trans = df['Translation'].tolist()

    console.print(f"[cyan]📊 Input: {len(src)} source lines, {len(trans)} translation lines[/cyan]")

    subtitle_set = load_key("subtitle")
    MAX_SUB_LENGTH = subtitle_set["max_length"]
    TARGET_SUB_MULTIPLIER = subtitle_set["target_multiplier"]

    # Initialize with original values — always a valid fallback
    split_src = src.copy()
    split_trans = trans.copy()
    remerged = trans.copy()

    for attempt in range(3):
        console.print(Panel(f"🔄 Split attempt {attempt + 1}", expand=False))
        try:
            result = split_align_subs(src.copy(), trans.copy())
            if not isinstance(result, tuple) or len(result) != 3:
                raise ValueError(
                    f"split_align_subs returned {type(result).__name__} "
                    f"with {len(result) if hasattr(result, '__len__') else '?'} elements, "
                    f"expected a 3-tuple (src, trans, remerged)"
                )
            split_src, split_trans, remerged = result
        except ValueError as e:
            if "not enough values to unpack" in str(e) or "expected 3" in str(e):
                console.print(f"[red]❌ Unpack error in split_align_subs: {e}[/red]")
            raise

        # Check if all subtitles meet length requirements (use str() for safety)
        all_src_ok = all(len(str(s)) <= MAX_SUB_LENGTH for s in split_src)
        all_trans_ok = all(calc_len(str(t)) * TARGET_SUB_MULTIPLIER <= MAX_SUB_LENGTH for t in split_trans)
        console.print(f"[cyan]📏 Length check: src_ok={all_src_ok}, trans_ok={all_trans_ok}[/cyan]")

        if all_src_ok and all_trans_ok:
            console.print("[green]✅ All subtitles within length limits[/green]")
            break

        # Update source data for next split iteration
        src, trans = split_src, split_trans
        console.print(f"[yellow]🔄 Some lines still too long, retrying (attempt {attempt+1}/3)...[/yellow]")

    # Ensure same length for downstream
    if len(src) > len(remerged):
        console.print(f"[yellow]⚠️ Padding remerged: {len(src) - len(remerged)} missing entries[/yellow]")
        remerged += [None] * (len(src) - len(remerged))
    elif len(remerged) > len(src):
        console.print(f"[yellow]⚠️ Padding src: {len(remerged) - len(src)} missing entries[/yellow]")
        src += [None] * (len(remerged) - len(src))

    console.print(f"[cyan]📊 Output: split_src={len(split_src)}, split_trans={len(split_trans)}, remerged={len(remerged)}[/cyan]")

    pd.DataFrame({'Source': split_src, 'Translation': split_trans}).to_excel(_5_SPLIT_SUB, index=False)
    pd.DataFrame({'Source': src, 'Translation': remerged}).to_excel(_5_REMERGED, index=False)
    console.print("[green]✅ Split results saved[/green]")

if __name__ == '__main__':
    split_for_sub_main()
