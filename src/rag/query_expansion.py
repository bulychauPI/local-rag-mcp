"""Step 1 of the hybrid pipeline: LLM-based keyword generation.

A 0.6B-3B model answers reliably only when the prompt is short and the output
format is trivial, so we ask for a single comma-separated line and parse it
through a ladder of increasingly forgiving strategies. Whatever happens, the
caller always gets a usable list (possibly empty) and never an exception.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ENABLE_QUERY_EXPANSION,
    KEYWORD_TEMPERATURE,
    KEYWORD_TIMEOUT,
    MAX_KEYWORDS,
)

# Deliberately tiny: one instruction, one example, no role-play, no reasoning.
KEYWORD_PROMPT = """Extract search keywords from the question.
Keep rare terms, abbreviations, file names and commands exactly as written.
Answer with keywords separated by commas. No explanation.

Question: How do I reset my VPN token?
Keywords: VPN token, reset VPN, vpn-reset

Question: {query}
Keywords:"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"[\[{].*[\]}]", re.DOTALL)
# Separators the model actually uses: commas, semicolons, newlines and the
# "1." / "2)" markers of a numbered list it was not asked for.
_ITEM_RE = re.compile(r"[\n;,]|(?:^|\s)\d+[.)]\s")


def _split_items(text: str):
    return _ITEM_RE.split(text)


def _strip_reasoning(text: str) -> str:
    """Remove qwen3 <think> blocks, including an unterminated trailing one."""
    text = _THINK_RE.sub(" ", text)
    text = _OPEN_THINK_RE.sub(" ", text)
    return text.strip()


def _clean_keyword(raw: str) -> str:
    kw = raw.strip().strip("\"'`*-•[]{}()")
    kw = re.sub(r"^\d+[.)]\s*", "", kw)  # numbered list leftovers
    kw = re.sub(r"^(keywords?|answer)\s*:\s*", "", kw, flags=re.IGNORECASE)
    return kw.strip()


def _dedupe(keywords, query: str):
    """Drop empties, duplicates and keywords longer than a search phrase."""
    result = []
    seen = {query.strip().lower()}
    for kw in keywords:
        kw = _clean_keyword(kw)
        if not kw or len(kw) > 60 or len(kw.split()) > 6:
            continue
        low = kw.lower()
        if low in seen:
            continue
        seen.add(low)
        result.append(kw)
        if len(result) >= MAX_KEYWORDS:
            break
    return result


def parse_keywords(raw_response: str, query: str = ""):
    """Parse an LLM response into a keyword list, tolerating bad formats.

    Fallback ladder: JSON -> comma/newline split -> bare word extraction -> [].
    """
    if not raw_response:
        return []

    text = _strip_reasoning(str(raw_response))
    if not text:
        return []

    # 1. Proper JSON (a list, or an object with a list-ish value).
    match = _JSON_RE.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        data = value
                        break
                else:
                    data = [v for v in data.values() if isinstance(v, str)]
            if isinstance(data, list):
                parsed = _dedupe([str(v) for v in data], query)
                if parsed:
                    return parsed
        except (ValueError, TypeError):
            pass

    # 2. The requested format: one line of comma-separated keywords.
    line = next((l for l in text.splitlines() if l.strip()), "")
    if "," in line or ("\n" not in text.strip() and line):
        parsed = _dedupe(_split_items(line), query)
        if parsed:
            return parsed

    # 3. The model wrote a bullet or numbered list: take one item per line.
    return _dedupe(_split_items(text), query)


def generate_keywords(query: str, llm=None):
    """Ask the LLM for keywords; return [] instead of raising on any failure.

    Args:
        query: the user question.
        llm: callable(prompt, options) -> str. Defaults to `rag.query.ask_llm`,
            imported lazily so this module stays importable without faiss.
    """
    if not ENABLE_QUERY_EXPANSION or not query.strip():
        return []

    if llm is None:
        try:
            from rag.query import ask_llm as llm
        except Exception:
            return []

    try:
        raw = llm(
            KEYWORD_PROMPT.format(query=query.strip()),
            options={
                "temperature": KEYWORD_TEMPERATURE,
                "num_predict": 64,
                "timeout": KEYWORD_TIMEOUT,
            },
        )
        return parse_keywords(raw, query)
    except Exception:
        # Timeout, connection error, unexpected payload shape - the caller
        # continues with the original query alone.
        return []


def build_fts_query(query: str, keywords):
    """Combine the raw question and the generated keywords into one FTS query."""
    if not keywords:
        return query
    return query + " " + " ".join(keywords)


if __name__ == "__main__":
    samples = [
        "VPN token, reset VPN, vpn-reset",
        '<think>The user wants...</think>\n["VPN token", "reset"]',
        "<think>hmm unterminated",
        "Keywords: 1. vacation policy 2. paid leave",
        '{"keywords": ["onboarding.md", "new hire"]}',
        "",
        "I am not sure what you mean.",
    ]
    for s in samples:
        print(repr(s[:40]), "->", parse_keywords(s, "test query"))
