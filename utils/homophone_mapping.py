from pathlib import Path
import json

import regex as re


BASE_DIR = Path(__file__).resolve().parent.parent
HOMOPHONE_MAPPING_JSON_PATH = BASE_DIR / "data" / "homophone_mapping.json"

custom_homophone_mapping = {}
_custom_homophone_terms = ()
_ascii_term_patterns = {}


def _normalize_mapping(mapping: dict) -> dict:
    return {
        str(key).strip(): str(value).strip()
        for key, value in (mapping or {}).items()
        if str(key).strip() and str(value).strip()
    }


def _load_custom_homophone_mapping() -> dict:
    if not HOMOPHONE_MAPPING_JSON_PATH.exists():
        return {}

    with HOMOPHONE_MAPPING_JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f) or {}

    if "homophone_mapping" in data and isinstance(data["homophone_mapping"], dict):
        return _normalize_mapping(data["homophone_mapping"])

    if isinstance(data, dict):
        return _normalize_mapping(data)

    return {}


def _build_custom_homophone_terms(mapping: dict) -> tuple:
    return tuple(sorted(mapping.keys(), key=len, reverse=True))


def _compile_ascii_term_patterns(terms: tuple) -> dict:
    patterns = {}
    for term in terms:
        if term.isascii() and term.replace("_", "").replace("-", "").isalnum():
            patterns[term] = re.compile(rf"(?<!\\w){re.escape(term)}(?!\\w)")
    return patterns


def _sync_custom_homophone_mapping(mapping: dict):
    global custom_homophone_mapping, _custom_homophone_terms, _ascii_term_patterns

    custom_homophone_mapping = _normalize_mapping(mapping)
    _custom_homophone_terms = _build_custom_homophone_terms(custom_homophone_mapping)
    _ascii_term_patterns = _compile_ascii_term_patterns(_custom_homophone_terms)


custom_homophone_mapping = _load_custom_homophone_mapping()
_custom_homophone_terms = _build_custom_homophone_terms(custom_homophone_mapping)
_ascii_term_patterns = _compile_ascii_term_patterns(_custom_homophone_terms)


def get_custom_homophone_mapping() -> dict:
    return dict(custom_homophone_mapping)


def set_custom_homophone_mapping(mapping: dict):
    HOMOPHONE_MAPPING_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized_mapping = _normalize_mapping(mapping)

    with HOMOPHONE_MAPPING_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump({"homophone_mapping": normalized_mapping}, f, ensure_ascii=False, indent=2)

    _sync_custom_homophone_mapping(normalized_mapping)


def reload_custom_homophone_mapping():
    _sync_custom_homophone_mapping(_load_custom_homophone_mapping())


def apply_homophone_mapping(text: str) -> str:
    if not text or not custom_homophone_mapping:
        return text

    mapped_text = text
    for term in _custom_homophone_terms:
        replacement = custom_homophone_mapping[term]
        pattern = _ascii_term_patterns.get(term)
        if pattern is not None:
            mapped_text = pattern.sub(replacement, mapped_text)
        else:
            mapped_text = mapped_text.replace(term, replacement)

    return mapped_text
