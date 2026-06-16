from pathlib import Path
from pathlib import Path
import json

import regex as re
import yaml

try:
    from config import config

    module = config.language_identification.language_identification_library.lower()
except (ImportError, AttributeError, KeyError):
    module = "langid"

langid_languages = ["af", "am", "an", "ar", "as", "az", "be", "bg", "bn", "br", "bs", "ca", "cs", "cy", "da", "de",
                    "dz", "el",
                    "en", "eo", "es", "et", "eu", "fa", "fi", "fo", "fr", "ga", "gl", "gu", "he", "hi", "hr", "ht",
                    "hu", "hy",
                    "id", "is", "it", "ja", "jv", "ka", "kk", "km", "kn", "ko", "ku", "ky", "la", "lb", "lo", "lt",
                    "lv", "mg",
                    "mk", "ml", "mn", "mr", "ms", "mt", "nb", "ne", "nl", "nn", "no", "oc", "or", "pa", "pl", "ps",
                    "pt", "qu",
                    "ro", "ru", "rw", "se", "si", "sk", "sl", "sq", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr",
                    "ug", "uk",
                    "ur", "vi", "vo", "wa", "xh", "zh", "zu"]

BASE_DIR = Path(__file__).resolve().parent.parent
LANGUAGE_MAPPING_JSON_PATH = BASE_DIR / "data" / "language_mapping.json"
LANGUAGE_MAPPING_YAML_PATH = BASE_DIR / "data" / "language_mapping.yaml"

classifier = None
custom_language_mapping = {}
_custom_language_terms = ()
_current_languages_signature = None


def _load_custom_language_mapping() -> dict:
    source_path = None
    if LANGUAGE_MAPPING_JSON_PATH.exists():
        source_path = LANGUAGE_MAPPING_JSON_PATH
    elif LANGUAGE_MAPPING_YAML_PATH.exists():
        source_path = LANGUAGE_MAPPING_YAML_PATH

    if source_path is None:
        return {}

    with source_path.open("r", encoding="utf-8") as f:
        if source_path.suffix == ".json":
            data = json.load(f) or {}
        else:
            data = yaml.safe_load(f) or {}

    mapping = data.get("language_mapping", {})
    if not isinstance(mapping, dict):
        return {}

    return {
        str(key).strip(): str(value).strip().lower()
        for key, value in mapping.items()
        if str(key).strip() and str(value).strip()
    }


def _build_custom_language_terms(mapping: dict) -> tuple:
    # Longer terms first so more specific phrases win when phrases overlap.
    return tuple(sorted(mapping.keys(), key=len, reverse=True))


def _term_in_text(term: str, text: str) -> bool:
    if term.isascii() and term.replace("_", "").replace("-", "").isalnum():
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        return re.search(pattern, text) is not None

    return term in text


custom_language_mapping = _load_custom_language_mapping()
_custom_language_terms = _build_custom_language_terms(custom_language_mapping)


def _sync_custom_language_mapping(mapping: dict):
    global custom_language_mapping, _custom_language_terms

    custom_language_mapping = {
        str(key).strip(): str(value).strip().lower()
        for key, value in mapping.items()
        if str(key).strip() and str(value).strip()
    }
    _custom_language_terms = _build_custom_language_terms(custom_language_mapping)


def get_custom_language_mapping() -> dict:
    return dict(custom_language_mapping)


def set_custom_language_mapping(mapping: dict):
    LANGUAGE_MAPPING_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized_mapping = {
        str(key).strip(): str(value).strip().lower()
        for key, value in mapping.items()
        if str(key).strip() and str(value).strip()
    }

    with LANGUAGE_MAPPING_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump({"language_mapping": normalized_mapping}, f, ensure_ascii=False, indent=2)

    _sync_custom_language_mapping(normalized_mapping)


def reload_custom_language_mapping():
    _sync_custom_language_mapping(_load_custom_language_mapping())


def init_classifier():
    global classifier

    if module == "fastlid" or module == "fasttext":
        from fastlid import fastlid
        classifier = fastlid
    elif module == "langid":
        import langid
        # init model
        langid.langid.load_model()
        classifier = langid.classify


def _supported_languages():
    if module == "fastlid" or module == "fasttext":
        from fastlid import supported_langs

        return supported_langs

    if module == "langid":
        return langid_languages

    raise ValueError("Wrong LANGUAGE_IDENTIFICATION_LIBRARY in config.yaml")


def _resolve_target_languages(langs):
    if not langs:
        return None

    supported_languages = set(_supported_languages())
    return tuple(dict.fromkeys(lang for lang in langs if lang in supported_languages))


def _apply_target_languages(langs):
    global _current_languages_signature

    if module == "fastlid" or module == "fasttext":
        from fastlid import fastlid

        target_languages = list(langs) if langs else list(_supported_languages())
        signature = tuple(target_languages)
        if signature == _current_languages_signature:
            return

        fastlid.set_languages = target_languages
        _current_languages_signature = signature
        return

    if module == "langid":
        import langid

        target_languages = list(langs) if langs else list(_supported_languages())
        signature = tuple(target_languages)
        if signature == _current_languages_signature:
            return

        langid.set_languages(target_languages)
        _current_languages_signature = signature
        return

    raise ValueError("Wrong LANGUAGE_IDENTIFICATION_LIBRARY in config.yaml")


init_classifier()


def set_languages(langs):
    _apply_target_languages(_resolve_target_languages(langs))


def classify_language(text: str, target_languages: list = None) -> str:
    global classifier

    normalized_text = text.strip()

    if normalized_text in custom_language_mapping:
        return custom_language_mapping[normalized_text]

    for term in _custom_language_terms:
        if _term_in_text(term, normalized_text):
            return custom_language_mapping[term]

    resolved_languages = _resolve_target_languages(target_languages)
    _apply_target_languages(resolved_languages)

    lang = classifier(normalized_text)[0]

    return lang


# def classify_zh_ja(text: str) -> str:
#     for idx, char in enumerate(text):
#         unicode_val = ord(char)
#
#         # 检测日语字符
#         if 0x3040 <= unicode_val <= 0x309F or 0x30A0 <= unicode_val <= 0x30FF:
#             return "ja"
#
#         # 检测汉字字符
#         if 0x4E00 <= unicode_val <= 0x9FFF:
#             # 检查周围的字符
#             next_char = text[idx + 1] if idx + 1 < len(text) else None
#
#             if next_char and (0x3040 <= ord(next_char) <= 0x309F or 0x30A0 <= ord(next_char) <= 0x30FF):
#                 return "ja"
#
#     return "zh"


def split_alpha_nonalpha(text, mode=1):
    """
    Splits the input text based on the specified mode.

    Parameters:
    - text (str): The input text to be split.
    - mode (int): The mode for splitting (1 or 2).
        - Mode 1: Splits based on the pattern - Chinese/Japanese followed by English or vice versa.
        - Mode 2: Splits based on the pattern - Chinese/Japanese followed by English/digit or vice versa.

    Returns:
    - list: A list of substrings after the split.
    """
    if mode == 1:
        pattern = r'(?<=[\u4e00-\u9fff\u3040-\u30FF\d\s])(?=[\p{Latin}])|(?<=[\p{Latin}\s])(?=[\u4e00-\u9fff\u3040-\u30FF\d])'
    elif mode == 2:
        pattern = r'(?<=[\u4e00-\u9fff\u3040-\u30FF\s])(?=[\p{Latin}\d])|(?<=[\p{Latin}\d\s])(?=[\u4e00-\u9fff\u3040-\u30FF])'
    else:
        raise ValueError("Invalid mode. Supported modes are 1 and 2.")

    return re.split(pattern, text)


if __name__ == "__main__":
    text = "这是一个测试文本"
    print(classify_language(text))
    # print(classify_zh_ja(text))  # "zh"

    text = "これはテストテキストです"
    print(classify_language(text))
    # print(classify_zh_ja(text))  # "ja"
