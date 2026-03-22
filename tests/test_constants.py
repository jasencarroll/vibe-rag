"""Tests for vibe_rag.constants — verify consistency between mappings."""
from vibe_rag.constants import CODE_EXTENSIONS, EXT_TO_LANG


def test_ext_to_lang_covers_all_code_extensions():
    """Every CODE_EXTENSION should have a corresponding entry in EXT_TO_LANG."""
    missing = CODE_EXTENSIONS - set(EXT_TO_LANG.keys())
    assert missing == set(), f"CODE_EXTENSIONS not in EXT_TO_LANG: {missing}"


def test_ext_to_lang_keys_are_code_extensions():
    """EXT_TO_LANG should not contain extensions outside CODE_EXTENSIONS."""
    extra = set(EXT_TO_LANG.keys()) - CODE_EXTENSIONS
    assert extra == set(), f"EXT_TO_LANG has extra keys not in CODE_EXTENSIONS: {extra}"
