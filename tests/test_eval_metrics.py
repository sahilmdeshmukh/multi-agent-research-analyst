"""Unit tests for eval metric helpers."""
from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from research_analyst.schemas import Memo, MemoSection, ResearchNote


def _make_note(url: str) -> ResearchNote:
    return ResearchNote(
        claim="test claim",
        source_url=url,
        source_title="Test",
        confidence=0.8,
    )


def test_unique_domains_counts_distinct():
    from eval.run_eval import _unique_domains

    notes = [
        _make_note("https://example.com/a"),
        _make_note("https://example.com/b"),
        _make_note("https://other.com/page"),
        _make_note("https://www.third.com/x"),
        _make_note("https://third.com/y"),
    ]
    assert _unique_domains(notes) == 3


def test_unique_domains_skips_malformed():
    from eval.run_eval import _unique_domains

    notes = [
        _make_note("not-a-url"),
        _make_note("https://valid.com/page"),
    ]
    assert _unique_domains(notes) == 1


def test_memo_word_count_sums_sections():
    from eval.run_eval import _memo_word_count

    memo = Memo(
        title="Test",
        summary="short summary",
        sections=[
            MemoSection(heading="A", body="one two three"),
            MemoSection(heading="B", body="four five"),
        ],
        sources=["https://example.com"],
    )
    assert _memo_word_count(memo) == 5


def test_memo_word_count_none_returns_zero():
    from eval.run_eval import _memo_word_count

    assert _memo_word_count(None) == 0
