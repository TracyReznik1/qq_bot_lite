"""Renderer tests: pure view over RenderState, deterministic and body-free."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path

from src.search.models import (
    AnswerBlock,
    Claim,
    DisclosureCode,
    EvidenceConflict,
    EvidenceConflictMember,
    RenderOutcome,
    RenderState,
    RequestSource,
    SearchTrace,
    SearchTier,
    WarningCode,
)
from src.search.renderer import render_plain_reply, render_search_reply
from tests.test_chat_retrieval_flow import item


def _state(
    *,
    blocks=(),
    claims=(),
    citation_map=None,
    sources=(),
    conflicts=(),
    disclosures=(),
    warnings=(),
    outcome=RenderOutcome.ANSWER,
):
    return RenderState(
        outcome=outcome,
        visible_blocks=tuple(blocks),
        visible_claims=tuple(claims),
        citation_map=dict(citation_map or {}),
        used_sources=tuple(sources),
        conflict_groups=tuple(conflicts),
        disclosure_codes=tuple(disclosures),
        warning_codes=tuple(warnings),
    )


def _conflict():
    return EvidenceConflict(
        "conflict-1",
        "版本",
        (
            EvidenceConflictMember("E1", "3.2", None, "contradicts"),
            EvidenceConflictMember("E2", "3.3", None, "contradicts"),
        ),
        topic_ids=("topic-1",),
    )


class PureRendererTests(unittest.TestCase):
    def test_success_hides_citations_and_sources_but_keeps_backend_metadata(self):
        block = AnswerBlock("B1", "factual", "版本是3.2", ("C1",))
        claim = Claim("C1", "B1", "版本是3.2", True, ("E1",))
        state = _state(
            blocks=(block,),
            claims=(claim,),
            citation_map={"E1": 1},
            sources=(item(eid="E1", url="https://a.example.com/page", title="Source A"),),
        )
        rendered = render_search_reply(state, qq_limit=1700)
        self.assertEqual("版本是3.2", rendered.text)
        for hidden in (
            "[1]",
            "来源：",
            "Source A",
            "https://a.example.com/page",
        ):
            self.assertNotIn(hidden, rendered.text)
        self.assertEqual(("E1",), rendered.used_evidence_ids)
        self.assertEqual(
            ("https://a.example.com/page",), rendered.shown_source_urls
        )
        for forbidden in ("检索完成", "搜索成功", "搜索状态：success"):
            self.assertNotIn(forbidden, rendered.text)

    def test_partial_disclosure_is_rendered(self):
        block = AnswerBlock("B1", "factual", "版本是3.2", ("C1",))
        claim = Claim("C1", "B1", "版本是3.2", True, ("E1",))
        state = _state(
            blocks=(block,),
            claims=(claim,),
            citation_map={"E1": 1},
            sources=(item(eid="E1"),),
            disclosures=(DisclosureCode.PARTIAL_EVIDENCE,),
            outcome=RenderOutcome.PARTIAL,
        )
        rendered = render_search_reply(state, qq_limit=1700)
        self.assertIn("以下只回答已获得证据支持的部分", rendered.text)

    def test_conflict_renders_natural_claims_without_source_details(self):
        block = AnswerBlock("B1", "factual", "版本有争议", ("C1",))
        claim = Claim("C1", "B1", "版本有争议", True, ("E1", "E2"))
        state = _state(
            blocks=(block,),
            claims=(claim,),
            citation_map={"E1": 1, "E2": 2},
            sources=(
                item(eid="E1", url="https://a.example.com", title="Source A"),
                item(eid="E2", url="https://b.example.com", title="Source B"),
            ),
            conflicts=(_conflict(),),
            disclosures=(DisclosureCode.SOURCE_CONFLICT,),
            outcome=RenderOutcome.CONFLICT,
        )
        rendered = render_search_reply(state, qq_limit=1700)
        self.assertIn("来源之间存在未解决差异", rendered.text)
        self.assertIn("3.2", rendered.text)
        self.assertIn("3.3", rendered.text)
        for hidden in (
            "[1]",
            "[2]",
            "Source A",
            "Source B",
            "https://a.example.com",
            "https://b.example.com",
            "来源：",
        ):
            self.assertNotIn(hidden, rendered.text)
        self.assertEqual(("E1", "E2"), rendered.used_evidence_ids)
        self.assertEqual(
            ("https://a.example.com", "https://b.example.com"),
            rendered.shown_source_urls,
        )

    def test_failure_renders_disclosure_without_sources(self):
        state = _state(
            disclosures=(DisclosureCode.ONLINE_VERIFICATION_FAILED,),
            outcome=RenderOutcome.FAILURE,
        )
        rendered = render_search_reply(state, qq_limit=1700)
        self.assertIn("无法完成在线核验", rendered.text)
        self.assertEqual((), rendered.shown_source_urls)

    def test_warning_rendered_exactly_once(self):
        block = AnswerBlock("B1", "factual", "剂量是99毫克", ("C1",))
        claim = Claim("C1", "B1", "剂量是99毫克", True, ("E1",))
        state = _state(
            blocks=(block,),
            claims=(claim,),
            citation_map={"E1": 1},
            sources=(item(eid="E1"),),
            warnings=(WarningCode.HIGH_CONSEQUENCE,),
        )
        rendered = render_search_reply(state, qq_limit=1700)
        self.assertEqual(1, rendered.text.count("不能替代适当的专业判断"))

    def test_no_warning_for_ordinary_text(self):
        block = AnswerBlock("B1", "factual", "医疗建议", ("C1",))
        claim = Claim("C1", "B1", "医疗建议", True, ("E1",))
        state = _state(
            blocks=(block,),
            claims=(claim,),
            citation_map={"E1": 1},
            sources=(item(eid="E1"),),
        )
        rendered = render_search_reply(state, qq_limit=1700)
        self.assertNotIn("不能替代适当的专业判断", rendered.text)

    def test_citations_never_appear_in_visible_text(self):
        block = AnswerBlock("B1", "factual", "正文", ("C1",))
        claim = Claim("C1", "B1", "正文", True, ("E1",))
        state = _state(
            blocks=(block,),
            claims=(claim,),
            citation_map={"E1": 1},
            sources=(item(eid="E1"),),
        )
        rendered = render_search_reply(state, qq_limit=1700)
        self.assertEqual("正文", rendered.text)
        self.assertNotIn("[1]", rendered.text)
        self.assertEqual(("E1",), rendered.used_evidence_ids)

    def test_same_url_evidence_keeps_each_backend_source_entry(self):
        block = AnswerBlock("B1", "factual", "正文", ("C1",))
        claim = Claim("C1", "B1", "正文", True, ("E1", "E2"))
        shared_url = "https://example.com/shared"
        state = _state(
            blocks=(block,),
            claims=(claim,),
            citation_map={"E1": 1, "E2": 2},
            sources=(
                item(eid="E1", url=shared_url, title="Source A"),
                item(eid="E2", url=shared_url, title="Source B"),
            ),
        )

        rendered = render_search_reply(state, qq_limit=1700)

        self.assertEqual("正文", rendered.text)
        self.assertNotIn("[1]", rendered.text)
        self.assertNotIn("[2]", rendered.text)
        self.assertNotIn(shared_url, rendered.text)
        self.assertEqual(("E1", "E2"), rendered.used_evidence_ids)
        self.assertEqual(
            (shared_url, shared_url), rendered.shown_source_urls
        )

    def test_unused_sources_are_rejected_by_render_state_contract(self):
        block = AnswerBlock("B1", "factual", "正文", ("C1",))
        claim = Claim("C1", "B1", "正文", True, ("E1",))
        with self.assertRaises(ValueError):
            _state(
                blocks=(block,),
                claims=(claim,),
                citation_map={"E1": 1},
                sources=(item(eid="E1"), item(eid="E2", url="https://unused.example.com")),
            )

    def test_long_source_url_stays_backend_only(self):
        block = AnswerBlock("B1", "factual", "正文", ("C1",))
        claim = Claim("C1", "B1", "正文", True, ("E1",))
        url = "https://example.com/" + "a" * 80
        state = _state(
            blocks=(block,),
            claims=(claim,),
            citation_map={"E1": 1},
            sources=(item(eid="E1", url=url, title="Long Title"),),
        )
        rendered = render_search_reply(state, qq_limit=60)
        self.assertEqual(("正文",), rendered.chunks)
        self.assertNotIn(url, rendered.text)
        self.assertEqual((url,), rendered.shown_source_urls)

    def test_render_plain_reply_never_adds_citations(self):
        trace = SearchTrace("req-1", RequestSource.CHAT, SearchTier.SKIP)
        rendered = render_plain_reply("普通回答", trace=trace, qq_limit=1700)
        self.assertEqual("普通回答", rendered.text)
        self.assertEqual((), rendered.used_evidence_ids)


class RendererOwnershipTests(unittest.TestCase):
    def test_renderer_does_not_import_or_inspect_semantic_search_state(self):
        source = Path("src/search/renderer.py").read_text(encoding="utf-8")
        for forbidden in (
            "SearchTier",
            "RiskContext",
            "FreshnessContext",
            "EvidenceState",
            "HIGH_CONSEQUENCE_ACTION",
            "medical",
            "legal",
            "financial",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
