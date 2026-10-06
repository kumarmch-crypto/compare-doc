"""Compare two versions of a document with Docling + LangChain agents.

Pipeline
--------
1. Both files are parsed with Docling into ``DoclingDocument`` objects
   (PDF, DOCX, PPTX, HTML, Markdown, images, ... anything Docling supports).
2. Each document is flattened into ordered elements (headings, paragraphs,
   list items, tables, pictures) tagged with their section and page.
3. A deterministic aligner (difflib) pairs the elements of both versions and
   produces candidate changes: added / removed / modified / moved.
4. An **analyst agent** inspects those changes through tools backed by the
   Docling documents and returns structured findings.
5. A **reporter agent** verifies the findings (same tools) and writes the
   final Markdown difference report.

Usage
-----
    python compare_docs.py v1.pdf v2.pdf
    python compare_docs.py v1.docx v2.docx -o report.md
    python compare_docs.py v1.pdf v2.pdf --base-url http://gpu-box:8000/v1 --model Qwen/Qwen3-32B
    python compare_docs.py v1.pdf v2.pdf --diff-only     # no LLM, raw diff

LLM backend: a locally served vLLM (OpenAI-compatible API). The agents rely on
tool calling, so start vLLM with tool calling enabled, e.g.:

    vllm serve Qwen/Qwen3-32B --enable-auto-tool-choice --tool-call-parser hermes

(pick the --tool-call-parser that matches your model: hermes, llama3_json,
mistral, ...). If --model is omitted, the first model served is used.
Settings can also come from env vars VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from docling.document_converter import DocumentConverter
from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field

DEFAULT_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
DEFAULT_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM ignores it unless started with --api-key
PAIR_THRESHOLD = 0.5  # min similarity for two elements to count as "modified"
MAX_TOOL_CHARS = 8000  # keep tool outputs within a sane context budget

Version = Literal["old", "new"]


# --------------------------------------------------------------------------- #
# 1. Parsing with Docling
# --------------------------------------------------------------------------- #
@dataclass
class Element:
    idx: int
    kind: str  # docling label: paragraph, list_item, section_header, table, ...
    section: str  # nearest preceding heading
    text: str  # plain text, or Markdown for tables
    page: int | None
    ref: str  # docling self_ref, e.g. "#/texts/12"


def parse_document(converter: DocumentConverter, path: Path) -> DoclingDocument:
    print(f"[docling] parsing {path} ...", file=sys.stderr)
    return converter.convert(str(path)).document


def extract_elements(doc: DoclingDocument) -> list[Element]:
    """Flatten the Docling body tree into reading-order elements."""
    elements: list[Element] = []
    section = "(preamble)"
    for item, _level in doc.iterate_items():
        if isinstance(item, TableItem):
            kind, text = "table", item.export_to_markdown(doc=doc)
        elif isinstance(item, PictureItem):
            kind, text = "picture", f"[picture] {item.caption_text(doc)}".strip()
        elif isinstance(item, TextItem):
            kind, text = item.label.value, item.text.strip()
            if isinstance(item, SectionHeaderItem) or item.label == DocItemLabel.TITLE:
                section = text or section
        else:
            continue
        if not text:
            continue
        page = item.prov[0].page_no if item.prov else None
        elements.append(Element(len(elements), kind, section, text, page, item.self_ref))
    return elements


# --------------------------------------------------------------------------- #
# 2. Deterministic alignment -> candidate changes
# --------------------------------------------------------------------------- #
@dataclass
class Change:
    id: int
    type: Literal["added", "removed", "modified", "moved"]
    old: Element | None
    new: Element | None
    similarity: float

    @property
    def section(self) -> str:
        return (self.new or self.old).section

    @property
    def kind(self) -> str:
        return (self.new or self.old).kind


def _norm(text: str) -> str:
    # Whitespace-insensitive but case-sensitive: capitalisation can change meaning
    # (e.g. "personal data" vs the defined term "Personal Data").
    return re.sub(r"\s+", " ", text).strip()


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b), autojunk=False).ratio()


def compute_changes(old: list[Element], new: list[Element]) -> list[Change]:
    raw: list[tuple[str, Element | None, Element | None, float]] = []
    sm = difflib.SequenceMatcher(
        None, [_norm(e.text) for e in old], [_norm(e.text) for e in new], autojunk=False
    )
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        olds, news = old[i1:i2], new[j1:j2]
        # Greedily pair the most similar old/new elements inside the block.
        candidates = sorted(
            ((_ratio(o.text, n.text), oi, ni) for oi, o in enumerate(olds) for ni, n in enumerate(news)
             if o.kind == n.kind or {o.kind, n.kind} <= {"text", "paragraph", "list_item"}),
            reverse=True,
        )
        used_o, used_n, pairs = set(), set(), {}
        for score, oi, ni in candidates:
            if score < PAIR_THRESHOLD:
                break
            if oi not in used_o and ni not in used_n:
                used_o.add(oi), used_n.add(ni)
                pairs[oi] = (ni, score)
        for oi, o in enumerate(olds):
            if oi in pairs:
                ni, score = pairs[oi]
                raw.append(("modified", o, news[ni], score))
            else:
                raw.append(("removed", o, None, 0.0))
        raw.extend(("added", None, n, 0.0) for ni, n in enumerate(news) if ni not in used_n)

    # Second pass across blocks: an element removed in one place and added
    # elsewhere was moved (verbatim) or moved and edited (similar).
    removed = [k for k, r in enumerate(raw) if r[0] == "removed"]
    added = [k for k, r in enumerate(raw) if r[0] == "added"]
    candidates = sorted(
        ((_ratio(raw[rk][1].text, raw[ak][2].text), rk, ak) for rk in removed for ak in added
         if raw[rk][1].kind == raw[ak][2].kind),
        reverse=True,
    )
    used = set()
    for score, rk, ak in candidates:
        if score < PAIR_THRESHOLD:
            break
        if rk in used or ak in used:
            continue
        used.update((rk, ak))
        o, n = raw[rk][1], raw[ak][2]
        raw[rk] = ("moved" if _norm(o.text) == _norm(n.text) else "modified", o, n, score)
        raw[ak] = ("", None, None, 0.0)

    raw = [r for r in raw if r[0]]  # already in document order (opcode order)
    return [Change(i, t, o, n, round(s, 2)) for i, (t, o, n, s) in enumerate(raw)]


def word_diff(a: str, b: str, context: int = 6) -> str:
    """Inline word diff: [-removed-] {+added+}, long unchanged runs elided."""
    aw, bw = a.split(), b.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, aw, bw, autojunk=False).get_opcodes():
        if tag == "equal":
            seg = aw[i1:i2]
            if len(seg) > 2 * context:
                seg = seg[:context] + ["…"] + seg[-context:]
            out.append(" ".join(seg))
        if tag in ("delete", "replace"):
            out.append("[-" + " ".join(aw[i1:i2]) + "-]")
        if tag in ("insert", "replace"):
            out.append("{+" + " ".join(bw[j1:j2]) + "+}")
    return " ".join(out)


def table_diff(a: str, b: str) -> str:
    return "\n".join(
        difflib.unified_diff(a.splitlines(), b.splitlines(), "old", "new", lineterm="", n=1)
    )


def _clip(text: str, limit: int = MAX_TOOL_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n… [truncated {len(text) - limit} chars]"


# --------------------------------------------------------------------------- #
# 3. Agent tools (closures over the parsed documents)
# --------------------------------------------------------------------------- #
class Workspace:
    def __init__(self, old_path: Path, new_path: Path, old_doc: DoclingDocument, new_doc: DoclingDocument):
        self.paths = {"old": old_path, "new": new_path}
        self.docs = {"old": old_doc, "new": new_doc}
        self.elements = {"old": extract_elements(old_doc), "new": extract_elements(new_doc)}
        self.changes = compute_changes(self.elements["old"], self.elements["new"])

    def sections(self, version: Version) -> list[str]:
        return list(dict.fromkeys(e.section for e in self.elements[version]))

    def overview(self) -> dict:
        return {
            v: {
                "file": self.paths[v].name,
                "pages": len(self.docs[v].pages),
                "elements": len(self.elements[v]),
                "element_kinds": dict(Counter(e.kind for e in self.elements[v])),
                "sections": self.sections(v),
            }
            for v in ("old", "new")
        } | {
            "total_candidate_changes": len(self.changes),
            "changes_by_type": dict(Counter(c.type for c in self.changes)),
            "changes_by_section": dict(Counter(c.section for c in self.changes)),
        }

    def change_detail(self, c: Change) -> dict:
        d = {"id": c.id, "type": c.type, "kind": c.kind, "section": c.section, "similarity": c.similarity}
        if c.old:
            d["old"] = {"page": c.old.page, "section": c.old.section, "text": _clip(c.old.text, 3000)}
        if c.new:
            d["new"] = {"page": c.new.page, "section": c.new.section, "text": _clip(c.new.text, 3000)}
        if c.type == "modified":
            d["diff"] = table_diff(c.old.text, c.new.text) if c.kind == "table" else word_diff(c.old.text, c.new.text)
        return d

    def build_tools(self) -> list:
        ws = self

        @tool
        def get_overview() -> str:
            """Overview of both document versions: files, page counts, element counts,
            section outlines, and counts of candidate changes by type and by section."""
            return _clip(json.dumps(ws.overview(), indent=1, ensure_ascii=False))

        @tool
        def list_changes(offset: int = 0, limit: int = 40, section: str | None = None,
                         change_type: str | None = None) -> str:
            """List candidate changes (paged) with a short preview.
            Optionally filter by section name (substring match) or change_type
            (added | removed | modified | moved). Use get_change for full detail."""
            items = [c for c in ws.changes
                     if (not section or section.lower() in c.section.lower())
                     and (not change_type or c.type == change_type)]
            page = items[offset: offset + limit]
            rows = []
            for c in page:
                preview = (word_diff(c.old.text, c.new.text) if c.type == "modified" and c.kind != "table"
                           else (c.new or c.old).text)
                rows.append({"id": c.id, "type": c.type, "kind": c.kind, "section": c.section,
                             "similarity": c.similarity, "preview": preview[:240]})
            return json.dumps({"total": len(items), "offset": offset, "returned": len(rows),
                               "changes": rows}, ensure_ascii=False)

        @tool
        def get_change(change_id: int) -> str:
            """Full detail of one change: old/new text, pages, sections and an inline
            diff ([-removed-] {+added+}) or a unified diff for tables."""
            if not 0 <= change_id < len(ws.changes):
                return f"No change with id {change_id}. Valid ids: 0..{len(ws.changes) - 1}"
            return json.dumps(ws.change_detail(ws.changes[change_id]), ensure_ascii=False)

        @tool
        def get_section_text(version: Version, section: str) -> str:
            """Full text of a section (all elements under the given heading, substring
            match) from the 'old' or 'new' version. Use for surrounding context."""
            els = [e for e in ws.elements[version] if section.lower() in e.section.lower()]
            if not els:
                return f"No section matching {section!r} in {version}. Sections: {ws.sections(version)}"
            return _clip("\n\n".join(f"[{e.kind}, p.{e.page}] {e.text}" for e in els))

        @tool
        def search_document(version: Version, query: str) -> str:
            """Case-insensitive text search in the 'old' or 'new' version. Returns up to
            20 matching elements with section and page. Useful to check whether content
            was relocated or reworded elsewhere."""
            q = query.lower()
            hits = [e for e in ws.elements[version] if q in e.text.lower()][:20]
            return json.dumps([{"section": e.section, "page": e.page, "kind": e.kind,
                                "text": e.text[:400]} for e in hits], ensure_ascii=False) or "[]"

        @tool
        def export_markdown(version: Version, offset: int = 0) -> str:
            """The whole 'old' or 'new' document as Markdown (Docling export), paged by
            character offset. Prefer the targeted tools; use this for global context."""
            md = ws.docs[version].export_to_markdown()
            chunk = md[offset: offset + MAX_TOOL_CHARS]
            more = f"\n… [{len(md) - offset - len(chunk)} more chars, next offset {offset + len(chunk)}]" \
                if offset + len(chunk) < len(md) else ""
            return chunk + more

        return [get_overview, list_changes, get_change, get_section_text, search_document, export_markdown]


# --------------------------------------------------------------------------- #
# 4. Agents
# --------------------------------------------------------------------------- #
class Finding(BaseModel):
    change_ids: list[int] = Field(description="Candidate change ids this finding covers.")
    section: str = Field(description="Section heading where the change occurs.")
    category: Literal[
        "substantive", "numeric/data", "table", "added content", "removed content",
        "structural/reordering", "editorial/formatting",
    ]
    significance: Literal["high", "medium", "low"]
    description: str = Field(description="What changed and its effect; quote old -> new for key wording.")


class DiffAnalysis(BaseModel):
    findings: list[Finding]
    noise_change_ids: list[int] = Field(
        default_factory=list,
        description="Change ids that are parsing artefacts (OCR noise, hyphenation, page breaks) and not real edits.",
    )


ANALYST_PROMPT = """You are a meticulous document-comparison analyst.
Two versions ("old" and "new") of the same document were parsed with Docling and
aligned automatically into numbered candidate changes. Your job is to examine them
and produce accurate, structured findings.

Method:
1. Call get_overview first.
2. Page through list_changes until you have seen EVERY change id.
3. Call get_change for anything non-trivial; use get_section_text / search_document
   for context, e.g. to tell a real deletion from content moved or reworded elsewhere.
4. Group related changes (e.g. a renumbered list, a reworded paragraph split in two)
   into one finding. Every change id must appear in exactly one finding or in
   noise_change_ids.

Significance: high = changes meaning, obligations, numbers, dates, amounts, names,
scope; medium = noticeable content additions/removals or restructuring;
low = wording, typos, punctuation, formatting.
Be factual. Never invent changes that are not supported by tool output."""

REPORTER_PROMPT = """You write the final difference report between two versions of a
document. You receive structured findings from an analyst. You have the same tools
to verify anything doubtful — spot-check high-significance findings with get_change
and correct mistakes.

Write the report in Markdown with these sections:
# Document Comparison: <old file> → <new file>
## Summary            (3-6 sentences: nature and scale of the revision)
## Key Changes        (high significance, each with section, old vs new quoted)
## Section-by-Section Changes   (grouped by section, medium + low items)
## Added / Removed Content
## Editorial & Formatting Changes   (concise list)
Cite change ids in brackets, e.g. [#12]. Output only the report."""


def _message_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def build_vllm_model(base_url: str, model_name: str | None, api_key: str,
                     max_tokens: int, temperature: float) -> ChatOpenAI:
    """Chat model backed by vLLM's OpenAI-compatible server."""
    if not model_name:
        try:
            served = [m.id for m in OpenAI(base_url=base_url, api_key=api_key).models.list()]
        except Exception as exc:  # connection refused, wrong URL, ...
            sys.exit(f"Cannot reach vLLM at {base_url}: {exc}")
        if not served:
            sys.exit(f"vLLM at {base_url} serves no models")
        model_name = served[0]
    print(f"[vllm] using model {model_name!r} at {base_url}", file=sys.stderr)
    return ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key,
                      max_tokens=max_tokens, temperature=temperature, timeout=600)


def run_agents(ws: Workspace, model: ChatOpenAI) -> str:
    tools = ws.build_tools()
    config = {"recursion_limit": 200}

    analyst = create_agent(model, tools=tools, system_prompt=ANALYST_PROMPT, response_format=ToolStrategy(DiffAnalysis))
    print(f"[analyst] reviewing {len(ws.changes)} candidate changes ...", file=sys.stderr)
    result = analyst.invoke(
        {"messages": [{"role": "user", "content": "Analyse all differences between the old and new document."}]},
        config=config,
    )
    analysis: DiffAnalysis = result["structured_response"]

    covered = {i for f in analysis.findings for i in f.change_ids} | set(analysis.noise_change_ids)
    missed = [c.id for c in ws.changes if c.id not in covered]
    print(f"[analyst] {len(analysis.findings)} findings, {len(missed)} change(s) not covered", file=sys.stderr)

    reporter = create_agent(model, tools=tools, system_prompt=REPORTER_PROMPT)
    brief = {
        "old_file": ws.paths["old"].name,
        "new_file": ws.paths["new"].name,
        "overview": {k: v for k, v in ws.overview().items() if k.startswith("change")},
        "findings": [f.model_dump() for f in analysis.findings],
        "uncovered_change_ids": missed,  # reporter should inspect these itself
    }
    print("[reporter] writing report ...", file=sys.stderr)
    result = reporter.invoke(
        {"messages": [{"role": "user", "content":
            "Analyst findings (JSON). Inspect any uncovered_change_ids with get_change and "
            "include them where relevant.\n\n" + json.dumps(brief, indent=1, ensure_ascii=False)}]},
        config=config,
    )
    return _message_text(result["messages"][-1])


# --------------------------------------------------------------------------- #
# 5. CLI
# --------------------------------------------------------------------------- #
def raw_diff_report(ws: Workspace) -> str:
    lines = [f"# Raw diff: {ws.paths['old'].name} → {ws.paths['new'].name}", ""]
    for c in ws.changes:
        d = ws.change_detail(c)
        lines.append(f"## #{c.id} {c.type} {c.kind} — {c.section}")
        if "diff" in d:
            lines.append(d["diff"])
        else:
            if c.old:
                lines.append(f"- old (p.{c.old.page}): {c.old.text}")
            if c.new:
                lines.append(f"+ new (p.{c.new.page}): {c.new.text}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two versions of a document using Docling + LangChain agents.")
    ap.add_argument("old", type=Path, help="Original version of the document")
    ap.add_argument("new", type=Path, help="Revised version of the document")
    ap.add_argument("-o", "--output", type=Path, help="Write the report to this file (default: stdout)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help=f"vLLM OpenAI-compatible endpoint (default: {DEFAULT_BASE_URL})")
    ap.add_argument("--model", default=os.getenv("VLLM_MODEL"),
                    help="Served model name (default: first model listed by the server)")
    ap.add_argument("--api-key", default=DEFAULT_API_KEY, help="API key if vLLM was started with --api-key")
    ap.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per model response")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--diff-only", action="store_true", help="Skip the agents; print the deterministic diff")
    args = ap.parse_args()

    for p in (args.old, args.new):
        if not p.is_file():
            ap.error(f"file not found: {p}")

    converter = DocumentConverter()
    ws = Workspace(args.old, args.new, parse_document(converter, args.old), parse_document(converter, args.new))

    if not ws.changes:
        report = f"No differences found between {args.old.name} and {args.new.name}."
    elif args.diff_only:
        report = raw_diff_report(ws)
    else:
        model = build_vllm_model(args.base_url, args.model, args.api_key, args.max_tokens, args.temperature)
        report = run_agents(ws, model)

    if args.output:
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(report)


if __name__ == "__main__":
    main()
