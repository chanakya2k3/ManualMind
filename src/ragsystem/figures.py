"""Find pages with images/diagrams and describe them with a vision model.

Manuals draw most diagrams as vector graphics, not embedded images, so
`page.get_images()` alone misses them. We detect graphics from both embedded
images and vector-drawing clusters, then send the whole rendered page to
Claude: seeing the page lets it tie callout numbers to the legend text and
keep multi-panel figures (step 1 > step 2 > step 3) together.
"""

import base64
import json
import pathlib
from typing import Literal

import anthropic
import pymupdf
from pydantic import BaseModel

MODEL = "claude-opus-5-5"
MIN_SIDE = 30    # pt; drops icons, bullets, horizontal rules, heading bars
MIN_AREA = 2500  # pt^2 (~50x50)


class Figure(BaseModel):
    kind: Literal["diagram", "illustration", "photo", "screenshot", "chart", "other"]
    title: str
    description: str
    labels: list[str]


class PageFigures(BaseModel):
    figures: list[Figure]


SYSTEM = """You describe figures in product user manuals so they can be found by \
text search and used to answer user questions.

For each meaningful figure on the page (diagrams, illustrations, photos, app \
screenshots, charts), write:
- title: a short, specific name, e.g. "Attaching the propellers to marked motors".
- description: what the figure shows and what a reader should learn from it. \
Spell out every step in a multi-panel sequence, every arrow or motion, and map \
each numbered or lettered callout to what it points at, using the page text \
for the names. Write it so it stands on its own without the image.
- labels: all text visible inside the figure (callout numbers, part names, \
button labels, screen text), verbatim.

Skip tables (they are extracted separately), page headers and footers, logos, \
and small decorative or warning icons. If the page has no meaningful figure, \
return an empty list."""


def graphic_regions(page: pymupdf.Page, exclude: list[pymupdf.Rect]) -> list[pymupdf.Rect]:
    """Bounding boxes of embedded images and vector drawings big enough to matter."""
    rects = [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()]
    rects += list(page.cluster_drawings())
    return [
        r for r in rects
        if min(r.width, r.height) >= MIN_SIDE
        and r.width * r.height >= MIN_AREA
        and not any(t.contains(r) for t in exclude)
    ]


def describe_page(client: anthropic.Anthropic, page: pymupdf.Page, page_text: str) -> PageFigures:
    png = page.get_pixmap(dpi=150).tobytes("png")
    response = client.beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "medium"},
        system=SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": base64.standard_b64encode(png).decode(),
                }},
                {"type": "text", "text": f"Page {page.number + 1}. Extracted page text for reference:\n\n{page_text}"},
            ],
        }],
        output_format=PageFigures,
    )
    if response.stop_reason == "refusal" or response.parsed_output is None:
        raise RuntimeError(f"page {page.number + 1}: no figure description (stop_reason={response.stop_reason})")
    return response.parsed_output


def extract_figures(
    client: anthropic.Anthropic | None,
    page: pymupdf.Page,
    page_text: str,
    table_rects: list[pymupdf.Rect],
    out: pathlib.Path,
) -> dict | None:
    """Write p###_figures.json for a page with graphics and return its record.

    With client=None, only records where graphics are (no API call). A page
    already described on an earlier run is reused so reruns cost nothing.
    """
    n = page.number + 1
    regions = graphic_regions(page, table_rects)
    if not regions:
        return None
    path = out / f"p{n:03}_figures.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if "figures" in cached or client is None:
            return cached

    record = {"page": n, "regions": [tuple(round(v, 1) for v in r) for r in regions]}
    if client is not None:
        record["figures"] = [f.model_dump() for f in describe_page(client, page, page_text).figures]
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record
