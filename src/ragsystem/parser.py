import argparse, pathlib

import anthropic
import pymupdf

from ragsystem.figures import extract_figures

ap = argparse.ArgumentParser()
ap.add_argument("--describe", action="store_true",
                help="describe figures with Claude (calls the API; needs ANTHROPIC_API_KEY)")
ap.add_argument("--pages", type=int, nargs="+", help="only these page numbers (1-based)")
args = ap.parse_args()

doc = pymupdf.open(r"D:\ManualMind\Manual Data\DJI_Air_2S_User_Manual_v1.0_enIV.pdf")
out = pathlib.Path("debug"); out.mkdir(exist_ok=True)
client = anthropic.Anthropic() if args.describe else None

for page in doc:
    n = page.number + 1
    if args.pages and n not in args.pages:
        continue
    tables = page.find_tables()          # add strategy="text" for borderless tables
    # 1. each table as Markdown
    for i, t in enumerate(tables.tables, 1):
        (out / f"p{n:03}_table{i}.md").write_text(t.to_markdown(), encoding="utf-8")

    # 2. remaining text, excluding anything inside a table
    boxes = [pymupdf.Rect(t.bbox) for t in tables.tables]
    blocks = [b for b in page.get_text("blocks", sort=True)
              if not any(pymupdf.Rect(b[:4]).intersects(r) for r in boxes)]
    text = "\n".join(b[4] for b in blocks)
    (out / f"p{n:03}.txt").write_text(text, encoding="utf-8")

    # 3. images and diagrams
    record = extract_figures(client, page, text, boxes, out)
    if record:
        described = f", {len(record['figures'])} figure(s) described" if "figures" in record else ""
        print(f"p{n:03}: {len(record['regions'])} graphic region(s){described}")
