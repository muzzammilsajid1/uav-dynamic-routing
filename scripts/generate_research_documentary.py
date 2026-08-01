"""Generate the interim UAV Dynamic Routing research documentary PDF.

The editable narrative is kept in:
    output/pdf/UAV_Dynamic_Routing_Research_Documentary_Interim.md

This generator intentionally uses only repository evidence already summarized
in that source. It creates vector diagrams with ReportLab so the PDF remains
sharp without requiring external image-generation services.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pypdf import PdfReader
from reportlab.graphics.shapes import Circle, Drawing, Line, Path as GPath, Polygon
from reportlab.graphics.shapes import Rect, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Flowable,
    Frame,
    KeepTogether,
    LongTable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf"
DEFAULT_SOURCE = OUTPUT_DIR / "UAV_Dynamic_Routing_Research_Documentary_Interim.md"
DEFAULT_OUTPUT = OUTPUT_DIR / "UAV_Dynamic_Routing_Research_Documentary_Interim.pdf"

PAGE_WIDTH, PAGE_HEIGHT = A4
LEFT_MARGIN = 18 * mm
RIGHT_MARGIN = 18 * mm
TOP_MARGIN = 20 * mm
BOTTOM_MARGIN = 18 * mm
CONTENT_WIDTH = PAGE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

NAVY = HexColor("#0B2239")
INK = HexColor("#17212B")
SLATE = HexColor("#536273")
TEAL = HexColor("#138A8A")
CYAN = HexColor("#4BB7C5")
ORANGE = HexColor("#F2A541")
GOLD = HexColor("#D8A628")
GREEN = HexColor("#2D9B73")
RED = HexColor("#C8553D")
PALE_BLUE = HexColor("#EAF4F7")
PALE_TEAL = HexColor("#EAF7F4")
PALE_ORANGE = HexColor("#FFF4E4")
PALE_RED = HexColor("#FCEDEA")
PALE_GREY = HexColor("#F3F5F7")
MID_GREY = HexColor("#D7DEE5")
WHITE = colors.white


def _register_fonts() -> None:
    candidates = {
        "Arial": Path("C:/Windows/Fonts/arial.ttf"),
        "Arial-Bold": Path("C:/Windows/Fonts/arialbd.ttf"),
        "Arial-Italic": Path("C:/Windows/Fonts/ariali.ttf"),
        "Arial-BoldItalic": Path("C:/Windows/Fonts/arialbi.ttf"),
        "Consolas": Path("C:/Windows/Fonts/consola.ttf"),
        "Consolas-Bold": Path("C:/Windows/Fonts/consolab.ttf"),
    }
    for name, path in candidates.items():
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        "Arial",
        normal="Arial",
        bold="Arial-Bold",
        italic="Arial-Italic",
        boldItalic="Arial-BoldItalic",
    )


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_kicker": ParagraphStyle(
            "CoverKicker",
            parent=base["Normal"],
            fontName="Arial-Bold",
            fontSize=10,
            leading=13,
            textColor=CYAN,
            spaceAfter=12,
            tracking=1.2,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Arial-Bold",
            fontSize=31,
            leading=34,
            textColor=WHITE,
            alignment=TA_LEFT,
            spaceAfter=14,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName="Arial",
            fontSize=14,
            leading=19,
            textColor=HexColor("#D8E7EF"),
            spaceAfter=18,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=base["Normal"],
            fontName="Arial",
            fontSize=9.4,
            leading=14,
            textColor=HexColor("#B9CCD7"),
        ),
        "h1": ParagraphStyle(
            "DocH1",
            parent=base["Heading1"],
            fontName="Arial-Bold",
            fontSize=21,
            leading=25,
            textColor=NAVY,
            spaceBefore=0,
            spaceAfter=13,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "DocH2",
            parent=base["Heading2"],
            fontName="Arial-Bold",
            fontSize=14.2,
            leading=18,
            textColor=TEAL,
            spaceBefore=13,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "DocH3",
            parent=base["Heading3"],
            fontName="Arial-Bold",
            fontSize=11.2,
            leading=14,
            textColor=INK,
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "DocBody",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=9.35,
            leading=13.3,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6.6,
            splitLongWords=True,
            allowWidows=0,
            allowOrphans=0,
        ),
        "bullet": ParagraphStyle(
            "DocBullet",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=9.2,
            leading=12.8,
            textColor=INK,
            leftIndent=14,
            firstLineIndent=-8,
            bulletIndent=4,
            spaceAfter=3.8,
            splitLongWords=True,
        ),
        "quote": ParagraphStyle(
            "DocQuote",
            parent=base["BodyText"],
            fontName="Arial-Bold",
            fontSize=9.3,
            leading=13.2,
            textColor=NAVY,
            backColor=PALE_BLUE,
            borderColor=CYAN,
            borderWidth=1,
            borderPadding=(8, 9, 8, 10),
            leftIndent=0,
            rightIndent=0,
            spaceBefore=6,
            spaceAfter=9,
        ),
        "equation": ParagraphStyle(
            "Equation",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=10.5,
            leading=15,
            textColor=NAVY,
            alignment=TA_CENTER,
            backColor=PALE_GREY,
            borderPadding=8,
            spaceBefore=5,
            spaceAfter=9,
            splitLongWords=True,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Consolas",
            fontSize=7.3,
            leading=10,
            textColor=INK,
            backColor=PALE_GREY,
            borderPadding=7,
            spaceBefore=5,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontName="Arial-Italic",
            fontSize=7.7,
            leading=10,
            textColor=SLATE,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=11,
        ),
        "toc_title": ParagraphStyle(
            "TOCTitle",
            parent=base["Heading1"],
            fontName="Arial-Bold",
            fontSize=22,
            leading=26,
            textColor=NAVY,
            spaceAfter=15,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Arial",
            fontSize=7.8,
            leading=10.5,
            textColor=SLATE,
        ),
    }


class DocumentaryDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, styles: dict[str, ParagraphStyle]) -> None:
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT_MARGIN,
            rightMargin=RIGHT_MARGIN,
            topMargin=TOP_MARGIN,
            bottomMargin=BOTTOM_MARGIN,
            title="From First Grid to Reproducible Study",
            author="UAV Dynamic Routing Research Team",
            subject="Interim documentary record of the UAV Dynamic Routing project",
        )
        self.styles = styles
        cover_frame = Frame(
            22 * mm,
            21 * mm,
            PAGE_WIDTH - 44 * mm,
            PAGE_HEIGHT - 42 * mm,
            id="cover-frame",
            showBoundary=0,
        )
        content_frame = Frame(
            LEFT_MARGIN,
            BOTTOM_MARGIN,
            CONTENT_WIDTH,
            PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN,
            id="content-frame",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            showBoundary=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(
                    id="cover",
                    frames=[cover_frame],
                    onPage=self._draw_cover_page,
                ),
                PageTemplate(
                    id="content",
                    frames=[content_frame],
                    # Draw furniture after flowables so large vector figures or
                    # split blocks cannot paint over the running header/footer.
                    onPageEnd=self._draw_content_page,
                ),
            ]
        )
        self._bookmark_index = 0

    def beforeDocument(self) -> None:
        self._bookmark_index = 0
        super().beforeDocument()

    def _draw_cover_page(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, fill=1, stroke=0)
        canvas.setFillColor(HexColor("#0E344D"))
        canvas.circle(PAGE_WIDTH - 45, PAGE_HEIGHT - 58, 120, fill=1, stroke=0)
        canvas.setFillColor(HexColor("#102D44"))
        canvas.circle(42, 58, 105, fill=1, stroke=0)

        cell = 9
        origin_x = PAGE_WIDTH - 190
        origin_y = 58
        canvas.setStrokeColor(HexColor("#35566A"))
        canvas.setLineWidth(0.45)
        for idx in range(15):
            canvas.line(origin_x + idx * cell, origin_y, origin_x + idx * cell, origin_y + 14 * cell)
            canvas.line(origin_x, origin_y + idx * cell, origin_x + 14 * cell, origin_y + idx * cell)
        blocks = [(2, 10), (3, 10), (4, 8), (6, 6), (7, 6), (9, 5), (11, 3)]
        canvas.setFillColor(HexColor("#2E5166"))
        for row, col in blocks:
            canvas.rect(origin_x + col * cell, origin_y + (13 - row) * cell, cell, cell, fill=1, stroke=0)

        route = [(1, 1), (2, 2), (3, 3), (4, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)]
        canvas.setStrokeColor(CYAN)
        canvas.setLineWidth(2.2)
        for left, right in zip(route, route[1:]):
            x1 = origin_x + left[1] * cell + cell / 2
            y1 = origin_y + (13 - left[0]) * cell + cell / 2
            x2 = origin_x + right[1] * cell + cell / 2
            y2 = origin_y + (13 - right[0]) * cell + cell / 2
            canvas.line(x1, y1, x2, y2)
        canvas.setFillColor(ORANGE)
        canvas.circle(origin_x + 1.5 * cell, origin_y + 12.5 * cell, 3.2, fill=1, stroke=0)
        canvas.setFillColor(GREEN)
        canvas.circle(origin_x + 13.5 * cell, origin_y + 1.5 * cell, 3.2, fill=1, stroke=0)

        canvas.setStrokeColor(CYAN)
        canvas.setLineWidth(2)
        canvas.line(22 * mm, PAGE_HEIGHT - 22 * mm, 72 * mm, PAGE_HEIGHT - 22 * mm)
        canvas.restoreState()

    def _draw_content_page(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(MID_GREY)
        canvas.setLineWidth(0.6)
        canvas.line(LEFT_MARGIN, PAGE_HEIGHT - 13.2 * mm, PAGE_WIDTH - RIGHT_MARGIN, PAGE_HEIGHT - 13.2 * mm)
        canvas.setFont("Arial-Bold", 7.3)
        canvas.setFillColor(SLATE)
        canvas.drawString(LEFT_MARGIN, PAGE_HEIGHT - 10.2 * mm, "UAV DYNAMIC ROUTING - INTERIM RESEARCH RECORD")
        canvas.setFont("Arial", 7.3)
        canvas.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, PAGE_HEIGHT - 10.2 * mm, "Evidence cutoff 2026-07-27")

        canvas.setStrokeColor(MID_GREY)
        canvas.line(LEFT_MARGIN, 12.7 * mm, PAGE_WIDTH - RIGHT_MARGIN, 12.7 * mm)
        canvas.setFont("Arial", 7.2)
        canvas.setFillColor(SLATE)
        canvas.drawString(LEFT_MARGIN, 8.7 * mm, "From First Grid to Reproducible Study")
        canvas.drawRightString(PAGE_WIDTH - RIGHT_MARGIN, 8.7 * mm, f"{doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable: Flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        levels = {"DocH1": 0, "DocH2": 1, "DocH3": 2}
        if style_name not in levels:
            return
        level = levels[style_name]
        text = flowable.getPlainText()
        self._bookmark_index += 1
        key = f"heading-{self._bookmark_index}"
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


def _inline_markup(text: str) -> str:
    safe = html.escape(text.strip())
    safe = re.sub(r"`([^`]+)`", r'<font name="Consolas" color="#0B5D6A">\1</font>', safe)
    safe = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", safe)
    return safe


def _equation_text(text: str) -> str:
    value = " ".join(text.split())
    if value.startswith(r"\text{deployment value}"):
        return "deployment value = f(success, route cost, decision cost, adaptation, scale, robustness)"
    if value.startswith("F(s,s')"):
        return "F(s,s') = γ Φ(s') - Φ(s),    γ = 0.99"
    if r"\frac{d_{octile}" in value:
        return "Φ(s) = - d_octile(s, g) / d_max"
    if value.startswith("d_{octile}"):
        return "d_octile = min(Δr, Δc) √2 + |Δr - Δc|"
    if value.startswith("O("):
        return "O((V + E) log V)"
    if value.startswith("Q(s,a)"):
        return "Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') - Q(s,a) ]"
    if value.startswith(r"\Delta C"):
        return "ΔC* = C*_after - C*_before"
    if value.startswith("y =") and r"Q_{target}" in value:
        return "y = r + (1 - d) γ max_a' Q_target(s', a')"
    if value.startswith("a^*"):
        return "a* = arg max_a' Q_online(s', a')    |    y = r + (1 - d) γ Q_target(s', a*)"
    if value.startswith(r"C = \sum"):
        return "C = Σ_(t=0)^(T-1) [ c_move(v_t, v_(t+1)) + p(v_(t+1)) ]"
    replacements = {
        r"\leftarrow": "←",
        r"\alpha": "α",
        r"\gamma": "γ",
        r"\Phi": "Φ",
        r"\Delta": "Δ",
        r"\sqrt{2}": "√2",
        r"\left": "",
        r"\right": "",
        r"\qquad": "    ",
        r"\text": "",
        r"\min": "min",
        r"\max": "max",
        r"\arg": "arg",
        r"\sum": "Σ",
        r"\in": "∈",
        r"\log": "log",
        r"\cdot": "·",
        r"\arrow": "→",
        "&": " ",
        "{": "",
        "}": "",
        "_{": "_",
        "^{": "^",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = value.replace("\\", "")
    return html.escape(value)


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_inline_markup(text), style)


def _table_flowable(rows: list[list[str]], styles: dict[str, ParagraphStyle]) -> LongTable:
    column_count = max(len(row) for row in rows)
    normalized = [row + [""] * (column_count - len(row)) for row in rows]
    header = normalized[0]
    body = normalized[2:] if len(normalized) > 1 and all(set(cell) <= {"-", ":"} for cell in normalized[1]) else normalized[1:]
    data = [header, *body]

    lengths = []
    for index in range(column_count):
        longest = max(len(re.sub(r"`|\*|<[^>]+>", "", row[index])) for row in data)
        lengths.append(max(8, min(longest, 42)))
    total = sum(lengths)
    widths = [CONTENT_WIDTH * value / total for value in lengths]

    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["small"],
        fontName="Arial",
        fontSize=7.5 if column_count <= 5 else 6.7,
        leading=9.4 if column_count <= 5 else 8.2,
        textColor=INK,
        splitLongWords=True,
    )
    head_style = ParagraphStyle(
        "TableHead",
        parent=cell_style,
        fontName="Arial-Bold",
        textColor=WHITE,
    )
    formatted = []
    for row_index, row in enumerate(data):
        formatted.append(
            [
                Paragraph(_inline_markup(cell), head_style if row_index == 0 else cell_style)
                for cell in row
            ]
        )

    table = LongTable(
        formatted,
        colWidths=widths,
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
        spaceBefore=5,
        spaceAfter=10,
    )
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, MID_GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in range(1, len(formatted)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE_GREY))
    table.setStyle(TableStyle(commands))
    return table


def _arrow(drawing: Drawing, x1: float, y1: float, x2: float, y2: float, color=TEAL, width=1.5) -> None:
    drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=width))
    angle_dx = x2 - x1
    angle_dy = y2 - y1
    length = max((angle_dx**2 + angle_dy**2) ** 0.5, 1)
    ux = angle_dx / length
    uy = angle_dy / length
    px = -uy
    py = ux
    size = 6
    drawing.add(
        Polygon(
            [
                x2,
                y2,
                x2 - size * ux + 2.5 * px,
                y2 - size * uy + 2.5 * py,
                x2 - size * ux - 2.5 * px,
                y2 - size * uy - 2.5 * py,
            ],
            fillColor=color,
            strokeColor=color,
        )
    )


def _box(
    drawing: Drawing,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    *,
    fill=PALE_BLUE,
    stroke=CYAN,
    title_color=NAVY,
    subtitle_color=SLATE,
    title_size=9.2,
    subtitle_size=6.9,
) -> None:
    drawing.add(Rect(x, y, width, height, rx=6, ry=6, fillColor=fill, strokeColor=stroke, strokeWidth=1.1))
    drawing.add(String(x + 8, y + height - 15, title, fontName="Arial-Bold", fontSize=title_size, fillColor=title_color))
    if subtitle:
        lines = _wrap_drawing_text(subtitle, max(16, int(width / (subtitle_size * 0.53))))
        for line_index, line in enumerate(lines[:4]):
            drawing.add(
                String(
                    x + 8,
                    y + height - 29 - line_index * (subtitle_size + 2),
                    line,
                    fontName="Arial",
                    fontSize=subtitle_size,
                    fillColor=subtitle_color,
                )
            )


def _wrap_drawing_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _title_on_drawing(d: Drawing, title: str, subtitle: str = "") -> None:
    d.add(String(0, d.height - 16, title, fontName="Arial-Bold", fontSize=12, fillColor=NAVY))
    if subtitle:
        d.add(String(0, d.height - 30, subtitle, fontName="Arial", fontSize=7.4, fillColor=SLATE))


def _figure_status() -> tuple[Drawing, str]:
    d = Drawing(CONTENT_WIDTH, 235)
    _title_on_drawing(d, "Evidence status at the documentary cutoff", "Implementation is further advanced than the final empirical evidence.")
    cards = [
        ("58 / 58", "tests passed", GREEN, PALE_TEAL),
        ("5 / 5", "full-method seeds", TEAL, PALE_BLUE),
        ("4 / 5", "vanilla-DQN seeds", ORANGE, PALE_ORANGE),
        ("310", "persisted scenarios", NAVY, PALE_GREY),
    ]
    card_y = 123
    card_w = (CONTENT_WIDTH - 27) / 4
    for index, (value, label, accent, fill) in enumerate(cards):
        x = index * (card_w + 9)
        d.add(Rect(x, card_y, card_w, 68, rx=7, ry=7, fillColor=fill, strokeColor=accent, strokeWidth=1))
        d.add(String(x + 10, card_y + 38, value, fontName="Arial-Bold", fontSize=20, fillColor=accent))
        d.add(String(x + 10, card_y + 19, label, fontName="Arial", fontSize=7.5, fillColor=SLATE))
    d.add(Rect(0, 24, CONTENT_WIDTH, 78, rx=7, ry=7, fillColor=NAVY, strokeColor=NAVY))
    d.add(String(14, 78, "INTERIM - cloud training active", fontName="Arial-Bold", fontSize=13, fillColor=WHITE))
    d.add(String(14, 58, "Active recorded task: dqn:seed055 on free Google Colab T4", fontName="Arial", fontSize=8.4, fillColor=HexColor("#D8E7EF")))
    d.add(String(14, 41, "Final 310-scenario RL evaluation, statistics, integrity report, and release paper: pending", fontName="Arial", fontSize=8.1, fillColor=HexColor("#D8E7EF")))
    d.add(Circle(CONTENT_WIDTH - 30, 64, 10, fillColor=ORANGE, strokeColor=WHITE, strokeWidth=1.2))
    return d, "Figure 1. Status labels reflect evidence inspected at 2026-07-27 22:03 PKT; they do not infer completion from partial checkpoints."


def _figure_concept_map() -> tuple[Drawing, str]:
    d = Drawing(CONTENT_WIDTH, 285)
    _title_on_drawing(d, "The research question is a multi-objective trade-off")
    center_x = CONTENT_WIDTH / 2 - 83
    center_y = 108
    _box(
        d,
        center_x,
        center_y,
        166,
        62,
        "Which routing approach?",
        "Compare learned reaction with full, heuristic, and incremental search.",
        fill=NAVY,
        stroke=NAVY,
        title_color=WHITE,
        subtitle_color=HexColor("#D8E7EF"),
    )
    nodes = [
        (8, 188, 110, 54, "Reliability", "Does the route reach the goal?", GREEN, PALE_TEAL),
        (135, 202, 110, 54, "Path quality", "How costly is the realized route?", TEAL, PALE_BLUE),
        (262, 202, 110, 54, "Decision cost", "Route total and per decision.", ORANGE, PALE_ORANGE),
        (389, 188, 110, 54, "Adaptation", "What follows a visible change?", RED, PALE_RED),
        (50, 30, 125, 54, "Scale", "15 to 100 cells per side.", NAVY, PALE_GREY),
        (192, 18, 125, 54, "Robustness", "Unseen maps, dynamics, noise.", TEAL, PALE_BLUE),
        (334, 30, 125, 54, "Reproducibility", "Seeds, hashes, raw Cartesian data.", GREEN, PALE_TEAL),
    ]
    for x, y, w, h, title, subtitle, stroke, fill in nodes:
        _box(d, x, y, w, h, title, subtitle, stroke=stroke, fill=fill, title_size=8.7, subtitle_size=6.3)
        target_x = x + w / 2
        target_y = y + h / 2
        origin_x = center_x + 83
        origin_y = center_y + 31
        if y > center_y:
            _arrow(d, origin_x, center_y + 62, target_x, y, color=stroke, width=0.9)
        else:
            _arrow(d, origin_x, center_y, target_x, y + h, color=stroke, width=0.9)
    return d, "Figure 2. The expanded study refuses to reduce routing quality to a single speed or success number."


def _figure_environment() -> tuple[Drawing, str]:
    d = Drawing(CONTENT_WIDTH, 330)
    _title_on_drawing(d, "Shared grid world, local observation, and eight actions")
    grid_x = 12
    grid_y = 38
    cell = 25
    size = 9
    obstacles = {(0, 6), (1, 1), (1, 5), (2, 4), (3, 1), (4, 6), (5, 3), (6, 2), (7, 6), (8, 5)}
    dynamic = {(2, 6), (6, 6)}
    start = (7, 1)
    goal = (1, 7)
    route = [(7, 1), (6, 1), (5, 2), (4, 3), (3, 4), (2, 5), (1, 6), (1, 7)]
    for row in range(size):
        for col in range(size):
            x = grid_x + col * cell
            y = grid_y + (size - 1 - row) * cell
            fill = WHITE
            if (row, col) in obstacles:
                fill = HexColor("#526778")
            if (row, col) in dynamic:
                fill = ORANGE
            d.add(Rect(x, y, cell, cell, fillColor=fill, strokeColor=MID_GREY, strokeWidth=0.55))
    for left, right in zip(route, route[1:]):
        x1 = grid_x + left[1] * cell + cell / 2
        y1 = grid_y + (size - 1 - left[0]) * cell + cell / 2
        x2 = grid_x + right[1] * cell + cell / 2
        y2 = grid_y + (size - 1 - right[0]) * cell + cell / 2
        d.add(Line(x1, y1, x2, y2, strokeColor=TEAL, strokeWidth=2.4))
    for node, label, fill in [(start, "S", GREEN), (goal, "G", RED)]:
        x = grid_x + node[1] * cell + cell / 2
        y = grid_y + (size - 1 - node[0]) * cell + cell / 2
        d.add(Circle(x, y, 8, fillColor=fill, strokeColor=WHITE, strokeWidth=1))
        d.add(String(x - 3.2, y - 3.1, label, fontName="Arial-Bold", fontSize=7.5, fillColor=WHITE))
    # Local 7x7 view around the current route position (4,3).
    view_row, view_col = (4, 3)
    d.add(
        Rect(
            grid_x + (view_col - 3) * cell,
            grid_y + (size - 1 - (view_row + 3)) * cell,
            7 * cell,
            7 * cell,
            fillColor=None,
            strokeColor=CYAN,
            strokeWidth=2.2,
            strokeDashArray=[5, 3],
        )
    )
    d.add(String(grid_x, 18, "Gray: static block   Orange: changing cell   Cyan box: 7x7 local observation", fontName="Arial", fontSize=7, fillColor=SLATE))

    compass_x = 382
    compass_y = 152
    d.add(Circle(compass_x, compass_y, 24, fillColor=PALE_BLUE, strokeColor=CYAN, strokeWidth=1))
    d.add(Circle(compass_x, compass_y, 5, fillColor=TEAL, strokeColor=TEAL))
    compass = [
        (0, 62, "N", 0, 47),
        (44, 44, "NE", 32, 34),
        (62, 0, "E", 49, -3),
        (44, -44, "SE", 32, -39),
        (0, -62, "S", -3, -51),
        (-44, -44, "SW", -55, -39),
        (-62, 0, "W", -66, -3),
        (-44, 44, "NW", -57, 34),
    ]
    for dx, dy, label, tx, ty in compass:
        _arrow(d, compass_x, compass_y, compass_x + dx * 0.62, compass_y + dy * 0.62, color=NAVY, width=1)
        d.add(String(compass_x + tx, compass_y + ty, label, fontName="Arial-Bold", fontSize=7.2, fillColor=NAVY))
    _box(d, 320, 42, 170, 64, "Move-then-observe", "Choose on current state; move; advance dynamics; expose the change before the next decision.", fill=PALE_TEAL, stroke=GREEN, title_size=9, subtitle_size=6.8)
    _box(d, 320, 224, 170, 57, "Cost contract", "Straight 1, diagonal sqrt(2), plus any non-negative destination penalty.", fill=PALE_ORANGE, stroke=ORANGE, title_size=9, subtitle_size=6.8)
    return d, "Figure 3. The same movement geometry and information timing govern classical planning and RL evaluation."


def _figure_learning() -> tuple[Drawing, str]:
    d = Drawing(CONTENT_WIDTH, 300)
    _title_on_drawing(d, "Goal-conditioned Double DQN + HER learning loop")
    _box(d, 5, 175, 120, 72, "61-value observation", "Position + goal displacement + 7x7 window + previous action.", fill=PALE_BLUE, stroke=CYAN)
    _box(d, 190, 175, 120, 72, "Online Q-network", "Two hidden layers, 128 units each; epsilon-greedy action.", fill=PALE_TEAL, stroke=GREEN)
    _box(d, 375, 175, 120, 72, "Environment", "Shared eight-action grid, rewards, dynamics, termination.", fill=PALE_ORANGE, stroke=ORANGE)
    _arrow(d, 125, 211, 190, 211)
    _arrow(d, 310, 211, 375, 211)
    _arrow(d, 435, 175, 435, 132, color=ORANGE)
    _box(d, 315, 58, 180, 68, "Replay buffer + HER", "Store transition; relabel future achieved goals; correct terminal flags.", fill=PALE_RED, stroke=RED)
    _arrow(d, 315, 92, 255, 92, color=RED)
    _box(d, 130, 58, 125, 68, "Double-DQN target", "Online network selects; target network evaluates.", fill=PALE_GREY, stroke=NAVY)
    _arrow(d, 192, 126, 235, 175, color=NAVY)
    d.add(String(15, 143, "Reward = sparse outcome + gamma Phi(s') - Phi(s)", fontName="Arial-Bold", fontSize=9, fillColor=NAVY))
    d.add(String(15, 128, "Dynamic HER uses obstacle-state-independent octile Phi.", fontName="Arial", fontSize=7.4, fillColor=SLATE))
    return d, "Figure 4. Correct reward and terminal semantics are required for HER relabeling to produce valid learning targets."


def _figure_timeline() -> tuple[Drawing, str]:
    d = Drawing(CONTENT_WIDTH, 330)
    _title_on_drawing(d, "Eleven days of rapid methodological expansion")
    x0 = 28
    x1 = CONTENT_WIDTH - 22
    y = 160
    d.add(Line(x0, y, x1, y, strokeColor=NAVY, strokeWidth=2))
    events = [
        ("16 Jul", "Grid +\nDijkstra", 0.00, TEAL, True),
        ("18 Jul", "HER speed +\nshaping", 0.18, ORANGE, False),
        ("19 Jul", "Terminal fix,\n300k model", 0.28, RED, True),
        ("21 Jul", "Dynamic\nreplanning", 0.47, TEAL, False),
        ("22 Jul", "50-pair result\n+ statistics", 0.57, GREEN, True),
        ("23-25", "Paper v1,\nA*, v2 split", 0.70, NAVY, False),
        ("26 Jul", "Expanded\nstudy build", 0.83, ORANGE, True),
        ("27 Jul", "Corrected seeds\n+ Colab", 1.00, GREEN, False),
    ]
    for date, label, fraction, color, above in events:
        x = x0 + fraction * (x1 - x0)
        d.add(Circle(x, y, 5.2, fillColor=color, strokeColor=WHITE, strokeWidth=1))
        stem_end = y + 57 if above else y - 57
        d.add(Line(x, y, x, stem_end, strokeColor=color, strokeWidth=1.1))
        box_y = stem_end if above else stem_end - 54
        box_x = min(max(x - 45, 0), CONTENT_WIDTH - 90)
        _box(d, box_x, box_y, 90, 51, date, label.replace("\n", " "), fill=WHITE, stroke=color, title_size=7.8, subtitle_size=6.1)
    d.add(String(8, 30, "Git records stop at f57ae15; the expanded implementation and cloud workflow were still uncommitted at the cutoff.", fontName="Arial-Italic", fontSize=7.2, fillColor=SLATE))
    return d, "Figure 5. The chronology explains why historical results and corrected v2 evidence must remain separate."


def _figure_evolution() -> tuple[Drawing, str]:
    d = Drawing(CONTENT_WIDTH, 270)
    _title_on_drawing(d, "From a two-method demonstration to an auditable study")
    stages = [
        (0, "Prototype", "Seeded grid\nDijkstra\nTabular Q / DQN", PALE_GREY, NAVY),
        (126, "Early paper", "One learned policy\n50 paired routes\nHonest Dijkstra advantage", PALE_ORANGE, ORANGE),
        (252, "Method repair", "Exact scenarios\nPost-move timing\nLocked source\nFive seeds", PALE_RED, RED),
        (378, "Expanded study", "A* + D* Lite\n310 scenarios\nAblations\nIntegrity-gated release", PALE_TEAL, GREEN),
    ]
    for x, title, subtitle, fill, stroke in stages:
        _box(d, x, 82, 112, 112, title, subtitle.replace("\n", " | "), fill=fill, stroke=stroke, title_size=10, subtitle_size=6.8)
        if x < 378:
            _arrow(d, x + 112, 138, x + 126, 138, color=TEAL, width=1.5)
    d.add(String(0, 49, "Core change:", fontName="Arial-Bold", fontSize=8.5, fillColor=NAVY))
    d.add(String(61, 49, "the project shifted from asking whether RL can route to asking exactly when, why, and with what uncertainty it is competitive.", fontName="Arial", fontSize=7.5, fillColor=SLATE))
    return d, "Figure 6. Each expansion was motivated by a concrete weakness in the earlier evidence."


def _figure_benchmark() -> tuple[Drawing, str]:
    d = Drawing(CONTENT_WIDTH, 320)
    _title_on_drawing(d, "Expanded benchmark composition", "310 persisted scenarios across 21 named splits")
    families = [
        ("Generalization", 90, TEAL),
        ("Fixed dynamics shift", 60, ORANGE),
        ("Scaling", 40, NAVY),
        ("Density", 20, GREEN),
        ("Stochastic", 20, CYAN),
        ("Moving", 20, RED),
        ("Energy", 20, GOLD),
        ("No-fly", 20, HexColor("#7A5FA0")),
        ("Sensor noise", 20, HexColor("#667788")),
    ]
    max_value = max(value for _, value, _ in families)
    start_y = 250
    for index, (label, value, color) in enumerate(families):
        y = start_y - index * 25
        d.add(String(0, y + 3, label, fontName="Arial", fontSize=7.4, fillColor=INK))
        bar_x = 105
        bar_width = 325 * value / max_value
        d.add(Rect(bar_x, y, 325, 12, fillColor=PALE_GREY, strokeColor=None))
        d.add(Rect(bar_x, y, bar_width, 12, fillColor=color, strokeColor=None))
        d.add(String(bar_x + bar_width + 7, y + 2, str(value), fontName="Arial-Bold", fontSize=7.5, fillColor=color))
    d.add(String(0, 19, "Grid sizes: 280 scenarios at 15x15; 10 each at 30x30, 50x50, and 100x100.", fontName="Arial-Italic", fontSize=7.2, fillColor=SLATE))
    return d, "Figure 7. Scenario identity is persisted once and reconstructed identically for every method."


def _figure_training_pipeline() -> tuple[Drawing, str]:
    d = Drawing(CONTENT_WIDTH, 300)
    _title_on_drawing(d, "Five-seed curriculum and controlled ablation matrix")
    stage_y = 185
    widths = [230, 85, 85]
    labels = [
        ("Static", "300k steps\ncurriculum", PALE_BLUE, TEAL),
        ("Dynamic mild", "100k steps\none toggle", PALE_ORANGE, ORANGE),
        ("Dynamic full", "100k steps\nthree toggles", PALE_RED, RED),
    ]
    x = 0
    for index, ((title, subtitle, fill, stroke), width) in enumerate(zip(labels, widths)):
        _box(d, x, stage_y, width, 60, title, subtitle.replace("\n", " | "), fill=fill, stroke=stroke, title_size=9.5, subtitle_size=7)
        if index < len(widths) - 1:
            _arrow(d, x + width, stage_y + 30, x + width + 12, stage_y + 30)
            x += width + 12
        else:
            x += width
    d.add(String(0, 157, "Policy seeds: 11, 22, 33, 44, 55   |   layout seed: 42   |   500,000 steps per seed", fontName="Arial-Bold", fontSize=8.1, fillColor=NAVY))
    variants = ["full", "dqn", "no_her", "no_shaping", "no_curriculum", "full_observation", "dynamic_scratch"]
    start_x = 0
    for index, variant in enumerate(variants):
        box_w = 68
        x = start_x + index * 71
        fill = PALE_TEAL if variant == "full" else PALE_GREY
        stroke = GREEN if variant == "full" else MID_GREY
        d.add(Rect(x, 78, box_w, 42, rx=4, ry=4, fillColor=fill, strokeColor=stroke, strokeWidth=0.8))
        short = variant.replace("_", "\n")
        lines = short.split("\n")
        for line_index, line in enumerate(lines[:2]):
            d.add(String(x + 5, 101 - line_index * 11, line, fontName="Arial-Bold" if variant == "full" else "Arial", fontSize=6.7, fillColor=NAVY))
    d.add(String(0, 51, "Configured total: 35 seed-variant trainings = 17.5 million steps", fontName="Arial-Bold", fontSize=9, fillColor=NAVY))
    d.add(String(0, 34, "Verified at cutoff: 5 full + 4 DQN seed completions = 4.5 million steps with complete metadata.", fontName="Arial", fontSize=7.4, fillColor=SLATE))
    return d, "Figure 8. The dynamic-from-scratch variant spends its equal 500k budget entirely in the full dynamic condition."


def _figure_cloud() -> tuple[Drawing, str]:
    d = Drawing(CONTENT_WIDTH, 305)
    _title_on_drawing(d, "Drive-backed free-Colab execution")
    _box(d, 0, 164, 105, 74, "Locked bundle", "Seven training-relevant files; aggregate SHA-256 checked.", fill=PALE_GREY, stroke=NAVY)
    _box(d, 135, 164, 105, 74, "Free Colab T4", "CUDA runtime; 20k-step dispatch benchmark.", fill=PALE_BLUE, stroke=CYAN)
    _box(d, 270, 164, 105, 74, "Seed worker", "Restore, train, capture provenance, sync completion.", fill=PALE_ORANGE, stroke=ORANGE)
    _box(d, 405, 164, 95, 74, "Google Drive", "Models, logs, metadata, status, backup.", fill=PALE_TEAL, stroke=GREEN)
    _arrow(d, 105, 201, 135, 201)
    _arrow(d, 240, 201, 270, 201)
    _arrow(d, 375, 201, 405, 201)
    _arrow(d, 452, 164, 452, 118, color=GREEN)
    _box(d, 340, 51, 160, 61, "Recover after disconnect", "A new runtime restores completed seeds and names the next incomplete task.", fill=WHITE, stroke=GREEN)
    _arrow(d, 340, 81, 302, 81, color=TEAL)
    _box(d, 120, 51, 182, 61, "Return for final evidence", "Restore artifacts locally; run the exact integrity, analysis, and paper pipeline.", fill=WHITE, stroke=TEAL)
    d.add(String(0, 24, "Dispatch snapshot: 243.64 steps/s on a free T4; local Python processes stopped.", fontName="Arial-Italic", fontSize=7.4, fillColor=SLATE))
    return d, "Figure 9. Drive improves recovery, but final validity still depends on provenance, stage coverage, and artifact integrity."


def _figure_evidence_tiers() -> tuple[Drawing, str]:
    d = Drawing(CONTENT_WIDTH, 300)
    _title_on_drawing(d, "Evidence tiers must not be blended")
    tiers = [
        (20, 42, 460, 44, "FINAL V2 EVIDENCE - not yet available", "310-scenario Cartesian outputs + passing integrity report", PALE_RED, RED),
        (50, 94, 400, 44, "CURRENT VERIFIED EVIDENCE", "58 tests, five full seeds, four DQN seeds, repeated A*/Dijkstra pilot", PALE_TEAL, GREEN),
        (80, 146, 340, 44, "HISTORICAL RESULTS", "40-pair static paper, 50-pair dynamic paper, seed-999 diagnostic", PALE_ORANGE, ORANGE),
        (110, 198, 280, 44, "PLANS AND INTERPRETATIONS", "Original question, improvement plan, conditional expectations", PALE_GREY, NAVY),
    ]
    for x, y, w, h, title, subtitle, fill, stroke in tiers:
        d.add(Rect(x, y, w, h, rx=5, ry=5, fillColor=fill, strokeColor=stroke, strokeWidth=1))
        d.add(String(x + 10, y + 26, title, fontName="Arial-Bold", fontSize=8.5, fillColor=stroke))
        d.add(String(x + 10, y + 11, subtitle, fontName="Arial", fontSize=6.7, fillColor=SLATE))
    return d, "Figure 10. Earlier measurements explain the project history but cannot substitute for the corrected expanded-study outputs."


def _figure_current_results() -> tuple[Drawing, str]:
    d = Drawing(CONTENT_WIDTH, 310)
    _title_on_drawing(d, "Repeated classical pilot: equal route cost, different search work")
    panels = [
        ("Mean route time (ms)", [("A*", 0.5982, TEAL), ("Dijkstra", 5.1742, ORANGE)], 6.0, 15),
        ("Mean node expansions", [("A*", 32.6, TEAL), ("Dijkstra", 251.14, ORANGE)], 280.0, 278),
    ]
    for title, data, max_value, x0 in panels:
        d.add(String(x0, 248, title, fontName="Arial-Bold", fontSize=9, fillColor=NAVY))
        chart_y = 72
        chart_h = 155
        bar_w = 62
        gap = 28
        for index, (label, value, color) in enumerate(data):
            x = x0 + 25 + index * (bar_w + gap)
            height = chart_h * value / max_value
            d.add(Rect(x, chart_y, bar_w, chart_h, fillColor=PALE_GREY, strokeColor=None))
            d.add(Rect(x, chart_y, bar_w, height, fillColor=color, strokeColor=None))
            d.add(String(x + 5, chart_y - 15, label, fontName="Arial-Bold", fontSize=7.5, fillColor=NAVY))
            label_text = f"{value:.4g}"
            d.add(String(x + 5, chart_y + height + 6, label_text, fontName="Arial-Bold", fontSize=7.2, fillColor=color))
    d.add(String(0, 35, "Both planners: 500/500 successes; mean path cost 10.5637.", fontName="Arial-Bold", fontSize=8.5, fillColor=GREEN))
    d.add(String(0, 20, "Hardware-specific Python measurements; final suite will add D* Lite and the full 310-scenario manifest.", fontName="Arial-Italic", fontSize=7.1, fillColor=SLATE))
    return d, "Figure 11. A* demonstrates why the final RL claim must be tested against stronger classical methods than naive Dijkstra alone."


def _figure_remaining() -> tuple[Drawing, str]:
    d = Drawing(CONTENT_WIDTH, 430)
    _title_on_drawing(d, "Path from interim status to final release")
    steps = [
        ("1", "Finish cloud training", "26 incomplete seed-variant records at the cutoff, including active DQN seed 55."),
        ("2", "Restore and verify provenance", "Reject mixed source, smoke runs, and superseded pre-observation-fix artifacts."),
        ("3", "Run all route evaluations", "Dijkstra, A*, D* Lite, full RL, and six ablations on the exact manifest."),
        ("4", "Pass Cartesian integrity", "Exact run IDs, hashes, repetitions, seeds, scenarios, and event parent coverage."),
        ("5", "Analyze and test", "Seed-level intervals, paired tests, effect sizes, Holm correction, adaptability."),
        ("6", "Generate evidence-bound paper", "Tables, figures, abstract, results, discussion, and conclusion from raw data."),
        ("7", "Compile and visually inspect", "Pinned PDF build, every page rendered, defects corrected, final gates rerun."),
    ]
    start_y = 345
    for index, (number, title, subtitle) in enumerate(steps):
        y = start_y - index * 48
        d.add(Circle(18, y + 14, 12, fillColor=TEAL if index < 2 else NAVY, strokeColor=WHITE, strokeWidth=1))
        d.add(String(14.3, y + 10.2, number, fontName="Arial-Bold", fontSize=8.5, fillColor=WHITE))
        d.add(String(42, y + 21, title, fontName="Arial-Bold", fontSize=9, fillColor=NAVY))
        lines = _wrap_drawing_text(subtitle, 96)
        for line_index, line in enumerate(lines[:2]):
            d.add(String(42, y + 7 - line_index * 9, line, fontName="Arial", fontSize=6.6, fillColor=SLATE))
        if index < len(steps) - 1:
            d.add(Line(18, y + 2, 18, y - 22, strokeColor=MID_GREY, strokeWidth=1.3))
    return d, "Figure 12. Completion is defined by admissible artifacts and verified release outputs, not by the end of training alone."


def _figure_final_pipeline() -> tuple[Drawing, str]:
    d = Drawing(CONTENT_WIDTH, 250)
    _title_on_drawing(d, "Integrity-gated artifact pipeline")
    nodes = [
        ("Tests", GREEN),
        ("Manifests", TEAL),
        ("Train", ORANGE),
        ("Evaluate", CYAN),
        ("Integrity", RED),
        ("Statistics", NAVY),
        ("Paper", GOLD),
        ("Visual QA", GREEN),
    ]
    box_w = 55
    gap = 8
    y = 124
    for index, (label, color) in enumerate(nodes):
        x = index * (box_w + gap)
        d.add(Rect(x, y, box_w, 45, rx=5, ry=5, fillColor=WHITE, strokeColor=color, strokeWidth=1.2))
        d.add(String(x + 7, y + 19, label, fontName="Arial-Bold", fontSize=6.9, fillColor=color))
        if index < len(nodes) - 1:
            _arrow(d, x + box_w, y + 22, x + box_w + gap, y + 22, color=SLATE, width=0.9)
    d.add(Rect(0, 47, CONTENT_WIDTH, 46, rx=5, ry=5, fillColor=PALE_RED, strokeColor=RED, strokeWidth=1))
    d.add(String(12, 73, "Release rule", fontName="Arial-Bold", fontSize=9, fillColor=RED))
    d.add(String(80, 73, "No passing integrity report or remaining placeholder -> no release PDF.", fontName="Arial-Bold", fontSize=8, fillColor=NAVY))
    d.add(String(12, 56, "After visual corrections, the final gates must run again so the inspected PDF and verified evidence remain synchronized.", fontName="Arial", fontSize=7.1, fillColor=SLATE))
    return d, "Figure 13. The paper compiler is downstream of evidence completeness and upstream of page-by-page visual inspection."


FIGURES: dict[str, Callable[[], tuple[Drawing, str]]] = {
    "status_snapshot": _figure_status,
    "concept_map": _figure_concept_map,
    "environment": _figure_environment,
    "learning_system": _figure_learning,
    "timeline": _figure_timeline,
    "evolution": _figure_evolution,
    "benchmark": _figure_benchmark,
    "training_pipeline": _figure_training_pipeline,
    "cloud": _figure_cloud,
    "evidence_tiers": _figure_evidence_tiers,
    "current_results": _figure_current_results,
    "remaining": _figure_remaining,
    "final_pipeline": _figure_final_pipeline,
}


def _figure_flowable(name: str, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    if name not in FIGURES:
        return [Paragraph(f"Unknown figure marker: {html.escape(name)}", styles["quote"])]
    drawing, caption = FIGURES[name]()
    return [Spacer(1, 4), drawing, Paragraph(html.escape(caption), styles["caption"])]


def _parse_markdown(source: str, styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    lines = source.splitlines()
    story: list[Flowable] = []
    paragraph_lines: list[str] = []
    list_lines: list[tuple[str, str]] = []
    quote_lines: list[str] = []
    table_lines: list[str] = []
    code_lines: list[str] = []
    equation_lines: list[str] = []
    in_code = False
    in_equation = False
    first_h1 = True

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            text = " ".join(item.strip() for item in paragraph_lines)
            story.append(_paragraph(text, styles["body"]))
            paragraph_lines = []

    def flush_list() -> None:
        nonlocal list_lines
        for marker, text in list_lines:
            bullet = marker if marker != "-" else "•"
            story.append(Paragraph(f"<b>{html.escape(bullet)}</b>  {_inline_markup(text)}", styles["bullet"]))
        list_lines = []

    def flush_quote() -> None:
        nonlocal quote_lines
        if quote_lines:
            story.append(Paragraph(_inline_markup(" ".join(quote_lines)), styles["quote"]))
            quote_lines = []

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            rows = []
            for line in table_lines:
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                rows.append(cells)
            story.append(_table_flowable(rows, styles))
            table_lines = []

    def flush_all() -> None:
        flush_paragraph()
        flush_list()
        flush_quote()
        flush_table()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            flush_all()
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["code"]))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if stripped == "$$":
            flush_all()
            if in_equation:
                story.append(Paragraph(_equation_text(" ".join(equation_lines)), styles["equation"]))
                equation_lines = []
                in_equation = False
            else:
                in_equation = True
            continue
        if in_equation:
            equation_lines.append(stripped)
            continue
        if not stripped:
            flush_all()
            continue
        figure_match = re.fullmatch(r"\[\[FIGURE:([a-z0-9_]+)\]\]", stripped)
        if figure_match:
            flush_all()
            story.extend(_figure_flowable(figure_match.group(1), styles))
            continue
        if stripped.startswith("# "):
            # The top-level title is represented by the designed cover.
            continue
        if stripped.startswith("## "):
            flush_all()
            heading_text = stripped[3:]
            if not first_h1:
                # These two chapters follow substantial figures/status material.
                # A hard break prevents their titles from becoming isolated at
                # the foot of the preceding page.
                if heading_text in {
                    "Results available at the cutoff",
                    "What remains before the research is finished",
                }:
                    story.append(PageBreak())
                else:
                    story.append(CondPageBreak(65 * mm))
            first_h1 = False
            story.append(Paragraph(_inline_markup(heading_text), styles["h1"]))
            continue
        if stripped.startswith("### "):
            flush_all()
            story.append(Paragraph(_inline_markup(stripped[4:]), styles["h2"]))
            continue
        if stripped.startswith("#### "):
            flush_all()
            story.append(Paragraph(_inline_markup(stripped[5:]), styles["h3"]))
            continue
        if stripped.startswith(">"):
            flush_paragraph()
            flush_list()
            flush_table()
            quote_lines.append(stripped.lstrip(">").strip())
            continue
        if re.match(r"^[-*]\s+", stripped):
            flush_paragraph()
            flush_quote()
            flush_table()
            list_lines.append(("-", re.sub(r"^[-*]\s+", "", stripped)))
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if numbered:
            flush_paragraph()
            flush_quote()
            flush_table()
            list_lines.append((f"{numbered.group(1)}.", numbered.group(2)))
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            flush_list()
            flush_quote()
            table_lines.append(stripped)
            continue
        if list_lines and line[:1].isspace():
            marker, prior = list_lines[-1]
            list_lines[-1] = (marker, f"{prior} {stripped}")
            continue
        paragraph_lines.append(stripped)

    flush_all()
    if code_lines:
        story.append(Preformatted("\n".join(code_lines), styles["code"]))
    if equation_lines:
        story.append(Paragraph(_equation_text(" ".join(equation_lines)), styles["equation"]))
    return story


def _cover_story(styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    return [
        Spacer(1, 30 * mm),
        Paragraph("INTERIM DOCUMENTARY EDITION", styles["cover_kicker"]),
        Paragraph("From First Grid to<br/>Reproducible Study", styles["cover_title"]),
        Paragraph(
            "The complete research record of the UAV Dynamic Routing project - "
            "its problem, system, experiments, failures, improvements, current "
            "evidence, cloud workflow, and path to final completion.",
            styles["cover_subtitle"],
        ),
        Spacer(1, 8 * mm),
        Table(
            [
                [
                    Paragraph("<b>Project</b><br/>UAV Dynamic Routing", styles["cover_meta"]),
                    Paragraph("<b>Research team</b><br/>Simra Imran and Muzzammil Sajid", styles["cover_meta"]),
                ],
                [
                    Paragraph("<b>Evidence cutoff</b><br/>2026-07-27 22:03 PKT", styles["cover_meta"]),
                    Paragraph("<b>Repository state</b><br/>Git f57ae15 + dirty expanded-study tree", styles["cover_meta"]),
                ],
            ],
            colWidths=[(CONTENT_WIDTH - 10) / 2] * 2,
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), HexColor("#102D44")),
                    ("BOX", (0, 0), (-1, -1), 0.8, HexColor("#35566A")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, HexColor("#35566A")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            ),
        ),
        Spacer(1, 12 * mm),
        Paragraph(
            "This edition is intentionally honest about incomplete evidence. "
            "Historical results are documented as history; final expanded-study "
            "claims remain pending until the integrity-gated experiment pipeline finishes.",
            styles["cover_meta"],
        ),
        NextPageTemplate("content"),
        PageBreak(),
    ]


def _toc_story(styles: dict[str, ParagraphStyle]) -> list[Flowable]:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOCLevel1",
            fontName="Arial-Bold",
            fontSize=8.8,
            leading=11.2,
            leftIndent=0,
            firstLineIndent=0,
            textColor=NAVY,
            spaceBefore=2,
        ),
        ParagraphStyle(
            "TOCLevel2",
            fontName="Arial",
            fontSize=7.2,
            leading=8.8,
            leftIndent=14,
            firstLineIndent=0,
            textColor=SLATE,
        ),
        ParagraphStyle(
            "TOCLevel3",
            fontName="Arial",
            fontSize=7.5,
            leading=10,
            leftIndent=28,
            firstLineIndent=0,
            textColor=SLATE,
        ),
    ]
    return [
        Paragraph("Contents", styles["toc_title"]),
        Paragraph(
            "The documentary is organized as a research narrative rather than a file changelog. "
            "Appendix-like technical detail, chronology, reproducibility, and glossary material "
            "appear after the remaining-work analysis.",
            styles["body"],
        ),
        Spacer(1, 5),
        toc,
        PageBreak(),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(source_path: Path, output_path: Path) -> dict[str, object]:
    _register_fonts()
    styles = _styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = source_path.read_text(encoding="utf-8")
    story = [
        *_cover_story(styles),
        *_toc_story(styles),
        *_parse_markdown(source, styles),
    ]
    document = DocumentaryDocTemplate(str(output_path), styles)
    document.multiBuild(story)

    reader = PdfReader(str(output_path))
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "document": str(output_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source": str(source_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "source_sha256": _sha256(source_path),
        "pdf_sha256": _sha256(output_path),
        "pages": len(reader.pages),
        "evidence_cutoff": "2026-07-27T22:03:00+05:00",
        "repository_git_head": "f57ae150068fbd7a00c1b37a8280cf60a5dd2ddc",
        "training_source_sha256": "00a4ff215b3d31f7be2a42d62f7467d19beacd3f3e78c8684efe2d233412f39d",
        "expanded_benchmark_sha256": "296196e774e5aa7f45b988f0321355a85f4dcbe21d1c727842cc818ebde0e0b5",
        "edition": "interim",
    }
    manifest_path = output_path.with_suffix(".build.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = generate(args.source.resolve(), args.output.resolve())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
