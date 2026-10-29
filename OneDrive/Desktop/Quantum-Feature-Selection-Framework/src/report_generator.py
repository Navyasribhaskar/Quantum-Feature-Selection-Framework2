"""
report_generator.py
=====================
Generates a professional PDF summary report of a completed pipeline run,
using reportlab. Used by the Download page's "PDF Report" export.

The report is deliberately table/text based (rather than embedding
rendered chart images) so it has zero dependency on optional plotting
backends (e.g. kaleido) and renders identically in every environment.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.utils import APP_SUBTITLE, APP_TITLE

_PRIMARY = colors.HexColor("#4F46E5")
_ACCENT = colors.HexColor("#8B5CF6")
_SOFT = colors.HexColor("#F8FAFC")
_BORDER = colors.HexColor("#E2E8F0")


def _styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="QFSFTitle", parent=styles["Title"], textColor=_PRIMARY, fontSize=22, spaceAfter=6
        )
    )
    styles.add(
        ParagraphStyle(
            name="QFSFSubtitle", parent=styles["Normal"], textColor=colors.HexColor("#475569"), fontSize=11
        )
    )
    styles.add(
        ParagraphStyle(
            name="QFSFHeading", parent=styles["Heading2"], textColor=_ACCENT, spaceBefore=16, spaceAfter=8
        )
    )
    styles.add(
        ParagraphStyle(
            name="QFSFBody", parent=styles["Normal"], fontSize=10, leading=14
        )
    )
    return styles


def _styled_table(data: list[list[Any]], col_widths: list[float] | None = None) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), _PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, _BORDER),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, _SOFT]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def generate_pdf_report(context: dict) -> bytes:
    """Build the full PDF report and return it as raw bytes.

    Expected `context` keys (all optional; missing sections are skipped
    gracefully):
        dataset_name, dataset_shape, target_column, dropped_metadata_columns,
        n_original_features, n_selected_features, reduction_percentage,
        n_qubits, shots, classical_metrics, quantum_metrics,
        class_mapping
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
    )
    styles = _styles()
    story = []

    # ---- Title page -----------------------------------------------------
    story.append(Paragraph(APP_TITLE, styles["QFSFTitle"]))
    story.append(Paragraph(APP_SUBTITLE, styles["QFSFSubtitle"]))
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles["QFSFBody"],
        )
    )
    story.append(Spacer(1, 12))

    # ---- Dataset summary --------------------------------------------------
    story.append(Paragraph("1. Dataset Summary", styles["QFSFHeading"]))
    dataset_rows = [
        ["Property", "Value"],
        ["Dataset", context.get("dataset_name", "N/A")],
        ["Shape", str(context.get("dataset_shape", "N/A"))],
        ["Target Column", context.get("target_column", "N/A")],
        ["Dropped Metadata Columns", ", ".join(context.get("dropped_metadata_columns", [])) or "None"],
        ["Class Mapping", str(context.get("class_mapping", "N/A"))],
    ]
    story.append(_styled_table(dataset_rows, col_widths=[2.2 * inch, 4 * inch]))

    # ---- Feature selection --------------------------------------------------
    story.append(Paragraph("2. Feature Selection (ANOVA F-test)", styles["QFSFHeading"]))
    fs_rows = [
        ["Metric", "Value"],
        ["Original Features", str(context.get("n_original_features", "N/A"))],
        ["Selected Features", str(context.get("n_selected_features", "N/A"))],
        ["Reduction Percentage", f"{context.get('reduction_percentage', 'N/A')}%"],
    ]
    story.append(_styled_table(fs_rows, col_widths=[2.2 * inch, 4 * inch]))

    # ---- Quantum configuration --------------------------------------------------
    story.append(Paragraph("3. Quantum Encoding Configuration", styles["QFSFHeading"]))
    q_rows = [
        ["Parameter", "Value"],
        ["Number of Qubits", str(context.get("n_qubits", "N/A"))],
        ["Shots per Circuit", str(context.get("shots", "N/A"))],
        ["Encoding Method", "Angle encoding (RY) + linear CX entanglement"],
        ["Simulator", "Qiskit Aer - AerSimulator"],
    ]
    story.append(_styled_table(q_rows, col_widths=[2.2 * inch, 4 * inch]))

    # ---- Model comparison --------------------------------------------------
    story.append(Paragraph("4. Classical vs. Quantum-Enhanced Model Comparison", styles["QFSFHeading"]))
    cm = context.get("classical_metrics", {})
    qm = context.get("quantum_metrics", {})
    comp_rows = [
        ["Metric", "Classical SVM", "Quantum-Enhanced SVM"],
        ["Accuracy", f"{cm.get('accuracy', 0):.2%}", f"{qm.get('accuracy', 0):.2%}"],
        ["Precision (macro)", f"{cm.get('precision_macro', 0):.2%}", f"{qm.get('precision_macro', 0):.2%}"],
        ["Recall (macro)", f"{cm.get('recall_macro', 0):.2%}", f"{qm.get('recall_macro', 0):.2%}"],
        ["F1-score (macro)", f"{cm.get('f1_macro', 0):.2%}", f"{qm.get('f1_macro', 0):.2%}"],
    ]
    story.append(_styled_table(comp_rows, col_widths=[2.2 * inch, 2 * inch, 2 * inch]))

    story.append(PageBreak())

    # ---- Classification reports --------------------------------------------------
    story.append(Paragraph("5. Classical SVM - Classification Report", styles["QFSFHeading"]))
    story.append(Paragraph(f"<pre>{cm.get('classification_report_text', 'N/A')}</pre>", styles["QFSFBody"]))

    story.append(Paragraph("6. Quantum-Enhanced SVM - Classification Report", styles["QFSFHeading"]))
    story.append(Paragraph(f"<pre>{qm.get('classification_report_text', 'N/A')}</pre>", styles["QFSFBody"]))

    # ---- Summary --------------------------------------------------
    story.append(Paragraph("7. Summary", styles["QFSFHeading"]))
    better = "Classical SVM" if cm.get("accuracy", 0) >= qm.get("accuracy", 0) else "Quantum-Enhanced SVM"
    story.append(
        Paragraph(
            f"On this run, the <b>{better}</b> achieved the higher test accuracy. This report "
            f"was generated automatically by the Quantum Feature Selection Framework dashboard.",
            styles["QFSFBody"],
        )
    )

    doc.build(story)
    return buffer.getvalue()
