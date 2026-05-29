"""Resume text extraction tests.

PDF and DOCX fixtures are generated in-memory to avoid binary blobs in git.
The relevant tests `importorskip` so a missing optional dependency doesn't
fail the suite.
"""
from __future__ import annotations

from io import BytesIO

import pytest

from job_scout.parsers.resume import UnsupportedResumeFormat, extract_resume_text

SAMPLE_TXT = "John Doe\nVP AI Transformation\n20 years life sciences.\n"


class TestTxt:
    def test_utf8_basic(self):
        out = extract_resume_text("resume.txt", SAMPLE_TXT.encode("utf-8"))
        assert "VP AI Transformation" in out

    def test_utf8_bom(self):
        out = extract_resume_text(
            "resume.txt", b"\xef\xbb\xbf" + SAMPLE_TXT.encode("utf-8")
        )
        assert out.startswith("John Doe")

    def test_latin1_fallback(self):
        # Smart quotes and accented chars encoded in latin-1 (not valid UTF-8).
        out = extract_resume_text("resume.txt", "caf\xe9".encode("latin-1"))
        assert "caf" in out


class TestUnsupported:
    def test_unsupported_format(self):
        with pytest.raises(UnsupportedResumeFormat):
            extract_resume_text("resume.doc", b"anything")

    def test_no_extension(self):
        with pytest.raises(UnsupportedResumeFormat):
            extract_resume_text("resume", b"anything")


class TestDocx:
    def test_docx_extraction(self):
        docx = pytest.importorskip("docx")
        d = docx.Document()
        d.add_paragraph("Jane Doe")
        d.add_paragraph("SVP Technology")
        d.add_paragraph("Life sciences leader for 20 years.")
        buf = BytesIO()
        d.save(buf)

        out = extract_resume_text("resume.docx", buf.getvalue())
        assert "Jane Doe" in out
        assert "SVP Technology" in out
        assert "Life sciences leader" in out

    def test_docx_table_cells_are_extracted(self):
        docx = pytest.importorskip("docx")
        d = docx.Document()
        table = d.add_table(rows=2, cols=2)
        table.rows[0].cells[0].text = "Role"
        table.rows[0].cells[1].text = "Chief Digital Officer"
        table.rows[1].cells[0].text = "Industry"
        table.rows[1].cells[1].text = "Pharma"
        buf = BytesIO()
        d.save(buf)

        out = extract_resume_text("resume.docx", buf.getvalue())
        assert "Chief Digital Officer" in out
        assert "Pharma" in out


class TestPdf:
    def test_pdf_extraction(self):
        pytest.importorskip("pypdf")
        pytest.importorskip("reportlab")
        from reportlab.pdfgen import canvas

        buf = BytesIO()
        c = canvas.Canvas(buf)
        c.drawString(100, 750, "Alex Smith")
        c.drawString(100, 730, "Chief Digital Officer")
        c.drawString(100, 710, "Life sciences technology leader.")
        c.showPage()
        c.save()

        out = extract_resume_text("resume.pdf", buf.getvalue())
        assert "Alex Smith" in out
        assert "Chief Digital Officer" in out
