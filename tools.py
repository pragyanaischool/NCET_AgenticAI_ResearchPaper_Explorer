import fitz
import requests
from utils import clean_text

# =========================
# 📄 PDF EXTRACTION
# =========================
def extract_pdf_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""

    for i, page in enumerate(doc):
        text += f"\n--- Page {i+1} ---\n"
        text += page.get_text()

    return clean_text(text)


def extract_multiple_pdfs(files):
    all_text = []

    for file in files:
        try:
            text = extract_pdf_text(file)
            labeled = f"""
========================
FILE: {file.name}
========================
{text}
"""
            all_text.append(labeled)
        except Exception as e:
            all_text.append(f"Error reading {file.name}: {str(e)}")

    return "\n\n".join(all_text)


# =========================
# 🌐 ARXIV
# =========================
def fetch_arxiv_paper(url):
    try:
        paper_id = url.split("/")[-1]
        api_url = f"http://export.arxiv.org/api/query?id_list={paper_id}"
        res = requests.get(api_url)

        return f"""
========================
ARXIV: {paper_id}
========================
{clean_text(res.text)}
"""
    except Exception as e:
        return f"Error: {str(e)}"
