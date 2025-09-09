# pip install openai faiss-cpu pypdf python-docx beautifulsoup4 lxml chardet
# export OPENAI_API_KEY="sk-..."

from openai import OpenAI
from pathlib import Path
import faiss, numpy as np, re, chardet
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup

# =================== CONFIG ===================
EMBED_MODEL = "text-embedding-3-large"
GEN_MODEL   = "gpt-4o-mini"

FILE_PATHS = [
    "~/Downloads/rossmoornews_20250903_RossmoorNews.pdf"
]

CHUNK_SIZE     = 1200
CHUNK_OVERLAP  = 150
TOP_K          = 6
# ==============================================

client = OpenAI()

# -------- File readers --------
def read_text_auto(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "\n".join([page.extract_text() or "" for page in PdfReader(str(path)).pages])
    if ext in (".txt", ".md", ".csv"):
        data = path.read_bytes()
        enc = chardet.detect(data)["encoding"] or "utf-8"
        return data.decode(enc, errors="ignore")
    if ext in (".html", ".htm"):
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]): tag.decompose()
        return re.sub(r"\s+\n", "\n", soup.get_text(separator=" ").strip())
    if ext == ".docx":
        return "\n".join(p.text for p in Document(str(path)).paragraphs if p.text)
    return path.read_text(encoding="utf-8", errors="ignore")

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, n, i = [], len(text), 0
    while i < n:
        j = min(i + size, n)
        chunk = text[i:j].strip()
        if chunk: chunks.append(chunk)
        if j >= n: break
        i = max(0, j - overlap)
    return chunks

# -------- Embedding & Index --------
def embed_texts(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs

def build_index(chunks):
    vecs = embed_texts(chunks)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index

def retrieve(question, index, chunks, k=TOP_K):
    qv = embed_texts([question])
    D, I = index.search(qv, k)
    return [chunks[i] for i in I[0] if i != -1]

# -------- QA --------
def answer(question, index, chunks):
    hits = retrieve(question, index, chunks, TOP_K)
    context = "\n\n".join(hits)
    prompt = f"""Use ONLY the context below to answer the question.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:"""

    resp = client.responses.create(
        model=GEN_MODEL,
        input=[{"role": "user", "content": prompt}],
    )
    print("\n--- RESPONSE ---\n")
    print(resp.output_text)

# -------- Main --------
def main():
    corpus = []
    for p in map(Path, FILE_PATHS):
        if not p.exists():
            print(f"[warn] file not found: {p}")
            continue
        corpus.extend(chunk_text(read_text_auto(p)))

    if not corpus:
        print("[error] No usable text extracted.")
        return

    index = build_index(corpus)
    print(f"[ok] Indexed {len(corpus)} chunks from {len(FILE_PATHS)} files.")

    try:
        while True:
            q = input("\nEnter your question (or Enter to quit): ").strip()
            if not q: break
            answer(q, index, corpus)
    except (KeyboardInterrupt, EOFError):
        print("\nbye")

if __name__ == "__main__":
    main()
