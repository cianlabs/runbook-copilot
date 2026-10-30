"""Retrieval layer for runbook-copilot: load runbooks, chunk them, search them.

Uses scikit-learn TF-IDF (small, local, no model download). The search()
function is the only thing the rest of the project should rely on, so the
embedding method can be swapped later (e.g. sentence-transformers).

Named failures (each message says what to do):
  NO_RUNBOOKS  - the runbooks folder is missing or has no .txt/.md files
  EMPTY_QUERY  - search() was called with an empty query
  BAD_SETTINGS - chunk_size / overlap values do not make sense

Word matching ignores simple endings, so "deployment", "deploys" and "deployed"
all match "deploy", and "failed", "fails" and "failing" all match "fail".
"""

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RUNBOOK_DIR = Path(__file__).parent / "runbooks"
CHUNK_SIZE = 400  # characters per chunk (target 300-500)
OVERLAP = 80  # characters shared between neighbouring chunks
MIN_TAIL = 120  # a last chunk shorter than this is merged into the one before it
TOP_K = 3
MIN_SCORE = 0.05  # scores below this count as "not relevant"


SUFFIXES = ("ments", "ment", "ings", "ing", "ures", "ure", "ed", "es", "s")


def stem(word):
    """Cut a simple ending off a word, so related forms match: deployment -> deploy.

    This is deliberately tiny, not a full stemmer. Both the runbooks and the query
    go through it, so the same word always becomes the same stem."""
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) >= 5:  # release / releases -> releas
        word = word[:-1]
    if len(word) >= 4 and word[-1] == word[-2]:  # running -> runn -> run
        word = word[:-1]
    return word


class StemmedVectorizer(TfidfVectorizer):
    """TF-IDF where every word is stemmed (single words and word pairs alike)."""

    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: [" ".join(stem(w) for w in term.split()) for term in analyzer(doc)]


class RetrieveError(Exception):
    """A named, human-readable failure. The first word of the message is the class."""

    def __init__(self, name, message):
        self.name = name
        super().__init__(f"{name}: {message}")


def load_runbooks(directory=RUNBOOK_DIR):
    """Return a list of (filename, text) for every .txt and .md file."""
    directory = Path(directory)
    if not directory.is_dir():
        raise RetrieveError("NO_RUNBOOKS", f"folder not found: {directory}. Create it and add .txt files.")
    files = sorted(list(directory.glob("*.txt")) + list(directory.glob("*.md")))
    docs = [(f.name, f.read_text(encoding="utf-8")) for f in files]
    docs = [(n, t) for n, t in docs if t.strip()]
    if not docs:
        raise RetrieveError("NO_RUNBOOKS", f"no non-empty .txt or .md files in {directory}.")
    return docs


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP, min_tail=MIN_TAIL):
    """Split text into overlapping pieces, preferring to cut at a space.

    A very short last piece (under min_tail characters) is merged into the
    piece before it, so a search never returns a tiny fragment."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise RetrieveError("BAD_SETTINGS", "need chunk_size > 0 and 0 <= overlap < chunk_size.")
    text = " ".join(text.split())  # collapse whitespace
    chunks = []
    starts = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + chunk_size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
            starts.append(start)
        if end >= len(text):
            break
        nxt = max(end - overlap, start + 1)
        space = text.find(" ", nxt, end)  # begin the next chunk at a word boundary
        start = space + 1 if space != -1 else nxt
    if len(chunks) > 1 and len(chunks[-1]) < min_tail:
        chunks.pop()
        chunks[-1] = text[starts[-2]:].strip()  # previous piece now runs to the end
    return chunks


def split_sections(text):
    """Split text at lines starting with "# ". Returns a list of (header, body).

    Text before the first header gets the header "(no header)".
    """
    sections = []
    header, lines = "(no header)", []
    for line in text.splitlines():
        if line.startswith("# "):
            if "".join(lines).strip():
                sections.append((header, "\n".join(lines)))
            header, lines = line[2:].strip(), []
        else:
            lines.append(line)
    if "".join(lines).strip():
        sections.append((header, "\n".join(lines)))
    return sections


def build_chunks(directory=RUNBOOK_DIR, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Return a list of {"source", "section", "text"} dicts.

    Each file is split into sections at "# " headers first, then each
    section is chunked by length on its own, so a chunk never mixes sections.
    """
    out = []
    for name, text in load_runbooks(directory):
        for header, body in split_sections(text):
            for piece in chunk_text(body, chunk_size, overlap):
                out.append({"source": name, "section": header, "text": piece})
    return out


class Index:
    """TF-IDF index over runbook chunks."""

    def __init__(self, directory=RUNBOOK_DIR, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
        self.chunks = build_chunks(directory, chunk_size, overlap)
        self.vectorizer = StemmedVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([c["section"] + " " + c["text"] for c in self.chunks])

    def search(self, query, top_k=TOP_K, min_score=MIN_SCORE):
        """Return up to top_k chunks as {"source", "section", "text", "score"}, best first.

        Chunks scoring below min_score are dropped, so an irrelevant query
        returns an empty list.
        """
        if not query or not query.strip():
            raise RetrieveError("EMPTY_QUERY", "give search() some words to look for.")
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked[:top_k]:
            if scores[i] >= min_score:
                results.append({**self.chunks[i], "score": round(float(scores[i]), 4)})
        return results


_default_index = None


def search(query, top_k=TOP_K):
    """Search the runbooks/ folder. Builds the index on first use."""
    global _default_index
    if _default_index is None:
        _default_index = Index()
    return _default_index.search(query, top_k=top_k)


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "database connection refused"
    try:
        for r in search(q):
            print(f"[{r['score']:.3f}] {r['source']} > {r['section']}: {r['text']}\n")
        print("(no relevant chunks)" if not search(q) else "")
    except RetrieveError as e:
        print(e)
