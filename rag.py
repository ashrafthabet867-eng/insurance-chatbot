"""RAG خفيف للنصوص القانونية العربية: تقسيم حسب المواد + بحث BM25 بدون مكتبات ثقيلة."""
import glob
import math
import os
import re
import unicodedata
from collections import Counter

MAX_CHARS = 1200      # أقصى طول للجزء الواحد
OVERLAP = 150
MIN_SCORE = 4.0       # أقل درجة تشابه مقبولة (اضبطها حسب تجربتك)

_TASHKEEL = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0640]")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال")
_SUFFIXES = ("ات", "ون", "ين", "ان", "ها", "ه", "ي")
_STOP = {
    "في", "من", "علي", "الي", "عن", "ان", "هل", "ما", "ماذا", "كيف", "هو", "هي",
    "هذا", "هذه", "ذلك", "التي", "الذي", "او", "ايه", "ازاي", "امتي", "فين",
    "عايز", "عاوز", "لو", "انا", "ده", "دي", "يعني", "لما", "كل", "مع", "ثم",
}
_ARTICLE_RE = re.compile(
    r"^\s*(?:ال)?ماد[ةه]\s*[\(\[]?\s*(?:([0-9٠-٩]+|[\u0621-\u064A]+)|\u27e6\u061f\u27e7)", re.M
)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # يحوّل أشكال العرض في PDF إلى حروف عادية
    text = _TASHKEEL.sub("", text).translate(_DIGITS)
    text = re.sub("[إأآ]", "ا", text)
    return text.replace("ى", "ي").replace("ة", "ه")


def _stem(tok: str) -> str:
    for p in _PREFIXES:
        if tok.startswith(p) and len(tok) - len(p) >= 3:
            tok = tok[len(p):]
            break
    for s in _SUFFIXES:
        if tok.endswith(s) and len(tok) - len(s) >= 3:
            tok = tok[: -len(s)]
            break
    return tok


def tokenize(text: str):
    toks = re.findall(r"[\u0621-\u064A0-9a-zA-Z]+", normalize(text))
    return [_stem(t) for t in toks if len(t) >= 2 and t not in _STOP]


# ---------------------------------------------------------------- التقسيم
def _windows(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + MAX_CHARS])
        i += MAX_CHARS - OVERLAP
    return out


def split_text(text: str):
    """يرجع قائمة (label, text). يقسم حسب المواد إن وُجدت، وإلا بنوافذ ثابتة."""
    clean = _TASHKEEL.sub("", text)
    matches = list(_ARTICLE_RE.finditer(clean))
    pieces = []
    if len(matches) >= 3:
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(clean)
            label = f"مادة {m.group(1)}" if m.group(1) else "مادة (رقمها غير مؤكد)"
            pieces.append((label, clean[m.start() : end].strip()))
    else:
        pieces = [(f"جزء {i}", w) for i, w in enumerate(_windows(clean), 1)]

    out = []
    for label, body in pieces:
        if len(body) <= MAX_CHARS:
            out.append((label, body))
        else:
            for j, w in enumerate(_windows(body), 1):
                out.append((f"{label} (جزء {j})", w))
    return out


def _read_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(path)
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        return unicodedata.normalize("NFKC", text)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


# ---------------------------------------------------------------- الفهرس
class Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        docs = [tokenize(c["label"] + " " + c["text"]) for c in chunks]
        self.lens = [len(d) for d in docs]
        self.tf = [Counter(d) for d in docs]
        n = len(docs)
        self.avg = (sum(self.lens) / n) if n else 1.0
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 4, min_score: float = MIN_SCORE):
        q = set(tokenize(query))
        scored = []
        for i, tf in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg)
            s = sum(
                self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
                for t in q
                if t in tf
            )
            scored.append((s, i))
        scored.sort(reverse=True)
        return [(self.chunks[i], s) for s, i in scored[:k] if s >= min_score]


def load_index(folder: str):
    """يبني الفهرس من ملفات txt/md/pdf داخل المجلد. يرجع None لو المجلد فارغ."""
    chunks = []
    paths = sorted(
        p
        for ext in ("txt", "md", "pdf")
        for p in glob.glob(os.path.join(folder, f"*.{ext}"))
        if os.path.basename(p).lower() != "readme.txt"
    )
    skip = ("_review", "_unverified", "_problems", "_compare", "_pages_")
    paths = [p for p in paths if not any(t in os.path.basename(p) for t in skip)]
    for path in paths:
        try:
            text = _read_file(path)
        except Exception:
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        for label, body in split_text(text):
            if body.strip():
                chunks.append({"source": name, "label": label, "text": body})
    return Index(chunks) if chunks else None
