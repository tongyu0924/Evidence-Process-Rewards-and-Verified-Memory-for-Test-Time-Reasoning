import re
import json
import math

STOP = set("""a an the of to and or for in on with when any each all is are be by as it its this that than over
under from into before after always explicitly use using avoid not do does may can should more most less such
which while who whom whose also both either other these those there their them they was were has have had
imaging image images finding findings clinical context diagnosis diagnoses option options feature features
history patient patients unlikely likely case cases correct answer suggest suggests suggesting seen shows show
showing typical typically usually often common commonly associated presence present noted demonstrate
demonstrates demonstrating""".split())

DX_GENERIC = {"syndrome", "disease", "tumor", "tumour", "type", "lesion", "mass", "with", "without", "left",
              "right", "bilateral", "of", "the", "and", "or", "in", "a", "an", "to", "due", "from", "s"}
# Single stems too broad to identify a diagnosis on their own
DX_TOO_BROAD = {"carcin", "adenoc", "sarcom", "neopla", "cancer", "metast", "infect", "inflam", "hemorr", "haemor",
                "fractu", "absces", "aneury", "thromb", "cyst", "lympho", "tumor", "tumour", "nodule", "fistul",
                "hernia", "malfor", "dissec", "stenos", "rupture", "ruptur", "trauma"}
DX_JACCARD = 0.6


def keywords(t):
    return {w for w in re.findall(r"[a-z][a-z\-]{2,}", str(t).lower()) if w not in STOP}


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(t).lower())).strip()


def dx_kw(t):
    # Keep type numbers (Chiari II != III); stem long words to 6 letters
    toks = [w for w in re.findall(r"[a-z0-9]+", norm(t)) if w not in DX_GENERIC]
    return {w[:6] if w.isalpha() and len(w) > 6 else w for w in toks}


def dx_match(a, b):
    """Whether two diagnosis names refer to the same disease."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ka, kb = dx_kw(a), dx_kw(b)
    if not ka or not kb:
        return False
    small, big = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
    if small <= big:
        if len(small) >= 2:
            return True
        w = next(iter(small))
        if w not in DX_TOO_BROAD and len(w) >= 6:
            return True
    return len(ka & kb) / len(ka | kb) >= DX_JACCARD


def parse_json(text):
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    s, e = t.find("{"), t.rfind("}")
    if s < 0 or e <= s:
        return None
    for cand in (t[s:e + 1], re.sub(r",\s*([}\]])", r"\1", t[s:e + 1])):
        try:
            return json.loads(cand)
        except Exception:
            pass
    return None


def extract_answer(text, valid):
    if not text:
        return None
    cls = "[" + valid + "]"
    for p in (r"[Aa]nswer\s*(?:is)?\s*[:：]?\s*\**\s*\(?(" + cls + r")\)?(?![A-Za-z])",
              r"\\boxed\{\s*\(?(" + cls + r")\)?"):
        ms = re.findall(p, text)
        if ms:
            return ms[-1]
    ms = re.findall(r"(?<![A-Za-z])\(?(" + cls + r")\)?(?![A-Za-z])", text[-60:])
    return ms[-1] if ms else None


def mcnemar(win, lose):
    """Exact two-sided McNemar test on discordant pairs."""
    n = win + lose
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, j) for j in range(min(win, lose) + 1)) / 2 ** n)
