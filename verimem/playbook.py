"""Disease Playbook: one entry per disease with knowledge (typical findings) and rules (discriminators, pitfalls),
copied from expert text, checked by quotes, and used only to verify candidate answers on the images."""
import re
import difflib
import collections

import numpy as np

from . import config as cfg
from .text import keywords, norm, dx_kw, dx_match, parse_json, extract_answer
from .serving import run_many, judge_text, judge_vision

NON_IMAGING = ("treatment", "endovascular", "surgery", "surgical", "management", "immunohisto", "immunostain",
               "staining", "stain ", "histolog", "biopsy", "s100", "cd68", "ablation", "therapy", "chemotherapy",
               "prognosis", "follow-up", "radiotherapy", "antibiotic")
IMAGING_RE = re.compile(r"\b(ct|mri?|mr|t1|t2|flair|dwi|adc|swi|stir|pet|spect|us|ultrasound|sonograph\w*|doppler|"
                        r"echo\w*|radiograph\w*|x-ray|fluoroscop\w*|angiograph\w*|mammograph\w*|scintigraph\w*|"
                        r"enhanc\w*|signal|intensity|hyper\w*|hypo\w*|iso\w*|attenuat\w*|dens\w*|calcif\w*|contrast|"
                        r"fluid|cystic|solid|septa\w*|margin\w*|wall|fat|air|gas|nodul\w*|opacit\w*|lytic|sclero\w*|"
                        r"cortical|periosteal|flow|vascular\w*|uptake|restrict\w*|diffusion|level|thicken\w*|dilat\w*)\b")
NON_IMG_RE = re.compile(r"\b(cd\d+|immunohisto\w*|immunostain\w*|histolog\w*|histopatholog\w*|patholog\w*|cytolog\w*|"
                        r"biopsy|serum|urine|laborator\w*|marker\w*|positivity|mutation|gene|cells?|treatment|surgery|"
                        r"surgical|therapy|prognosis|management|antibiotic\w*|sputum|culture|acid-fast|endoscop\w*|"
                        r"clinical\w*|symptom\w*|craniofacial|hypertelorism|palate)\b")
# Case-specific text (figure numbers, "this patient", sizes, past-tense case description)
CASE_RE = re.compile(r"\b(fig\w*|figure|our (case|patient)|this (case|patient)|the patient|present case|"
                     r"in (our|this) (case|series))\b|\d+(\.\d+)?\s?(mm|cm|x\s?\d)")
CASE_VERB_RE = re.compile(r"\b(showed|revealed|demonstrated|was (seen|detected|found|noted|identified|performed|"
                          r"visualized|confirmed)|were (seen|detected|found|noted|identified|evident))\b")
CONTRAST_RE = re.compile(r"\b(unlike|differentiat\w*|distinguish\w*|whereas|in contrast|rather than|compared|"
                         r"versus|vs\.?|while|than|other than|from other)\b")
MOD_FAM = [("ct", r"\b(ct|computed tomograph\w*|hounsfield|hu)\b"),
           ("mr", r"\b(mri?|mr|magnetic resonance|t1\w*|t2\w*|flair|dwi|adc|swi|stir|gadolinium|diffusion)\b"),
           ("us", r"\b(us|ultrasound|sonograph\w*|doppler|echogen\w*|hypoechoic|hyperechoic|anechoic|echocardiograph\w*)\b"),
           ("xr", r"\b(x-?ray|radiograph\w*|plain film|chest film|cxr)\b"),
           ("nm", r"\b(pet|spect|scintigraph\w*|bone scan|fdg|uptake|tracer)\b"),
           ("angio", r"\b(angiograph\w*|dsa|venograph\w*)\b"),
           ("mammo", r"\b(mammograph\w*|mammogram)\b")]
VSCORE = {"SUPPORTED": 1.0, "UNCLEAR": 0.5, "CONTRADICTED": 0.0}


def imaging_ok(t):
    lo = str(t).lower()
    return bool(IMAGING_RE.search(lo)) and not NON_IMG_RE.search(lo) and not any(k in lo for k in NON_IMAGING)


def verify_evidence(quote, case):
    """The quote must appear verbatim in the expert text (ignoring case, punctuation and spacing)."""
    q = norm(quote)
    src = norm(case["findings"] + " " + case["discussion"])
    if len(q.split()) < 5 or not src:
        return False
    if q in src:
        return True
    m = difflib.SequenceMatcher(None, q, src, autojunk=False).find_longest_match(0, len(q), 0, len(src))
    return m.size >= 0.9 * len(q)


def quote_ok(quote, case):
    """Looser check for rules: verbatim, verbatim fragments around '...', or 85% word overlap with one sentence."""
    if verify_evidence(quote, case):
        return True
    src_raw = case["findings"] + " " + case["discussion"]
    src = norm(src_raw)
    frags = [norm(f) for f in re.split(r"\.\.\.|…|\[\.\.\.\]", str(quote)) if len(norm(f).split()) >= 4]
    if frags and sum(len(f.split()) for f in frags) >= 6 and all(f in src for f in frags):
        return True
    qt = [w for w in norm(quote).split() if len(w) >= 3]
    if len(qt) < 6:
        return False
    best = 0.0
    for sent in re.split(r"(?<=[.;!?])\s+|\n", src_raw):
        st = set(norm(sent).split())
        if st:
            best = max(best, sum(w in st for w in qt) / len(qt))
    return best >= 0.85


def mod_fams(t):
    lo = str(t).lower()
    return {k for k, rx in MOD_FAM if re.search(rx, lo)}


def mod_ok(text, case):
    """Only check findings of a modality that this case actually has."""
    if "_fams" not in case:
        case["_fams"] = mod_fams(" ".join(case.get("mods", set())))
    f, c = mod_fams(text), case["_fams"]
    return not f or not c or bool(f & c)


class Playbook:
    def __init__(self):
        self.b = collections.OrderedDict()
        self.n = 0

    def active(self):
        return [i for i in self.b if not self.b[i]["banned"]]

    def find(self, name):
        # Merge by canonical name only, so that different diseases are not merged through aliases
        return next((i for i, x in self.b.items() if dx_match(name, x["name"])), None)

    def add(self, name, aliases, region, mods, feats, source):
        name = " ".join(str(name).split())[:80]
        if len(name) < 3:
            return 0
        aliases = [" ".join(str(a).split())[:80] for a in aliases if str(a).strip()][:3]
        i = self.find(name)
        if i is None:
            self.n += 1
            i = f"entry-{self.n:04d}"
            self.b[i] = dict(name=name, aliases=set(), region=str(region)[:60], scope_mod=set(), feats=[],
                             banned=False, source=source, s_sum=0.0, n_upd=0)
        x = self.b[i]
        x["aliases"] |= set(aliases)
        x["scope_mod"] |= set(mods)
        added = 0
        for f in feats:
            kw = keywords(f["text"])
            dup = next((g for g in x["feats"] if g["type"] == f["type"] and kw and
                        len(kw & keywords(g["text"])) / len(kw | keywords(g["text"])) >= 0.5), None)
            if dup:
                dup["seen"] += 1
                continue
            x["feats"].append(dict(f, seen=1, source=source))
            added += 1
        x["feats"] = sorted(x["feats"], key=lambda g: (-g["seen"], g["type"] == "observed"))[:cfg.MAX_FEAT]
        return added

    def matched(self, case):
        """{entry id: option letter} for entries whose name (or else alias) matches an option."""
        out = {}
        act = self.active()
        for L, o in case["opts"].items():
            hits = [i for i in act if dx_match(self.b[i]["name"], o)] or \
                   [i for i in act if any(dx_match(a, o) for a in self.b[i]["aliases"])]
            for i in hits:
                out.setdefault(i, L)
        return out

    def credit(self, i):
        x = self.b[i]
        return x["s_sum"] / x["n_upd"] if x["n_upd"] else 0.5

    def update_credit(self, i, s):
        x = self.b[i]
        x["s_sum"] += s
        x["n_upd"] += 1
        if x["n_upd"] >= cfg.MIN_UPD and self.credit(i) < cfg.EVICT:
            x["banned"] = True

    def usable(self, i):
        return not self.b[i]["n_upd"] or self.credit(i) >= 0.5

    def dump(self):
        rows = []
        for i, x in self.b.items():
            al = f" (aliases: {', '.join(sorted(x['aliases']))})" if x["aliases"] else ""
            rows.append(f"[{i}] {'BANNED ' if x['banned'] else ''}{x['name']}{al} | {x['region']} | "
                        f"credit={self.credit(i):.2f} ({x['n_upd']} updates) | source={x['source']}")
            for g in x["feats"]:
                tag = {"feature": "knowledge", "rule": f"rule vs {g['vs']}", "pitfall": f"pitfall vs {g['vs']}",
                       "observed": "test observation"}[g["type"]]
                rows.append(f"    - [{tag}] {g['text']} (seen {g['seen']}x)\n      quote: {g['quote']}")
        return "\n".join(rows)


# ---------------------------------------------------------------- construction
def entry_prompt(case, student_out, student_pred):
    wrong = student_pred and student_pred != case["gold"]
    stu = (f"\n\nA student chose {student_pred}. {case['opts'].get(student_pred, '')} (WRONG). Student reasoning "
           f"(end):\n{(student_out or '')[-1500:]}" if wrong else "")
    return (f"Expert radiology teaching case.\n{case['ctx']}\n\nCorrect answer: {case['gold']}. {case['gold_text']}\n\n"
            f"EXPERT FINDINGS:\n{case['findings'][:2500]}\n\nEXPERT DISCUSSION:\n{case['discussion'][:3500]}{stu}\n\n"
            "Write a disease knowledge card for the CORRECT diagnosis using ONLY facts stated in the expert text.\n"
            "- name: canonical disease name, short and general (2-6 words; no laterality, no patient details), "
            "e.g. 'tailgut cyst', 'renal angiomyolipoma'\n- aliases: 0-3 other common names\n"
            "- region: organ or body region\n"
            "- aliases must be true synonyms of the SAME disease; never list related, similar or differential diseases\n"
            "- features: up to 3 typical IMAGING findings of this disease (modality + appearance), each <= 25 words, "
            "stated as GENERAL knowledge that holds for other patients (prefer the general statements of the "
            "DISCUSSION; do not describe this patient, sizes, figures or case details)\n"
            "The card has two parts: KNOWLEDGE (features above) and a PLAYBOOK of decision rules:\n"
            "- rules: up to 3 decision rules against specific wrong options, each {\"vs\": short canonical name of that "
            "option, \"if\": ONE imaging finding (modality + appearance, <= 18 words) that, when SEEN, favours the "
            "correct diagnosis over 'vs'}" + (" (a rule against the student's wrong choice first)" if wrong else "") +
            "; only if the expert text itself states the difference (its quote must name or contrast the other disease)\n" +
            ("- pitfalls: up to 1 lesson from the student's mistake: {\"vs\": the student's wrong choice, \"trap\": the "
             "imaging finding that misled the student because both diseases can show it, \"check\": ONE imaging finding "
             "to check instead that separates them (stated in the expert text)}\n" if wrong else "") +
            "- differentials: for EACH WRONG option whose typical imaging appearance the expert text itself "
            "describes (often in the DISCUSSION), give its canonical short name and up to 2 GENERAL imaging findings "
            "of THAT disease; skip wrong options the text does not describe\n"
            "For EVERY feature/rule/pitfall add 'quote': a VERBATIM fragment (8-40 words) of the expert text that "
            "states it. No histology, immunostaining, lab values, treatment or clinical history alone.\n"
            'Return ONLY JSON: {"name": "...", "aliases": ["..."], "region": "...", '
            '"features": [{"text": "...", "quote": "..."}], "rules": [{"vs": "...", "if": "...", "quote": "..."}], '
            '"pitfalls": [{"vs": "...", "trap": "...", "check": "...", "quote": "..."}], '
            '"differentials": [{"name": "...", "features": [{"text": "...", "quote": "..."}]}]}')


def support_prompt(f, disease):
    if f["type"] == "feature":
        claim = f"Imaging finding of {disease}: \"{f['text']}\""
        ask = (f"Answer YES only if the EVIDENCE explicitly states this same imaging finding FOR {disease} (not for "
               "another disease); answer NO if the statement adds anything that the evidence does not say.")
    else:
        claim = f"Difference between {disease} and {f['vs']}: \"{f['text']}\""
        ask = (f"Answer YES only if the EVIDENCE explicitly states this difference between {disease} and "
               f"{f['vs']}; answer NO if the evidence only describes one disease or one patient.")
    return (f"EVIDENCE (verbatim from an expert radiology teaching case):\n\"{f['quote']}\"\n\n{claim}\n\n{ask} "
            "Answer with exactly one word: YES or NO.")


def check_item(f, c, stat, gold_txt):
    f["text"] = " ".join(str(f["text"]).split())[:300]
    f["quote"] = " ".join(str(f["quote"]).split())
    tl, ql = f["text"].lower(), f["quote"].lower()
    if len(f["text"]) < 20:
        stat["too short"] += 1
    elif not imaging_ok(f["text"]):
        stat["not imaging"] += 1
    elif not (verify_evidence(f["quote"], c) if f["type"] == "feature" else quote_ok(f["quote"], c)):
        stat["quote not in text"] += 1
    elif CASE_RE.search(tl) or CASE_RE.search(ql) or (f["type"] == "feature" and CASE_VERB_RE.search(ql)):
        stat["case specific"] += 1
    elif f["type"] != "feature" and not (CONTRAST_RE.search(ql) or (dx_kw(f["vs"]) & dx_kw(f["quote"]))):
        stat["rule without contrast"] += 1
    elif f["type"] != "feature" and dx_match(f["vs"], gold_txt):
        stat["rule against itself"] += 1
    else:
        return True
    return False


def _clean(s, n=80):
    return " ".join(str(s or "").split())[:n]


def build_batch(pb, batch, outs, stat):
    """Add entries for the correct diagnosis of each case, and for wrong options that the expert text describes."""
    preds = [extract_answer(o, "".join(c["opts"])) for c, o in zip(batch, outs)]
    cards = run_many(lambda x: judge_text(entry_prompt(*x), 1300, "entry_writer"), list(zip(batch, outs, preds)),
                     cfg.JUDGE_CONC, "playbook entries")
    entries = []
    for c, p, t in zip(batch, preds, cards):
        j = parse_json(t) or {}
        if not j.get("name"):
            stat["bad format"] += 1
            continue
        gold_txt = c["opts"][c["gold"]]
        others = [x for L, x in c["opts"].items() if L != c["gold"]]
        wrong_txt = c["opts"].get(p, "") if p and p != c["gold"] else ""
        fs = [dict(type="feature", text=f.get("text", ""), vs="", quote=f.get("quote", ""))
              for f in (j.get("features") or [])[:3] if isinstance(f, dict)]
        fs += [dict(type="rule", text=f.get("if") or f.get("text", ""), vs=_clean(f.get("vs")), quote=f.get("quote", ""))
               for f in (j.get("rules") or [])[:3] if isinstance(f, dict)]
        if wrong_txt:
            fs += [dict(type="pitfall", text=f.get("check", ""), vs=_clean(f.get("vs")) or wrong_txt,
                        quote=f.get("quote", ""), trap=_clean(f.get("trap"), 150))
                   for f in (j.get("pitfalls") or [])[:1] if isinstance(f, dict)]
        ok = [f for f in fs if check_item(f, c, stat, gold_txt)]
        vs_names = [f["vs"] for f in fs if f["vs"]]
        al = []
        for a in (j.get("aliases") or []):
            a = " ".join(str(a).split())
            if a and not any(dx_match(a, x) for x in others + vs_names):
                al.append(a)
        entries.append((c, j["name"], al, j.get("region", ""), ok))
        for d in (j.get("differentials") or [])[:4]:
            if not isinstance(d, dict) or not d.get("name"):
                continue
            dn = _clean(d["name"])
            if not any(dx_match(dn, x) for x in others) or dx_match(dn, gold_txt):
                continue
            dfs = [dict(type="feature", text=f.get("text", ""), vs="", quote=f.get("quote", ""))
                   for f in (d.get("features") or [])[:2] if isinstance(f, dict)]
            dok = [f for f in dfs if check_item(f, c, stat, gold_txt)]
            if dok:
                entries.append((c, dn, [], j.get("region", ""), dok))
    checks = [(k, f) for k, e in enumerate(entries) for f in e[4]]
    verdicts = run_many(lambda x: judge_text(support_prompt(x[1], entries[x[0]][1]), 3, "support_check"), checks,
                        cfg.JUDGE_CONC, "quote support check")
    keep = collections.defaultdict(list)
    for (k, f), v in zip(checks, verdicts):
        if (v or "").strip().upper().startswith("YES"):
            keep[k].append(f)
        else:
            stat["quote does not support"] += 1
    added = 0
    for k, (c, name, al, region, _) in enumerate(entries):
        if keep[k]:
            added += pb.add(name, al, region, c.get("mods", set()), keep[k], f"train:{c['id']}")
    return added


# ---------------------------------------------------------------- Playbook check
def checklist(pb, case, letter):
    """Findings that should be visible if `letter` is correct; rules against other options come first."""
    items = []
    others = [o for L, o in case["opts"].items() if L != letter]
    for i, L in pb.matched(case).items():
        if L != letter or not pb.usable(i):
            continue
        for f in sorted(pb.b[i]["feats"], key=lambda g: -g["seen"]):
            if not mod_ok(f["text"], case):
                continue
            if f["type"] in ("rule", "pitfall"):
                if any(dx_match(f["vs"], o) for o in others):
                    items.insert(sum(1 for x in items if x[2]), (i, f["text"], True))
            else:
                items.append((i, f["text"], False))
    return [(i, t) for i, t, _ in items[:cfg.CHECK_ITEMS]]


def check_prompt(case, items):
    note = (f" Only images 1-{cfg.VERIFY_IMAGES} are shown; claims about other images are UNCLEAR."
            if len(case["imgs"]) > cfg.VERIFY_IMAGES else "")
    lst = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(items))
    return (f"{case['ctx']}\n\nClaims about the images:\n{lst}\n\n"
            "Look at the images yourself and judge each claim: SUPPORTED (clearly visible), "
            "CONTRADICTED (the images show otherwise) or UNCLEAR (cannot be judged)." + note +
            "\nAnswer with one line per claim, exactly in the form '<number>: SUPPORTED|CONTRADICTED|UNCLEAR'.")


_CHECK_CACHE = {}


def check_scores(reqs):
    """reqs: [(pb, case, letter)] -> [(q, entry ids)]; q = share of decided checklist items that hold
    (UNCLEAR items are ignored; q is None if nothing can be decided)."""
    lists = [checklist(pb, c, L) for pb, c, L in reqs]
    res = [(None, []) for _ in reqs]
    jobs = []
    for k, ((_, c, _), cl) in enumerate(zip(reqs, lists)):
        if not cl:
            continue
        key = (c["id"], tuple(t for _, t in cl))
        if key in _CHECK_CACHE:
            q = _CHECK_CACHE[key]
            res[k] = (q, sorted({i for i, _ in cl}) if q is not None else [])
        else:
            jobs.append((k, c, [t for _, t in cl]))
    outs = run_many(lambda jb: judge_vision(jb[1], check_prompt(jb[1], jb[2]), 12 * len(jb[2]) + 40, "playbook_check"),
                    jobs, cfg.JUDGE_CONC, "playbook check")
    for (k, c, cl), txt in zip(jobs, outs):
        got = {int(m.group(1)) - 1: VSCORE[m.group(2).upper()]
               for m in re.finditer(r"(\d+)\s*[:.)\-]\s*\**(SUPPORTED|CONTRADICTED|UNCLEAR)", txt or "", re.I)}
        dec = [v for v in (got.get(i, 0.5) for i in range(len(cl))) if v != 0.5]
        q = float(np.mean(dec)) if dec else None
        _CHECK_CACHE[(c["id"], tuple(cl))] = q
        res[k] = (q, sorted({i for i, _ in lists[k]}) if q is not None else [])
    return res


def measure_credit(pb, cases, update=True):
    """Checklists should hold for the correct option (credit s = q) and fail for wrong ones (s = 1 - q).
    Returns mean q on correct options, mean q on wrong options and their counts."""
    reqs = [(c, L) for c in cases for L in sorted(set(pb.matched(c).values())) if checklist(pb, c, L)]
    qc, qw = [], []
    for (c, L), (q, ids) in zip(reqs, check_scores([(pb, c, L) for c, L in reqs])):
        if q is None:
            continue
        ok = L == c["gold"]
        (qc if ok else qw).append(q)
        if update:
            for i in ids:
                pb.update_credit(i, q if ok else 1 - q)
    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
    return mean(qc), mean(qw), len(qc), len(qw)
