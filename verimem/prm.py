"""Evidence process reward model and answer selection."""
import re
import copy
import math
import collections

import numpy as np

from . import config as cfg
from .text import extract_answer
from .serving import run_many, judge_vision
from .playbook import checklist, check_scores

VSC = {"SUPPORTED": 1.0, "UNCLEAR": 0.5, "CONTRADICTED": 0.0, "NA": None}
RSC = {"VALID": 1.0, "WEAK": 0.5, "INVALID": 0.0, "NA": None}


def split_steps(r):
    """Split a path into at most PRM_MAX_STEPS steps (lines or sentences), dropping the answer line."""
    body = re.split(r"\n\s*(?:Answer|Entries used|Notes used)\s*:", r or "")[0]
    lines = [l.strip(" -*•\t#") for l in body.splitlines()]
    lines = [l for l in lines if len(l) >= 15]
    if len(lines) < 3:
        lines = [x.strip() for x in re.split(r"(?<=[.!?])\s+", " ".join(lines) or body) if len(x.strip()) >= 15]
    if len(lines) > cfg.PRM_MAX_STEPS:
        g = math.ceil(len(lines) / cfg.PRM_MAX_STEPS)
        lines = [" ".join(lines[i:i + g]) for i in range(0, len(lines), g)]
    return [" ".join(l.split())[:400] for l in lines]


def prm_prompt(case, steps, answer_text):
    note = (f" Only images 1-{cfg.VERIFY_IMAGES} are shown; findings about other images are UNCLEAR."
            if len(case["imgs"]) > cfg.VERIFY_IMAGES else "")
    lst = "\n".join(f"{i + 1}. {st}" for i, st in enumerate(steps))
    return (f"{case['ctx']}\n\nA student's reasoning, step by step (final answer: {answer_text}):\n{lst}\n\n"
            "Grade EACH step on TWO separate questions, as a process reward model:\n"
            "V = visual evidence: are the image findings this step states really present? Look at the images. "
            "SUPPORTED (clearly visible) / UNCLEAR (cannot be judged) / CONTRADICTED (the images show "
            "otherwise) / NA (the step states no image finding)." + note + "\n"
            "R = reasoning: ASSUMING the findings it relies on are true, is the medical inference of this step "
            "sound (correct radiology knowledge, follows from those findings and the history, does not "
            "overreach)? VALID / WEAK (plausible but incomplete or overreaching) / INVALID (medically wrong or "
            "does not follow) / NA (pure description, no inference).\n"
            "Judge V and R independently: a step can describe a real finding but draw a wrong conclusion, or "
            "reason correctly from a finding that is not there.\n"
            "Answer with one line per step, exactly in the form '<number>: V=<label> R=<label>'.")


def aggregate(vs):
    if not vs:
        return 0.5
    if cfg.PRM_AGG == "min":
        return min(vs)
    if cfg.PRM_AGG == "mean":
        return float(np.mean(vs))
    return float(np.exp(np.mean(np.log(np.clip(vs, 0.05, 1.0)))))


def _prm_scores(items):
    jobs, steps_all = [], []
    for ci, (case, paths) in enumerate(items):
        valid = "".join(case["opts"])
        st = [split_steps(r) for r in paths]
        steps_all.append(st)
        for k, (r, ss) in enumerate(zip(paths, st)):
            a = extract_answer(r, valid)
            if ss:
                jobs.append((ci, k, case, ss, f"{a}. {case['opts'][a]}" if a else "none"))
    outs = run_many(lambda jb: judge_vision(jb[2], prm_prompt(jb[2], jb[3], jb[4]), 16 * len(jb[3]) + 40, "prm"),
                    jobs, cfg.JUDGE_CONC, "evidence PRM")
    got = {}
    for (ci, k, _, ss, _), txt in zip(jobs, outs):
        g = {}
        for m in re.finditer(r"(\d+)\s*[:.)\-]\s*\**\s*V\s*=\s*\**([A-Za-z]+)\**[\s,;|]*R\s*=\s*\**([A-Za-z]+)", txt or ""):
            g[int(m.group(1)) - 1] = (VSC.get(m.group(2).upper(), 0.5), RSC.get(m.group(3).upper(), 0.5))
        got[(ci, k)] = [g.get(i, (0.5, 0.5)) for i in range(len(ss))]
    res = []
    for ci, (_, paths) in enumerate(items):
        detail, Sv, Sr, S, claims = [], [], [], [], []
        for k in range(len(paths)):
            d = [(st, v, r) for st, (v, r) in zip(steps_all[ci][k], got.get((ci, k), []))]
            sv = aggregate([v for _, v, _ in d if v is not None])
            sr = aggregate([r for _, _, r in d if r is not None])
            detail.append(d)
            Sv.append(sv)
            Sr.append(sr)
            S.append(float(sv ** cfg.ALPHA * sr ** (1 - cfg.ALPHA)))
            claims.append([(st, min(x for x in (v, r) if x is not None) if (v, r) != (None, None) else 0.5)
                           for st, v, r in d])
        res.append(dict(S=S, S_vis=Sv, S_rea=Sr, detail=detail, claims=claims))
    return res


_PRM_CACHE = {}


def prm_scores(items):
    """items: [(case, paths)] -> per item dict(S, S_vis, S_rea, detail, claims). Results are cached."""
    keys = [(c["id"], tuple(hash(r) for r in paths)) for c, paths in items]
    todo = {k: it for k, it in zip(keys, items) if k not in _PRM_CACHE}
    for k, r in zip(todo, _prm_scores(list(todo.values()))):
        _PRM_CACHE[k] = r
    return [copy.deepcopy(_PRM_CACHE[k]) for k in keys]


def select(items, pb=None):
    """items: [(case, paths)] -> [(best index, confidence, evidence or None)].
    Unanimous answers are kept without grading. With a Playbook, the score of a path whose answer has
    an entry is multiplied by ELIM_MIN + (1 - ELIM_MIN) * q, where q is the checklist score on the images."""
    out, need = [None] * len(items), []
    for t, (case, paths) in enumerate(items):
        valid = "".join(case["opts"])
        idx = [i for i, r in enumerate(paths) if extract_answer(r, valid)]
        if len({extract_answer(paths[i], valid) for i in idx}) <= 1:
            out[t] = (idx[0] if idx else 0, 1.0, None)
        else:
            need.append(t)
    evs = dict(zip(need, prm_scores([items[t] for t in need])))
    elim = {}
    if pb is not None:
        reqs = []
        for t in need:
            case, paths = items[t]
            valid = "".join(case["opts"])
            for L in sorted({extract_answer(r, valid) for r in paths} - {None}):
                if checklist(pb, case, L):
                    reqs.append((t, L))
        for (t, L), r in zip(reqs, check_scores([(pb, items[t][0], L) for t, L in reqs])):
            if r[0] is not None:
                elim[(t, L)] = r
    for t in need:
        case, paths = items[t]
        ev = evs[t]
        ans = [extract_answer(r, "".join(case["opts"])) for r in paths]
        S = list(ev["S"])
        ev["check"] = {}
        for k, a in enumerate(ans):
            if a and (t, a) in elim:
                q, ids = elim[(t, a)]
                S[k] *= cfg.ELIM_MIN + (1 - cfg.ELIM_MIN) * q
                ev["check"][a] = dict(q=q, ids=ids)
        ev["S_final"] = S
        A = collections.defaultdict(float)
        for k, a in enumerate(ans):
            if a:
                A[a] += S[k]
        ya = max(A, key=A.get)
        best = max((k for k, a in enumerate(ans) if a == ya), key=lambda k: S[k])
        out[t] = (best, A[ya] / (sum(A.values()) + 1e-9), ev)
    return out
