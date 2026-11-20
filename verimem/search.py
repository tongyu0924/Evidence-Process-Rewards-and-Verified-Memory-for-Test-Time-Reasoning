"""Strategy design, recursion with contrastive reflection, and verified writing."""
import re
import collections

from . import config as cfg
from .text import extract_answer, parse_json
from .serving import run_many, judge_text, gen_k
from .playbook import imaging_ok
from .prm import prm_scores, select

BASE_INSTR = ("Reason step by step concisely: describe the key findings in the images, integrate them with the "
              "clinical history, and weigh the options. Then end with a final line exactly 'Answer: X' "
              "where X is one option letter.")

FALLBACK = [
    "Describe every image systematically (modality, sequence, location, signal/density) before looking at the options.",
    "Start from the single most discriminating imaging finding and test each option against it.",
    "Try to falsify each option in turn using the clinical history and the images; keep the one that survives.",
    "Rank the options by prior probability from age, sex and history, then check whether the imaging confirms the top one.",
]


def strategy_instr(strategy, lessons=()):
    return lessons_block(lessons) + f"Strategy for this attempt: {strategy}\n\n" + BASE_INSTR


def lessons_block(lessons):
    if not lessons:
        return ""
    return ("Lessons from comparing your earlier attempts on THIS case (they come from self-comparison, not from "
            "the answer key; verify them against the images):\n" + "\n".join(lessons[-6:]) + "\n\n")


def design_prompt(case, n, lessons):
    kl = "\n".join(lessons[-6:]) if lessons else "(empty)"
    return (f"{case['ctx']}\n\nKnowledge list from earlier attempts on THIS case:\n{kl}\n\n"
            f"Design {n} DIFFERENT strategies for a junior radiologist who will solve this case from the images. "
            "Each strategy is 1-2 sentences describing HOW to approach the case (which findings or images to "
            "examine first, which differential to rule out, which pitfall to avoid); strategies must differ in "
            "approach, must follow the knowledge list, and must NOT state a final answer.\n"
            'Return ONLY JSON: {"strategies": [{"strategy": "..."}]}')


def design(cases, n, lessons_list, role="designer"):
    """n strategies per case. The Playbook is never shown to the designer or the generator."""
    outs = run_many(lambda p: judge_text(p, 700, role),
                    [design_prompt(c, n, l) for c, l in zip(cases, lessons_list)], cfg.JUDGE_CONC, "strategy design")
    res = []
    for t in outs:
        j = parse_json(t) or {}
        items = []
        for x in (j.get("strategies") or []):
            st = " ".join(str(x.get("strategy", "") if isinstance(x, dict) else x).split())
            if len(st) >= 15:
                items.append(st[:400])
        items = (items + [f for f in FALLBACK if f not in items])[:n]
        res.append(items)
    return res


def contrast_prompt(case, paths, best, valid):
    others = [i for i in range(len(paths)) if i != best and extract_answer(paths[i], valid)
              and extract_answer(paths[i], valid) != extract_answer(paths[best], valid)][:2]
    if not others:
        return None
    body = "\n\n".join(f"[OTHER reasoning, answer {extract_answer(paths[i], valid)}]\n{paths[i][-1800:]}" for i in others)
    return (f"{case['ctx']}\n\n[PREFERRED reasoning, answer {extract_answer(paths[best], valid)}]\n{paths[best][-2200:]}\n\n"
            f"{body}\n\nCompare the PREFERRED reasoning with the OTHER reasonings. Which findings or steps does the "
            "preferred one handle better, and which mistakes or unsupported assumptions do the others make? "
            "Write 2-4 short lessons (Do's and Don'ts) for re-solving THIS case. Do not just state a final answer. "
            "Each line must start with '- '.")


def contradicted(ev, limit=3):
    """Steps graded as contradicted or invalid become lessons for the next round."""
    if not ev or not ev.get("detail"):
        return []
    vis = list(dict.fromkeys(st for d in ev["detail"] for st, v, _ in d if v == 0.0))
    rea = list(dict.fromkeys(st for d in ev["detail"] for st, v, r in d if r == 0.0 and v != 0.0))
    out = [f"- The images do NOT show this; don't rely on it: {st}" for st in vis[:limit]]
    out += [f"- This inference is medically invalid; don't repeat it: {st}" for st in rea[:max(0, limit - len(out))]]
    return out


def recurse(cases, pb, pools, sels):
    """While paths disagree and confidence >= TAU: reflect, update the knowledge list, design K - 1 new
    strategies and re-select among the new paths and the previously selected one."""
    pools = [list(p) for p in pools]
    evs = [e for _, _, e in sels]
    sels = [(b, c) for b, c, _ in sels]
    rounds = [1] * len(cases)
    lessons = [[] for _ in cases]
    prev = [None] * len(cases)
    active = list(range(len(cases)))
    for rnd in range(1, cfg.ROUNDS + 1):
        nxt, cps = [], []
        for k in active:
            valid = "".join(cases[k]["opts"])
            b, conf = sels[k]
            ans = [extract_answer(r, valid) for r in pools[k]]
            stable = prev[k] is not None and prev[k] == ans[b]
            prev[k] = ans[b]
            if rnd >= cfg.ROUNDS or stable:
                continue
            if len({a for a in ans if a}) > 1 and conf >= cfg.TAU:
                cp = contrast_prompt(cases[k], pools[k], b, valid)
                if cp:
                    nxt.append(k)
                    cps.append((k, cp))
        if not nxt:
            break
        refl = run_many(lambda x: judge_text(x[1], 400, "reflection"), cps, cfg.JUDGE_CONC, "contrastive reflection")
        for (k, _), t in zip(cps, refl):
            lessons[k] += [l.strip() for l in (t or "").splitlines() if l.strip().startswith("-")][:4]
            lessons[k] += [x for x in contradicted(evs[k]) if x not in lessons[k]]
        des = design([cases[k] for k in nxt], cfg.K - 1, [lessons[k] for k in nxt], "designer_recurse")
        jobs = [(k, st) for k, d in zip(nxt, des) for st in d]
        outs = gen_k([(cases[k], strategy_instr(st, lessons[k])) for k, st in jobs], 1, "gen_recurse")
        newp = collections.defaultdict(list)
        for (k, _), o in zip(jobs, outs):
            newp[k].append(o[0])
        new_pools = [newp[k] + [pools[k][sels[k][0]]] for k in nxt]
        for k, p, (b, c, e) in zip(nxt, new_pools, select([(cases[k], p) for k, p in zip(nxt, new_pools)], pb=pb)):
            pools[k], sels[k], evs[k], rounds[k] = p, (b, c), e, rnd + 1
        active = nxt
    return pools, sels, lessons, rounds


def write_verified(cases, pools, sels, pb, log):
    """Add up to two imaging steps of the selected path to the entry of the chosen disease, only when
    confidence >= WRITE_CONF and no step is contradicted or invalid."""
    todo, claims = [], {}
    for t, (paths, (b, conf, ev)) in enumerate(zip(pools, sels)):
        if conf < cfg.WRITE_CONF:
            log["low confidence"] += 1
            continue
        todo.append(t)
        if ev is not None and ev.get("claims"):
            claims[t] = ev["claims"][b]
    need = [t for t in todo if t not in claims]
    for t, ev in zip(need, prm_scores([(cases[t], [pools[t][sels[t][0]]]) for t in need])):
        claims[t] = ev["claims"][0]
    written = []
    for t in todo:
        c, paths, (b, _, _) = cases[t], pools[t], sels[t]
        ans = extract_answer(paths[b], "".join(c["opts"]))
        cl = claims.get(t) or []
        good = [s for s, v in cl if v == 1.0 and imaging_ok(s) and 20 <= len(s) <= 250
                and not re.search(r"\b(option|answer|diagnosis|therefore|consistent with|likely)\b", s.lower())]
        if len(cl) < 2 or any(v == 0.0 for _, v in cl) or not good or not ans:
            log["not verified"] += 1
            continue
        fs = [dict(type="observed", text=s, vs="", quote="(test time, verified on images)") for s in good[:2]]
        n = pb.add(c["opts"][ans], [], "", c.get("mods", set()), fs, f"test:{c['id']}")
        log["written"] += n
        if n:
            written.append((c["id"], c["opts"][ans], [f["text"] for f in fs]))
    return written
