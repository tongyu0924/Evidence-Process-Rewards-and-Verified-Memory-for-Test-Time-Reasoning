#!/usr/bin/env python3
"""Build the Playbook on the training split, then run test-time search on the test split.

Example:
    python run.py --limit 720 --gen-model Qwen/Qwen3-VL-8B-Instruct
"""
import os
import sys
import json
import time
import copy
import signal
import atexit
import random
import argparse
import collections

import numpy as np

from verimem import config as cfg


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=cfg.DATASET)
    ap.add_argument("--limit", type=int, default=cfg.LIMIT)
    ap.add_argument("--train-n", type=int, default=cfg.TRAIN_N)
    ap.add_argument("--credit-n", type=int, default=cfg.CREDIT_N)
    ap.add_argument("--val-n", type=int, default=cfg.VAL_N)
    ap.add_argument("--gen-model", default=cfg.GEN_MODEL)
    ap.add_argument("--judge-model", default=cfg.JUDGE_MODEL)
    ap.add_argument("--gpus", default=cfg.GPUS)
    ap.add_argument("--k", type=int, default=cfg.K)
    ap.add_argument("--rounds", type=int, default=cfg.ROUNDS)
    ap.add_argument("--out-dir", default=cfg.OUT_DIR)
    a = ap.parse_args()
    for key, val in vars(a).items():
        setattr(cfg, key.upper(), val)


parse_args()

from verimem import serving                                              # noqa: E402
from verimem.data import load_test, load_train                           # noqa: E402
from verimem.text import extract_answer, mcnemar                         # noqa: E402
from verimem.playbook import Playbook, build_batch, measure_credit       # noqa: E402
from verimem.prm import prm_scores, select                               # noqa: E402
from verimem.search import BASE_INSTR, design, strategy_instr, recurse, write_verified  # noqa: E402


def start_servers():
    serving.launch("gen", cfg.GEN_MODEL, cfg.GEN_PORT, cfg.GEN_MAX_LEN, cfg.MAX_IMAGES, cfg.GEN_NEED_GB,
                   aliases=("judge",))
    extra = ["--quantization", cfg.JUDGE_QUANT] if cfg.JUDGE_QUANT and "awq" not in cfg.JUDGE_MODEL.lower() else []
    try:
        serving.launch("judge", cfg.JUDGE_MODEL, cfg.JUDGE_PORT, cfg.JUDGE_MAX_LEN, cfg.VERIFY_IMAGES,
                       cfg.JUDGE_NEED_GB, extra=extra + ["--max-num-seqs", "32"])
        return cfg.JUDGE_MODEL
    except RuntimeError as e:
        # Fall back to a single GPU: the generator also acts as the judge
        print(f">>> judge server unavailable ({str(e).splitlines()[0]}); using the generator as judge", flush=True)
        cfg.JUDGE_PORT = cfg.GEN_PORT
        return cfg.GEN_MODEL + " (shared)"


def answer(case, paths, best):
    return extract_answer(paths[best], "".join(case["opts"]))


def majority(case, paths):
    cnt = collections.Counter(a for a in (extract_answer(r, "".join(case["opts"])) for r in paths) if a)
    return cnt.most_common(1)[0][0] if cnt else None


def prm_pair_accuracy(cases):
    """Share of (correct, wrong) path pairs on labelled cases where the PRM ranks the correct path higher."""
    outs = serving.gen_k([(c, BASE_INSTR) for c in cases], cfg.VAL_SAMPLES, "val_gen", 1024)
    pairs = []
    for c, paths in zip(cases, outs):
        valid = "".join(c["opts"])
        good = [p for p in paths if extract_answer(p, valid) == c["gold"]]
        bad = [p for p in paths if extract_answer(p, valid) not in (None, c["gold"])]
        pp = [(c, g, b) for g in good for b in bad]
        random.Random(hash(c["id"]) % 10 ** 6).shuffle(pp)
        pairs += pp[:cfg.PAIR_MAX]
    ev = prm_scores([(c, [g, b]) for c, g, b in pairs])
    acc = lambda key: float(np.mean([1.0 if e[key][0] > e[key][1] else 0.5 if e[key][0] == e[key][1] else 0.0
                                     for e in ev])) if ev else float("nan")
    return dict(pairs=len(pairs), combined=acc("S"), visual=acc("S_vis"), reasoning=acc("S_rea"))


def main():
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    atexit.register(serving.stop_servers)
    signal.signal(signal.SIGTERM, lambda *a: (serving.stop_servers(), sys.exit(1)))
    t0 = time.time()
    judge = start_servers()

    test = load_test(cfg.LIMIT)
    allv = load_train(cfg.TRAIN_N + cfg.CREDIT_N + cfg.VAL_N)
    tr = allv[:cfg.TRAIN_N]
    cr = allv[cfg.TRAIN_N:cfg.TRAIN_N + cfg.CREDIT_N]
    val = allv[cfg.TRAIN_N + cfg.CREDIT_N:]
    print(f">>> playbook {len(tr)} / credit {len(cr)} / val {len(val)} / test {len(test)} cases", flush=True)

    # 1) Build the Playbook from expert text
    pb = Playbook()
    stat = collections.Counter()
    for s in range(0, len(tr), cfg.BATCH):
        batch = tr[s:s + cfg.BATCH]
        outs = serving.gen_greedy([(c, BASE_INSTR) for c in batch], "train_gen", 1024)
        added = build_batch(pb, batch, outs, stat)
        print(f">>> playbook batch {s // cfg.BATCH + 1}: +{added} items, {len(pb.b)} diseases", flush=True)
    print("    rejected: " + ", ".join(f"{k} {v}" for k, v in stat.items()), flush=True)
    if not pb.b:
        sys.exit(">>> the Playbook is empty; increase --train-n")

    # 2) Entry credit on labelled cases
    credit = dict(train=measure_credit(pb, cr, update=True), val=measure_credit(pb, val, update=False))
    measure_credit(pb, val, update=True)
    print(f">>> checklist score on correct / wrong options: {credit['val'][0]:.2f} / {credit['val'][1]:.2f}", flush=True)
    prm_acc = prm_pair_accuracy(val)
    print(f">>> PRM pair accuracy on validation: {prm_acc['combined'] * 100:.1f}%", flush=True)

    # 3) Test stream (no labels)
    pb_static, pb_online = copy.deepcopy(pb), copy.deepcopy(pb)
    n = len(test)
    keys = ("cot", "self_consistency", "prm", "strategies", "playbook_check", "verimem")
    P = {k: [None] * n for k in keys}
    rounds_all, lessons_all, written_all = [1] * n, [[] for _ in range(n)], []
    wlog = collections.Counter()
    for s in range(0, n, cfg.STREAM_B):
        B = test[s:s + cfg.STREAM_B]
        print(f">>> test {s + 1}-{s + len(B)} | {len(pb_online.active())} Playbook entries", flush=True)
        greedy = serving.gen_greedy([(c, BASE_INSTR) for c in B], "test_greedy")
        anchors = serving.gen_k([(c, BASE_INSTR) for c in B], cfg.K, "test_anchor")
        a_sel = select(list(zip(B, anchors)))
        des = design(B, cfg.SLOTS, [[] for _ in B])
        jobs = [(t, st) for t, d in enumerate(des) for st in d]
        outs = serving.gen_k([(B[t], strategy_instr(st)) for t, st in jobs], 1, "test_strategy")
        extra = collections.defaultdict(list)
        for (t, _), o in zip(jobs, outs):
            extra[t].append(o[0])
        pools0 = [list(anchors[t][:max(1, cfg.K - len(extra[t]))]) + extra[t] for t in range(len(B))]
        s_sel = select(list(zip(B, pools0)))
        st_sel = select(list(zip(B, pools0)), pb=pb_static)
        fu_sel = select(list(zip(B, pools0)), pb=pb_online)
        written_all += write_verified(B, pools0, fu_sel, pb_online, wlog)
        pools, sels, lessons, rounds = recurse(B, pb_online, pools0, fu_sel)
        for t, c in enumerate(B):
            k = s + t
            P["cot"][k] = extract_answer(greedy[t], "".join(c["opts"]))
            P["self_consistency"][k] = majority(c, anchors[t])
            P["prm"][k] = answer(c, anchors[t], a_sel[t][0])
            P["strategies"][k] = answer(c, pools0[t], s_sel[t][0])
            P["playbook_check"][k] = answer(c, pools0[t], st_sel[t][0])
            P["verimem"][k] = answer(c, pools[t], sels[t][0])
            rounds_all[k], lessons_all[k] = rounds[t], lessons[t]

    # 4) Report
    gold = [c["gold"] for c in test]
    rows = {}
    print("\n" + "=" * 72)
    print(f"{n} test cases | generator {cfg.GEN_MODEL} | judge {judge} | K={cfg.K}")
    for key in keys:
        acc = sum(p == y for p, y in zip(P[key], gold)) / n
        w = sum(p == y and b != y for p, b, y in zip(P[key], P["cot"], gold))
        l = sum(p != y and b == y for p, b, y in zip(P[key], P["cot"], gold))
        rows[key] = dict(acc=acc, vs_cot=dict(win=w, lose=l, p=mcnemar(w, l)))
        print(f"{key:<18} {acc * 100:5.1f}%" + ("" if key == "cot" else f"   vs CoT +{w}/-{l}  p={mcnemar(w, l):.3f}"))
    print(f"mean rounds {np.mean(rounds_all):.2f} | written {len(written_all)} | {(time.time() - t0) / 60:.1f} min")

    tag = f"{cfg.GEN_MODEL.split('/')[-1]}_K{cfg.K}_R{cfg.ROUNDS}_test{n}"
    cost = {k: dict(calls=v[0], prompt=v[1], completion=v[2]) for k, v in serving.COST.items()}
    with open(os.path.join(cfg.OUT_DIR, f"summary_{tag}.json"), "w") as f:
        json.dump(dict(n=n, gen_model=cfg.GEN_MODEL, judge_model=judge, methods=rows, prm_pair_accuracy=prm_acc,
                       checklist_score=credit, playbook_rejections=dict(stat), write_log=dict(wlog),
                       playbook_size=len(pb.b), cost=cost), f, indent=1)
    with open(os.path.join(cfg.OUT_DIR, f"per_case_{tag}.jsonl"), "w") as f:
        for k, c in enumerate(test):
            f.write(json.dumps(dict(id=c["id"], gold=c["gold"], **{m: P[m][k] for m in keys},
                                    rounds=rounds_all[k], lessons=lessons_all[k])) + "\n")
    with open(os.path.join(cfg.OUT_DIR, f"playbook_{tag}.txt"), "w") as f:
        f.write(pb_online.dump())
    print("results saved to", cfg.OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
