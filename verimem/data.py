import io
import os
import re
import json
import base64

from . import config as cfg

LETTERS = "ABCDEFGHIJ"


def parse_options(v):
    if isinstance(v, str) and v.strip()[:1] in "[{":
        try:
            v = json.loads(v)
        except Exception:
            pass
    out = {}
    if isinstance(v, dict):
        for k, t in v.items():
            k = str(k).strip().strip("().:").upper()
            if len(k) == 1 and k in LETTERS and t:
                out[k] = str(t).strip()
    elif isinstance(v, (list, tuple)):
        for i, t in enumerate(v[:len(LETTERS)]):
            m = re.match(r"^\(?([A-J])[\.\):]\s+(.*)$", str(t).strip(), re.S)
            out[m.group(1) if m else LETTERS[i]] = (m.group(2) if m else str(t)).strip()
    elif isinstance(v, str):
        parts = re.split(r"(?:^|\n|\s)\(?([A-J])[\.\):]\s+", v.strip())
        for i in range(1, len(parts) - 1, 2):
            out[parts[i]] = parts[i + 1].strip()
    return dict(sorted(out.items()))


def to_b64(v):
    from PIL import Image
    try:
        if isinstance(v, Image.Image):
            im = v
        elif isinstance(v, dict) and v.get("bytes"):
            im = Image.open(io.BytesIO(v["bytes"]))
        elif isinstance(v, dict) and v.get("path"):
            return to_b64(v["path"])
        elif isinstance(v, str) and v.strip():
            from huggingface_hub import hf_hub_download
            p = v if os.path.isfile(v) else hf_hub_download(cfg.DATASET, v.strip().lstrip("./"), repo_type="dataset")
            im = Image.open(p)
        else:
            return None
        im = im.convert("RGB")
        w, h = im.size
        s = cfg.IMG_SIDE / max(w, h)
        if s < 1:
            im = im.resize((max(32, int(w * s)), max(32, int(h * s))))
        w, h = im.size
        if min(w, h) < 32:
            im = im.resize((max(32, w), max(32, h)))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print("  WARN image:", str(e)[:120])
        return None


def img_cols(keys):
    return sorted([c for c in keys if re.fullmatch(r"image_?\d+", c)], key=lambda c: int(re.sub(r"\D", "", c)))


def make_case(ex, cols, cid):
    opts = parse_options(ex.get("options"))
    gold = str(ex.get("correct_answer") or "").strip().strip("().").upper()[:1]
    if gold not in opts:
        t = str(ex.get("correct_answer_text") or "").strip().lower()
        gold = next((L for L, o in opts.items() if o.lower() == t), None)
    if not opts or not gold:
        return None
    imgs, mods = [], set()
    for c in cols:
        v = ex.get(c)
        if v is None:
            v = ex.get(c + "_path")
        if v is None or v == "":
            continue
        b = to_b64(v)
        if b:
            imgs.append(b)
            for mk in (c + "_modality", c + "_sub_modality"):
                m = str(ex.get(mk) or "").strip().lower()
                if m and m not in ("none", "nan"):
                    mods.add(m)
        if len(imgs) >= cfg.MAX_IMAGES:
            break
    hist = str(ex.get("CLINICAL_HISTORY") or "(not provided)").strip()
    ctx = ("Clinical history:\n" + hist + "\n\nQuestion: What is the most likely diagnosis?\nOptions:\n"
           + "\n".join(f"{L}. {t}" for L, t in opts.items()))
    return dict(id=cid, gold=gold, gold_text=opts[gold], opts=opts, imgs=imgs, mods=mods, ctx=ctx,
                findings=str(ex.get("IMAGING_FINDINGS") or ""), discussion=str(ex.get("discussion") or ""))


def load_test(n):
    from datasets import load_dataset
    ds = load_dataset(cfg.DATASET, split="test")
    cols = img_cols(ds.column_names)
    return [c for c in (make_case(ds[i], cols, f"test-{i}") for i in range(min(n, len(ds)))) if c]


def load_train(n):
    """Training cases that have expert text, in a fixed shuffled order."""
    from datasets import load_dataset
    ds = load_dataset(cfg.DATASET, split="train", streaming=True).shuffle(seed=0, buffer_size=300)
    out, cols = [], None
    for i, ex in enumerate(ds):
        cols = cols or img_cols(ex.keys())
        c = make_case(ex, cols, f"train-{i}")
        if c and (c["findings"] or c["discussion"]):
            out.append(c)
        if len(out) >= n:
            break
    return out
