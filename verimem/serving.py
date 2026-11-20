"""vLLM servers and OpenAI-compatible chat calls."""
import os
import re
import sys
import json
import time
import shutil
import signal
import subprocess
import collections
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import config as cfg

SYS = ("You are an expert radiologist. You will be given a clinical history and several radiology "
       "images from one patient case, plus answer options. Choose the single best diagnosis.")

SERVERS = []
LAUNCHED = {}
COST = collections.defaultdict(lambda: [0, 0, 0])   # role -> [calls, prompt tokens, completion tokens]


def stop_servers(*_):
    for p in SERVERS:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                p.terminate()
    for p in SERVERS:
        try:
            p.wait(timeout=30)
        except Exception:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                pass


def gpu_inventory():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    inv = []
    for line in out.strip().splitlines():
        try:
            i, free, total = [x.strip() for x in line.split(",")]
            inv.append((int(i), float(free) / 1024, float(total) / 1024))
        except ValueError:
            pass
    if cfg.GPUS != "auto":
        allow = {int(x) for x in str(cfg.GPUS).split(",") if x.strip()}
        inv = [g for g in inv if g[0] in allow]
    return inv


def _log_path(name):
    return os.path.join(cfg.OUT_DIR, f"server_{name}.log")


def start_server(model, name, gpu, port, max_len, mm_limit, util, aliases=(), extra=()):
    exe = shutil.which("vllm") or os.path.join(os.path.dirname(sys.executable), "vllm")
    cmd = ([exe, "serve", model] if os.path.exists(exe) else
           [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", model])
    cmd += ["--served-model-name", name, *aliases, "--host", "127.0.0.1", "--port", str(port),
            "--max-model-len", str(max_len), "--gpu-memory-utilization", f"{util:.3f}",
            "--enable-prefix-caching", "--trust-remote-code",
            "--limit-mm-per-prompt", json.dumps({"image": mm_limit}), *extra]
    logf = open(_log_path(name), "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    SERVERS.append(p)
    print(f">>> starting {name}: {model} on GPU {gpu}", flush=True)
    return p


def wait_ready(p, name, port, timeout=2400):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if p.poll() is not None:
            txt = open(_log_path(name), errors="replace").read()
            key = [l for l in txt.splitlines()
                   if re.search(r"(Error|ERROR|out of memory|OOM|insufficient|Exception)", l)][-10:]
            raise RuntimeError(f"{name} server failed:\n" + "\n".join(key))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    print(f">>> {name} ready ({(time.time() - t0) / 60:.1f} min)", flush=True)
                    return
        except Exception:
            pass
        time.sleep(10)
    raise RuntimeError(f"{name} server not ready after {timeout // 60} min")


def launch(name, model, port, max_len, mm_limit, need, tries=4, aliases=(), extra=()):
    """Start a server on the lowest-numbered GPU with enough free memory; try the next GPU on failure."""
    bad = set()
    for _ in range(tries):
        inv = [g for g in gpu_inventory() if g[0] not in bad]
        left = {i: f - 1.0 - LAUNCHED.get(i, 0) for i, f, _ in inv}
        total = {i: t for i, _, t in inv}
        g = next((i for i in sorted(left) if left[i] >= need), None)
        if g is None:
            raise RuntimeError(f"{name} needs {need:.0f} GB but no GPU has that much free memory")
        p = start_server(model, name, g, port, max_len, mm_limit, min(0.95, need / total[g]), aliases, extra)
        LAUNCHED[g] = LAUNCHED.get(g, 0) + need
        try:
            wait_ready(p, name, port)
            return p, g
        except RuntimeError as e:
            print(f"    {e}", flush=True)
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                p.kill()
            SERVERS.remove(p)
            LAUNCHED[g] -= need
            bad.add(g)
            time.sleep(10)
    raise RuntimeError(f"{name} server failed on {tries} GPUs")


def chat(port, name, messages, max_tokens, role, n=1, temperature=0.0):
    body = {"model": name, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "n": n}
    if temperature > 0:
        body["top_p"] = 0.95
    data = json.dumps(body).encode()
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                d = json.loads(r.read())
            u = d.get("usage") or {}
            c = COST[role]
            c[0] += 1
            c[1] += u.get("prompt_tokens", 0)
            c[2] += u.get("completion_tokens", 0)
            return d["choices"]
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))
    print(f"    WARN {role} request failed: {str(last)[:200]}", flush=True)
    return []


def run_many(fn, items, conc, label):
    if not items:
        return []
    print(f"    {label}: {len(items)} requests", flush=True)
    with ThreadPoolExecutor(max_workers=conc) as ex:
        return list(ex.map(fn, items))


def image_content(case, max_images=None):
    imgs = case["imgs"][:max_images] if max_images else case["imgs"]
    content = []
    for i, b in enumerate(imgs):
        content += [{"type": "text", "text": f"Image {i + 1}:"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}}]
    return content


def vision_msgs(case, text, max_images=None):
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": image_content(case, max_images) + [{"type": "text", "text": text}]}]


def text_msgs(text):
    return [{"role": "system", "content": "You are an expert radiologist and medical educator."},
            {"role": "user", "content": text}]


def gen_k(cases_instrs, k, role, max_tokens=1536):
    """k sampled paths per case from the generator."""
    f = lambda ci: [ch["message"]["content"] or "" for ch in
                    chat(cfg.GEN_PORT, "gen", vision_msgs(ci[0], ci[0]["ctx"] + "\n\n" + ci[1]), max_tokens, role,
                         n=k, temperature=cfg.TEMP)]
    outs = run_many(f, cases_instrs, max(1, cfg.GEN_CONC // max(1, k)) * 2, f"generator x{k}")
    return [o + [""] * (k - len(o)) for o in outs]


def gen_greedy(cases_instrs, role, max_tokens=1536):
    def f(ci):
        ch = chat(cfg.GEN_PORT, "gen", vision_msgs(ci[0], ci[0]["ctx"] + "\n\n" + ci[1]), max_tokens, role)
        return (ch[0]["message"]["content"] or "") if ch else ""
    return run_many(f, cases_instrs, cfg.GEN_CONC, "generator (greedy)")


def judge_text(prompt, max_tokens, role):
    ch = chat(cfg.JUDGE_PORT, "judge", text_msgs(prompt), max_tokens, role)
    return (ch[0]["message"]["content"] or "") if ch else ""


def judge_vision(case, text, max_tokens, role):
    ch = chat(cfg.JUDGE_PORT, "judge", vision_msgs(case, text, cfg.VERIFY_IMAGES), max_tokens, role)
    return (ch[0]["message"]["content"] or "") if ch else ""
