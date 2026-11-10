import glob, hashlib, json, math, os, re, statistics, sys
W = "/workspace"
def latest(pattern):
    fs = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
    return fs[-1] if fs else None

print("=" * 70, "\n[A] DPO TRAINING SUMMARY")
st = latest(f"{W}/dpo_run/trainer_state.json") or latest("/root/outputs/checkpoint-*/trainer_state.json")
if st:
    s = json.load(open(st)); log = s["log_history"]
    print(f"file={st}  global_step={s.get('global_step')}  epochs={s.get('epoch')}")
    keys = ["loss", "rewards/accuracies", "rewards/margins", "rewards/chosen", "rewards/rejected",
            "logps/chosen", "logps/rejected", "grad_norm"]
    train = [e for e in log if "loss" in e and "eval_loss" not in e and "train_runtime" not in e]
    nan = sum(1 for e in train if any(isinstance(e.get(k), float) and math.isnan(e[k]) for k in keys))
    print(f"logged train steps={len(train)}  steps with NaN={nan}")
    print("train means per epoch:")
    for ep in range(1, 4):
        part = [e for e in train if ep - 1 < e.get("epoch", 0) <= ep + 1e-6]
        if part:
            print(f"  epoch {ep}: " + "  ".join(
                f"{k}={statistics.fmean(e[k] for e in part if k in e):.4f}" for k in keys if any(k in e for e in part)))
    print("first 10 steps mean: " + "  ".join(
        f"{k}={statistics.fmean(e[k] for e in train[:10] if k in e):.4f}" for k in keys[:3]))
    print("held-out evaluations:")
    for e in log:
        if "eval_loss" in e:
            print(f"  epoch={e.get('epoch', 0):.3f} step={e.get('step')} " + "  ".join(
                f"{k}={float(e['eval_' + k]):.4f}" for k in keys[:-1] if ('eval_' + k) in e))
    for e in log:
        if "train_runtime" in e:
            print(f"train_runtime={e['train_runtime']:.0f}s ({e['train_runtime']/3600:.2f} h)  "
                  f"train_loss={e.get('train_loss')}  samples/s={e.get('train_samples_per_second')}")
else:
    print("trainer_state.json not found")

print("=" * 70, "\n[B] PAIRED McNEMAR TEST  SFT vs DPO (exact, two-sided)")
BENCH = {"ARC-Easy": ("arc_easy", "acc_norm", "none"), "ARC-Challenge": ("arc_challenge", "acc_norm", "none"),
         "HellaSwag": ("hellaswag", "acc_norm", "none"), "MMLU": ("mmlu", "acc", "none"),
         "WinoGrande": ("winogrande", "acc", "none"), "GSM8K": ("gsm8k", "exact_match", "flexible-extract")}
def items(run, task, metric, flt):
    pat = f"{W}/results/{run}/**/samples_{'mmlu_' if task == 'mmlu' else task + '_'}*.jsonl"
    best = {}
    for f in glob.glob(pat, recursive=True):
        sub = os.path.basename(f)[8:].rsplit("_", 1)[0]
        if task != "mmlu" and sub != task: continue
        if sub not in best or os.path.getmtime(f) > os.path.getmtime(best[sub]): best[sub] = f
    out = {}
    for sub, f in best.items():
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("filter", "none") == flt and metric in r:
                out[(sub, r["doc_id"])] = int(float(r[metric]) >= 0.5)
    return out
def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
print(f"{'benchmark':<14}{'n':>7}{'SFT%':>8}{'DPO%':>8}{'SFTok-DPOx':>12}{'SFTx-DPOok':>12}{'p':>10}")
for name, (t, m, fl) in BENCH.items():
    a, b = items("sft", t, m, fl), items("dpo", t, m, fl)
    ks = a.keys() & b.keys()
    if not ks: print(f"{name:<14} no samples"); continue
    x = sum(a[k] and not b[k] for k in ks); y = sum(b[k] and not a[k] for k in ks)
    print(f"{name:<14}{len(ks):>7}{100*sum(a[k] for k in ks)/len(ks):>8.2f}{100*sum(b[k] for k in ks)/len(ks):>8.2f}"
          f"{x:>12}{y:>12}{mcnemar(x, y):>10.4f}")

print("=" * 70, "\n[C] GSM8K GENERATED ANSWER LENGTH (words)")
for run in ["sft", "dpo"]:
    f = latest(f"{W}/results/{run}/**/samples_gsm8k_*.jsonl")
    if not f: continue
    L = []
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        if r.get("filter") != "flexible-extract": continue
        resp = r.get("resps", [[""]])[0]
        resp = resp[0] if isinstance(resp, list) else resp
        L.append(len(str(resp).split()))
    L2 = sorted(L)
    print(f"{run.upper()}: n={len(L)} mean={statistics.fmean(L):.1f} median={L2[len(L2)//2]} "
          f"p90={L2[int(0.9*(len(L2)-1))]} max={L2[-1]}")

print("=" * 70, "\n[D] OVERLAP: Orca-DPO pairs vs SlimOrca (SFT data)")
try:
    from datasets import load_dataset
    h = lambda t: hashlib.sha1(re.sub(r"\s+", " ", (t or "").strip().lower()).encode()).hexdigest()
    slim = load_dataset("Open-Orca/SlimOrca", split="train")
    sft5k = slim.shuffle(seed=42).select(range(5000))
    def index(ds):
        P, A = set(), set()
        for ex in ds:
            for t in ex["conversations"]:
                if t["from"] == "human": P.add(h(t["value"]))
                elif t["from"] == "gpt": A.add(h(t["value"]))
        return P, A
    P5, A5 = index(sft5k); Pall, Aall = index(slim)
    n = p5 = c5 = pa = ca = 0
    for line in open(f"{W}/train.jsonl", encoding="utf-8"):
        ex = json.loads(line); n += 1
        u = next(mm["content"] for mm in ex["messages"] if mm["role"] == "user")
        ch = ex["chosen"]["content"] if isinstance(ex["chosen"], dict) else ex["chosen"]
        p5 += h(u) in P5; c5 += h(ch) in A5; pa += h(u) in Pall; ca += h(ch) in Aall
    print(f"DPO pairs={n}  SlimOrca full={len(slim)}  SFT subset=5000")
    print(f"vs SFT 5K subset : prompt overlap={p5} ({100*p5/n:.2f}%)  chosen overlap={c5} ({100*c5/n:.2f}%)")
    print(f"vs full SlimOrca : prompt overlap={pa} ({100*pa/n:.2f}%)  chosen overlap={ca} ({100*ca/n:.2f}%)")
except Exception as e:
    print("overlap check skipped:", repr(e))
print("=" * 70)
