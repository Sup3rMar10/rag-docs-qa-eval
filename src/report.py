"""Build results/RESULTS.md and charts from results/retrieval.json and results/generation.*.json."""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = "results"
COLORS = {"bm25": "#9aa5b1", "dense": "#2a9d8f", "hybrid": "#6c8ebf", "hybrid+rerank": "#e9a23b"}


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def retrieval_section(data):
    lines = ["## Retrieval: does the right passage come back?\n",
             "A chunk is a hit if it contains the question's verbatim evidence sentence. 40 answerable questions.\n",
             "| Method | Chunk size (words) | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | Avg context @5 (words) | ms/query |",
             "|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(data, key=lambda k: (data[k]["chunk_size"], list(COLORS).index(data[k]["method"]))):
        r = data[key]
        lines.append(f"| {r['method']} | {r['chunk_size']} | {r['hit@1']:.0%} | {r['hit@3']:.0%} | {r['hit@5']:.0%} | "
                     f"{r['hit@10']:.0%} | {r['mrr@10']:.3f} | {r['avg_ctx_words@5']:.0f} | {r['avg_latency_ms']:.0f} |")
    lines += ["", "Hit@5 by question difficulty:\n", "| Method | Chunk size | Easy | Medium | Hard |", "|---|---|---|---|---|"]
    for key in sorted(data, key=lambda k: (data[k]["chunk_size"], list(COLORS).index(data[k]["method"]))):
        r = data[key]
        d = r["by_difficulty"]
        lines.append(f"| {r['method']} | {r['chunk_size']} | " +
                     " | ".join(f"{d[x]['hit@5']:.0%} (n={d[x]['n']})" if x in d else "-" for x in ("easy", "medium", "hard")) + " |")
    return lines


def plot_retrieval(data):
    sizes = sorted({r["chunk_size"] for r in data.values()})
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 3.6), sharey=True, squeeze=False)
    for ax, size in zip(axes[0], sizes, strict=True):
        rows = [r for r in data.values() if r["chunk_size"] == size]
        rows.sort(key=lambda r: list(COLORS).index(r["method"]))
        xs = range(len(rows))
        w = 0.38
        ax.bar([x - w / 2 for x in xs], [r["hit@5"] * 100 for r in rows], w, label="Hit@5",
               color=[COLORS[r["method"]] for r in rows])
        ax.bar([x + w / 2 for x in xs], [r["hit@1"] * 100 for r in rows], w, label="Hit@1",
               color=[COLORS[r["method"]] for r in rows], alpha=0.45)
        for x, r in zip(xs, rows, strict=True):
            ax.text(x - w / 2, r["hit@5"] * 100 + 1.5, f"{r['hit@5']:.0%}", ha="center", fontsize=8)
            ax.text(x + w / 2, r["hit@1"] * 100 + 1.5, f"{r['hit@1']:.0%}", ha="center", fontsize=8)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([r["method"] for r in rows], fontsize=8)
        ax.set_title(f"chunks of up to {size} words (solid = Hit@5, faded = Hit@1)", fontsize=9)
        ax.set_ylim(0, 105)
    axes[0][0].set_ylabel("% of questions")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "retrieval.png"), dpi=150)


def audited_accuracy(run, audit):
    """Scorer accuracy adjusted by the manual false-negative / false-positive overrides, or None if not audited."""
    a = audit.get(run["name"])
    if a is None:
        return None
    s = run["summary"]
    correct = round(s["answer_accuracy"] * s["n_answerable"])
    return (correct + len(a.get("false_negatives", {})) - len(a.get("false_positives", {}))) / s["n_answerable"]


def generation_section(runs, audit):
    if not runs:
        return []
    lines = ["", "## Generation: does the answer come out right?\n",
             "Scored by deterministic checks (no LLM judge). 40 answerable + 8 unanswerable questions. "
             "`Audited` applies the manual overrides in `data/manual_audit.json` (single reviewer, each with a reason).\n",
             "| Run | Accuracy (scorer) | Accuracy (audited) | Citation grounded | Falsely abstains | "
             "Abstains on unanswerable | Hallucinates on unanswerable | Acc. when evidence in prompt | "
             "Acc. when retrieval missed | s/question |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for run in runs:
        s = run["summary"]
        cites = pct(s["citation_grounded_rate"]) if run["mode"] == "rag" else "-"
        aud = audited_accuracy(run, audit)
        lines.append(f"| {run['name']} | {pct(s['answer_accuracy'])} | {pct(aud) if aud is not None else 'not audited'} | "
                     f"{cites} | {pct(s['false_abstention_rate'])} | "
                     f"{pct(s['abstention_rate_on_unanswerable'])} | {pct(s['hallucination_rate_on_unanswerable'])} | "
                     f"{pct(s['accuracy_when_evidence_in_context'])} (n={s['n_evidence_in_context']}) | "
                     f"{pct(s['accuracy_when_retrieval_missed'])} | {s['avg_seconds_per_question']:.0f} |")
    return lines


def plot_generation(runs, audit):
    labels = [r["name"].replace(" (", "\n(") for r in runs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    xs = range(len(runs))
    w = 0.38
    scorer = [r["summary"]["answer_accuracy"] * 100 for r in runs]
    aud = [(audited_accuracy(r, audit) or 0) * 100 for r in runs]
    axes[0].bar([x - w / 2 for x in xs], scorer, w, color="#2a9d8f", label="strict scorer")
    axes[0].bar([x + w / 2 for x in xs], aud, w, color="#2a9d8f", alpha=0.45, label="manually audited")
    for x, a, b in zip(xs, scorer, aud, strict=True):
        axes[0].text(x - w / 2, a + 1.5, f"{a:.0f}%", ha="center", fontsize=8)
        if b:
            axes[0].text(x + w / 2, b + 1.5, f"{b:.0f}%", ha="center", fontsize=8)
    axes[0].set_title("Answer accuracy (40 answerable questions)", fontsize=10)
    axes[0].legend(fontsize=8, loc="upper left")
    abst = [r["summary"]["abstention_rate_on_unanswerable"] * 100 for r in runs]
    hall = [r["summary"]["hallucination_rate_on_unanswerable"] * 100 for r in runs]
    axes[1].bar([x - w / 2 for x in xs], abst, w, color="#6c8ebf", label="correctly says 'I don't know'")
    axes[1].bar([x + w / 2 for x in xs], hall, w, color="#d1495b", label="invents an answer")
    for x, a, b in zip(xs, abst, hall, strict=True):
        axes[1].text(x - w / 2, a + 1.5, f"{a:.0f}%", ha="center", fontsize=8)
        axes[1].text(x + w / 2, b + 1.5, f"{b:.0f}%", ha="center", fontsize=8)
    axes[1].set_title("Unanswerable questions (n=8, very noisy)", fontsize=10)
    axes[1].legend(fontsize=8, loc="upper left")
    for ax in axes:
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylim(0, 112)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "generation.png"), dpi=150)


def main():
    lines = ["# Results\n", "Generated by `python -m src.report` from `results/*.json`.\n"]
    path = os.path.join(RES, "retrieval.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        lines += retrieval_section(data)
        plot_retrieval(data)
    runs = []
    for p in sorted(glob.glob(os.path.join(RES, "generation.*.json"))):
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
    runs.sort(key=lambda r: (r["mode"] != "closed-book", r.get("prompt", "")))   # closed-book, then v1, then v2
    audit = {}
    if os.path.exists(os.path.join("data", "manual_audit.json")):
        with open(os.path.join("data", "manual_audit.json"), encoding="utf-8") as f:
            audit = json.load(f)
    lines += generation_section(runs, audit)
    if runs:
        plot_generation(runs, audit)
    with open(os.path.join(RES, "RESULTS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote RESULTS.md" + (" and retrieval.png" if os.path.exists(path) else ""))


if __name__ == "__main__":
    main()
