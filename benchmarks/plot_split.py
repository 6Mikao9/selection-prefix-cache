"""Render the checked-in A100 split samples; no CUDA or model dependency.

Usage: python benchmarks/plot_split.py
       python benchmarks/plot_split.py --report-only --small SMALL.json --long LONG.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def load(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    for mode in data["modes"].values():
        samples = [s["wall_seconds"] * 1000 for s in mode["samples"]]
        assert len(samples) == data["config"]["repeats"]
        assert mode["all_indices_exact"]
        for key, actual in (("median", statistics.median(samples)),
                            ("min", min(samples)), ("max", max(samples))):
            assert abs(mode[f"wall_ms_{key}"] - actual) < 1e-9
    return data


def cell(mode):
    return (f"{mode['wall_ms_median']:.3f} "
            f"[{mode['wall_ms_min']:.3f}, {mode['wall_ms_max']:.3f}]")


def report(small, large):
    lines = [
        "# Head recompute / tail selection cache: A100 split sweep",
        "",
        "These are **synthetic QK + top-k/sort + selected-value gather** measurements, "
        "not model prefill latency or model TTFT. No production QSA kernel, complete "
        "attention, projection, MLP, GDN, or end-to-end model is measured.",
        "",
        "NVIDIA A100-SXM4-80GB; PyTorch 2.8.0+cu128. Each mode has seven "
        "samples, interleaved in a seeded shuffled order; two replay buffers and "
        "prefetch are enabled. The associated A100 validation run passed **25 tests**. "
        "That count describes that run only, not later local test additions. All "
        "reported modes passed exact sorted-index and consumer-output checks.",
        "",
        "![Capacity and latency split sweep](split_tradeoff.png)",
        "",
        "The figure shows every sampled split and every codec/source combination. "
        "Markers are medians; whiskers are observed minima and maxima, not confidence "
        "intervals. Both panels share a zero-based runtime axis. The upper axes give "
        "auto-codec retained payload in MiB at selected splits; raw payloads are in "
        "the tables. Payload is **not** total runtime memory.",
        "",
        "## Findings",
        "",
        "For the long configuration, auto retained payload falls from **199,324 B** "
        "at head=0 to **151,682 B** at head=4 and **101,330 B** at head=8 "
        "(23.9% and 49.2% reductions). Corresponding DRAM medians are "
        "**4.416 → 4.348 / 4.348 ms**. This is a capacity/latency tradeoff with "
        "approximately unchanged observed latency; the roughly 1.5% median change "
        "has overlapping sample ranges and does not establish a significant speedup. "
        "Full recomputation is about 4.95–4.97 ms here. Warm-file auto medians also "
        "decrease across these splits, but cannot establish cold-storage performance.",
        "",
        "For the small configuration, **full recomputation is faster than every "
        "retained-tail split** in both sources and both codec policies. Thus tail "
        "caching does not win universally; the small result must accompany the "
        "long result. At head=16, codec/source labels identify separate repeated "
        "measurements of the same no-tail computation, so small differences there "
        "are measurement variation rather than a codec or storage effect.",
        "",
    ]
    for title, data, filename in (("Small", small, "split_small_a100.json"),
                                  ("Long", large, "split_long_a100.json")):
        c = data["config"]
        lines += [f"## {title} configuration", "",
                  f"16 chunks; rows={c['rows']}; k={c['k']}; universe={c['universe']}; "
                  f"head_dim={c['head_dim']}; query noise={c['query_noise']}; seed={c['seed']}. "
                  f"Raw samples: [{filename}]({filename}).", "",
                  "Times are median [minimum, maximum] in milliseconds. Capacities "
                  "are retained encoded **payload bytes**, identical for DRAM and warm file.", "",
                  "| Head chunks / fraction | Auto payload B | Raw payload B | Auto DRAM ms | Auto warm file ms | Raw DRAM ms | Raw warm file ms |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for h in (0, 1, 2, 4, 8, 16):
            modes = data["modes"]
            auto = modes[f"memory:auto:head{h}"]
            raw = modes[f"memory:raw_u16:head{h}"]
            row = [f"{h} / {100*h/16:g}%", f"{auto['capacity']['payload_bytes']:,}",
                   f"{raw['capacity']['payload_bytes']:,}"]
            row += [cell(modes[f"{source}:{codec}:head{h}"])
                    for codec, source in (("auto", "memory"), ("auto", "disk"),
                                          ("raw_u16", "memory"), ("raw_u16", "disk"))]
            lines.append("| " + " | ".join(row) + " |")
        lines += ["", "Replay buffers at head=0 (bytes; independent of source):", "",
                  "| Policy | Device output | Device input | Pinned staging | CPU read-ahead payload upper bound | Manifest |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for codec in ("auto", "raw_u16"):
            mode = data["modes"][f"memory:{codec}:head0"]
            s = mode["samples"][0]
            vals = [s[key] for key in ("output_buffer_bytes", "device_input_buffer_bytes",
                                      "pinned_buffer_bytes", "cpu_read_ahead_payload_upper_bound_bytes")]
            vals.append(mode["capacity"]["manifest_bytes"])
            lines.append("| " + codec + " | " + " | ".join(f"{v:,}" for v in vals) + " |")
        lines.append("")
    lines += ["## Measurement scope and accounting", "",
              "- Timing starts after synchronizing prior compute/preparation work and ends "
              "after both streams complete. It includes head QK/top-k/sort, tail read/H2D/decode, "
              "every consumer, per-pass startup, and completion. Encoding, initial capture/D2H, "
              "store construction/validation, replay-buffer allocation, and JIT warmup are outside samples.",
              "- DRAM means encoded packets read from MemoryStore into pinned staging. Warm file "
              "means DiskStore reads after construction, validation, and warmup; OS page cache is "
              "not flushed. No cold SSD latency, artificial read delay, or disk bandwidth is claimed.",
              "- Only retained tail chunks are encoded. Raw payload is 2 × rows × k × tail_chunks. "
              "Auto independently chooses raw_u16 or union_bitmap for each retained chunk. "
              "Manifest bytes are separate; even head=16 has a small manifest. Resident replay "
              "buffers, CPU read-ahead, Python objects, Q/K/V and recomputation intermediates are "
              "additional costs. The buffer limit covers pinned staging plus device input/output, "
              "not the entire request's memory.",
              "- All modes and validation references coexist in the harness. Its process-wide "
              "peak allocation cannot be interpreted as a per-mode or per-request GPU requirement. "
              "Payload savings do not imply equal reductions in GPU peak memory.",
              "- Chunk candidate counts grow from an existing prefix with at least k candidates. "
              "Rows within a chunk share one candidate count; there is no per-row causal mask. "
              "Recomputed chunk fraction therefore differs from QK score-work fraction.",
              "- The split is static and chunk-aligned: floor(total_chunks × requested fraction). "
              "The caller supplies ordering, dependencies and validated split/cache identity. "
              "This experiment does not implement adaptive switching, eviction, production prefix "
              "matching, or model integration.",
              "", "## Reproduction", "",
              "Run `python benchmarks/plot_split.py` from the repository root with matplotlib "
              "available. The script reads only the two checked-in JSON files and regenerates "
              "this report and the figure; it performs no GPU work. `--report-only` requires "
              "only the Python standard library.", ""]
    return "\n".join(lines)


def plot(small, large, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 12, "savefig.dpi": 220})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    heads = (0, 1, 2, 4, 8, 16)
    colors = {"auto": "#176D9C", "raw_u16": "#C26825"}
    maximum = 0
    for ax, data, title in zip(axes, (small, large),
                               ("Small: 8K candidates, k=128", "Long: 64K candidates, k=512")):
        for codec in ("auto", "raw_u16"):
            for source in ("memory", "disk"):
                modes = [data["modes"][f"{source}:{codec}:head{h}"] for h in heads]
                y = [m["wall_ms_median"] for m in modes]
                error = [[m["wall_ms_median"] - m["wall_ms_min"] for m in modes],
                         [m["wall_ms_max"] - m["wall_ms_median"] for m in modes]]
                maximum = max(maximum, max(m["wall_ms_max"] for m in modes))
                ax.errorbar([h/16 for h in heads], y, yerr=error,
                            color=colors[codec], linestyle="-" if source == "memory" else "--",
                            marker="o" if source == "memory" else "s", markersize=4,
                            linewidth=1.5, capsize=2, elinewidth=.7, alpha=.9)
        ax.set_title(title, pad=55, loc="left")
        ax.set_xlabel("Recomputed head fraction")
        ax.set_xlim(-.025, 1.025)
        ax.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        ax.grid(axis="y", alpha=.18)
        top = ax.secondary_xaxis("top")
        top.set_xticks([0, .25, .5, 1],
                      [f"{data['modes'][f'memory:auto:head{h}']['capacity']['payload_bytes']/2**20:.3f}"
                       for h in (0, 4, 8, 16)])
        top.set_xlabel("Retained auto payload (MiB)", labelpad=7)
    axes[0].set_ylabel("Completed pass wall time (ms)")
    axes[0].set_ylim(0, maximum * 1.10)
    legend = [Line2D([0], [0], color=colors[c], linestyle=ls, marker=marker,
                     markersize=4, label=f"{label} / {source}")
              for c, label in (("auto", "Auto"), ("raw_u16", "Raw u16"))
              for source, ls, marker in (("DRAM", "-", "o"), ("Warm file", "--", "s"))]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .045))
    fig.text(.5, .015, "Synthetic QK + top-k + selected-value gather; 7 samples per mode. "
             "Median with min-max whiskers. Not model TTFT.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.69, bottom=.24, left=.065, right=.98, wspace=.13)
    fig.savefig(output, facecolor="white")
    plt.close(fig)


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--small", type=Path, default=directory / "split_small_a100.json")
    parser.add_argument("--long", type=Path, default=directory / "split_long_a100.json")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    small, large = load(args.small), load(args.long)
    (directory / "split_results.md").write_text(report(small, large), encoding="utf-8")
    if not args.report_only:
        plot(small, large, directory / "split_tradeoff.png")


if __name__ == "__main__":
    main()
