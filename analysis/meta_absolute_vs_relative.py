#!/usr/bin/env python3
"""Absolute versus relative estimation error (supporting m1).

Reads the main training parquet and writes
results/compare/absolute_vs_relative_error.md, the table cited in the
manuscript's discussion of relative-error amplification.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
RAW = os.path.join(BASE, "results", "raw", "meta_training_data.parquet")
OUT = os.path.join(BASE, "results", "compare", "absolute_vs_relative_error.md")


def stats(df: pd.DataFrame) -> dict:
    return {
        "n": len(df),
        "mean_abs": df["abs_error_pp"].mean(),
        "median_abs": df["abs_error_pp"].median(),
        "mean_rel": df["abs_bias_pct"].mean(),
        "median_rel": df["abs_bias_pct"].median(),
        "corr": df["abs_error_pp"].corr(df["abs_bias_pct"]),
    }


def main() -> None:
    df = pd.read_parquet(RAW)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    lines = [
        "# Absolute vs relative estimation error",
        "",
        "| group | n | mean abs error (pp) | median abs error (pp) | "
        "mean rel error (%) | median rel error (%) | corr(abs, rel) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = [("ALL", stats(df))]
    for model in sorted(df["model_type"].unique()):
        rows.append((model, stats(df[df["model_type"] == model])))
    for name, s in rows:
        lines.append(
            f"| {name} | {s['n']} | {s['mean_abs']:.4f} | {s['median_abs']:.4f} | "
            f"{s['mean_rel']:.1f} | {s['median_rel']:.1f} | {s['corr']:.3f} |"
        )

    lines += [
        "",
        "## By true-ACE quintile",
        "",
        "| true ACE quintile | mean true ACE | mean abs error (pp) | mean rel error (%) |",
        "|---|---:|---:|---:|",
    ]
    df = df.copy()
    df["ace_quintile"] = pd.qcut(df["true_ace"], 5, duplicates="drop")
    for interval, group in df.groupby("ace_quintile", observed=True):
        lines.append(
            f"| {interval} | {group['true_ace'].mean():.4f} | "
            f"{group['abs_error_pp'].mean():.4f} | {group['abs_bias_pct'].mean():.1f} |"
        )

    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"[OK] Saved {OUT}")
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
