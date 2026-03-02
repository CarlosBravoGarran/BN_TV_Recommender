#!/usr/bin/env python3
"""
Professional LLM Evaluation Suite for BN TV Recommender
Tests intent classification and attribute extraction with real metrics.

Metrics produced:
  - Per-class Precision / Recall / F1
  - Macro & Weighted averages
  - Confusion matrix
  - Per-field accuracy for attribute extraction
  - Hallucination detection (invented non-null values)
  - Latency (mean, p50, p95) and token usage / estimated cost
"""

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, "main")

from LLM_agent import classify_intent, extract_attributes_llm, colorize

# ============================================================================
# CONFIGURATION
# ============================================================================

# Rough estimate: gpt-4o pricing (input / output per 1M tokens, Feb 2026)
COST_PER_1K_INPUT_TOKENS  = 0.0025   # $2.50 / 1M
COST_PER_1K_OUTPUT_TOKENS = 0.010    # $10.00 / 1M

VALID_INTENTS = {"RECOMMEND", "ALTERNATIVE", "FEEDBACK_POS", "FEEDBACK_NEG", "SMALLTALK", "OTHER"}
VALID_ATTRIBUTES = {
    "UserAge":        {"young", "adult", "senior"},
    "UserGender":     {"male", "female"},
    "HouseholdType":  {"single", "couple", "family"},
    "TimeOfDay":      {"morning", "afternoon", "night"},
    "DayType":        {"weekday", "weekend"},
    "ProgramType":    {"movie", "series", "news", "documentary", "entertainment"},
    "ProgramGenre":   {"comedy", "drama", "horror", "romance", "news", "documentary",
                       "entertainment", "action", "thriller", "sci-fi", "fantasy"},
    "ProgramDuration":{"short", "medium", "long"},
}

# ============================================================================
# TEST DATASETS
# ============================================================================

# Each tuple: (message, expected_intent, difficulty, notes)
INTENT_TEST_CASES = [
    # ── ALTERNATIVE: genre rejections ───────────────────────────────────────
    ("No me gusta el drama",                "ALTERNATIVE", "easy",   "genre rejection"),
    ("Nada de terror",                      "ALTERNATIVE", "easy",   "genre rejection"),
    ("No quiero ver comedias",              "ALTERNATIVE", "easy",   "genre rejection"),
    ("Algo que no sea romántico",           "ALTERNATIVE", "easy",   "genre rejection"),
    ("Prefiero otro género",                "ALTERNATIVE", "medium", "vague genre alternative"),
    ("Sin drama por favor",                 "ALTERNATIVE", "easy",   "genre rejection"),
    ("Nada de ciencia ficción",             "ALTERNATIVE", "easy",   "genre rejection"),
    # ── ALTERNATIVE: simple alternative requests ────────────────────────────
    ("Dame otra opción",                    "ALTERNATIVE", "easy",   "simple alternative"),
    ("Otra cosa",                           "ALTERNATIVE", "easy",   "simple alternative"),
    ("Algo diferente",                      "ALTERNATIVE", "easy",   "simple alternative"),
    ("¿Hay algo más?",                      "ALTERNATIVE", "medium", "implicit alternative"),
    # ── ALTERNATIVE: ambiguous / edge cases ─────────────────────────────────
    ("No es lo que busco",                  "ALTERNATIVE", "hard",   "implicit rejection, no genre/title"),
    ("Prefiero ver otra cosa",              "ALTERNATIVE", "medium", "alternative phrasing"),
    # ── FEEDBACK_NEG: specific content rejection ─────────────────────────────
    ("Esa no me gusta",                     "FEEDBACK_NEG", "easy",  "specific rejection"),
    ("Ya la vi",                            "FEEDBACK_NEG", "easy",  "already seen"),
    ("No me convence esa película",         "FEEDBACK_NEG", "easy",  "specific rejection"),
    ("Esa es muy larga",                    "FEEDBACK_NEG", "medium","attribute-based rejection"),
    ("No, esa no",                          "FEEDBACK_NEG", "medium","short negative"),
    ("La he visto ya",                      "FEEDBACK_NEG", "easy",  "already seen variant"),
    ("Esa película no me llama la atención","FEEDBACK_NEG", "medium","paraphrase"),
    # ── FEEDBACK_POS: acceptance ────────────────────────────────────────────
    ("Me gusta",                            "FEEDBACK_POS", "easy",  "simple positive"),
    ("Perfecto",                            "FEEDBACK_POS", "easy",  "simple positive"),
    ("La veo",                              "FEEDBACK_POS", "easy",  "commitment"),
    ("De acuerdo",                          "FEEDBACK_POS", "easy",  "agreement"),
    ("Buena idea",                          "FEEDBACK_POS", "medium","implicit positive"),
    ("Sí, me apetece",                      "FEEDBACK_POS", "medium","positive with desire"),
    ("Esa me parece bien",                  "FEEDBACK_POS", "medium","acceptance"),
    # ── RECOMMEND: new recommendation requests ───────────────────────────────
    ("Quiero ver una película",             "RECOMMEND", "easy",    "type request"),
    ("Qué puedo ver",                       "RECOMMEND", "easy",    "open recommendation"),
    ("Recomiéndame algo",                   "RECOMMEND", "easy",    "explicit request"),
    ("Quiero ver una comedia",              "RECOMMEND", "easy",    "genre + type"),
    ("Ponme algo de acción",                "RECOMMEND", "easy",    "genre request"),
    ("¿Qué hay bueno hoy?",                 "RECOMMEND", "medium",  "temporal rec request"),
    ("Busco una serie entretenida",         "RECOMMEND", "medium",  "series request"),
    # ── SMALLTALK: chitchat ─────────────────────────────────────────────────
    ("Hola",                                "SMALLTALK", "easy",    "greeting"),
    ("Gracias",                             "SMALLTALK", "easy",    "thanks"),
    ("Adiós",                               "SMALLTALK", "easy",    "farewell"),
    ("¿Cómo estás?",                        "SMALLTALK", "medium",  "social question"),
    ("Muy bien",                            "SMALLTALK", "hard",    "ambiguous positive"),
    # ── HARD / AMBIGUOUS edge cases ─────────────────────────────────────────
    ("No quiero ver nada de eso",           "ALTERNATIVE", "hard",  "vague broad rejection"),
    ("Me aburre un poco",                   "FEEDBACK_NEG", "hard", "implicit mild rejection"),
    ("Quiero algo más corto",               "ALTERNATIVE", "hard",  "duration alternative"),
]

# Each tuple: (message, expected_attrs_dict)
# Only non-null fields are checked; others must be null (hallucination guard).
EXTRACTION_TEST_CASES = [
    (
        "Quiero ver una comedia de 90 minutos",
        {"ProgramType": "movie", "ProgramGenre": "comedy", "ProgramDuration": "long"},
    ),
    (
        "Soy un chico de 32 años",
        {"UserAge": "young", "UserGender": "male"},
    ),
    (
        "Somos una familia y queremos ver una serie esta tarde",
        {"HouseholdType": "family", "ProgramType": "series", "TimeOfDay": "afternoon"},
    ),
    (
        "Quiero ver algo de terror por la noche",
        {"ProgramGenre": "horror", "TimeOfDay": "night"},
    ),
    (
        "Soy una señora mayor de 70 años y vivo sola",
        {"UserAge": "senior", "UserGender": "female", "HouseholdType": "single"},
    ),
    (
        "Un documental corto para el fin de semana",
        {"ProgramType": "documentary", "ProgramDuration": "short", "DayType": "weekend"},
    ),
    (
        "Algo de acción, somos una pareja",
        {"ProgramGenre": "action", "HouseholdType": "couple"},
    ),
    (
        "Quiero ver las noticias",
        {"ProgramType": "news", "ProgramGenre": "news"},
    ),
    (
        "Una película de ciencia ficción larga",
        {"ProgramType": "movie", "ProgramGenre": "sci-fi", "ProgramDuration": "long"},
    ),
    (
        "Hola, ¿qué hay?",
        {},  # No attributes — everything must be null (hallucination test)
    ),
    (
        "Algo entretenido para esta noche entre semana",
        {"ProgramGenre": "entertainment", "TimeOfDay": "night", "DayType": "weekday"},
    ),
    (
        "Tengo 45 años y vivo en pareja",
        {"UserAge": "adult", "HouseholdType": "couple"},
    ),
]


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class IntentResult:
    message: str
    expected: str
    predicted: str
    correct: bool
    difficulty: str
    notes: str
    latency_ms: float
    tokens_in: int
    tokens_out: int


@dataclass
class ExtractionResult:
    message: str
    expected: dict
    predicted: dict
    field_hits: int           # correctly predicted non-null fields
    field_misses: int         # expected non-null fields predicted wrong / null
    hallucinations: int       # fields predicted non-null when expected null
    invalid_values: int       # values outside the allowed set
    latency_ms: float
    tokens_in: int
    tokens_out: int


# ============================================================================
# METRIC HELPERS
# ============================================================================

def confusion_matrix(results: list[IntentResult], classes: list[str]) -> dict:
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    mat = [[0] * n for _ in range(n)]
    for r in results:
        if r.expected in idx and r.predicted in idx:
            mat[idx[r.expected]][idx[r.predicted]] += 1
        elif r.expected in idx:
            mat[idx[r.expected]][idx.get("OTHER", 0)] += 1
    return {"classes": classes, "matrix": mat}


def per_class_metrics(results: list[IntentResult], classes: list[str]) -> dict:
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for r in results:
        if r.predicted == r.expected:
            tp[r.expected] += 1
        else:
            fp[r.predicted] += 1
            fn[r.expected] += 1

    metrics = {}
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        support = tp[c] + fn[c]
        metrics[c] = {"precision": p, "recall": r, "f1": f1, "support": support}
    return metrics


def macro_avg(per_class: dict) -> dict:
    keys = list(per_class.keys())
    p = sum(per_class[k]["precision"] for k in keys) / len(keys)
    r = sum(per_class[k]["recall"] for k in keys) / len(keys)
    f1 = sum(per_class[k]["f1"] for k in keys) / len(keys)
    return {"precision": p, "recall": r, "f1": f1}


def weighted_avg(per_class: dict) -> dict:
    total = sum(per_class[k]["support"] for k in per_class)
    if total == 0:
        return {"precision": 0, "recall": 0, "f1": 0}
    p = sum(per_class[k]["precision"] * per_class[k]["support"] for k in per_class) / total
    r = sum(per_class[k]["recall"] * per_class[k]["support"] for k in per_class) / total
    f1 = sum(per_class[k]["f1"] * per_class[k]["support"] for k in per_class) / total
    return {"precision": p, "recall": r, "f1": f1}


def latency_percentiles(values: list[float]) -> dict:
    s = sorted(values)
    n = len(s)
    def pct(p):
        idx = int(p / 100 * (n - 1))
        return s[idx] if n > 0 else 0.0
    return {
        "mean_ms": sum(s) / n if n else 0,
        "p50_ms":  pct(50),
        "p95_ms":  pct(95),
        "max_ms":  s[-1] if s else 0,
    }


def estimate_cost(total_in: int, total_out: int) -> float:
    return (total_in / 1000 * COST_PER_1K_INPUT_TOKENS +
            total_out / 1000 * COST_PER_1K_OUTPUT_TOKENS)


# ============================================================================
# TEST RUNNERS
# ============================================================================

def run_intent_tests(verbose: bool = True) -> tuple[list[IntentResult], dict]:
    """Run intent classification tests and return results + summary metrics."""
    print("\n" + "=" * 80)
    print(" INTENT CLASSIFICATION EVALUATION")
    print("=" * 80)

    results: list[IntentResult] = []

    for message, expected, difficulty, notes in INTENT_TEST_CASES:
        t0 = time.perf_counter()
        # We patch classify_intent to also capture token usage
        try:
            # Import client directly to capture usage
            import LLM_agent as _agent
            resp = _agent.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": _agent.INTENT_PROMPT},
                    {"role": "user", "content": message},
                ],
                temperature=0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            content = _agent.clean_json_response(resp.choices[0].message.content)
            try:
                predicted = json.loads(content).get("intent", "OTHER")
            except json.JSONDecodeError:
                predicted = "OTHER"
            tokens_in  = resp.usage.prompt_tokens
            tokens_out = resp.usage.completion_tokens
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            predicted = "OTHER"
            tokens_in, tokens_out = 0, 0
            print(f"  ERROR calling API: {e}")

        correct = predicted == expected
        r = IntentResult(
            message=message, expected=expected, predicted=predicted,
            correct=correct, difficulty=difficulty, notes=notes,
            latency_ms=latency_ms, tokens_in=tokens_in, tokens_out=tokens_out,
        )
        results.append(r)

        if verbose:
            status  = colorize("PASS", "\033[92m") if correct else colorize("FAIL", "\033[91m")
            exp_col = colorize(f"{expected:<15}", "\033[93m")
            got_col = colorize(f"{predicted:<15}", "\033[92m" if correct else "\033[91m")
            diff    = colorize(f"[{difficulty}]", "\033[90m")
            print(f"  {status} {diff} {message:<42} | exp {exp_col} | got {got_col} | {latency_ms:6.0f}ms")

    # ── Per-class metrics ──────────────────────────────────────────────────
    classes = sorted(VALID_INTENTS)
    per_class = per_class_metrics(results, classes)
    macro   = macro_avg(per_class)
    weighted = weighted_avg(per_class)
    correct  = sum(1 for r in results if r.correct)
    accuracy = correct / len(results) if results else 0.0
    lat      = latency_percentiles([r.latency_ms for r in results])
    total_in  = sum(r.tokens_in for r in results)
    total_out = sum(r.tokens_out for r in results)
    cost      = estimate_cost(total_in, total_out)

    # ── Per-difficulty breakdown ────────────────────────────────────────────
    diff_groups: dict[str, list] = defaultdict(list)
    for r in results:
        diff_groups[r.difficulty].append(r.correct)
    diff_acc = {d: sum(v)/len(v) for d, v in diff_groups.items()}

    summary = {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(results),
        "per_class": per_class,
        "macro_avg": macro,
        "weighted_avg": weighted,
        "by_difficulty": diff_acc,
        "latency": lat,
        "tokens": {"input": total_in, "output": total_out},
        "estimated_cost_usd": cost,
        "confusion_matrix": confusion_matrix(results, classes),
    }
    return results, summary


def run_extraction_tests(verbose: bool = True) -> tuple[list[ExtractionResult], dict]:
    """Run attribute extraction tests and return results + summary metrics."""
    print("\n" + "=" * 80)
    print(" ATTRIBUTE EXTRACTION EVALUATION")
    print("=" * 80)

    results: list[ExtractionResult] = []

    for message, expected_attrs in EXTRACTION_TEST_CASES:
        t0 = time.perf_counter()
        try:
            import LLM_agent as _agent
            resp = _agent.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": _agent.EXTRACTION_PROMPT},
                    {"role": "user", "content": message},
                ],
                temperature=0,
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            content = _agent.clean_json_response(resp.choices[0].message.content)
            try:
                predicted_raw: dict = json.loads(content)
            except json.JSONDecodeError:
                predicted_raw = {}
            tokens_in  = resp.usage.prompt_tokens
            tokens_out = resp.usage.completion_tokens
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            predicted_raw = {}
            tokens_in, tokens_out = 0, 0
            print(f"  ERROR calling API: {e}")

        # Normalise: treat empty string / "null" string as None
        predicted: dict = {}
        for k, v in predicted_raw.items():
            predicted[k] = None if v in (None, "", "null", "NULL") else v

        # All BN attribute keys expected in output
        all_keys = list(VALID_ATTRIBUTES.keys())

        hits = misses = hallucinations = invalid_values = 0

        field_details: list[str] = []
        for key in all_keys:
            exp_val  = expected_attrs.get(key, None)   # None means "must be null"
            pred_val = predicted.get(key, None)

            expected_null = exp_val is None

            if expected_null:
                if pred_val is not None:
                    hallucinations += 1
                    field_details.append(f"    HALLUCINATION {key}: expected null, got '{pred_val}'")
            else:
                allowed = VALID_ATTRIBUTES.get(key, set())
                if pred_val == exp_val:
                    hits += 1
                    field_details.append(f"    OK  {key}: '{pred_val}'")
                elif pred_val is None:
                    misses += 1
                    field_details.append(f"    MISS {key}: expected '{exp_val}', got null")
                elif pred_val not in allowed:
                    invalid_values += 1
                    misses += 1
                    field_details.append(f"    INVALID {key}: expected '{exp_val}', got '{pred_val}' (not in allowed set)")
                else:
                    misses += 1
                    field_details.append(f"    WRONG {key}: expected '{exp_val}', got '{pred_val}'")

        r = ExtractionResult(
            message=message, expected=expected_attrs, predicted=predicted,
            field_hits=hits, field_misses=misses,
            hallucinations=hallucinations, invalid_values=invalid_values,
            latency_ms=latency_ms, tokens_in=tokens_in, tokens_out=tokens_out,
        )
        results.append(r)

        if verbose:
            expected_count = len(expected_attrs)
            status = colorize("PASS", "\033[92m") if (misses == 0 and hallucinations == 0) else colorize("FAIL", "\033[91m")
            print(f"\n  {status} \"{message}\"")
            print(f"         hits={hits}/{expected_count}  misses={misses}  hallucinations={hallucinations}  "
                  f"invalid={invalid_values}  {latency_ms:.0f}ms")
            for detail in field_details:
                if any(tag in detail for tag in ("MISS", "HALLUCINATION", "WRONG", "INVALID")):
                    print(colorize(detail, "\033[91m"))
                else:
                    print(colorize(detail, "\033[90m"))

    # ── Aggregate metrics ──────────────────────────────────────────────────
    total_expected = sum(len(r.expected) for r in results)
    total_hits     = sum(r.field_hits for r in results)
    total_hallucs  = sum(r.hallucinations for r in results)
    total_invalid  = sum(r.invalid_values for r in results)
    perfect_cases  = sum(1 for r in results if r.field_misses == 0 and r.hallucinations == 0)

    lat       = latency_percentiles([r.latency_ms for r in results])
    total_in  = sum(r.tokens_in for r in results)
    total_out = sum(r.tokens_out for r in results)
    cost      = estimate_cost(total_in, total_out)

    # Per-field hit rate
    field_hits_map: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        for key, exp_val in r.expected.items():
            pred_val = r.predicted.get(key)
            field_hits_map[key].append(pred_val == exp_val)

    per_field_accuracy = {
        k: sum(v) / len(v) for k, v in field_hits_map.items()
    }

    summary = {
        "perfect_cases": perfect_cases,
        "total_cases": len(results),
        "perfect_rate": perfect_cases / len(results) if results else 0,
        "field_hit_rate": total_hits / total_expected if total_expected else 0,
        "total_hallucinations": total_hallucs,
        "total_invalid_values": total_invalid,
        "per_field_accuracy": per_field_accuracy,
        "latency": lat,
        "tokens": {"input": total_in, "output": total_out},
        "estimated_cost_usd": cost,
    }
    return results, summary


# ============================================================================
# REPORT PRINTERS
# ============================================================================

def print_intent_report(summary: dict) -> None:
    print("\n" + "=" * 80)
    print(" INTENT CLASSIFICATION — REPORT")
    print("=" * 80)
    acc = summary["accuracy"]
    color = "\033[92m" if acc >= 0.9 else "\033[93m" if acc >= 0.75 else "\033[91m"
    print(f"\n  Accuracy: {colorize(f'{acc:.1%}', color)}  "
          f"({summary['correct']}/{summary['total']} correct)\n")

    # Per-class table
    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>9} {'F1':>8} {'Support':>9}")
    print(f"  {'-'*15} {'-'*10} {'-'*9} {'-'*8} {'-'*9}")
    for cls, m in sorted(summary["per_class"].items()):
        p_col = f"{m['precision']:.2f}"
        r_col = f"{m['recall']:.2f}"
        f_col = f"{m['f1']:.2f}"
        row_color = "\033[92m" if m["f1"] >= 0.8 else "\033[93m" if m["f1"] >= 0.5 else "\033[91m"
        print(colorize(f"  {cls:<15} {p_col:>10} {r_col:>9} {f_col:>8} {m['support']:>9}", row_color))
    macro    = summary["macro_avg"]
    weighted = summary["weighted_avg"]
    print(f"  {'-'*15} {'-'*10} {'-'*9} {'-'*8} {'-'*9}")
    print(f"  {'macro avg':<15} {macro['precision']:>10.2f} {macro['recall']:>9.2f} {macro['f1']:>8.2f}")
    print(f"  {'weighted avg':<15} {weighted['precision']:>10.2f} {weighted['recall']:>9.2f} {weighted['f1']:>8.2f}")

    # By difficulty
    print("\n  Accuracy by difficulty:")
    for d, a in sorted(summary["by_difficulty"].items()):
        bar_color = "\033[92m" if a >= 0.9 else "\033[93m" if a >= 0.7 else "\033[91m"
        bar = "█" * int(a * 20) + "░" * (20 - int(a * 20))
        print(f"    {d:<8} {colorize(bar, bar_color)} {a:.1%}")

    # Confusion matrix
    cm = summary["confusion_matrix"]
    classes = cm["classes"]
    mat = cm["matrix"]
    print("\n  Confusion matrix (rows=expected, cols=predicted):")
    header = "".join(f"{c[:5]:>7}" for c in classes)
    print(f"  {'':>13}{header}")
    for i, cls in enumerate(classes):
        row = "".join(
            colorize(f"{mat[i][j]:>7}", "\033[92m") if i == j and mat[i][j] > 0
            else colorize(f"{mat[i][j]:>7}", "\033[91m") if i != j and mat[i][j] > 0
            else f"{mat[i][j]:>7}"
            for j in range(len(classes))
        )
        print(f"  {cls:<13}{row}")

    # Latency & cost
    lat = summary["latency"]
    tok = summary["tokens"]
    print(f"\n  Latency — mean:{lat['mean_ms']:.0f}ms  p50:{lat['p50_ms']:.0f}ms  p95:{lat['p95_ms']:.0f}ms  max:{lat['max_ms']:.0f}ms")
    print(f"  Tokens  — in:{tok['input']}  out:{tok['output']}  total:{tok['input']+tok['output']}")
    print(f"  Est. cost — ${summary['estimated_cost_usd']:.4f} USD")


def print_extraction_report(summary: dict) -> None:
    print("\n" + "=" * 80)
    print(" ATTRIBUTE EXTRACTION — REPORT")
    print("=" * 80)
    rate = summary["perfect_rate"]
    hit  = summary["field_hit_rate"]
    color = "\033[92m" if rate >= 0.8 else "\033[93m" if rate >= 0.6 else "\033[91m"
    print(f"\n  Perfect cases:   {colorize(f'{rate:.1%}', color)}  ({summary['perfect_cases']}/{summary['total_cases']})")
    print(f"  Field hit rate:  {hit:.1%}")
    print(f"  Hallucinations:  {summary['total_hallucinations']}  (non-null values invented when null expected)")
    print(f"  Invalid values:  {summary['total_invalid_values']}  (values outside allowed set)")

    print("\n  Per-field accuracy (on cases where field was expected):")
    for f_name, acc in sorted(summary["per_field_accuracy"].items(), key=lambda x: -x[1]):
        bar_color = "\033[92m" if acc >= 0.9 else "\033[93m" if acc >= 0.7 else "\033[91m"
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"    {f_name:<18} {colorize(bar, bar_color)} {acc:.1%}")

    lat = summary["latency"]
    tok = summary["tokens"]
    print(f"\n  Latency — mean:{lat['mean_ms']:.0f}ms  p50:{lat['p50_ms']:.0f}ms  p95:{lat['p95_ms']:.0f}ms  max:{lat['max_ms']:.0f}ms")
    print(f"  Tokens  — in:{tok['input']}  out:{tok['output']}  total:{tok['input']+tok['output']}")
    print(f"  Est. cost — ${summary['estimated_cost_usd']:.4f} USD")


def print_executive_summary(intent_summary: dict, extraction_summary: dict) -> None:
    print("\n" + "=" * 80)
    print(" EXECUTIVE SUMMARY")
    print("=" * 80)

    ia  = intent_summary["accuracy"]
    iw  = intent_summary["weighted_avg"]["f1"]
    er  = extraction_summary["perfect_rate"]
    eh  = extraction_summary["total_hallucinations"]
    total_cost = (intent_summary["estimated_cost_usd"] +
                  extraction_summary["estimated_cost_usd"])

    grade_i = "A" if ia >= 0.92 else "B" if ia >= 0.80 else "C" if ia >= 0.65 else "D"
    grade_e = "A" if er >= 0.85 else "B" if er >= 0.70 else "C" if er >= 0.55 else "D"
    grade_col = lambda g: "\033[92m" if g == "A" else "\033[93m" if g == "B" else "\033[91m"

    print(f"""
  ┌──────────────────────────────────────────────┐
  │  Intent Classification                        │
  │    Accuracy         {f"{ia:.1%}":<10}  Grade: {colorize(grade_i, grade_col(grade_i))}        │
  │    Weighted F1      {iw:.1%}                        │
  │                                              │
  │  Attribute Extraction                        │
  │    Perfect cases    {f"{er:.1%}":<10}  Grade: {colorize(grade_e, grade_col(grade_e))}        │
  │    Hallucinations   {eh:<10}                    │
  │                                              │
  │  Total estimated cost: ${total_cost:<8.4f} USD         │
  └──────────────────────────────────────────────┘""")

    # Actionable insights
    print("\n  Actionable insights:")
    worst_classes = sorted(
        intent_summary["per_class"].items(),
        key=lambda x: x[1]["f1"]
    )[:2]
    for cls, m in worst_classes:
        if m["support"] > 0 and m["f1"] < 0.85:
            print(f"    ⚠  Intent '{cls}' has low F1={m['f1']:.2f} — "
                  f"review INTENT_PROMPT examples for this class")

    if eh > 0:
        print(f"    ⚠  {eh} hallucination(s) in extraction — "
              f"strengthen null rules in EXTRACTION_PROMPT")

    if intent_summary["by_difficulty"].get("hard", 1.0) < 0.7:
        print("    ⚠  Hard cases underperform — consider adding few-shot examples for edge cases")

    if ia >= 0.92 and er >= 0.85:
        print("    ✓  Both prompts performing well. Consider running on a larger blind test set.")


# ============================================================================
# MAIN
# ============================================================================

def main(save_json: bool = False):
    print("\nBN TV Recommender — LLM Prompt Engineering Evaluation")
    print(f"Model: gpt-4o | Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    intent_results, intent_summary     = run_intent_tests(verbose=True)
    extraction_results, extr_summary   = run_extraction_tests(verbose=True)

    print_intent_report(intent_summary)
    print_extraction_report(extr_summary)
    print_executive_summary(intent_summary, extr_summary)

    if save_json:
        out = {
            "intent":    intent_summary,
            "extraction": extr_summary,
        }
        path = "llm_eval_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n  Full results saved to {path}")

    overall_ok = (
        intent_summary["accuracy"] >= 0.80
        and extr_summary["perfect_rate"] >= 0.60
        and extr_summary["total_hallucinations"] == 0
    )
    print()
    return overall_ok


if __name__ == "__main__":
    success = main(save_json=True)
    sys.exit(0 if success else 1)
