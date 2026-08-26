"""Turn labelled measurements into tuned thresholds.

``frame.THRESHOLDS`` currently holds cut points fitted to synthetic cards.
Real frames will land elsewhere. This module closes that loop: label a batch
CSV, fit cut points to the labels, and report how well they separate.

Fitting is deliberately simple. The tiers are *ordered* (common < uncommon <
rare < holo) and the feature is one-dimensional, so the best split for each
adjacent pair can be found exactly by scanning candidate cut points -- no
optimiser, no training run, and the result is a number you can read and
argue with.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path

from .models import FrameTier

# Ordered by how ornate the border is, which is what the 1-D fit splits on.
TIER_ORDER = [FrameTier.NORMAL, FrameTier.OTHER, FrameTier.E]


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key) or default)
    except (TypeError, ValueError):
        return default


def best_split(low_vals: list[float], high_vals: list[float]) -> tuple[float, float]:
    """Cut point separating two ordered groups, plus the accuracy it achieves.

    Scans every midpoint between observed values and keeps the one that
    misclassifies fewest samples. Exact for 1-D, which is all we need.
    """
    if not low_vals or not high_vals:
        return (0.0, 0.0)
    vals = sorted(set(low_vals + high_vals))
    cands = [(a + b) / 2 for a, b in zip(vals, vals[1:])] or [vals[0]]
    total = len(low_vals) + len(high_vals)
    best_t, best_correct = cands[0], -1
    for t in cands:
        correct = sum(1 for v in low_vals if v < t) + sum(1 for v in high_vals if v >= t)
        if correct > best_correct:
            best_t, best_correct = t, correct
    return (round(best_t, 4), best_correct / total)


def fit_thresholds(rows: list[dict], feature: str = "ornateness") -> dict:
    """Fit cut points for each adjacent tier pair from labelled rows."""
    by_tier: dict[str, list[float]] = {t.value: [] for t in TIER_ORDER}
    for r in rows:
        lab = (r.get("true_frame") or "").strip().lower()
        if lab in by_tier:
            by_tier[lab].append(_f(r, feature))

    labelled = sum(len(v) for v in by_tier.values())
    out = {"feature": feature, "labelled": labelled, "counts": {k: len(v) for k, v in by_tier.items()},
           "thresholds": {}, "splits": []}
    if labelled < 4:
        return out

    prev = 0.0
    for lo, hi in zip(TIER_ORDER, TIER_ORDER[1:]):
        lo_v, hi_v = by_tier[lo.value], by_tier[hi.value]
        t, acc = best_split(lo_v, hi_v)
        if not lo_v or not hi_v:
            out["splits"].append({"boundary": f"{lo.value}|{hi.value}", "threshold": None,
                                  "accuracy": None, "note": "no labelled samples on one side"})
            continue
        t = max(t, prev)                      # keep cut points monotonic
        prev = t
        out["thresholds"][hi.value] = t
        out["splits"].append({"boundary": f"{lo.value}|{hi.value}", "threshold": t,
                              "accuracy": round(acc, 3), "n": len(lo_v) + len(hi_v)})

    out["overall_accuracy"] = round(_apply_accuracy(by_tier, out["thresholds"]), 3)
    return out


def _apply_accuracy(by_tier: dict[str, list[float]], thresholds: dict[str, float]) -> float:
    correct = total = 0
    for tier_name, vals in by_tier.items():
        for v in vals:
            total += 1
            if _classify(v, thresholds) == tier_name:
                correct += 1
    return correct / total if total else 0.0


def _classify(v: float, thresholds: dict[str, float]) -> str:
    got = FrameTier.NORMAL.value
    for tier in TIER_ORDER[1:]:
        t = thresholds.get(tier.value)
        if t is not None and v >= t:
            got = tier.value
    return got


def ocr_report(rows: list[dict]) -> dict:
    """Accuracy of the badge reader against ``true_print`` labels.

    Reports the error that actually costs something separately: a numbered
    card written off as an E, or an E promoted to a number.
    """
    checked = wrong = 0
    conf_ok: list[float] = []
    conf_bad: list[float] = []
    misses: list[dict] = []
    e_as_number = number_as_e = 0

    for r in rows:
        truth = (r.get("true_print") or "").strip().upper()
        if not truth:
            continue
        got = "E" if str(r.get("no_number")) == "1" else (r.get("print_no") or "").strip()
        checked += 1
        conf = _f(r, "ocr_conf")
        if got == truth:
            conf_ok.append(conf)
            continue
        wrong += 1
        conf_bad.append(conf)
        if truth == "E" and got not in ("E", ""):
            e_as_number += 1
        if truth != "E" and got == "E":
            number_as_e += 1
        if len(misses) < 25:
            misses.append({"image": r.get("image"), "slot": r.get("slot"),
                           "want": truth, "got": got or "?", "conf": conf,
                           "raw": (r.get("ocr_text") or "")[:24]})

    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    return {
        "checked": checked,
        "correct": checked - wrong,
        "accuracy": round((checked - wrong) / checked, 3) if checked else None,
        "number_read_as_E": number_as_e,
        "E_read_as_number": e_as_number,
        "mean_conf_correct": mean(conf_ok),
        "mean_conf_wrong": mean(conf_bad),
        "misses": misses,
    }


# --- labelling sheet ------------------------------------------------------

_SHEET_CSS = """
body{font:14px system-ui,sans-serif;background:#1e1f22;color:#e7e7ea;margin:0;padding:20px}
h1{font-size:18px;margin:0 0 4px}p.sub{color:#9a9aa2;margin:0 0 18px}
.bar{position:sticky;top:0;background:#1e1f22;padding:12px 0;border-bottom:1px solid #3a3b40;z-index:5}
button{background:#5865f2;color:#fff;border:0;border-radius:6px;padding:9px 16px;font-size:14px;cursor:pointer}
button:hover{background:#4752c4}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px;margin-top:16px}
.card{background:#2b2d31;border:1px solid #3a3b40;border-radius:8px;padding:10px}
.card img{width:100%;border-radius:5px;display:block;background:#000}
.meta{font-size:11px;color:#9a9aa2;margin:7px 0 4px;line-height:1.5;word-break:break-all}
.flag{color:#f0b232;font-weight:600}
label{display:block;font-size:11px;color:#9a9aa2;margin-top:6px}
select,input{width:100%;box-sizing:border-box;background:#1e1f22;color:#e7e7ea;
  border:1px solid #4a4b52;border-radius:5px;padding:5px;font-size:13px;margin-top:2px}
"""

_SHEET_JS = """
function esc(v){v=(v==null?'':String(v));return /[",\\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v;}
function save(){
  const hdr=JSON.parse(document.getElementById('hdr').textContent);
  const out=[hdr.join(',')];
  document.querySelectorAll('.card').forEach(c=>{
    const row=JSON.parse(c.dataset.row);
    row.true_print=c.querySelector('.tp').value.trim().toUpperCase();
    row.true_frame=c.querySelector('.tf').value;
    out.push(hdr.map(h=>esc(row[h])).join(','));
  });
  const blob=new Blob([out.join('\\n')],{type:'text/csv'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download='labels.csv';a.click();
}
function fillAll(){
  const v=document.getElementById('bulk').value; if(!v) return;
  document.querySelectorAll('.tf').forEach(s=>s.value=v);
}
"""


def build_sheet(rows: list[dict], out_html: str | Path, fields: list[str]) -> int:
    """Write a local page for labelling crops and exporting labels.csv.

    Typing labels into a spreadsheet for a few hundred crops is where a
    calibration effort usually dies, so the crop, its measurements and the
    two label controls sit together and the page exports the CSV itself.
    """
    out_html = Path(out_html)
    cards = []
    tiers = "".join(f'<option value="{t.value}">{t.value}</option>' for t in TIER_ORDER)
    for r in rows:
        img = f"{Path(r['image']).stem}__slot{r['slot']}.png"
        got = "E" if str(r.get("no_number")) == "1" else (r.get("print_no") or "?")
        conf = _f(r, "ocr_conf")
        flag = f'<span class="flag">{html.escape(r.get("flags") or "")}</span>' if r.get("flags") else ""
        import json as _json
        cards.append(f"""
<div class="card" data-row='{html.escape(_json.dumps(r), quote=True)}'>
  <img src="{html.escape(img)}" loading="lazy" alt="">
  <div class="meta">{html.escape(r['image'])} · slot {r['slot']}<br>
    read <b>{html.escape(str(got))}</b> ({conf:.0%}) · {html.escape(r.get('frame',''))}
    · idx {_f(r,'ornateness'):.3f}<br>{flag}</div>
  <label>true print (number or E)<input class="tp" value="{html.escape(str(got))}"></label>
  <label>true frame<select class="tf">{tiers}</select></label>
</div>""")
        cards[-1] = cards[-1].replace(
            f'<option value="{r.get("frame","")}">', f'<option value="{r.get("frame","")}" selected>', 1)

    import json as _json
    doc = f"""<!doctype html><meta charset="utf-8"><title>gacha_vision labelling</title>
<style>{_SHEET_CSS}</style>
<h1>Label these crops</h1>
<p class="sub">Prefilled with what the pipeline read. Fix what is wrong, then export —
feed labels.csv to <code>python -m gacha_vision fit labels.csv</code>.</p>
<div class="bar">
  <button onclick="save()">Download labels.csv</button>
  &nbsp; set every frame to
  <select id="bulk" onchange="fillAll()" style="width:auto;display:inline-block">
    <option value="">—</option>{tiers}</select>
  &nbsp; <span style="color:#9a9aa2">{len(rows)} crops</span>
</div>
<div class="grid">{''.join(cards)}</div>
<script id="hdr" type="application/json">{_json.dumps(fields)}</script>
<script>{_SHEET_JS}</script>"""
    out_html.write_text(doc, encoding="utf-8")
    return len(rows)
