"""
Q2: Crossing with Conflicting Evidence — GO / SLOW / STOP decision system
SeDriCa Perception Assignment 2026-27

Reads city/crossing_evidence.csv and the three 24-frame image sequences.
Outputs per-frame decisions with reasons, timelines, and stopping-distance checks.

Author: Shubham Kumar
Run: python3 q2_crossing_decision.py
"""

import os
import csv
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.join(os.path.dirname(__file__), '..')
EV_CSV    = os.path.join(BASE_DIR, 'city', 'crossing_evidence.csv')
REF_CSV   = os.path.join(BASE_DIR, 'city', 'crossing_reference.csv')
IMG_DIR   = os.path.join(BASE_DIR, 'city', 'crossing')
OUT_DIR   = os.path.join(os.path.dirname(__file__), 'outputs', 'q2')
os.makedirs(OUT_DIR, exist_ok=True)

EPISODES = ['clear_green', 'late_red', 'person_crossing']

# ── Decision thresholds ───────────────────────────────────────────────────────
# These are the tunable parameters of the baseline rule.
RED_THRESH     = 0.45   # red_score above this → red-light evidence
PERSON_THRESH  = 0.65   # person_score above this → person-in-crossing evidence
STALE_AGE_S    = 0.4    # V2X message older than this is considered stale (seconds)

# ── Physics ───────────────────────────────────────────────────────────────────
TAU   = 0.15   # total processing + actuation delay (s)
A_DEC = 0.8    # comfortable deceleration (m/s²)


def stopping_distance(v_mps: float) -> float:
    return v_mps * TAU + v_mps**2 / (2 * A_DEC)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_evidence():
    data = {ep: [] for ep in EPISODES}
    with open(EV_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = row['episode']
            data[ep].append({
                'frame':         int(row['frame']),
                'time_s':        float(row['time_s']),
                'dist_m':        float(row['distance_to_line_m']),
                'speed_mps':     float(row['speed_mps']),
                'red_score':     float(row['red_score']),
                'person_score':  float(row['person_score']),
                'v2x_light':     row['v2x_light'],
                'v2x_time_s':    float(row['v2x_sample_time_s']),
            })
    return data


def load_reference():
    ref = {ep: [] for ep in EPISODES}
    with open(REF_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = row['episode']
            ref[ep].append({
                'frame':       int(row['frame']),
                'true_light':  row['true_light'],
                'person':      int(row['person_on_crossing']),
                'stop_req':    int(row['stop_required']),
            })
    return ref


# ── Decision functions ────────────────────────────────────────────────────────

def decide_baseline(row: dict) -> tuple[str, str]:
    """
    Baseline: use raw scores + V2X (no staleness check).
    Returns (decision, reason).
    """
    red  = row['red_score']   > RED_THRESH
    pers = row['person_score'] > PERSON_THRESH
    v2x  = row['v2x_light']

    if red or pers:
        if row['dist_m'] < stopping_distance(row['speed_mps']):
            return 'STOP', f'red_score={row["red_score"]:.2f} or person_score={row["person_score"]:.2f}'
        return 'STOP', f'red_score={row["red_score"]:.2f} or person_score={row["person_score"]:.2f}'
    if v2x == 'red':
        return 'STOP', 'V2X=red'
    if v2x == 'green' and not red and not pers:
        return 'GO', 'V2X=green, scores low'
    return 'SLOW', 'Uncertain evidence'


def decide_revised(row: dict) -> tuple[str, str]:
    """
    Revised rule: accounts for V2X staleness and score-message disagreement.
    When stale or disagreement → SLOW until resolved.
    """
    red   = row['red_score']   > RED_THRESH
    pers  = row['person_score'] > PERSON_THRESH
    v2x   = row['v2x_light']
    age   = row['time_s'] - row['v2x_time_s']
    stale = age > STALE_AGE_S

    # Hard-stop conditions (high confidence)
    if red and pers:
        return 'STOP', f'Both red_score={row["red_score"]:.2f} and person on crossing'
    if red:
        if stale and v2x == 'green':
            return 'SLOW', f'red_score high but V2X stale ({age:.2f}s old), disagree → SLOW'
        return 'STOP', f'red_score={row["red_score"]:.2f}'
    if pers:
        return 'STOP', f'person_score={row["person_score"]:.2f} (person in path)'

    # V2X check with staleness
    if stale:
        return 'SLOW', f'V2X msg stale ({age:.2f}s); relying on vision only → caution'
    if v2x == 'red':
        return 'STOP', 'V2X=red (fresh)'
    if v2x == 'green':
        return 'GO', 'V2X=green (fresh), vision scores low'

    return 'SLOW', 'Uncertain'


# ── Run decisions ─────────────────────────────────────────────────────────────

def run_decisions(evidence: dict, decision_fn) -> dict:
    results = {}
    for ep, rows in evidence.items():
        results[ep] = [
            {'frame': r['frame'],
             'time_s': r['time_s'],
             **{k: r[k] for k in ['dist_m','speed_mps','red_score','person_score','v2x_light','v2x_time_s']},
             'decision': decision_fn(r)[0],
             'reason':   decision_fn(r)[1]}
            for r in rows
        ]
    return results


# ── Stopping-distance check ───────────────────────────────────────────────────

def check_stopping(decisions: dict, label='baseline'):
    print(f'\n--- Stopping-distance check ({label}) ---')
    for ep, rows in decisions.items():
        first_stop = next((r for r in rows if r['decision'] == 'STOP'), None)
        if first_stop is None:
            print(f'  {ep}: No STOP issued')
            continue
        d_stop = stopping_distance(first_stop['speed_mps'])
        d_avail = first_stop['dist_m']
        ok = d_avail >= d_stop
        print(f'  {ep}: First STOP at frame {first_stop["frame"]} '
              f'(t={first_stop["time_s"]:.1f}s) | '
              f'd_stop={d_stop:.2f}m, d_avail={d_avail:.2f}m → '
              f'{"✓ safe" if ok else "✗ TOO LATE"}')


# ── Timeline plot ─────────────────────────────────────────────────────────────

DECISION_Y = {'GO': 0, 'SLOW': 1, 'STOP': 2}
COLORS     = {'GO': 'green', 'SLOW': 'orange', 'STOP': 'red'}


def plot_timeline(baseline: dict, revised: dict, evidence: dict, ref: dict):
    fig, axes = plt.subplots(len(EPISODES), 1, figsize=(14, 9), sharex=False)
    fig.suptitle('Q2 — Decision timelines: Baseline vs Revised', fontsize=12)

    for ax, ep in zip(axes, EPISODES):
        rows  = evidence[ep]
        b_dec = [r['decision'] for r in baseline[ep]]
        r_dec = [r['decision'] for r in revised[ep]]
        ref_stop = [r['stop_req'] for r in ref[ep]]
        times = [r['time_s'] for r in rows]

        # Plot baseline as filled blocks
        for i, (t, d) in enumerate(zip(times, b_dec)):
            ax.barh(2.2, 0.09, left=t, color=COLORS[d], alpha=0.8, height=0.4)

        # Plot revised as filled blocks offset down
        for i, (t, d) in enumerate(zip(times, r_dec)):
            ax.barh(1.6, 0.09, left=t, color=COLORS[d], alpha=0.8, height=0.4)

        # Reference STOP regions
        for i, (t, s) in enumerate(zip(times, ref_stop)):
            if s:
                ax.axvspan(t, t + 0.1, color='red', alpha=0.15)

        # Scores (secondary axis)
        ax2 = ax.twinx()
        rs = [r['red_score'] for r in rows]
        ps = [r['person_score'] for r in rows]
        ax2.plot(times, rs, 'm-', lw=1.2, label='red_score')
        ax2.plot(times, ps, 'b-', lw=1.2, label='person_score')
        ax2.set_ylim(0, 1.3)
        ax2.set_ylabel('Scores', fontsize=7)
        ax2.legend(fontsize=7, loc='upper right')

        ax.set_yticks([1.6, 2.2])
        ax.set_yticklabels(['Revised', 'Baseline'], fontsize=8)
        ax.set_title(ep, fontsize=9)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_xlim(-0.05, 2.4)

    legend_els = [mpatches.Patch(color=c, label=d) for d, c in COLORS.items()]
    legend_els.append(mpatches.Patch(color='red', alpha=0.2, label='Ref STOP region'))
    fig.legend(handles=legend_els, loc='lower center', ncol=4, fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fname = os.path.join(OUT_DIR, 'decision_timelines.png')
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


# ── Monte Carlo uncertainty test ──────────────────────────────────────────────

def mc_test(evidence: dict, ref: dict, n_trials=30, seed=42):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    results_base  = {'missed': {ep: 0 for ep in EPISODES},
                     'false':  {ep: 0 for ep in EPISODES},
                     'delay':  {ep: [] for ep in EPISODES}}
    results_rev   = {'missed': {ep: 0 for ep in EPISODES},
                     'false':  {ep: 0 for ep in EPISODES},
                     'delay':  {ep: [] for ep in EPISODES}}

    for trial in range(n_trials):
        noise_std = 0.05
        for ep, rows in evidence.items():
            ref_rows = ref[ep]
            perturbed = []
            for i, r in enumerate(rows):
                # Drop frame with 10% probability
                if rng.random() < 0.10:
                    perturbed.append(None)
                    continue
                pr = dict(r)
                pr['red_score']    = float(np.clip(r['red_score']    + np_rng.normal(0, noise_std), 0, 1))
                pr['person_score'] = float(np.clip(r['person_score'] + np_rng.normal(0, noise_std), 0, 1))
                perturbed.append(pr)

            def score(dec_fn, res_dict):
                first_required = next((i for i, rr in enumerate(ref_rows) if rr['stop_req']), None)
                first_issued   = None
                for i, pr in enumerate(perturbed):
                    if pr is None:
                        continue
                    d, _ = dec_fn(pr)
                    ref_r = ref_rows[i]
                    if d == 'STOP' and first_issued is None:
                        first_issued = i
                    if d == 'STOP' and not ref_r['stop_req']:
                        res_dict['false'][ep] += 1
                    if ref_r['stop_req'] and d != 'STOP':
                        res_dict['missed'][ep] += 1
                if first_required is not None and first_issued is not None:
                    delay = rows[first_issued]['time_s'] - rows[first_required]['time_s']
                    res_dict['delay'][ep].append(delay)

            score(decide_baseline, results_base)
            score(decide_revised,  results_rev)

    print('\n=== Monte Carlo Uncertainty Test (seed=42, n=30 trials) ===')
    print(f'{"Episode":<20} {"Base missed":>12} {"Base false":>10} '
          f'{"Rev missed":>10} {"Rev false":>10} {"Avg delay (rev)":>15}')
    for ep in EPISODES:
        avg_d = np.mean(results_rev['delay'][ep]) if results_rev['delay'][ep] else float('nan')
        print(f'{ep:<20} {results_base["missed"][ep]:>12} {results_base["false"][ep]:>10} '
              f'{results_rev["missed"][ep]:>10} {results_rev["false"][ep]:>10} {avg_d:>15.3f}')


# ── Save decision CSV ──────────────────────────────────────────────────────────

def save_decision_csv(decisions: dict, label: str):
    fname = os.path.join(OUT_DIR, f'decisions_{label}.csv')
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'frame', 'time_s', 'decision', 'reason'])
        for ep, rows in decisions.items():
            for r in rows:
                w.writerow([ep, r['frame'], r['time_s'], r['decision'], r['reason']])
    print(f'  Saved {fname}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading data …')
    evidence = load_evidence()
    ref      = load_reference()

    print('\nRunning baseline decisions …')
    baseline = run_decisions(evidence, decide_baseline)
    print('\nRunning revised decisions …')
    revised  = run_decisions(evidence, decide_revised)

    print('\nPrinting a few sample decisions:')
    for ep in EPISODES:
        print(f'\n  {ep}:')
        for r in baseline[ep][:6]:
            print(f'    frame {r["frame"]:2d} | {r["decision"]:4s} | {r["reason"]}')

    check_stopping(baseline, 'baseline')
    check_stopping(revised,  'revised')

    print('\nPlotting timelines …')
    plot_timeline(baseline, revised, evidence, ref)

    save_decision_csv(baseline, 'baseline')
    save_decision_csv(revised,  'revised')

    mc_test(evidence, ref)

    print('\nDone. Outputs in:', OUT_DIR)


if __name__ == '__main__':
    main()
