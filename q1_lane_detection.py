"""
Q1: Lane Centre Detection — "The road disappears"
SeDriCa Perception Assignment 2026-27

Detects lane centre at two rows (near: v_n=260, far: v_f=170) for all 72 frames
across three sequences: clear, shadow, missing.

Author: Shubham Kumar
Dependencies: numpy, opencv-python, matplotlib
Run: python3 q1_lane_detection.py
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'city', 'lane')
REF_CSV  = os.path.join(os.path.dirname(__file__), '..', 'city', 'lane_reference.csv')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'outputs', 'q1')
os.makedirs(OUT_DIR, exist_ok=True)

SEQUENCES = ['clear', 'shadow', 'missing']

# Rows of interest (image coords: v increases downward)
V_NEAR = 260   # near row
V_FAR  = 170   # far row

# ROI: only look at columns 60..420, rows 155..275
COL_MIN, COL_MAX = 60, 420
ROW_MIN, ROW_MAX = 155, 275

# Assumed lane half-width in pixels at near row (calibrated visually ~90px)
LANE_HALF = 90

# Confidence thresholds
MIN_PEAK_STRENGTH = 15   # min pixel count in a 5-px window to trust a boundary


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_frame(seq: str, frame_idx: int) -> np.ndarray:
    path = os.path.join(DATA_DIR, seq, f'{frame_idx:03d}.png')
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img


def extract_edge_map(bgr: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale, apply Gaussian blur, then Canny.
    We also layer in an S-channel threshold (HLS) to catch white/yellow markings
    even under shadow.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 90)

    # S-channel helps separate markings from shadow
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)
    l_ch = hls[:, :, 1]
    # White-pixel mask (bright areas in L channel)
    white_mask = (l_ch > 180).astype(np.uint8) * 255
    # Combine
    combined = cv2.bitwise_or(edges, white_mask)
    return combined


def row_profile(edge_map: np.ndarray, row: int) -> np.ndarray:
    """Return the edge/brightness profile along a single horizontal row."""
    return edge_map[row, COL_MIN:COL_MAX].astype(float)


def find_boundaries(profile: np.ndarray, min_strength: int = MIN_PEAK_STRENGTH):
    """
    Find left and right lane boundaries in a 1-D profile.
    Strategy: look for the leftmost strong peak (left boundary) and rightmost
    strong peak (right boundary) in the valid column range.
    Returns (left_col, right_col, confidence) where cols are in image coords.
    confidence ∈ [0, 1].
    """
    # Smooth profile with a small box
    kernel = np.ones(7) / 7
    smooth = np.convolve(profile, kernel, mode='same')

    n = len(smooth)
    # Search left half for left boundary
    left_half = smooth[:n//2]
    right_half = smooth[n//2:]

    def peak_col(arr, offset=0):
        if arr.max() < min_strength:
            return None, 0.0
        idx = int(np.argmax(arr))
        strength = float(arr[idx])
        conf = min(strength / 255.0, 1.0)
        return idx + offset, conf

    left_idx, left_conf  = peak_col(left_half, 0)
    right_idx, right_conf = peak_col(right_half, n//2)

    # Convert back to image column
    left_col  = (COL_MIN + left_idx)  if left_idx  is not None else None
    right_col = (COL_MIN + right_idx) if right_idx is not None else None

    return left_col, right_col, left_conf, right_conf


def centre_from_boundaries(left_col, right_col, left_conf, right_conf,
                            assumed_half=LANE_HALF):
    """
    Compute lane centre and overall confidence from boundary detections.
    Falls back to single-boundary + assumed half-width when only one is found.
    """
    if left_col is not None and right_col is not None:
        centre = (left_col + right_col) / 2.0
        conf = (left_conf + right_conf) / 2.0
        return centre, conf
    elif left_col is not None:
        centre = left_col + assumed_half
        return centre, left_conf * 0.6   # single-boundary penalty
    elif right_col is not None:
        centre = right_col - assumed_half
        return centre, right_conf * 0.6
    else:
        return None, 0.0


# ── Main detector ─────────────────────────────────────────────────────────────

def detect_lane_centre(bgr: np.ndarray,
                       prev_near=None, prev_far=None):
    """
    Returns (u_near, u_far, conf_near, conf_far).
    u_near / u_far are None if estimate is 'unknown'.
    prev_near / prev_far: previous-frame estimates for temporal smoothing.
    """
    edge = extract_edge_map(bgr)

    # --- Near row ---
    prof_n = row_profile(edge, V_NEAR)
    ln, rn, lcn, rcn = find_boundaries(prof_n)
    u_near, c_near = centre_from_boundaries(ln, rn, lcn, rcn)

    # --- Far row ---
    prof_f = row_profile(edge, V_FAR)
    lf, rf, lcf, rcf = find_boundaries(prof_f)
    u_far, c_far = centre_from_boundaries(lf, rf, lcf, rcf)

    # Temporal smoothing: if current frame confidence is low, blend with prev
    ALPHA = 0.4   # weight of current frame
    if prev_near is not None and u_near is not None and c_near < 0.5:
        u_near = ALPHA * u_near + (1 - ALPHA) * prev_near
        c_near = c_near * 0.8  # still lower confidence
    if prev_far is not None and u_far is not None and c_far < 0.5:
        u_far = ALPHA * u_far + (1 - ALPHA) * prev_far
        c_far = c_far * 0.8

    # Sanity check: near and far centres should be close (straight road)
    if u_near is not None and u_far is not None:
        if abs(u_near - u_far) > 60:
            # Inconsistency → lower confidence of both
            c_near *= 0.5
            c_far  *= 0.5

    # Mark as unknown below threshold
    UNKNOWN_THRESH = 0.15
    if c_near < UNKNOWN_THRESH:
        u_near = None
    if c_far < UNKNOWN_THRESH:
        u_far  = None

    return u_near, u_far, c_near, c_far


# ── Run all sequences ─────────────────────────────────────────────────────────

def run_sequence(seq: str):
    results = []
    prev_near, prev_far = None, None
    for frame_idx in range(24):
        bgr = load_frame(seq, frame_idx)
        u_n, u_f, c_n, c_f = detect_lane_centre(bgr, prev_near, prev_far)
        results.append({
            'frame': frame_idx,
            'u_near': u_n, 'u_far': u_f,
            'conf_near': c_n, 'conf_far': c_f
        })
        if u_n is not None:
            prev_near = u_n
        if u_f is not None:
            prev_far = u_f
    return results


# ── Evaluation against reference ──────────────────────────────────────────────

def load_reference():
    ref = {}
    with open(REF_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq   = row['sequence']
            frame = int(row['frame'])
            ref[(seq, frame)] = {
                'near': float(row['near_center_x_px']),
                'far':  float(row['far_center_x_px'])
            }
    return ref


def evaluate(seq_results: dict, ref: dict):
    errors = {}
    for seq, results in seq_results.items():
        ne_list, fe_list = [], []
        detected_n, detected_f = 0, 0
        for r in results:
            gt = ref.get((seq, r['frame']))
            if gt is None:
                continue
            if r['u_near'] is not None:
                ne_list.append(abs(r['u_near'] - gt['near']))
                detected_n += 1
            if r['u_far'] is not None:
                fe_list.append(abs(r['u_far'] - gt['far']))
                detected_f += 1
        errors[seq] = {
            'near_MAE': np.mean(ne_list) if ne_list else float('nan'),
            'far_MAE':  np.mean(fe_list) if fe_list else float('nan'),
            'near_detected': detected_n,
            'far_detected':  detected_f,
            'total': 24
        }
    return errors


# ── Visualisation helpers ──────────────────────────────────────────────────────

def visualise_stages(seq: str, frame_idx: int, tag: str):
    """
    4-panel figure: original | edge map | ROI edge | annotated result
    """
    bgr = load_frame(seq, frame_idx)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    edge = extract_edge_map(bgr)

    # ROI crop
    roi_edge = edge[ROW_MIN:ROW_MAX, COL_MIN:COL_MAX]

    u_n, u_f, c_n, c_f = detect_lane_centre(bgr)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'Q1 | seq={seq} frame={frame_idx:02d} ({tag})', fontsize=11)

    axes[0].imshow(rgb)
    axes[0].axhline(V_NEAR, color='cyan',  lw=1.2, ls='--', label=f'v_near={V_NEAR}')
    axes[0].axhline(V_FAR,  color='yellow', lw=1.2, ls='--', label=f'v_far={V_FAR}')
    axes[0].set_title('Original')
    axes[0].legend(fontsize=7)

    axes[1].imshow(edge, cmap='gray')
    axes[1].set_title('Edge map')

    axes[2].imshow(roi_edge, cmap='gray')
    axes[2].set_title(f'ROI edge\n(cols {COL_MIN}–{COL_MAX}, rows {ROW_MIN}–{ROW_MAX})')

    axes[3].imshow(rgb)
    # Draw detections
    if u_n is not None:
        axes[3].plot(u_n, V_NEAR, 'co', ms=8, label=f'u_n={u_n:.1f} (c={c_n:.2f})')
    else:
        axes[3].text(240, V_NEAR-5, 'u_near: UNKNOWN', color='cyan', fontsize=8)
    if u_f is not None:
        axes[3].plot(u_f, V_FAR, 'yo', ms=8, label=f'u_f={u_f:.1f} (c={c_f:.2f})')
    else:
        axes[3].text(240, V_FAR-5, 'u_far: UNKNOWN', color='yellow', fontsize=8)
    axes[3].axhline(V_NEAR, color='cyan',  lw=0.8, ls='--')
    axes[3].axhline(V_FAR,  color='yellow', lw=0.8, ls='--')
    axes[3].set_title('Lane centre estimate')
    axes[3].legend(fontsize=7)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f'stages_{seq}_{frame_idx:02d}_{tag}.png')
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def plot_sequence_results(seq: str, results: list, ref: dict):
    """Plot u_near and u_far vs frame, with reference overlay."""
    frames = [r['frame'] for r in results]
    u_n    = [r['u_near'] for r in results]
    u_f    = [r['u_far']  for r in results]
    c_n    = [r['conf_near'] for r in results]
    c_f    = [r['conf_far']  for r in results]

    ref_n = [ref.get((seq, f), {}).get('near') for f in frames]
    ref_f = [ref.get((seq, f), {}).get('far')  for f in frames]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Q1 — Sequence: {seq}', fontsize=13)

    # Near row
    ax = axes[0]
    ax.plot(frames, ref_n, 'k--', lw=1.5, label='Reference u_near')
    det_frames = [f for f, u in zip(frames, u_n) if u is not None]
    det_vals   = [u for u in u_n if u is not None]
    ax.plot(det_frames, det_vals, 'co-', ms=5, lw=1.2, label='Predicted u_near')
    unk_frames = [f for f, u in zip(frames, u_n) if u is None]
    for xf in unk_frames:
        ax.axvline(xf, color='red', alpha=0.3)
    ax.set_ylabel('u_near (px)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Far row
    ax = axes[1]
    ax.plot(frames, ref_f, 'k--', lw=1.5, label='Reference u_far')
    det_frames2 = [f for f, u in zip(frames, u_f) if u is not None]
    det_vals2   = [u for u in u_f if u is not None]
    ax.plot(det_frames2, det_vals2, 'yo-', ms=5, lw=1.2, label='Predicted u_far')
    unk_frames2 = [f for f, u in zip(frames, u_f) if u is None]
    for xf in unk_frames2:
        ax.axvline(xf, color='red', alpha=0.3)
    ax.set_ylabel('u_far (px)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Confidence
    ax = axes[2]
    ax.fill_between(frames, c_n, alpha=0.5, color='cyan',   label='conf_near')
    ax.fill_between(frames, c_f, alpha=0.5, color='yellow', label='conf_far')
    ax.axhline(0.15, color='red', ls='--', lw=0.8, label='Unknown threshold')
    ax.set_ylabel('Confidence')
    ax.set_xlabel('Frame')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f'sequence_{seq}.png')
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def plot_comparison_panel(seq_clear, seq_shadow, seq_missing, ref):
    """2-row × 3-col comparison: clear | shadow | missing for one chosen frame."""
    chosen = 12   # a frame where shadow/missing is clearly different
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Q1 — Stage comparison: clear vs shadow vs missing (frame 12)',
                 fontsize=12)

    for col_idx, (seq, results) in enumerate([('clear', seq_clear),
                                               ('shadow', seq_shadow),
                                               ('missing', seq_missing)]):
        bgr = load_frame(seq, chosen)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        edge = extract_edge_map(bgr)

        r = results[chosen]
        u_n, u_f, c_n, c_f = r['u_near'], r['u_far'], r['conf_near'], r['conf_far']

        # Row 0: original + detections
        ax = axes[0, col_idx]
        ax.imshow(rgb)
        if u_n is not None:
            ax.plot(u_n, V_NEAR, 'co', ms=9)
        if u_f is not None:
            ax.plot(u_f, V_FAR, 'yo', ms=9)
        ax.axhline(V_NEAR, color='cyan',   lw=0.8, ls='--')
        ax.axhline(V_FAR,  color='yellow', lw=0.8, ls='--')
        ref_n = ref.get((seq, chosen), {}).get('near')
        ref_f = ref.get((seq, chosen), {}).get('far')
        if ref_n:
            ax.plot(ref_n, V_NEAR, 'r+', ms=10, mew=2)
        if ref_f:
            ax.plot(ref_f, V_FAR, 'r+', ms=10, mew=2)
        u_n_str = f'{u_n:.1f}' if u_n is not None else '?'
        u_f_str = f'{u_f:.1f}' if u_f is not None else '?'
        ax.set_title(f'{seq}\nu_n={u_n_str} c={c_n:.2f}\nu_f={u_f_str} c={c_f:.2f}',
                     fontsize=8)
        ax.axis('off')

        # Row 1: edge map
        ax2 = axes[1, col_idx]
        ax2.imshow(edge, cmap='gray')
        ax2.axhline(V_NEAR, color='cyan',   lw=0.8, ls='--')
        ax2.axhline(V_FAR,  color='yellow', lw=0.8, ls='--')
        ax2.set_title('Edge map', fontsize=8)
        ax2.axis('off')

    legend_elements = [
        mpatches.Patch(color='cyan',   label='Predicted (near row)'),
        mpatches.Patch(color='yellow', label='Predicted (far row)'),
        mpatches.Patch(color='red',    label='Reference (red cross)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fname = os.path.join(OUT_DIR, 'comparison_panel.png')
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved {fname}')


def save_predictions_csv(seq_results: dict):
    fname = os.path.join(OUT_DIR, 'predictions.csv')
    with open(fname, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sequence', 'frame', 'u_near', 'u_far', 'conf_near', 'conf_far'])
        for seq, results in seq_results.items():
            for r in results:
                w.writerow([
                    seq, r['frame'],
                    f"{r['u_near']:.2f}" if r['u_near'] is not None else 'unknown',
                    f"{r['u_far']:.2f}"  if r['u_far']  is not None else 'unknown',
                    f"{r['conf_near']:.3f}",
                    f"{r['conf_far']:.3f}"
                ])
    print(f'  Saved {fname}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print('Loading reference …')
    ref = load_reference()

    print('\nRunning detector on all sequences …')
    seq_results = {}
    for seq in SEQUENCES:
        print(f'  → {seq}')
        seq_results[seq] = run_sequence(seq)

    print('\nEvaluating …')
    errors = evaluate(seq_results, ref)
    for seq, e in errors.items():
        print(f'  {seq:10s}  near MAE={e["near_MAE"]:6.2f}px ({e["near_detected"]}/24 detected)'
              f'  far MAE={e["far_MAE"]:6.2f}px ({e["far_detected"]}/24 detected)')

    print('\nGenerating per-sequence timeline plots …')
    for seq in SEQUENCES:
        plot_sequence_results(seq, seq_results[seq], ref)

    print('\nGenerating stage panels …')
    # Clear frame 5 (good), shadow frame 12 (hard), missing frame 14 (hard)
    visualise_stages('clear',   5,  'easy')
    visualise_stages('shadow',  12, 'shadow_hard')
    visualise_stages('missing', 14, 'missing_hard')

    print('\nGenerating 3-way comparison panel …')
    plot_comparison_panel(seq_results['clear'],
                          seq_results['shadow'],
                          seq_results['missing'], ref)

    print('\nSaving predictions CSV …')
    save_predictions_csv(seq_results)

    # Print summary table
    print('\n=== Summary Table ===')
    print(f'{"Sequence":<12} {"Near MAE":>10} {"Far MAE":>10} {"Det N":>8} {"Det F":>8}')
    for seq, e in errors.items():
        print(f'{seq:<12} {e["near_MAE"]:>10.2f} {e["far_MAE"]:>10.2f}'
              f' {e["near_detected"]:>8} {e["far_detected"]:>8}')


if __name__ == '__main__':
    main()
