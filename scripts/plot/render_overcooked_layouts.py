"""Render the five original Overcooked layouts side by side (paper Fig 3/14).

    python scripts/plot/render_overcooked_layouts.py --out figures/oc_original_5_layouts
"""
import argparse
from pathlib import Path

import jax
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from cecp.envs.overcooked import OvercookedEnv
from cecp.eval.display import OC_ORIGINAL_LAYOUT_NAMES
from cecp.eval.plotting import save_fig


def crop_frame(frame: np.ndarray) -> np.ndarray:
    """Crop the grey margin (the top-left pixel's colour) around a rendered frame."""
    mask = np.any(frame != frame[0, 0], axis=2)
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    return frame[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='figures/oc_original_5_layouts', help='output path prefix')
    parser.add_argument('--formats', nargs='+', default=['pdf', 'png'])
    args = parser.parse_args()

    rng = jax.random.key(0)
    frames = []
    for name in OC_ORIGINAL_LAYOUT_NAMES:
        env = OvercookedEnv(layout=name, train_layout_mode='fixed', heldout_layout_num_samples=0)
        rng, rng_reset = jax.random.split(rng)
        state, _ = env.reset(rng_reset, env.default_const)
        frames.append(crop_frame(np.asarray(env.create_frame_renderer()(state, agent_view_size=5))))

    widths = [f.shape[1] for f in frames]
    fig = plt.figure(figsize=(sum(w / widths[0] * 3 for w in widths), 3))
    grid = gridspec.GridSpec(1, len(frames), width_ratios=widths, wspace=0.05)
    for i, (frame, name) in enumerate(zip(frames, OC_ORIGINAL_LAYOUT_NAMES)):
        ax = fig.add_subplot(grid[i])
        ax.imshow(frame)
        ax.set_title(OC_ORIGINAL_LAYOUT_NAMES[name], fontsize=11)
        ax.axis('off')
    save_fig(fig, Path(args.out), formats=args.formats)


if __name__ == '__main__':
    main()
