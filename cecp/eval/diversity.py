"""Population disagreement estimators: entropy gap (Delta H) and pairwise Jensen-Shannon divergence.

For a population {pi_1 .. pi_N} at a state s, with a uniform prior over members:

    pi_bar(.|s) = (1/N) sum_j pi_j(.|s)
    Delta H(s)  = H(pi_bar(.|s)) - (1/N) sum_j H(pi_j(.|s)) = I(member; action | s) >= 0

Delta H is the uniform-weight generalized JSD of the N action distributions, in nats, bounded by
log(num_actions). The pairwise JSD matrix d_ij = E_s[JSD(pi_i || pi_j)] is Delta H of the two-member
population {pi_i, pi_j}, bounded by log 2. Used by evaluate_population_diversity (Table 2),
evaluate_cell_action_distributions (Fig 5/13) and evaluate_partner_stage_mi (with partner stages in
place of members, Fig 18).
"""
from typing import Sequence

import numpy as np
from scipy.special import xlogy

NUM_SEATS = 2
JSD_UPPER_NATS = float(np.log(2.0))

TOL_DELTA_H_NONNEG = 1e-9
TOL_DELTA_H_UPPER = 1e-8
TOL_DELTA_H_VS_H_BAR = 1e-9
TOL_PROBS_NORMALIZED = 1e-4
TOL_FORM_AGREEMENT = 1e-8
TOL_JSD_NONNEG = 1e-9
TOL_JSD_UPPER = 1e-8


def entropy_gap_from_probs(probs: np.ndarray) -> dict[str, np.ndarray]:
    """Delta H from an [N, B, T, A] probability tensor (N members over B x T states).

    Returns {'delta_h', 'h_bar', 'h_mean'}, each [B, T] float64 in nats. Delta H is computed as
    mean_j KL(p_j || p_bar), which is >= 0 term by term and avoids cancellation when both entropies
    are tiny; it is checked against H(p_bar) - mean_j H(p_j).
    """
    p = np.asarray(probs, dtype=np.float64)
    if not np.allclose(p.sum(axis=-1), 1.0, atol=TOL_PROBS_NORMALIZED):
        raise AssertionError(f'Action probabilities are not normalized: max deviation {np.abs(p.sum(-1) - 1.0).max():.3e}')
    p = p / p.sum(axis=-1, keepdims=True)
    p_bar = p.mean(axis=0)
    h_mean = (-np.sum(xlogy(p, p), axis=-1)).mean(axis=0)
    h_bar = -np.sum(xlogy(p_bar, p_bar), axis=-1)
    delta_h = np.mean(np.sum(xlogy(p, p) - xlogy(p, p_bar[None]), axis=-1), axis=0)
    gap = float(np.max(np.abs(delta_h - (h_bar - h_mean))))
    if gap > TOL_FORM_AGREEMENT:
        raise AssertionError(f'mean-KL and entropy-difference forms of Delta H disagree by {gap:.3e}')
    return {'delta_h': delta_h, 'h_bar': h_bar, 'h_mean': h_mean}


def assert_entropy_gap_bounds(stats: dict[str, np.ndarray], num_actions: int, context: str) -> None:
    """0 <= Delta H <= H(pi_bar) <= log(num_actions)."""
    upper = float(np.log(num_actions))
    if float(stats['delta_h'].min()) < -TOL_DELTA_H_NONNEG:
        raise AssertionError(f'{context}: Delta H must be >= 0, got {float(stats["delta_h"].min()):.3e}')
    if float(stats['delta_h'].max()) > upper + TOL_DELTA_H_UPPER:
        raise AssertionError(f'{context}: Delta H must be <= log({num_actions}), got {float(stats["delta_h"].max()):.6f}')
    if float(stats['h_bar'].max()) > upper + TOL_DELTA_H_UPPER:
        raise AssertionError(f'{context}: H(pi_bar) must be <= log({num_actions}), got {float(stats["h_bar"].max()):.6f}')
    excess = float(np.max(stats['delta_h'] - stats['h_bar']))
    if excess > TOL_DELTA_H_VS_H_BAR:
        raise AssertionError(f'{context}: Delta H exceeds H(pi_bar) by {excess:.3e}')


def pairwise_jsd_from_probs(probs: np.ndarray) -> np.ndarray:
    """[N, N] state-averaged pairwise JSD (nats) from an [N, B, T, A] probability tensor: the divergence
    is taken per state and then averaged over the B x T states."""
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum(axis=-1, keepdims=True)
    n = p.shape[0]
    h_each = -np.sum(xlogy(p, p), axis=-1)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            m = 0.5 * (p[i] + p[j])
            jsd = 0.5 * (np.sum(xlogy(p[i], p[i]) - xlogy(p[i], m), axis=-1)
                         + np.sum(xlogy(p[j], p[j]) - xlogy(p[j], m), axis=-1))
            gap = float(np.max(np.abs(jsd - (-np.sum(xlogy(m, m), axis=-1) - 0.5 * (h_each[i] + h_each[j])))))
            if gap > TOL_FORM_AGREEMENT:
                raise AssertionError(f'mean-KL and entropy-difference forms of JSD disagree for ({i}, {j}) by {gap:.3e}')
            d[i, j] = d[j, i] = float(jsd.mean())
    return d


def assert_pairwise_jsd_bounds(jsd: np.ndarray, context: str) -> None:
    if float(np.nanmin(jsd)) < -TOL_JSD_NONNEG:
        raise AssertionError(f'{context}: pairwise JSD must be >= 0, got {float(np.nanmin(jsd)):.3e}')
    if float(np.nanmax(jsd)) > JSD_UPPER_NATS + TOL_JSD_UPPER:
        raise AssertionError(f'{context}: pairwise JSD must be <= log 2, got {float(np.nanmax(jsd)):.6f}')


class LayoutAccumulator:
    """Running sums for one layout, with a leading trajectory-source axis (pooled = sum over axis 0)."""

    def __init__(self, source_labels: Sequence[str], num_steps: int, agent_ids: Sequence[int], grid_size: int) -> None:
        self.source_labels = [str(s) for s in source_labels]
        self.agent_ids = [int(a) for a in agent_ids]
        n, t, g, k = len(self.source_labels), num_steps, grid_size, len(self.agent_ids)
        self.pair_jsd_sum = np.zeros((n, k, k))
        self.heat_dh_sum = np.zeros((n, g, g))
        self.heat_count = np.zeros((n, g, g), np.int64)
        self.heat_hbar_sum = np.zeros((n, g, g))
        self.t_dh_sum = np.zeros((n, t))
        self.t_dh_sqsum = np.zeros((n, t))
        self.t_count = np.zeros((n, t), np.int64)
        self.t_hbar_sum = np.zeros((n, t))
        self.seat_dh_sum = np.zeros((n, NUM_SEATS))
        self.seat_count = np.zeros((n, NUM_SEATS), np.int64)
        self.src_dh_sum = np.zeros(n)
        self.src_dh_sqsum = np.zeros(n)
        self.src_count = np.zeros(n, np.int64)
        self.src_hbar_sum = np.zeros(n)
        self.src_hmean_sum = np.zeros(n)
        self.src_delta_h_min = np.full(n, np.inf)
        self.src_delta_h_max = np.full(n, -np.inf)

    def add(self, src: int, stats: dict[str, np.ndarray], pos: np.ndarray, pair_jsd: np.ndarray) -> None:
        """Add one source segment. stats: [B, T] arrays; pos: [B, T, 2] (row, col), where B flattens
        (episode, seat) so seat = b % NUM_SEATS; pair_jsd: [N, N] averaged over this segment's states."""
        b, t = stats['delta_h'].shape
        delta_h, h_bar = stats['delta_h'], stats['h_bar']
        rows, cols = pos[..., 0], pos[..., 1]
        t_idx = np.broadcast_to(np.arange(t), (b, t))
        seat_idx = np.broadcast_to((np.arange(b) % NUM_SEATS)[:, None], (b, t))
        np.add.at(self.heat_dh_sum[src], (rows, cols), delta_h)
        np.add.at(self.heat_count[src], (rows, cols), 1)
        np.add.at(self.heat_hbar_sum[src], (rows, cols), h_bar)
        np.add.at(self.t_dh_sum[src], t_idx, delta_h)
        np.add.at(self.t_dh_sqsum[src], t_idx, delta_h ** 2)
        np.add.at(self.t_count[src], t_idx, 1)
        np.add.at(self.t_hbar_sum[src], t_idx, h_bar)
        np.add.at(self.seat_dh_sum[src], seat_idx, delta_h)
        np.add.at(self.seat_count[src], seat_idx, 1)
        self.pair_jsd_sum[src] += np.asarray(pair_jsd, dtype=np.float64) * float(delta_h.size)
        self.src_dh_sum[src] += float(delta_h.sum())
        self.src_dh_sqsum[src] += float((delta_h ** 2).sum())
        self.src_count[src] += int(delta_h.size)
        self.src_hbar_sum[src] += float(h_bar.sum())
        self.src_hmean_sum[src] += float(stats['h_mean'].sum())
        self.src_delta_h_min[src] = min(self.src_delta_h_min[src], float(delta_h.min()))
        self.src_delta_h_max[src] = max(self.src_delta_h_max[src], float(delta_h.max()))


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise num / den, NaN where den == 0."""
    out = np.full(np.shape(num), np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=(den > 0))
    return out


def mean_sem(sum_x: float, sum_x2: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return float('nan'), float('nan')
    mean = sum_x / n
    if n < 2:
        return float(mean), 0.0
    var = max(0.0, (sum_x2 - n * mean * mean) / (n - 1))
    return float(mean), float(np.sqrt(var / n))
