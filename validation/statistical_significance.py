"""Bootstrap confidence intervals and Ledoit-Wolf Sharpe ratio test."""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    """Result of bootstrap confidence interval estimation."""
    metric_name: str
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    n_bootstrap: int
    bootstrap_distribution: np.ndarray


@dataclass
class SharpeTestResult:
    """Result of Ledoit-Wolf Sharpe ratio difference test."""
    sharpe_a: float
    sharpe_b: float
    sharpe_diff: float
    se_diff: float
    t_stat: float
    p_value: float
    significant: bool
    alpha: float


def block_bootstrap_indices(n: int, block_size: int, n_bootstrap: int,
                             rng: np.random.Generator = None) -> np.ndarray:
    """Generate circular block bootstrap index arrays.

    Block bootstrap preserves serial correlation in daily returns.

    Args:
        n: Length of original series.
        block_size: Block length (typically 21 for monthly blocks).
        n_bootstrap: Number of bootstrap samples.
        rng: Random generator.

    Returns:
        (n_bootstrap, n) array of resampled indices.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_blocks = int(np.ceil(n / block_size))
    indices = np.empty((n_bootstrap, n), dtype=int)

    for b in range(n_bootstrap):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) % n for s in starts])
        indices[b] = idx[:n]

    return indices


def bootstrap_sharpe(daily_returns: pd.Series, n_bootstrap: int = 10000,
                      ci_level: float = 0.95, block_size: int = 21,
                      seed: int = 42) -> BootstrapResult:
    """Bootstrap confidence interval for annualized Sharpe ratio.

    Uses circular block bootstrap with 21-day blocks to preserve
    autocorrelation structure in daily returns.
    """
    s = daily_returns.dropna().values
    n = len(s)
    rng = np.random.default_rng(seed)

    # Point estimate
    point = (s.mean() * 252) / (s.std() * np.sqrt(252)) if s.std() > 0 else 0

    # Bootstrap
    indices = block_bootstrap_indices(n, block_size, n_bootstrap, rng)
    sharpes = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = s[indices[b]]
        mu = sample.mean() * 252
        vol = sample.std() * np.sqrt(252)
        sharpes[b] = mu / vol if vol > 0 else 0

    alpha = (1 - ci_level) / 2
    ci_lo = np.percentile(sharpes, alpha * 100)
    ci_hi = np.percentile(sharpes, (1 - alpha) * 100)

    return BootstrapResult(
        metric_name="Sharpe Ratio",
        point_estimate=point,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        bootstrap_distribution=sharpes,
    )


def bootstrap_terminal_wealth(daily_returns: pd.Series, n_bootstrap: int = 10000,
                               ci_level: float = 0.95, block_size: int = 21,
                               seed: int = 42) -> BootstrapResult:
    """Bootstrap confidence interval for terminal wealth ($1 invested)."""
    s = daily_returns.dropna().values
    n = len(s)
    rng = np.random.default_rng(seed)

    point = np.prod(1 + s)

    indices = block_bootstrap_indices(n, block_size, n_bootstrap, rng)
    terminals = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        terminals[b] = np.prod(1 + s[indices[b]])

    alpha = (1 - ci_level) / 2
    ci_lo = np.percentile(terminals, alpha * 100)
    ci_hi = np.percentile(terminals, (1 - alpha) * 100)

    return BootstrapResult(
        metric_name="Terminal Wealth",
        point_estimate=point,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        bootstrap_distribution=terminals,
    )


def bootstrap_max_drawdown(daily_returns: pd.Series, n_bootstrap: int = 10000,
                            ci_level: float = 0.95, block_size: int = 21,
                            seed: int = 42) -> BootstrapResult:
    """Bootstrap confidence interval for maximum drawdown."""
    s = daily_returns.dropna().values
    n = len(s)
    rng = np.random.default_rng(seed)

    cum = np.cumprod(1 + s)
    running_max = np.maximum.accumulate(cum)
    point = np.min((cum - running_max) / running_max)

    indices = block_bootstrap_indices(n, block_size, n_bootstrap, rng)
    dds = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = s[indices[b]]
        cum_b = np.cumprod(1 + sample)
        rm_b = np.maximum.accumulate(cum_b)
        dds[b] = np.min((cum_b - rm_b) / rm_b)

    alpha = (1 - ci_level) / 2
    ci_lo = np.percentile(dds, alpha * 100)
    ci_hi = np.percentile(dds, (1 - alpha) * 100)

    return BootstrapResult(
        metric_name="Max Drawdown",
        point_estimate=point,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
        bootstrap_distribution=dds,
    )


def ledoit_wolf_sharpe_test(returns_a: pd.Series, returns_b: pd.Series,
                             alpha: float = 0.05) -> SharpeTestResult:
    """Ledoit-Wolf (2008) test for difference in Sharpe ratios.

    Accounts for non-normality and serial correlation using HAC
    standard errors. Tests H0: Sharpe_A = Sharpe_B.

    Reference: Ledoit & Wolf (2008), "Robust performance hypothesis
    testing with the Sharpe ratio", JEF.
    """
    # Align
    common = returns_a.dropna().index.intersection(returns_b.dropna().index)
    ra = returns_a.loc[common].values
    rb = returns_b.loc[common].values
    n = len(ra)

    mu_a, mu_b = ra.mean(), rb.mean()
    sig_a, sig_b = ra.std(ddof=1), rb.std(ddof=1)

    sharpe_a = (mu_a * 252) / (sig_a * np.sqrt(252)) if sig_a > 0 else 0
    sharpe_b = (mu_b * 252) / (sig_b * np.sqrt(252)) if sig_b > 0 else 0

    # Compute the test statistic using the delta method
    # Following Ledoit-Wolf: define f(mu, sigma^2) = mu/sqrt(sigma^2)
    # gradient for Sharpe: df/dmu = 1/sigma, df/dsigma2 = -mu/(2*sigma^3)

    v_a = ra ** 2  # second moment proxy
    v_b = rb ** 2

    # Stack moments: [mu_a, mu_b, v_a, v_b]
    y = np.column_stack([ra, rb, v_a, v_b])
    y_mean = y.mean(axis=0)

    # HAC (Newey-West) variance with automatic bandwidth
    max_lag = int(np.floor(n ** (1/3)))

    # Center
    y_centered = y - y_mean

    # Gamma_0
    gamma_0 = (y_centered.T @ y_centered) / n

    # Newey-West weighted sum
    S = gamma_0.copy()
    for j in range(1, max_lag + 1):
        w = 1 - j / (max_lag + 1)  # Bartlett kernel
        gamma_j = (y_centered[j:].T @ y_centered[:-j]) / n
        S += w * (gamma_j + gamma_j.T)

    # Gradient of Sharpe difference w.r.t. [mu_a, mu_b, sigma2_a, sigma2_b]
    sigma2_a = sig_a ** 2
    sigma2_b = sig_b ** 2

    grad = np.array([
        1.0 / sig_a if sig_a > 0 else 0,                    # d/d(mu_a)
        -1.0 / sig_b if sig_b > 0 else 0,                   # d/d(mu_b)
        -mu_a / (2 * sig_a ** 3) if sig_a > 0 else 0,       # d/d(sigma2_a)
        mu_b / (2 * sig_b ** 3) if sig_b > 0 else 0,        # d/d(sigma2_b)
    ])

    # Variance of Sharpe difference
    var_diff = grad @ S @ grad / n
    se_diff = np.sqrt(max(var_diff, 1e-20))

    sharpe_diff = sharpe_a - sharpe_b
    # The delta method gives daily Sharpe diff; annualize
    # Actually, we computed annualized Sharpes, but the gradient was for daily.
    # Correct: work in daily, then multiply by sqrt(252) at the end.
    daily_sharpe_a = mu_a / sig_a if sig_a > 0 else 0
    daily_sharpe_b = mu_b / sig_b if sig_b > 0 else 0
    daily_diff = daily_sharpe_a - daily_sharpe_b

    t_stat = daily_diff / se_diff if se_diff > 0 else 0

    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(t_stat)))

    return SharpeTestResult(
        sharpe_a=sharpe_a,
        sharpe_b=sharpe_b,
        sharpe_diff=sharpe_diff,
        se_diff=se_diff * np.sqrt(252),  # annualized SE
        t_stat=t_stat,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
    )


def run_all_significance_tests(strategy_returns: pd.Series,
                                benchmark_returns: pd.Series,
                                n_bootstrap: int = 10000,
                                seed: int = 42) -> dict:
    """Run all statistical significance tests.

    Returns dict with all bootstrap CIs and Ledoit-Wolf test results.
    """
    results = {}

    # Bootstrap CIs on strategy
    results["sharpe_ci"] = bootstrap_sharpe(strategy_returns, n_bootstrap, seed=seed)
    results["terminal_ci"] = bootstrap_terminal_wealth(strategy_returns, n_bootstrap, seed=seed)
    results["maxdd_ci"] = bootstrap_max_drawdown(strategy_returns, n_bootstrap, seed=seed)

    # Ledoit-Wolf test vs benchmark
    results["lw_test"] = ledoit_wolf_sharpe_test(strategy_returns, benchmark_returns)

    # Key pass/fail checks
    results["sharpe_above_zero"] = results["sharpe_ci"].ci_lower > 0
    results["sharpe_above_half"] = results["sharpe_ci"].ci_lower > 0.5
    results["beats_benchmark"] = results["lw_test"].sharpe_diff > 0

    return results
