"""Vectorized performance analytics for quantitative strategies.

All functions accept pandas Series of arithmetic daily returns
and return scalar metrics. No Python loops over rows.

References:
    - López de Prado, M. (2018). Advances in Financial Machine Learning.
    - Sharpe, W. (1994). The Sharpe Ratio. Journal of Portfolio Management.
    - Sortino & Satchell (2001). Managing Downside Risk in Financial Markets.
"""

import math

import pandas as pd
from scipy import stats


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, freq: int = 252) -> float:
    """Annualized Sharpe Ratio.

    SR = (mean(r) - rf_daily) / std(r) * sqrt(freq)

    Parameters
    ----------
    returns : pd.Series
        Arithmetic daily returns
    rf : float
        Annual risk-free rate (e.g. 0.04 for 4%)
    freq : int
        Trading periods per year (252 for daily, 52 for weekly)

    Returns
    -------
    float
        Annualized Sharpe Ratio
    """
    rf_period = (1 + rf) ** (1 / freq) - 1
    excess = returns - rf_period
    std = excess.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(excess.mean() / std * math.sqrt(freq))


def sortino_ratio(returns: pd.Series, rf: float = 0.0, freq: int = 252) -> float:
    """Annualized Sortino Ratio — penalizes only downside volatility.

    Sortino = (mean(r) - rf) / downside_std * sqrt(freq)
    where downside_std = std of returns below the minimum acceptable return (rf).
    """
    rf_period = (1 + rf) ** (1 / freq) - 1
    excess = returns - rf_period
    downside = excess[excess < 0]
    if len(downside) == 0:
        # No negative returns — ratio is infinite, return a large finite value
        return float(excess.mean() * math.sqrt(freq) / 1e-12)
    downside_std = downside.std(ddof=1)
    if downside_std < 1e-12:
        return 0.0
    return float(excess.mean() / downside_std * math.sqrt(freq))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown.

    Returns
    -------
    float
        Drawdown as a negative fraction, e.g. -0.35 means -35%.
        Returns 0.0 if there is no drawdown.
    """
    equity = (1 + returns).cumprod()
    rolling_peak = equity.cummax()
    drawdown = equity / rolling_peak - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, freq: int = 252) -> float:
    """Calmar Ratio: annualized return divided by absolute max drawdown.

    Returns
    -------
    float
        Calmar Ratio. Returns inf if drawdown is zero.
    """
    n = len(returns)
    total = float((1 + returns).prod() - 1.0)
    ann_return = (1 + total) ** (freq / max(n, 1)) - 1
    mdd = abs(max_drawdown(returns))
    if mdd < 1e-10:
        return float("inf") if ann_return > 0 else 0.0
    return float(ann_return / mdd)


def probabilistic_sharpe(
    returns: pd.Series,
    sr_star: float = 0.0,
    freq: int = 252,
) -> float:
    """Probabilistic Sharpe Ratio (PSR) — López de Prado (2018).

    PSR(SR*) = Φ[ (SR_hat - SR*) / SE(SR_hat) ]
    SE(SR_hat) = sqrt( (1 - γ3*SR_hat + (γ4-1)/4 * SR_hat²) / (T-1) )

    where γ3 = skewness, γ4 = excess kurtosis.

    Parameters
    ----------
    returns : pd.Series
        Daily returns
    sr_star : float
        Benchmark Sharpe Ratio (annualized). Default 0 — test against no-skill.
    freq : int
        Trading periods per year

    Returns
    -------
    float
        Probability in [0, 1] that true SR > sr_star.
    """
    t = len(returns)
    if t < 5:
        return 0.5

    sr_hat = sharpe_ratio(returns, freq=freq)
    # De-annualize SR* for the SE formula (which uses period-normalized SR)
    sr_hat_period = sr_hat / math.sqrt(freq)
    sr_star_period = sr_star / math.sqrt(freq)

    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns))  # excess kurtosis (Fisher)

    se_sq = (1 - skew * sr_hat_period + (kurt / 4) * sr_hat_period**2) / (t - 1)
    if se_sq <= 0:
        se_sq = 1e-12

    z = (sr_hat_period - sr_star_period) / math.sqrt(se_sq)
    return float(stats.norm.cdf(z))


def information_coefficient(signals: pd.Series, forward_returns: pd.Series) -> float:
    """Spearman rank Information Coefficient between signals and forward returns.

    IC measures the predictive power of signals. IC = 0 → no skill, IC = 1 → perfect.

    Returns
    -------
    float
        Spearman rank correlation in [-1, +1].
    """
    aligned = pd.concat([signals, forward_returns], axis=1).dropna()
    if len(aligned) < 3:
        return 0.0
    ic, _ = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(ic)


def annual_turnover(positions: pd.Series, freq: int = 252) -> float:
    """Annualized portfolio turnover as a percentage.

    Turnover = sum(|Δposition|) / avg(|position|) * (freq / T) * 100

    Returns
    -------
    float
        Annual turnover percentage. 100 = 100% of portfolio turned per year.
    """
    if positions.abs().sum() < 1e-12:
        return 0.0

    changes = positions.diff().abs().sum()
    avg_exposure = positions.abs().mean()
    if avg_exposure < 1e-12:
        return 0.0

    t = len(positions)
    return float(changes / avg_exposure * (freq / t) * 100)
