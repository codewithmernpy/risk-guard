# RiskGuard Quantitative Methodology

This document records the exact mathematical definitions, formulas, and modeling assumptions implemented in `riskguard/`.

---

## 1. Per-Position & Liquidation Analytics (`riskguard/analytics/margin.py`, `positions.py`)

### 1.1 Binance USDT-M Linear Perpetual Liquidation Price
For a linear USDT-margined perpetual contract with side $s \in \{+1 \text{ (long)}, -1 \text{ (short)}\}$, positive quantity $Q > 0$, entry price $EP > 0$, maintenance margin rate $\text{MMR}$, and continuous bracket deduction $\text{cumB}$:

$$
LP = \frac{WB + \text{cumB} - s \cdot Q \cdot EP}{Q \cdot \text{MMR} - s \cdot Q}
$$

- **Bracket Continuity ($\text{cumB}$):** For tier $k > 0$ starting above cutoff $C_{k-1}$:
  $$\text{cumB}_k = \text{cumB}_{k-1} + C_{k-1} \cdot (\text{MMR}_k - \text{MMR}_{k-1})$$
  so that $MM(N) = N \cdot \text{MMR}_k - \text{cumB}_k$ is continuous across bracket boundaries.
- **Isolated Margin:** $WB = \frac{Q \cdot EP}{L}$ where $L$ is position leverage.
- **Cross Margin (Documented Approximation):**
  $$WB_{\text{cross}, i} = \max\left(0,\; \text{Equity} - \text{uPnL}_i - \sum_{j \ne i} M_j\right)$$
  where $M_j$ is the collateral position $j$ takes out of the cross pool: its maintenance margin $\max(0, N_j \cdot \text{MMR}_j - \text{cumB}_j)$ for other cross positions, and its ring-fenced isolated equity $\max(0, \text{margin}_j + \text{uPnL}_j)$ for isolated positions. Subtracting $\text{uPnL}_i$ avoids double-counting position $i$'s unrealized PnL inside the entry-anchored numerator; when $\text{Mark} = EP$, $WB_{\text{cross}, i}$ reduces directly to $\text{Equity} - \sum_{j \ne i} M_j$.
- **No liquidation price:** a short always has a positive $LP$. A long has $LP \le 0$ when its collateral covers a fall to \$0 (an isolated 1x long, or a cross long whose shared equity exceeds its notional). It is then reported as `liquidation_price = None` with `liquidation_note` "none at current equity (cross)" or "none (isolated margin covers a fall to \$0)"; account-level cross liquidation is still tested in the stress scenarios (§2.3).

### 1.2 Distance to Liquidation & Stop Safety
- **Percentage Distance:** $\frac{\text{Mark} - LP}{\text{Mark}} \times 100$ (long) or $\frac{LP - \text{Mark}}{\text{Mark}} \times 100$ (short).
- **ATR Multiples:** $\frac{s \cdot (\text{Mark} - LP)}{\text{ATR}_{1D}(14)}$, where $\text{ATR}_{1D}(14)$ is Wilder's 14-day Average True Range computed from daily UTC resampled OHLCV bars.
- **Risk to Stop:** $s \cdot Q \cdot (\text{Mark} - S)$ in USD and as % of equity (a stop trailed into profit still gives back $\text{Mark} - S$).
- **DANGER Triggers:**
  - `stop_beyond_liquidation`: $S \le LP$ (long) or $S \ge LP$ (short).
  - `missing_stop`: no protective stop-loss provided.
  - `invalid_stop_direction` (stop already crossed): $\text{Mark} \le S$ (long) or $\text{Mark} \ge S$ (short). The stop should already have filled, so it is not protecting the position: risk to stop is reported as unbounded (`None`), and the position counts as unprotected in the portfolio worst case and in the mandatory-stop rule.
  - `at_or_past_liquidation`: mark price has reached or crossed $LP$.
- **WARN:** liquidation within 1.5 daily ATRs of the mark.

### 1.3 Funding Cost Estimate
Daily funding cost is normalized to 24 hours:
- Binance `lastFundingRate` (8h interval): $\text{Funding}_{\text{daily}} = 3 \times r_{8h}$.
- Hyperliquid `funding` (1h interval): $\text{Funding}_{\text{daily}} = 24 \times r_{1h}$.
- Estimated daily cost: $\text{Cost}_{\text{daily}} = s \cdot \text{Notional} \cdot \text{Funding}_{\text{daily}}$ (positive = cost paid, negative = funding received).

---

## 2. Portfolio Exposure, Tail Risk & Stress Tests (`riskguard/analytics/portfolio.py`)

### 2.1 Exposure, BTC Beta & Concentration
- **BTC Beta ($\beta_i$):** OLS slope against `BTCUSDT` using aligned 90-day (2,160-hour) hourly simple returns:
  $$\beta_i = \frac{\text{Cov}(r_i, r_{\text{BTC}})}{\text{Var}(r_{\text{BTC}})}, \quad \beta_{\text{BTC}} = 1.0$$
- **BTC-Beta-Adjusted Net Exposure:** $\sum_i s_i \cdot \text{Notional}_i \cdot \beta_i$.
- **Correlation Matrices:** Pearson correlations computed over trailing 30d (720h) and 90d (2,160h) hourly returns.
- **Herfindahl-Hirschman Index (HHI):** $\text{HHI} = \sum_a w_a^2$ over per-asset gross notional shares $w_a$, with effective bet count $N_{\text{eff}} = 1 / \text{HHI}$.

### 2.2 1-Day VaR & CVaR (Historical & Parametric Cornish-Fisher)
Using current signed asset notionals $E_a$ applied to aligned daily simple returns $R_{t, a}$ to produce historical portfolio daily PnL $P_t = \sum_a E_a R_{t, a}$ and losses $L_t = -P_t$:
- **Historical VaR / CVaR ($\alpha \in \{0.95, 0.99\}$):**
  $$\text{VaR}_\alpha = \text{Quantile}_\alpha(L), \quad \text{CVaR}_\alpha = \mathbb{E}[L \mid L \ge \text{VaR}_\alpha]$$
- **Parametric Cornish-Fisher VaR / CVaR:** With sample mean $\mu$, standard deviation $\sigma$, Fisher skewness $S$, and excess kurtosis $K$ of $P_t$, and standard normal lower-tail quantile $z = \Phi^{-1}(1 - \alpha)$:
  $$z_{\text{CF}} = z + \frac{z^2 - 1}{6} S + \frac{z^3 - 3z}{24} K - \frac{2z^3 - 5z}{36} S^2$$
  $$\text{VaR}_{\text{CF},\alpha} = \max\left(0,\; -(\mu + z_{\text{CF}} \sigma)\right)$$
  Cornish-Fisher CVaR is the exact tail average of the Cornish-Fisher quantile function, using the truncated normal moments $\int_{-\infty}^{c} z^k \phi(z)\,dz$ ($k = 0..3$) with $c = \Phi^{-1}(1-\alpha)$:
  $$\text{CVaR}_{\text{CF},\alpha} = -\mu + \frac{\sigma\,\phi(c)}{1-\alpha}\left[1 + \frac{S c}{6} - \frac{K (1 - c^2)}{24} + \frac{S^2 (1 - 2c^2)}{36}\right]$$
  floored at $\text{VaR}_{\text{CF},\alpha}$. With $S = K = 0$ both reduce exactly to Gaussian VaR and Expected Shortfall.

### 2.3 Stress Scenarios & Liquidation Checks
Each scenario computes shocked mark prices $P_{\text{shock}, i} = P_{\text{mark}, i}(1 + \delta_i)$, portfolio PnL $\sum_i s_i \cdot \text{Notional}_i \cdot \delta_i$ (linear mark-to-market), shocked equity, and liquidations:
- **Isolated positions:** liquidated when the shocked mark reaches the position's own $LP$; the position's scenario loss is capped at $\max(0, \text{margin} + \text{uPnL})$ (an isolated position cannot lose more than its own margin).
- **Cross positions (account level):** cross pool $= \text{Equity} - \sum_{\text{isolated}} \max(0, \text{margin}_j + \text{uPnL}_j)$; if cross pool $+ \sum_{\text{cross}} \text{PnL}_j \le \sum_{\text{cross}} MM_j(\text{shocked notional})$, every cross position is liquidated. Offsetting cross hedges therefore net, as on the exchange.

1. **BTC -5% / -10% / -20%:** $\delta_i = \beta_i \cdot \delta_{\text{BTC}}$.
2. **Alt Crash (1.5x BTC):** $\delta_{\text{BTC}} = -10\%$, $\delta_{\text{alt}} = -15\%$.
3. **Correlation $\to$ 1:** every asset moves together by $\max(10\%, 2.5 \cdot \sigma_{1d, i})$; both the crash (all down) and the squeeze (all up) are evaluated and the one with the worse PnL is reported, so net-short and hedged books are stressed too.
4. **Funding Spike:** $+0.10\%$ per 8h ($+0.30\%/\text{day}$) sustained for 7 days; longs pay, shorts receive: $\text{PnL}_i = -s_i \cdot \text{Notional}_i \cdot 0.003 \cdot 7$.

---

## 3. Position Sizing & Pre-Trade Check (`riskguard/analytics/sizing.py`)

Given entry price $EP$, stop price $SP$, stop distance $\Delta = |EP - SP|$, daily ATR, and equity $E$:
1. **Fixed Fractional:** $Q = \frac{E \cdot r}{ \Delta }$.
2. **ATR Volatility-Targeted:** $Q = \frac{E \cdot r}{\text{ATR}_{1D}}$.
3. **Capped Fractional Kelly:** Full Kelly risk fraction $f^* = p - \frac{1 - p}{b}$ (for win rate $p$ and payoff ratio $b$), scaled by fractional multiplier (0.25) and capped.
4. **Max-Leverage-Safe Size:** Computes maximum isolated leverage keeping $LP$ at least $k \cdot \text{ATR}_{1D}$ beyond $SP$:
   - Long ($LP_{\text{target}} = SP - k \cdot \text{ATR}$): $L_{\text{safe}} = \left[1 - \frac{LP_{\text{target}}}{EP}(1 - \text{MMR})\right]^{-1}$; if $LP_{\text{target}} \le 0$ only $1\times$ is safe.
   - Short ($LP_{\text{target}} = SP + k \cdot \text{ATR}$): $L_{\text{safe}} = \left[\frac{LP_{\text{target}}}{EP}(1 + \text{MMR}) - 1\right]^{-1}$
   - Clamped to $[1\times, 100\times]$; ignoring cumB is conservative (cumB moves $LP$ further away). Isolated size cap: $E \cdot L_{\text{safe}} / EP$.
   For cross margin, analytically caps quantity so cross $LP$ respects $LP_{\text{target}}$: long $Q \le (WB + \text{cumB}) / (EP - LP_{\text{target}}(1-\text{MMR}))$, short $Q \le (WB + \text{cumB}) / (LP_{\text{target}}(1+\text{MMR}) - EP)$.

**Pre-Trade Check:** the proposed position is evaluated as if filled at its entry price (mark = entry; existing positions keep the live mark). `NO-GO`: circuit breaker tripped, missing stop under the mandatory-stop rule, stop on the wrong side of entry, stop beyond liquidation, or too many positions. `REDUCE SIZE`: leverage above the cap, risk per trade or total open risk above limits, new correlated-exposure breach, or less than $k$ ATR between stop and liquidation. The recommended quantity is the largest size inside both the risk budget ($\min$ of per-trade limit and remaining total-risk budget) and the liquidation-safety cap.

**Rule boundaries:** caps (risk per trade, total open risk, leverage, positions, correlated exposure) are `BREACH` only above the limit; loss limits (daily, weekly, drawdown) are `BREACH` once the loss reaches the limit, matching the circuit breaker. `WARN` from `warn_fraction` (default 80%) of the limit. Comparisons use a 1e-9 relative tolerance so float noise cannot move a boundary.

---

## 4. Behavioural Detectors & Discipline Score (`riskguard/analytics/journal.py`)

- **Revenge Trading:** New trade opened within 30 minutes after a losing trade closes with entry notional $\ge 1.15\times$ the losing trade's notional.
- **Overtrading:** Daily trade count $\ge \max(6,\; 2.0 \times \text{rolling 14-day median trades/day})$.
- **Size Escalation:** After $\ge 2$ consecutive losses, next trade notional $\ge 1.4\times$ streak median notional. The streak only counts trades already closed when the new trade was opened (no lookahead).
- **Disposition Effect:** Average losing-trade hold duration $\ge 1.5\times$ average winning-trade hold duration.
- **Stop Violation:** Trade with `planned_stop` exits beyond planned stop with ex-fee loss $> 1.15\times$ planned risk.
- **Tilt Session:** 4-hour cluster with $\ge 3$ losing trades plus $\ge 2$ behavioral flags or $\ge 3\%$ equity loss.
- **Discipline Score (0–100):** Weighted sum of Stop Adherence (25), Size & Revenge Control (25), Pace & Tilt Control (20), Disposition Control (15), and Tail Loss Consistency (15).
- **Drawdown:** depth = $(\text{peak} - \text{equity}) / \text{peak}$ on the closed-trade equity curve; duration = time from the peak until equity is back at or above it (or until the reference time for an open drawdown).
- **Today / This Week:** a real journal is evaluated against the current UTC date; a journal made only of `SYNTHETIC` demo trades is evaluated as of its last closed trade. The evaluated date is shown next to the circuit breaker.

---

## 5. Drawdown Schedules & Risk of Ruin (`riskguard/analytics/drawdown.py`)

- **Step-Down Schedule:** Scales risk multiplier cleanly across drawdown tiers (`1.0x -> 0.75x -> 0.50x -> 0.25x -> 0.0x HALT`).
- **CPPI Multiplier:** Defines floor $F = \text{Peak} \cdot (1 - \text{DD}_{\max})$, cushion $C = \max(0, E - F)$, and scales risk proportionally to cushion ratio.
- **Ruin definition (all estimates):** equity ever falls to $\text{Start} \cdot (1 - \text{DD}_{\text{ruin}})$.
- **Analytical Risk of Ruin (infinite horizon):** Solves $\mathbb{E}[u^R] = p u^b + (1 - p) u^{-1} = 1$ for $u \in (0, 1)$ via Brent root-finding (`scipy.optimize.brentq`) when expectancy $> 0$, giving $\text{RoR} = u^N$ for compounding drawdown units $N = \frac{\ln(1 - \text{DD}_{\text{ruin}})}{\ln(1 - r)}$; $\text{RoR} = 1$ when expectancy $\le 0$.
- **Monte Carlo Simulation (finite horizon):** Vectorized NumPy paths that bootstrap the journal's realised $R$-multiples (when at least 10 are available, otherwise $+b$ / $-1$ Bernoulli trades) under both flat risk and the drawdown step-down schedule; the 95th percentile worst peak-to-trough drawdown of the flat paths is reported separately.
