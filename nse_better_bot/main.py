import yfinance as yf
import pandas as pd
import numpy as np
import os
import random
from numba import njit
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION (Index Options Universe)
# ==========================================
INDICES = {
    'NIFTY':        {'ticker': '^NSEI',               'expiry_day': 3}, # Thursday
    'BANKNIFTY':    {'ticker': '^NSEBANK',            'expiry_day': 2}, # Wednesday
    'FINNIFTY':     {'ticker': 'NIFTY_FIN_SERVICE.NS', 'expiry_day': 1}, # Tuesday
    'MIDCPNIFTY':   {'ticker': 'NIFTY_MID_SELECT.NS',  'expiry_day': 0}  # Monday
}

# Timeframe Settings
TIMEFRAMES = ['5m'] 
MAX_HISTORY_DAYS = "59d" # Yahoo Limit for 5m

# Strategy Settings
NUM_ITERATIONS = 100000       
MIN_TRADES_TOTAL = 40        # Across all indices
ROBUSTNESS_THRESHOLD = 0.70  # Neighbor stability score required

# Option Sim Settings
OPTION_COST_PCT = 0.001      # 0.1% per trade (Spread + Comm)
DELTA = 0.55                 # ATM Delta
EXIT_TIME_MINUTES = 920      # 15:20 Hard Exit

if not os.path.exists("data_universal"):
    os.makedirs("data_universal")

# ==========================================
# 2. ROBUST DATA LOADING
# ==========================================
def clean_yfinance_data(df):
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        try: df.columns = df.columns.droplevel(1) 
        except: pass
    df.dropna(inplace=True)
    return df

def fetch_data_universe():
    data_store = {}
    print("--- 1. Fetching 5m Data (Max History) ---")
    
    for idx_name, info in INDICES.items():
        ticker = info['ticker']
        for tf in TIMEFRAMES:
            file_path = f"data_universal/{idx_name}_{tf}.csv"
            df = None
            
            # 1. Try Cache
            if os.path.exists(file_path):
                try: df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
                except: df = None

            # 2. Download Fresh
            if df is None:
                try:
                    df = yf.download(ticker, period=MAX_HISTORY_DAYS, interval=tf, progress=False)
                    df = clean_yfinance_data(df)
                    if not df.empty and len(df) > 100: df.to_csv(file_path)
                except: pass

            # 3. Store in Memory
            if df is not None and len(df) > 200:
                times_min = df.index.hour.values * 60 + df.index.minute.values
                weekdays = df.index.weekday.values
                
                data_store[idx_name] = {
                    'opens': np.ascontiguousarray(df['Open'].values, dtype=np.float64),
                    'highs': np.ascontiguousarray(df['High'].values, dtype=np.float64),
                    'lows': np.ascontiguousarray(df['Low'].values, dtype=np.float64),
                    'closes': np.ascontiguousarray(df['Close'].values, dtype=np.float64),
                    'times': np.ascontiguousarray(times_min, dtype=np.int64),
                    'weekdays': np.ascontiguousarray(weekdays, dtype=np.int64),
                    'expiry_day': info['expiry_day'],
                }
    
    print(f"Loaded {len(data_store)} Indices. Ready for Optimization.")
    return data_store

# ==========================================
# 3. NUMBA INDICATORS
# ==========================================
@njit(fastmath=True, cache=True)
def calc_ema(prices, period):
    n = len(prices)
    ema = np.empty(n); ema[:] = np.nan
    if n < period: return ema
    ema[period-1] = np.mean(prices[:period])
    mult = 2 / (period + 1)
    for i in range(period, n):
        ema[i] = (prices[i] - ema[i-1]) * mult + ema[i-1]
    return ema

@njit(fastmath=True, cache=True)
def calc_sma(prices, period):
    n = len(prices)
    sma = np.empty(n); sma[:] = np.nan
    if n < period: return sma
    w_sum = np.sum(prices[:period])
    sma[period-1] = w_sum / period
    for i in range(period, n):
        w_sum = w_sum - prices[i-period] + prices[i]
        sma[i] = w_sum / period
    return sma

# ==========================================
# 4. OPTION BACKTEST ENGINE (Simulated)
# ==========================================
@njit(fastmath=True)
def backtest_options(opens, highs, lows, closes, 
                     weekdays, times_min, expiry_day_idx,
                     short_ma, long_ma, super_ma, 
                     trail_pct, target_pct, filter_mode): 
    
    n = len(closes)
    total_roi = 0.0
    
    in_pos = False 
    entry_spot_price = 0.0
    high_spot_since_entry = 0.0
    
    # Option Sim Vars
    entry_premium_est = 0.0
    days_to_expiry_entry = 0
    entry_time_idx = 0
    
    trades = 0
    wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    
    start_idx = 300 
    
    for i in range(start_idx, n - 1):
        curr_open = opens[i+1]
        curr_high = highs[i+1]
        curr_low = lows[i+1]
        curr_close = closes[i+1]
        curr_time = times_min[i+1]
        curr_day = weekdays[i+1]
        
        # --- EXIT LOGIC ---
        if in_pos:
            high_spot_since_entry = max(high_spot_since_entry, curr_high)
            should_exit = False
            exit_spot_price = 0.0
            
            # 1. Expiry Hard Exit (15:20)
            if curr_day == expiry_day_idx and curr_time >= EXIT_TIME_MINUTES:
                should_exit = True
                exit_spot_price = curr_close 
            
            # 2. Spot Trailing Stop
            if not should_exit:
                stop_price = high_spot_since_entry * (1.0 - trail_pct / 100.0)
                if curr_low <= stop_price:
                    should_exit = True
                    exit_spot_price = curr_open if curr_open < stop_price else stop_price
            
            # 3. Spot Target
            if not should_exit:
                target_price = entry_spot_price * (1.0 + target_pct / 100.0)
                if curr_high >= target_price:
                    should_exit = True
                    exit_spot_price = target_price
            
            # 4. Cross Reversal
            if not should_exit:
                if short_ma[i-1] >= long_ma[i-1] and short_ma[i] < long_ma[i]:
                    should_exit = True
                    exit_spot_price = curr_open
            
            if should_exit:
                # --- PnL CALCULATION (Option Sim) ---
                spot_points = exit_spot_price - entry_spot_price
                option_gross_pts = spot_points * DELTA
                
                # Theta Decay
                mins_held = (i - entry_time_idx) * 5 # 5m candles
                if mins_held < 5: mins_held = 5
                
                dte_decay_mult = 3.0 - (days_to_expiry_entry * 0.4) 
                if dte_decay_mult < 1.0: dte_decay_mult = 1.0
                theta_loss = entry_premium_est * (0.0001 * dte_decay_mult) * mins_held
                
                net_pts = option_gross_pts - theta_loss
                trade_roi = (net_pts / entry_premium_est) - OPTION_COST_PCT
                
                total_roi += trade_roi
                
                if trade_roi > 0: 
                    wins += 1
                    gross_win += trade_roi
                else:
                    gross_loss += abs(trade_roi)
                    
                trades += 1
                in_pos = False

        # --- ENTRY LOGIC ---
        elif not in_pos:
            if 555 < curr_time < 900: # 9:15 to 15:00
                crossover = short_ma[i-1] <= long_ma[i-1] and short_ma[i] > long_ma[i]
                
                # SuperTrend Filter Logic
                trend_ok = True
                if filter_mode == 1: # Only Buy if Price > SuperMA
                    if closes[i] <= super_ma[i]: trend_ok = False
                elif filter_mode == 2: # Only Buy if Price < SuperMA (Mean Rev)
                    if closes[i] >= super_ma[i]: trend_ok = False
                
                if crossover and trend_ok:
                    in_pos = True
                    entry_spot_price = curr_open
                    high_spot_since_entry = curr_open
                    entry_time_idx = i
                    
                    # Premium Est
                    dte = (expiry_day_idx - curr_day)
                    if dte < 0: dte += 7
                    days_to_expiry_entry = dte
                    premium_pct = 0.003 + (0.002 * dte) 
                    entry_premium_est = curr_open * premium_pct

    return total_roi, trades, wins, gross_win, gross_loss

# ==========================================
# 5. EVALUATOR & NEIGHBORS
# ==========================================
def evaluate_params(p, data_store):
    if p['s_ma'] >= p['l_ma']: return -999, {}, {}
    
    index_scores = []
    total_trades_all = 0
    global_gross_win = 0.0
    global_gross_loss = 0.0
    
    detailed_results = {} # Store ROI per index

    for name, d in data_store.items():
        # 1. Calculate Indicators based on Type (SMA/EMA)
        if p['t_s'] == 1: arr_s = calc_ema(d['closes'], p['s_ma'])
        else:             arr_s = calc_sma(d['closes'], p['s_ma'])
            
        if p['t_l'] == 1: arr_l = calc_ema(d['closes'], p['l_ma'])
        else:             arr_l = calc_sma(d['closes'], p['l_ma'])
            
        if p['t_sl'] == 1: arr_sl = calc_ema(d['closes'], p['sl_ma'])
        else:              arr_sl = calc_sma(d['closes'], p['sl_ma'])
            
        # 2. Run Backtest
        roi, tr, _, gw, gl = backtest_options(
            d['opens'], d['highs'], d['lows'], d['closes'],
            d['weekdays'], d['times'], d['expiry_day'],
            arr_s, arr_l, arr_sl, 
            p['trail_pct'], p['tgt_pct'], p['f_mode']
        )
        
        # 3. Kill Switch: If any index loses > 20% total, kill strat
        if roi < -0.20: return -999, {}, {}
        
        index_scores.append(roi)
        detailed_results[name] = roi
        total_trades_all += tr
        global_gross_win += gw
        global_gross_loss += gl

    if total_trades_all < MIN_TRADES_TOTAL:
        return -999, {}, {}

    # Profit Factor
    pf = global_gross_win / global_gross_loss if global_gross_loss > 0 else 1.5
    if pf < 1.05: return -999, {}, {}

    # Score = Median ROI * PF
    # (Median ensures we don't just win big on Nifty and fail others)
    median_roi = np.median(index_scores)
    score = median_roi * pf
    
    metrics = {'pf': pf, 'trades': total_trades_all, 'roi_map': detailed_results}
    return score, metrics, detailed_results

def get_neighbors(p):
    neighbors = []
    # Create variations (±5%) to check stability
    n1 = p.copy(); n1['s_ma'] = max(5, int(p['s_ma'] * 0.95)); n1['l_ma'] = max(10, int(p['l_ma'] * 0.95)); neighbors.append(n1)
    n2 = p.copy(); n2['s_ma'] = int(p['s_ma'] * 1.05); n2['l_ma'] = int(p['l_ma'] * 1.05); neighbors.append(n2)
    # Tweak Trail
    n3 = p.copy(); n3['trail_pct'] = p['trail_pct'] * 0.9; neighbors.append(n3)
    return neighbors

# ==========================================
# 6. MAIN LOOP
# ==========================================
def run_optimization():
    print("\n--- 2. Starting Universal Option Optimization ---")
    data = fetch_data_universe()
    if not data: return

    print(f"Goal: Maximize Median ROI * PF across {list(data.keys())}")
    
    best_score = -9999
    
    # Pre-generate Random Params
    # t_s: 0=SMA, 1=EMA
    # f_mode: 0=OFF, 1=Price>Super, 2=Price<Super
    
    r_short = np.random.randint(5, 50, NUM_ITERATIONS)
    r_long = np.random.randint(10, 100, NUM_ITERATIONS)
    r_super = np.random.randint(50, 300, NUM_ITERATIONS)
    r_ts = np.random.randint(0, 2, NUM_ITERATIONS) 
    r_tl = np.random.randint(0, 2, NUM_ITERATIONS)
    r_tsl = np.random.randint(0, 2, NUM_ITERATIONS)
    r_trail = np.random.uniform(0.1, 1.5, NUM_ITERATIONS)
    r_tgt = np.random.uniform(0.5, 4.0, NUM_ITERATIONS)
    r_filt = np.random.randint(0, 3, NUM_ITERATIONS)

    for i in tqdm(range(NUM_ITERATIONS)):
        p = {
            's_ma': r_short[i], 'l_ma': r_long[i], 'sl_ma': r_super[i],
            't_s': r_ts[i], 't_l': r_tl[i], 't_sl': r_tsl[i],
            'trail_pct': r_trail[i], 'tgt_pct': r_tgt[i], 'f_mode': r_filt[i]
        }
        
        score, metrics, det_res = evaluate_params(p, data)
        
        if score > best_score:
            # Robustness Check (Neighbors)
            neighbors = get_neighbors(p)
            n_scores = []
            for np_p in neighbors:
                ns, _, _ = evaluate_params(np_p, data)
                if ns > -100: n_scores.append(ns)
            
            # If stable (neighbors perform relatively close to main)
            avg_n = sum(n_scores)/len(n_scores) if n_scores else 0
            
            # If Stability is good OR score is exceptionally high with decent stability
            if len(n_scores) > 0 and avg_n > (score * 0.6): 
                best_score = score
                
                # --- FORMATTED OUTPUT ---
                ts = "EMA" if p['t_s'] else "SMA"
                tl = "EMA" if p['t_l'] else "SMA"
                tsl = "EMA" if p['t_sl'] else "SMA"
                
                f_str = ["DISABLED", "Buy Only > Super (Trend)", "Buy Only < Super (MeanRev)"][p['f_mode']]
                
                res_str = ""
                for k, v in det_res.items():
                    res_str += f"{k}:{v*100:.1f}% | "
                
                tqdm.write(f"\n💎 NEW BEST FIND! Score: {score:.4f} (PF: {metrics['pf']:.2f})")
                tqdm.write(f"   Cross: {ts} {p['s_ma']} crosses {tl} {p['l_ma']}")
                tqdm.write(f"   Filter: {f_str} (using {tsl} {p['sl_ma']})")
                tqdm.write(f"   Exits: Trail {p['trail_pct']:.2f}% | Target {p['tgt_pct']:.2f}%")
                tqdm.write(f"   Returns: {res_str}")
                tqdm.write(f"   Total Trades: {metrics['trades']}")

if __name__ == "__main__":
    run_optimization()