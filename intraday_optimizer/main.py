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
# 1. CONFIGURATION
# ==========================================
INDICES = {
    'NIFTY':        {'ticker': '^NSEI',       'expiry_day': 3}, # Thursday
    'BANKNIFTY':    {'ticker': '^NSEBANK',    'expiry_day': 2}, # Wednesday
    'FINNIFTY':     {'ticker': 'NIFTY_FIN_SERVICE.NS', 'expiry_day': 1}, # Tuesday
    'MIDCPNIFTY':   {'ticker': 'NIFTY_MID_SELECT.NS',  'expiry_day': 0}  # Monday
}

# Updated Timeframes
TIMEFRAMES = ['3m', '5m', '15m']

# Weighting Config (Higher timeframe = Lower weight as per request)
TF_WEIGHTS = {'3m': 3.0, '5m': 2.0, '15m': 1.0}

# Strategy Constraints
NUM_ITERATIONS = 50000       
COST_PCT = 0.0005            # 0.05% per trade
EXIT_TIME_MINUTES = 920      # 15:20 Expiry Exit
MIN_TRADES_TOTAL = 100       # Minimum trades across ALL charts to be valid

if not os.path.exists("data_universal"):
    os.makedirs("data_universal")

# ==========================================
# 2. DATA LOADING (BATCH)
# ==========================================
def fetch_data_batch():
    """
    Loads all combinations into a dictionary of Numpy arrays.
    """
    data_store = {}
    print("--- 1. Fetching/Loading Data for ALL 12 Charts ---")
    
    for idx_name, info in INDICES.items():
        for tf in TIMEFRAMES:
            ticker = info['ticker']
            file_path = f"data_universal/{idx_name}_{tf}.csv"
            
            # Yahoo limit for intraday is 60 days
            period = "59d" 
            
            df = None
            # Download if missing
            if not os.path.exists(file_path):
                try:
                    tqdm.write(f"Downloading {idx_name} {tf}...")
                    df = yf.download(ticker, period=period, interval=tf, progress=False, multi_level_index=False)
                    if len(df) > 100:
                        df.dropna(inplace=True)
                        df.to_csv(file_path)
                except: pass
            else:
                try:
                    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
                except: pass

            if df is not None and len(df) > 500:
                # Pre-process for Numba
                data_store[f"{idx_name}|{tf}"] = {
                    'opens': np.ascontiguousarray(df['Open'].values, dtype=np.float64),
                    'highs': np.ascontiguousarray(df['High'].values, dtype=np.float64),
                    'lows': np.ascontiguousarray(df['Low'].values, dtype=np.float64),
                    'closes': np.ascontiguousarray(df['Close'].values, dtype=np.float64),
                    'times': np.ascontiguousarray(df.index.hour.values * 60 + df.index.minute.values, dtype=np.int64),
                    'weekdays': np.ascontiguousarray(df.index.weekday.values, dtype=np.int64),
                    'expiry_day': info['expiry_day'],
                    'tf_weight': TF_WEIGHTS[tf]
                }
            else:
                print(f"⚠️ Warning: Could not load data for {idx_name} {tf}")

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

# ==========================================
# 4. CORE BACKTEST LOGIC
# ==========================================
@njit(fastmath=True)
def backtest_single_chart(opens, highs, lows, closes, 
                          weekdays, times_min, expiry_day_idx,
                          short_ma, long_ma, super_ma, 
                          trail_pct, target_pct): 
    
    n = len(closes)
    # Start with 0 capital logic for pure PnL summing
    # We will sum % returns to normalize across indices with different prices
    total_pct_return = 0.0
    
    in_pos = False 
    entry_price = 0.0
    high_since_entry = 0.0
    trades = 0
    wins = 0
    gross_win_pct = 0.0
    gross_loss_pct = 0.0
    
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
            high_since_entry = max(high_since_entry, curr_high)
            should_exit = False
            exit_price = 0.0
            
            # 1. Expiry Hard Exit
            if curr_day == expiry_day_idx and curr_time >= EXIT_TIME_MINUTES:
                should_exit = True
                exit_price = curr_close 
            
            # 2. Trailing Stop
            if not should_exit:
                stop_price = high_since_entry * (1.0 - trail_pct / 100.0)
                if curr_low <= stop_price:
                    should_exit = True
                    exit_price = curr_open if curr_open < stop_price else stop_price
            
            # 3. Target
            if not should_exit:
                target_price = entry_price * (1.0 + target_pct / 100.0)
                if curr_high >= target_price:
                    should_exit = True
                    exit_price = target_price
            
            # 4. Cross Exit
            if not should_exit:
                if short_ma[i-1] >= long_ma[i-1] and short_ma[i] < long_ma[i]:
                    should_exit = True
                    exit_price = curr_open
            
            if should_exit:
                trade_pnl = (exit_price - entry_price) / entry_price - COST_PCT
                total_pct_return += trade_pnl
                
                if trade_pnl > 0: 
                    wins += 1
                    gross_win_pct += trade_pnl
                else:
                    gross_loss_pct += abs(trade_pnl)
                    
                trades += 1
                in_pos = False
                high_since_entry = 0.0

        # --- ENTRY LOGIC ---
        elif not in_pos:
            # Entry Window: 9:15 to 3:00 PM
            if 555 < curr_time < 900: 
                crossover = short_ma[i-1] <= long_ma[i-1] and short_ma[i] > long_ma[i]
                trend_ok = closes[i] > super_ma[i]
                
                if crossover and trend_ok:
                    in_pos = True
                    entry_price = curr_open
                    high_since_entry = curr_open

    return total_pct_return, trades, wins, gross_win_pct, gross_loss_pct

# ==========================================
# 5. UNIVERSAL OPTIMIZER
# ==========================================
def run_universal_optimization():
    
    # 1. LOAD ALL DATA
    data_map = fetch_data_batch()
    if not data_map:
        print("No data found. Check internet or data folder.")
        return

    keys = list(data_map.keys()) # e.g. ['NIFTY|3m', 'BANKNIFTY|5m'...]
    print(f"\n--- 2. Starting Universal Optimization ---")
    print(f"Charts Loaded: {len(keys)}")
    print(f"Goal: Find ONE parameter set that survives ALL {len(keys)} charts.")
    print(f"Weights: {TF_WEIGHTS}")

    best_weighted_score = -99999
    best_p = {}
    best_breakdown = {} # To store how it performed on each chart

    # Random Search Arrays
    r_short = np.random.randint(5, 60, NUM_ITERATIONS)
    r_long = np.random.randint(15, 120, NUM_ITERATIONS)
    r_super = np.random.randint(60, 400, NUM_ITERATIONS)
    r_trail = np.random.uniform(0.1, 2.0, NUM_ITERATIONS) # Tighter for 3m
    r_target = np.random.uniform(0.3, 4.0, NUM_ITERATIONS)

    for i in tqdm(range(NUM_ITERATIONS), desc="Simulating Universe"):
        s_ma = r_short[i]
        l_ma = r_long[i]
        sl_ma = r_super[i]
        t_pct = r_trail[i]
        tgt_pct = r_target[i]

        if s_ma >= l_ma: continue
        
        # Iteration Stats
        total_weighted_roi = 0.0
        total_trades_all = 0
        is_universal_failure = False
        
        current_breakdown = {}

        # --- SUB-LOOP: TEST ON EVERY CHART ---
        for k in keys:
            d = data_map[k]
            
            # Recalculate indicators for this specific chart's close data
            # (Cannot pre-calc because params change every iteration)
            arr_s = calc_ema(d['closes'], s_ma)
            arr_l = calc_ema(d['closes'], l_ma)
            arr_sl = calc_ema(d['closes'], sl_ma)
            
            roi, tr, wins, g_win, g_loss = backtest_single_chart(
                d['opens'], d['highs'], d['lows'], d['closes'],
                d['weekdays'], d['times'], d['expiry_day'],
                arr_s, arr_l, arr_sl, t_pct, tgt_pct
            )
            
            # --- UNIVERSALITY CHECKS ---
            pf = g_win / g_loss if g_loss > 0 else 1.5
            
            # 1. Kill Switch: If Profit Factor on ANY single chart is < 0.6, kill the strategy.
            # We allow small losses (0.6-1.0) on some charts, but no disasters.
            if pf < 0.6: 
                is_universal_failure = True
                break
            
            # 2. Weighted Score Accumulation
            # Score = ROI * Weight
            total_weighted_roi += (roi * d['tf_weight'])
            total_trades_all += tr
            
            # Store stats for this key
            current_breakdown[k] = f"ROI:{roi*100:.1f}%(PF:{pf:.1f})"

        # --- EVALUATE UNIVERSAL SCORE ---
        if not is_universal_failure and total_trades_all > MIN_TRADES_TOTAL:
            
            # Final Score is simply the sum of weighted ROIs
            if total_weighted_roi > best_weighted_score:
                best_weighted_score = total_weighted_roi
                best_p = {
                    'SMA': s_ma, 'LMA': l_ma, 'Super': sl_ma,
                    'Trail%': t_pct, 'Tgt%': tgt_pct
                }
                best_breakdown = current_breakdown
                best_breakdown['Total_Trades'] = total_trades_all

    # ==========================================
    # 6. RESULTS DISPLAY (SAFEGUARDED)
    # ==========================================
    print("\n" + "="*60)
    if not best_p:
        print("❌ NO UNIVERSAL STRATEGY FOUND.")
        print("The criteria were too strict. No single parameter set worked across all 12 charts.")
        print("Try: Lowering weights, reducing 'Universality Check' strictness, or checking data.")
    else:
        print(f"💎 UNIVERSAL PARAMETER SET FOUND")
        print(f"   Score (Weighted ROI): {best_weighted_score:.4f}")
        print(f"   Parameters: SMA {best_p['SMA']} / LMA {best_p['LMA']} | SuperTrend EMA {best_p['Super']}")
        print(f"   Risk Mgmt : Trail {best_p['Trail%']:.2f}% | Target {best_p['Tgt%']:.2f}%")
        print("-" * 60)
        print("PERFORMANCE BREAKDOWN:")
        
        # Pretty print the breakdown by Timeframe
        sorted_keys = sorted(best_breakdown.keys())
        for k in sorted_keys:
            if k == 'Total_Trades': continue
            print(f"   {k: <15} : {best_breakdown[k]}")
            
        print(f"\n   Total Trades Executed: {best_breakdown['Total_Trades']}")
    print("="*60)

if __name__ == "__main__":
    run_universal_optimization()