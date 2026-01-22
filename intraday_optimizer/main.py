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
    'NIFTY':        {'ticker': '^NSEI',               'expiry_day': 3}, # Thursday
    'BANKNIFTY':    {'ticker': '^NSEBANK',            'expiry_day': 2}, # Wednesday
    'FINNIFTY':     {'ticker': 'NIFTY_FIN_SERVICE.NS', 'expiry_day': 1}, # Tuesday
    'MIDCPNIFTY':   {'ticker': 'NIFTY_MID_SELECT.NS',  'expiry_day': 0}  # Monday
}

TIMEFRAMES = ['5m', '15m']
TF_WEIGHTS = {'5m': 1.5, '15m': 1.0}

# Strategy Constraints
NUM_ITERATIONS = 50000       
EXIT_TIME_MINUTES = 920      # 15:20 PM
MIN_TRADES_REQUIRED = 30

# OPTIONS SETTINGS (The "Synthetic" Model)
OPTION_COST_PCT = 0.001     # Higher costs (0.1%) for Options (Spread + Brokerage)
DELTA = 0.55                # Avg Delta for ATM/ITM winning trades
THETA_FACTOR = 0.05         # Multiplier for time decay intensity

if not os.path.exists("data_universal"):
    os.makedirs("data_universal")

# ==========================================
# 2. DATA LOADING (Standard Spot Data)
# ==========================================
def clean_yfinance_data(df):
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        try: df.columns = df.columns.droplevel(1) 
        except: pass
    df.dropna(inplace=True)
    return df

def fetch_data_batch():
    data_store = {}
    print("\n--- 1. Fetching Spot Data for Option Simulation ---")
    
    for idx_name, info in INDICES.items():
        ticker = info['ticker']
        for tf in TIMEFRAMES:
            file_path = f"data_universal/{idx_name}_{tf}.csv"
            df = None
            
            # Local Cache
            if os.path.exists(file_path):
                try: df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
                except: df = None

            # Download
            if df is None or len(df) < 100:
                try:
                    df = yf.download(ticker, period="59d", interval=tf, progress=False)
                    df = clean_yfinance_data(df)
                    if not df.empty and len(df) > 100: df.to_csv(file_path)
                except: pass

            if df is not None and len(df) > 200:
                times_min = df.index.hour.values * 60 + df.index.minute.values
                weekdays = df.index.weekday.values
                
                data_store[f"{idx_name}|{tf}"] = {
                    'opens': np.ascontiguousarray(df['Open'].values, dtype=np.float64),
                    'highs': np.ascontiguousarray(df['High'].values, dtype=np.float64),
                    'lows': np.ascontiguousarray(df['Low'].values, dtype=np.float64),
                    'closes': np.ascontiguousarray(df['Close'].values, dtype=np.float64),
                    'times': np.ascontiguousarray(times_min, dtype=np.int64),
                    'weekdays': np.ascontiguousarray(weekdays, dtype=np.int64),
                    'expiry_day': info['expiry_day'],
                    'tf_weight': TF_WEIGHTS[tf],
                    'interval_mins': 5 if tf == '5m' else 15
                }
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
# 4. SYNTHETIC OPTION BACKTEST ENGINE
# ==========================================
@njit(fastmath=True)
def backtest_options_sim(opens, highs, lows, closes, 
                         weekdays, times_min, expiry_day_idx, interval_mins,
                         short_ma, long_ma, super_ma, 
                         trail_pct, target_pct): 
    
    n = len(closes)
    # ROI is now based on OPTION PREMIUM, not Spot Price
    total_option_roi = 0.0
    
    in_pos = False 
    entry_spot_price = 0.0
    high_spot_since_entry = 0.0
    
    # Option Simulation Vars
    entry_premium_est = 0.0
    days_to_expiry_entry = 0
    entry_time_idx = 0
    
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
            high_spot_since_entry = max(high_spot_since_entry, curr_high)
            should_exit = False
            exit_spot_price = 0.0
            
            # 1. Expiry Hard Exit
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
                # =========================================
                # 💰 CALCULATE OPTION PnL
                # =========================================
                
                # A. Spot Movement Points
                spot_points = exit_spot_price - entry_spot_price
                
                # B. Delta Profit (Assuming 0.55 Delta)
                option_gross_pts = spot_points * DELTA
                
                # C. Theta Decay (Time Penalty)
                # Calculate minutes held
                mins_held = (i - entry_time_idx) * interval_mins
                if mins_held < 0: mins_held = 5 # Safety
                
                # Decay Rate depends on DTE (Closer to expiry = Higher decay)
                # DTE 4 (Far) -> Decay Factor 1.0
                # DTE 0 (Expiry) -> Decay Factor 3.0
                dte_decay_mult = 3.0 - (days_to_expiry_entry * 0.4) 
                if dte_decay_mult < 1.0: dte_decay_mult = 1.0
                
                # Decay formula: Premium * (Small Factor) * Multiplier * Duration
                theta_loss = entry_premium_est * (0.0001 * dte_decay_mult) * mins_held
                
                # D. Net Option Profit
                net_option_pts = option_gross_pts - theta_loss
                
                # E. ROI Calculation
                # ROI = Net Pts / Capital Deployed (Premium) - Transaction Costs
                trade_roi = (net_option_pts / entry_premium_est) - OPTION_COST_PCT
                
                total_option_roi += trade_roi
                
                if trade_roi > 0: 
                    wins += 1
                    gross_win_pct += trade_roi
                else:
                    gross_loss_pct += abs(trade_roi)
                    
                trades += 1
                in_pos = False

        # --- ENTRY LOGIC ---
        elif not in_pos:
            if 555 < curr_time < 900: 
                crossover = short_ma[i-1] <= long_ma[i-1] and short_ma[i] > long_ma[i]
                trend_ok = closes[i] > super_ma[i]
                
                if crossover and trend_ok:
                    in_pos = True
                    entry_spot_price = curr_open
                    high_spot_since_entry = curr_open
                    entry_time_idx = i
                    
                    # 🧮 ESTIMATE ATM OPTION PREMIUM
                    # Logic: Calculate DTE (Days To Expiry)
                    # Expiry=3 (Thu), Today=1 (Tue) -> DTE = 2
                    dte = (expiry_day_idx - curr_day)
                    if dte < 0: dte += 7 # Wrap around if data includes prev week
                    days_to_expiry_entry = dte
                    
                    # Premium Estimation Model (Approximation)
                    # DTE 0: 0.3% of Spot | DTE 4: 1.1% of Spot
                    premium_pct = 0.003 + (0.002 * dte) 
                    entry_premium_est = curr_open * premium_pct

    return total_option_roi, trades, wins, gross_win_pct, gross_loss_pct

# ==========================================
# 5. UNIVERSAL OPTIMIZER (OPTION MODE)
# ==========================================
def run_universal_optimization():
    
    data_map = fetch_data_batch()
    if not data_map:
        print("❌ No data loaded.")
        return

    keys = list(data_map.keys()) 
    print(f"\n--- 2. Optimization Started (Option Simulation Mode) ---")
    print("   * PnL assumes ATM Option Buy")
    print("   * Includes simulated Delta (0.55) and Theta Decay")
    
    best_weighted_score = -99999
    best_p = {}
    best_breakdown = {}

    r_short = np.random.randint(5, 40, NUM_ITERATIONS)
    r_long = np.random.randint(10, 80, NUM_ITERATIONS)
    r_super = np.random.randint(50, 300, NUM_ITERATIONS)
    r_trail = np.random.uniform(0.1, 1.5, NUM_ITERATIONS)
    r_target = np.random.uniform(0.5, 4.0, NUM_ITERATIONS)

    for i in tqdm(range(NUM_ITERATIONS), desc="Simulating Options"):
        s_ma = r_short[i]
        l_ma = r_long[i]
        sl_ma = r_super[i]
        t_pct = r_trail[i]
        tgt_pct = r_target[i]

        if s_ma >= l_ma: continue
        
        total_weighted_roi = 0.0
        total_trades_all = 0
        is_fail = False
        current_breakdown = {}

        for k in keys:
            d = data_map[k]
            
            arr_s = calc_ema(d['closes'], s_ma)
            arr_l = calc_ema(d['closes'], l_ma)
            arr_sl = calc_ema(d['closes'], sl_ma)
            
            roi, tr, wins, g_win, g_loss = backtest_options_sim(
                d['opens'], d['highs'], d['lows'], d['closes'],
                d['weekdays'], d['times'], d['expiry_day'], d['interval_mins'],
                arr_s, arr_l, arr_sl, t_pct, tgt_pct
            )
            
            # Option Strategy Filter: Needs higher Profit Factor to justify risk
            pf = g_win / g_loss if g_loss > 0 else 1.5
            if pf < 0.8: # Options are risky, reject anything with < 0.8 PF on any chart
                is_fail = True
                break
            
            total_weighted_roi += (roi * d['tf_weight'])
            total_trades_all += tr
            current_breakdown[k] = f"ROI:{roi*100:.1f}%(PF:{pf:.1f})"

        if not is_fail and total_trades_all > MIN_TRADES_REQUIRED:
            if total_weighted_roi > best_weighted_score:
                best_weighted_score = total_weighted_roi
                best_p = {'SMA': s_ma, 'LMA': l_ma, 'Super': sl_ma, 'Trail': t_pct, 'Tgt': tgt_pct}
                best_breakdown = current_breakdown
                best_breakdown['Total_Trades'] = total_trades_all

    print("\n" + "="*60)
    if not best_p:
        print("❌ No Universal Option Strategy Found.")
        print("   Options decay killed all random strategies. Try relaxing filters.")
    else:
        print(f"💎 BEST UNIVERSAL OPTION STRATEGY")
        print(f"   Score (Weighted Option ROI): {best_weighted_score:.4f}")
        print(f"   Settings: SMA {best_p['SMA']} / LMA {best_p['LMA']} | Filter {best_p['Super']}")
        print(f"   Risk: Spot Trail {best_p['Trail']:.2f}% | Spot Target {best_p['Tgt']:.2f}%")
        print("-" * 60)
        for k in sorted(best_breakdown.keys()):
            if k != 'Total_Trades': print(f"   {k: <15} : {best_breakdown[k]}")
        print(f"   Total Trades: {best_breakdown['Total_Trades']}")
    print("="*60)

if __name__ == "__main__":
    run_universal_optimization()