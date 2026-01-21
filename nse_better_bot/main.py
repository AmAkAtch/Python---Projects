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
# 1. ROBUST CONFIGURATION
# ==========================================

NIFTY_UNIVERSE = [
    # --- GIANTS ---
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'ITC.NS', 'SBIN.NS', 
    'BHARTIARTL.NS', 'LT.NS', 'HINDUNILVR.NS', 'TATAMOTORS.NS', 'M&M.NS', 'MARUTI.NS',
    'SUNPHARMA.NS', 'TITAN.NS', 'BAJFINANCE.NS', 'NTPC.NS', 'POWERGRID.NS', 'ULTRACEMCO.NS',
    
    # --- MIDCAP & MOMENTUM ---
    'TRENT.NS', 'BEL.NS', 'HAL.NS', 'POLYCAB.NS', 'DIXON.NS', 'COFORGE.NS', 'LTIM.NS',
    'ZOMATO.NS', 'VBL.NS', 'ABB.NS', 'SIEMENS.NS', 'TVSMOTOR.NS', 'CHOLAFIN.NS',
    'PERSISTENT.NS', 'APOLLOHOSP.NS', 'DLF.NS', 'JINDALSTEL.NS', 'VEDL.NS', 'HAVELLS.NS',
    'TATACHEM.NS', 'VOLTAS.NS', 'ESCORTS.NS', 'CUMMINSIND.NS', 'TORNTPHARM.NS'
]

START_DATE = "2015-01-01"
INITIAL_CAPITAL = 10000.0   
NUM_ITERATIONS = 100000      
MIN_TRADES_REQUIRED = 50     

COST_PCT = 0.0020  # 0.20% Slippage/Tax

# --- SCORING LOGIC SETTINGS (IMPORTED FROM SCRIPT 1) ---
DECAY_RATE = 0.10          # 10% decay per year (High recency bias)
ROBUSTNESS_THRESHOLD = 0.70 # Neighbors must be 70% as good as the best
MAX_YEAR_DRAWDOWN = -0.15   # Kill Switch: If any year loses >15%, discard.

if not os.path.exists("data"):
    os.makedirs("data")

# ==========================================
# 2. DATA LOADING (WITH YEAR PRE-PROCESSING)
# ==========================================
def get_stock_data(ticker):
    file_path = f"data/{ticker}.csv"
    if os.path.exists(file_path):
        try: return pd.read_csv(file_path)
        except: pass
    try:
        df = yf.download(ticker, start=START_DATE, progress=False, multi_level_index=False)
        if len(df) > 500:
            df.to_csv(file_path)
            return df
    except Exception: return None
    return None

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
# 4. BACKTEST LOGIC (YEARLY PNL TRACKING)
# ==========================================
@njit(fastmath=True)
def backtest_logic_yearly(opens, highs, lows, closes, years,
                          short_ma, long_ma, super_ma, 
                          use_trailing, trail_pct, filter_mode,
                          min_year): 
    
    # Yearly PnL Bucket (Index 0 = min_year, Index 1 = min_year+1...)
    # Assuming max 20 years of history
    yearly_pnl = np.zeros(25, dtype=np.float64) 
    
    n = len(closes)
    in_pos = False
    entry_price = 0.0
    high_since_entry = 0.0
    
    trades = 0
    wins = 0
    
    # Start loop
    start_idx = 0
    for i in range(n):
        if not np.isnan(super_ma[i]) and not np.isnan(long_ma[i]):
            start_idx = i; break
    if start_idx == 0: start_idx = 200
            
    for i in range(start_idx, n - 1):
        # Determine Current Year Index
        yr_idx = years[i+1] - min_year
        if yr_idx < 0: yr_idx = 0
        
        next_open = opens[i+1]
        next_low = lows[i+1]
        next_high = highs[i+1]
        curr_close = closes[i]
        
        # --- EXIT LOGIC ---
        if in_pos:
            high_since_entry = max(high_since_entry, next_high)
            should_exit = False
            exit_price = 0.0
            
            # Trailing Stop
            if use_trailing:
                stop_price = high_since_entry * (1.0 - trail_pct / 100.0)
                if next_low <= stop_price:
                    should_exit = True
                    exit_price = next_open if next_open < stop_price else stop_price
            
            # Crossover Exit
            if not should_exit:
                if short_ma[i-1] >= long_ma[i-1] and short_ma[i] < long_ma[i]:
                    should_exit = True
                    exit_price = next_open
            
            if should_exit:
                # Calculate PnL
                gross_pnl_pct = (exit_price - entry_price) / entry_price
                net_pnl_pct = gross_pnl_pct - COST_PCT
                
                # Assume fixed bet size per trade (simulating equal weight portfolio)
                # Logic: We track % returns per year, not absolute dollars to normalize across stocks
                yearly_pnl[yr_idx] += net_pnl_pct
                
                trades += 1
                if net_pnl_pct > 0: wins += 1
                
                in_pos = False
                high_since_entry = 0.0

        # --- ENTRY LOGIC ---
        elif not in_pos:
            crossover = short_ma[i-1] <= long_ma[i-1] and short_ma[i] > long_ma[i]
            
            trend_ok = False
            if filter_mode == 0: trend_ok = True
            elif filter_mode == 1: 
                if curr_close > super_ma[i]: trend_ok = True
            elif filter_mode == 2: 
                if curr_close < super_ma[i]: trend_ok = True
            
            if crossover and trend_ok:
                in_pos = True
                entry_price = next_open
                high_since_entry = next_open

    return yearly_pnl, trades, wins

# ==========================================
# 5. CORE EVALUATOR (THE BRIDGE)
# ==========================================
def evaluate_params(p, data_store, yearly_weights, min_year, max_year):
    """
    Runs the parameters on ALL stocks and aggregates the Yearly PnL.
    Applies Kill Switch and Recency Weighting.
    """
    
    # 1. Architecture Decoding
    if p['s_ma'] >= p['l_ma']: return -999, {}, []
    
    total_trades_all = 0
    total_wins_all = 0
    
    # Aggregate Portfolio PnL per Year (Summing % returns of all stocks)
    # This represents the return of an equal-weighted portfolio of these stocks
    global_yearly_pnl = np.zeros(25, dtype=np.float64) 
    
    valid_stocks = 0
    
    for ticker, data in data_store.items():
        closes = data['close']
        
        # Indicator Switching
        if p['t_s'] == 0: arr_s = calc_sma(closes, p['s_ma'])
        else: arr_s = calc_ema(closes, p['s_ma'])
            
        if p['t_l'] == 0: arr_l = calc_sma(closes, p['l_ma'])
        else: arr_l = calc_ema(closes, p['l_ma'])
            
        if p['t_sl'] == 0: arr_sl = calc_sma(closes, p['sl_ma'])
        else: arr_sl = calc_ema(closes, p['sl_ma'])
        
        y_pnl, tr, w = backtest_logic_yearly(
            data['open'], data['high'], data['low'], closes, data['years'],
            arr_s, arr_l, arr_sl, 
            p['use_tr'], p['tr_pct'], p['f_mode'], 
            min_year
        )
        
        if tr > 0:
            global_yearly_pnl += y_pnl
            total_trades_all += tr
            total_wins_all += w
            valid_stocks += 1

    if valid_stocks == 0 or total_trades_all < MIN_TRADES_REQUIRED:
        return -999, {}, []

    # Normalize Yearly PnL by number of stocks (Average Return per Stock per Year)
    # This keeps the numbers relatable (e.g., 0.20 = 20% avg return)
    global_yearly_pnl = global_yearly_pnl / valid_stocks
    
    # --- SCORING LOGIC FROM SCRIPT 1 ---
    
    # 1. Kill Switch: Check Recent Years for Drawdown
    # Check years from (max_year - 5) to max_year
    recent_start_idx = max_year - 5 - min_year
    recent_end_idx = max_year - min_year
    
    for i in range(max(0, recent_start_idx), recent_end_idx + 1):
        if global_yearly_pnl[i] < MAX_YEAR_DRAWDOWN:
            return -999, {}, [] # Immediate Rejection (Too risky)

    # 2. Recency Weighted Score
    weighted_score = 0.0
    for y in range(min_year, max_year + 1):
        idx = y - min_year
        if 0 <= idx < 25:
            weighted_score += (global_yearly_pnl[idx] * yearly_weights.get(y, 0))

    metrics = {
        'trades': total_trades_all,
        'win_rate': total_wins_all / total_trades_all if total_trades_all > 0 else 0,
        'yearly_pnl': global_yearly_pnl
    }
    
    return weighted_score, metrics, global_yearly_pnl

# ==========================================
# 6. NEIGHBOR GENERATOR (ROBUSTNESS)
# ==========================================
def get_neighbors(p):
    neighbors = []
    
    # Neighbor 1: Slightly Faster MAs
    n1 = p.copy()
    n1['s_ma'] = max(5, int(p['s_ma'] * 0.95))
    n1['l_ma'] = max(10, int(p['l_ma'] * 0.95))
    neighbors.append(n1)
    
    # Neighbor 2: Slightly Slower MAs
    n2 = p.copy()
    n2['s_ma'] = int(p['s_ma'] * 1.05)
    n2['l_ma'] = int(p['l_ma'] * 1.05)
    neighbors.append(n2)
    
    # Neighbor 3: Tighter Trail
    if p['use_tr']:
        n3 = p.copy()
        n3['tr_pct'] = max(1.0, p['tr_pct'] - 2.0)
        neighbors.append(n3)
    
    # Neighbor 4: Looser Trail
    if p['use_tr']:
        n4 = p.copy()
        n4['tr_pct'] = p['tr_pct'] + 2.0
        neighbors.append(n4)
        
    return neighbors

# ==========================================
# 7. MAIN OPTIMIZATION LOOP
# ==========================================
def run_robust_optimization():
    print("--- 1. Loading & Aligning Data ---")
    
    data_store = {}
    all_years = []
    
    for ticker in tqdm(NIFTY_UNIVERSE):
        df = get_stock_data(ticker)
        if df is not None and len(df) > 500:
            df['Year'] = pd.to_datetime(df.index).year
            all_years.extend(df['Year'].unique())
            
            data_store[ticker] = {
                'open': np.ascontiguousarray(df['Open'].values.flatten()),
                'high': np.ascontiguousarray(df['High'].values.flatten()),
                'low': np.ascontiguousarray(df['Low'].values.flatten()),
                'close': np.ascontiguousarray(df['Close'].values.flatten()),
                'years': np.ascontiguousarray(df['Year'].values.flatten(), dtype=np.int32)
            }
            
    min_year = min(all_years)
    max_year = max(all_years)
    print(f"Loaded {len(data_store)} stocks. History: {min_year} - {max_year}")
    
    # Pre-calc Weights
    yearly_weights = {}
    print("\n--- Recency Weights ---")
    for y in range(min_year, max_year + 1):
        age = max_year - y
        weight = 1.0 / ((1 + DECAY_RATE) ** age)
        yearly_weights[y] = weight
        # print(f"{y}: {weight:.2f}")

    print(f"\n--- 2. Starting Robust Search ({NUM_ITERATIONS} iter) ---")
    print(f"Logic: Kill Switch < {MAX_YEAR_DRAWDOWN*100}% | Neighbor Check > {ROBUSTNESS_THRESHOLD*100}%")

    best_score = -9999
    best_params = None
    
    # Random Seeds
    r_short = np.random.randint(5, 61, NUM_ITERATIONS)
    r_long = np.random.randint(10, 91, NUM_ITERATIONS)
    r_super = np.random.randint(50, 301, NUM_ITERATIONS)
    r_pct = np.random.uniform(2.0, 25.0, NUM_ITERATIONS) # Tighter range for equities
    r_use = np.random.randint(0, 2, NUM_ITERATIONS)
    r_type_s = np.random.randint(0, 2, NUM_ITERATIONS)
    r_type_l = np.random.randint(0, 2, NUM_ITERATIONS)
    r_type_sl = np.random.randint(0, 2, NUM_ITERATIONS)
    r_filt = np.random.randint(0, 3, NUM_ITERATIONS)

    for i in tqdm(range(NUM_ITERATIONS)):
        
        # 1. Candidate Generation
        p = {
            's_ma': r_short[i], 'l_ma': r_long[i], 'sl_ma': r_super[i],
            't_s': r_type_s[i], 't_l': r_type_l[i], 't_sl': r_type_sl[i],
            'use_tr': bool(r_use[i]), 'tr_pct': r_pct[i], 'f_mode': r_filt[i]
        }
        
        score, metrics, yearly_arr = evaluate_params(p, data_store, yearly_weights, min_year, max_year)
        
        # 2. Initial Filter
        if score > best_score:
            
            # 3. THE NEIGHBOR TEST (Validation)
            neighbors = get_neighbors(p)
            neighbor_scores = []
            
            for np_params in neighbors:
                n_score, _, _ = evaluate_params(np_params, data_store, yearly_weights, min_year, max_year)
                if n_score > -900: # If it didn't hit the Kill Switch
                    neighbor_scores.append(n_score)
            
            # Calculate Stability
            if len(neighbor_scores) > 0:
                avg_n_score = sum(neighbor_scores) / len(neighbor_scores)
                # Avoid division by zero if score is tiny
                if abs(score) < 0.001: ratio = 0
                else: ratio = avg_n_score / score
                
                # 4. Confirmation
                if ratio >= ROBUSTNESS_THRESHOLD:
                    best_score = score
                    best_params = p
                    
                    # Formatting Output
                    t_s_str = "EMA" if p['t_s'] else "SMA"
                    t_l_str = "EMA" if p['t_l'] else "SMA"
                    f_str = ["OFF", "Price > Super", "Price < Super"][p['f_mode']]
                    tr_str = f"{p['tr_pct']:.1f}%" if p['use_tr'] else "OFF"
                    
                    # Yearly Breakdown String (Last 5 years)
                    recent_years_str = ""
                    for y in range(max_year-4, max_year+1):
                        idx = y - min_year
                        if idx >= 0:
                            val = yearly_arr[idx] * 100
                            recent_years_str += f"{y}:{val:.0f}% | "
                    
                    tqdm.write(f"\n🌟 ROBUST FIND! Score: {score:.2f} (Stability: {ratio*100:.0f}%)")
                    tqdm.write(f"Params: {t_s_str}{p['s_ma']} / {t_l_str}{p['l_ma']} | Filter: {f_str}")
                    tqdm.write(f"Risk: Trail {tr_str} | Trades: {metrics['trades']}")
                    tqdm.write(f"Recent Returns: {recent_years_str}")

    print("\n" + "="*60)
    print("🏆 FINAL ROBUST PARAMETERS 🏆")
    print("="*60)
    print(best_params)

if __name__ == "__main__":
    run_robust_optimization()