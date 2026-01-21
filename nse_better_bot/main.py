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
# 1. CONFIGURATION (Nifty 200 Universe)
# ==========================================
NIFTY_UNIVERSE = [
    # --- GIANTS (Nifty 50) ---
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS', 'HINDUNILVR.NS', 
    'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'BAJFINANCE.NS', 'KOTAKBANK.NS', 'LT.NS', 
    'HCLTECH.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'MARUTI.NS', 'TITAN.NS', 'SUNPHARMA.NS', 
    'BAJAJFINSV.NS', 'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'ONGC.NS', 'NTPC.NS', 
    'POWERGRID.NS', 'JSWSTEEL.NS', 'TATASTEEL.NS', 'M&M.NS', 'TECHM.NS', 'HDFCLIFE.NS', 
    'ADANIENT.NS', 'ADANIPORTS.NS', 'COALINDIA.NS', 'TATAMOTORS.NS', 'GRASIM.NS', 
    'DIVISLAB.NS', 'SBILIFE.NS', 'DRREDDY.NS', 'CIPLA.NS', 'BPCL.NS', 'BRITANNIA.NS', 
    'EICHERMOT.NS', 'INDUSINDBK.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'TATACONSUM.NS', 
    'UPL.NS', 'APOLLOHOSP.NS', 'PIDILITIND.NS', 
    
    # --- MIDCAP / NEXT 50 FAVORITES ---
    'GODREJCP.NS', 'DABUR.NS', 'SHREECEM.NS', 'VEDL.NS', 'DLF.NS', 'HAVELLS.NS', 'SRF.NS', 
    'ICICIPRULI.NS', 'JINDALSTEL.NS', 'GAIL.NS', 'AMBUJACEM.NS', 'CHOLAFIN.NS', 'BERGEPAINT.NS', 
    'BANKBARODA.NS', 'CANBK.NS', 'BEL.NS', 'SIEMENS.NS', 'BOSCHLTD.NS', 'MCDOWELL-N.NS', 
    'MARICO.NS', 'IOC.NS', 'TORNTPHARM.NS', 'PIIND.NS', 'NAUKRI.NS', 'HAL.NS', 'TRENT.NS', 
    'TVSMOTOR.NS', 'ZOMATO.NS', 'VBL.NS', 'ABB.NS', 'PAGEIND.NS', 'COLPAL.NS',
    
    # --- HIGH MOMENTUM MIDCAPS (The "Juice" for Moving Averages) ---
    'POLYCAB.NS', 'DIXON.NS', 'PERSISTENT.NS', 'LTIM.NS', 'KPITTECH.NS', 'COFORGE.NS',
    'ASTRAL.NS', 'BALKRISIND.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'ASHOKLEY.NS',
    'CUMMINSIND.NS', 'OBEROIRLTY.NS', 'ESCORTS.NS', 'JUBLFOOD.NS', 'MRF.NS', 'MUTHOOTFIN.NS',
    'PETRONET.NS', 'PFC.NS', 'RECLTD.NS', 'SHRIRAMFIN.NS', 'TATACHEM.NS', 'TATAPOWER.NS',
    'VOLTAS.NS', 'ZEEL.NS', 'AUROPHARMA.NS', 'LUPIN.NS', 'ALKEM.NS', 'BHEL.NS', 'SAIL.NS'
]

START_DATE = "2015-01-01"
NUM_ITERATIONS = 100000      
MIN_TRADES_REQUIRED = 50     
COST_PCT = 0.0020  # 0.20% Slippage/Tax

# SCORING SETTINGS (Crypto Logic)
DECAY_RATE = 0.10          # High recency bias
ROBUSTNESS_THRESHOLD = 0.70 
MAX_YEAR_DRAWDOWN = -0.15   # Kill switch

if not os.path.exists("data"):
    os.makedirs("data")

# ==========================================
# 2. DATA LOADING (BUG FIXED 🔧)
# ==========================================
def get_stock_data(ticker):
    file_path = f"data/{ticker}.csv"
    
    # Try loading from CSV first
    if os.path.exists(file_path):
        try:
            # FIX: Explicitly tell Pandas to parse the 'Date' column and make it the index
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            return df
        except Exception as e:
            pass # If fail, re-download

    # Download if missing or broken
    try:
        df = yf.download(ticker, start=START_DATE, progress=False, multi_level_index=False)
        if len(df) > 200:
            df.to_csv(file_path) # Saves with 'Date' column
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
# 4. BACKTEST LOGIC
# ==========================================
@njit(fastmath=True)
def backtest_logic_yearly(opens, highs, lows, closes, years,
                          short_ma, long_ma, super_ma, 
                          use_trailing, trail_pct, filter_mode,
                          min_year): 
    
    # PnL array: Index 0 = min_year, Index 1 = min_year+1...
    yearly_pnl = np.zeros(30, dtype=np.float64) 
    
    n = len(closes)
    in_pos = False
    entry_price = 0.0
    high_since_entry = 0.0
    trades = 0
    wins = 0
    
    start_idx = 0
    for i in range(n):
        if not np.isnan(super_ma[i]) and not np.isnan(long_ma[i]):
            start_idx = i; break
    if start_idx == 0: start_idx = 200
            
    for i in range(start_idx, n - 1):
        # Calculate Year Index
        yr_idx = years[i+1] - min_year
        if yr_idx < 0: yr_idx = 0
        if yr_idx >= 30: yr_idx = 29
        
        # --- EXIT ---
        if in_pos:
            high_since_entry = max(high_since_entry, highs[i+1])
            should_exit = False
            exit_price = 0.0
            
            # Trailing Stop
            if use_trailing:
                stop_price = high_since_entry * (1.0 - trail_pct / 100.0)
                if lows[i+1] <= stop_price:
                    should_exit = True
                    exit_price = opens[i+1] if opens[i+1] < stop_price else stop_price
            
            # Crossover Exit
            if not should_exit:
                if short_ma[i-1] >= long_ma[i-1] and short_ma[i] < long_ma[i]:
                    should_exit = True
                    exit_price = opens[i+1]
            
            if should_exit:
                pnl = (exit_price - entry_price) / entry_price - COST_PCT
                yearly_pnl[yr_idx] += pnl
                trades += 1
                if pnl > 0: wins += 1
                in_pos = False
                high_since_entry = 0.0

        # --- ENTRY ---
        elif not in_pos:
            crossover = short_ma[i-1] <= long_ma[i-1] and short_ma[i] > long_ma[i]
            trend_ok = True
            if filter_mode == 1 and closes[i] <= super_ma[i]: trend_ok = False
            if filter_mode == 2 and closes[i] >= super_ma[i]: trend_ok = False
            
            if crossover and trend_ok:
                in_pos = True
                entry_price = opens[i+1]
                high_since_entry = opens[i+1]

    return yearly_pnl, trades, wins

# ==========================================
# 5. CORE EVALUATOR (MEDIAN + ROBUSTNESS)
# ==========================================
def evaluate_params(p, data_store, yearly_weights, min_year, max_year):
    if p['s_ma'] >= p['l_ma']: return -999, {}, []
    
    stock_scores = []
    total_trades_all = 0
    
    # We also track the "Global Portfolio" just for the Kill Switch check
    portfolio_yearly_pnl = np.zeros(30, dtype=np.float64) 
    valid_stocks_count = 0

    for ticker, data in data_store.items():
        closes = data['close']
        if p['t_s'] == 0: arr_s = calc_sma(closes, p['s_ma'])
        else: arr_s = calc_ema(closes, p['s_ma'])
        
        if p['t_l'] == 0: arr_l = calc_sma(closes, p['l_ma'])
        else: arr_l = calc_ema(closes, p['l_ma'])
        
        if p['t_sl'] == 0: arr_sl = calc_sma(closes, p['sl_ma'])
        else: arr_sl = calc_ema(closes, p['sl_ma'])
        
        y_pnl, tr, _ = backtest_logic_yearly(
            data['open'], data['high'], data['low'], closes, data['years'],
            arr_s, arr_l, arr_sl, 
            p['use_tr'], p['tr_pct'], p['f_mode'], min_year
        )
        
        if tr > 0:
            total_trades_all += tr
            portfolio_yearly_pnl += y_pnl
            valid_stocks_count += 1
            
            # --- INDIVIDUAL STOCK SCORING (The "Universal Soldier" Logic) ---
            # Calculate score for THIS stock using Recency Weights
            this_score = 0.0
            for y in range(min_year, max_year + 1):
                idx = y - min_year
                if 0 <= idx < 30:
                    this_score += (y_pnl[idx] * yearly_weights.get(y, 0))
            
            stock_scores.append(this_score)

    if len(stock_scores) < 10 or total_trades_all < MIN_TRADES_REQUIRED:
        return -999, {}, []

    # 1. KILL SWITCH (Portfolio Level)
    # Check if the "Average Portfolio" crashed in any recent year
    avg_portfolio_pnl = portfolio_yearly_pnl / valid_stocks_count
    
    recent_start = max_year - 5 - min_year
    for i in range(max(0, recent_start), max_year - min_year + 1):
        if avg_portfolio_pnl[i] < MAX_YEAR_DRAWDOWN:
            return -999, {}, []

    # 2. FINAL SCORE = MEDIAN of Individual Scores
    # This prevents 'Hero Stocks' from faking the results.
    final_score = np.median(stock_scores)
    
    metrics = {
        'trades': total_trades_all,
        'yearly_pnl': avg_portfolio_pnl
    }
    
    return final_score, metrics, avg_portfolio_pnl

# ==========================================
# 6. NEIGHBOR GENERATOR
# ==========================================
def get_neighbors(p):
    neighbors = []
    # Tweaks for neighbors
    n1 = p.copy(); n1['s_ma'] = max(5, int(p['s_ma'] * 0.95)); n1['l_ma'] = max(10, int(p['l_ma'] * 0.95)); neighbors.append(n1)
    n2 = p.copy(); n2['s_ma'] = int(p['s_ma'] * 1.05); n2['l_ma'] = int(p['l_ma'] * 1.05); neighbors.append(n2)
    return neighbors

# ==========================================
# 7. MAIN
# ==========================================
def run_robust_optimization():
    print("--- 1. Loading Data (With Date Fix) ---")
    
    data_store = {}
    all_years = []
    
    for ticker in tqdm(NIFTY_UNIVERSE):
        df = get_stock_data(ticker)
        if df is not None and len(df) > 500:
            df['Year'] = df.index.year # NOW THIS WORKS CORRECTLY
            all_years.extend(df['Year'].unique())
            
            data_store[ticker] = {
                'open': np.ascontiguousarray(df['Open'].values),
                'high': np.ascontiguousarray(df['High'].values),
                'low': np.ascontiguousarray(df['Low'].values),
                'close': np.ascontiguousarray(df['Close'].values),
                'years': np.ascontiguousarray(df['Year'].values, dtype=np.int32)
            }
            
    min_year = int(min(all_years))
    max_year = int(max(all_years))
    print(f"Loaded {len(data_store)} stocks. Time: {min_year} -> {max_year}")
    
    yearly_weights = {}
    for y in range(min_year, max_year + 1):
        age = max_year - y
        yearly_weights[y] = 1.0 / ((1 + DECAY_RATE) ** age)

    print(f"\n--- 2. Optimization (Median Logic + Kill Switch) ---")
    best_score = -9999
    
    # Pre-generate random params
    r_short = np.random.randint(5, 61, NUM_ITERATIONS)
    r_long = np.random.randint(10, 100, NUM_ITERATIONS)
    r_super = np.random.randint(50, 250, NUM_ITERATIONS)
    r_pct = np.random.uniform(3.0, 20.0, NUM_ITERATIONS)
    r_use = np.random.randint(0, 2, NUM_ITERATIONS)
    r_filt = np.random.randint(0, 3, NUM_ITERATIONS)
    r_type = np.random.randint(0, 2, (NUM_ITERATIONS, 3)) # s, l, sl

    for i in tqdm(range(NUM_ITERATIONS)):
        p = {
            's_ma': r_short[i], 'l_ma': r_long[i], 'sl_ma': r_super[i],
            't_s': r_type[i][0], 't_l': r_type[i][1], 't_sl': r_type[i][2],
            'use_tr': bool(r_use[i]), 'tr_pct': r_pct[i], 'f_mode': r_filt[i]
        }
        
        score, metrics, yearly_arr = evaluate_params(p, data_store, yearly_weights, min_year, max_year)
        
        if score > best_score:
            # Neighbor Check
            neighbors = get_neighbors(p)
            n_scores = []
            for n_p in neighbors:
                ns, _, _ = evaluate_params(n_p, data_store, yearly_weights, min_year, max_year)
                if ns > -900: n_scores.append(ns)
            
            stability = (sum(n_scores)/len(n_scores))/score if (n_scores and abs(score)>0.001) else 0
            
            if stability >= ROBUSTNESS_THRESHOLD:
                best_score = score
                
                # Format Output
                f_str = ["OFF", "Price > Super", "Price < Super"][p['f_mode']]
                ts = "EMA" if p['t_s'] else "SMA"
                tl = "EMA" if p['t_l'] else "SMA"
                
                # Recent Years Display (Last 6 years)
                rec_str = ""
                display_years = range(max_year-5, max_year+1)
                for y in display_years:
                    idx = y - min_year
                    if idx >= 0:
                        val = yearly_arr[idx] * 100
                        rec_str += f"{y}:{val:.1f}% | "
                
                tqdm.write(f"\n🔥 MEDIAN FIND! Score: {score:.2f} (Stability: {stability*100:.0f}%)")
                tqdm.write(f"Params: {ts}{p['s_ma']} / {tl}{p['l_ma']} | Filter: {f_str}")
                tqdm.write(f"Returns (Avg Stock): {rec_str}")

if __name__ == "__main__":
    run_robust_optimization()