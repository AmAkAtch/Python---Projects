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
# 1. CONFIGURATION: INDICES & TIMEFRAMES
# ==========================================
# Tickers map to Yahoo Finance symbols. 
# Note: FinNifty/Midcap ticker data on Yahoo can be spotty; standard tickers used below.
INDICES = {
    'NIFTY':        {'ticker': '^NSEI',       'expiry_day': 3}, # Thursday
    'BANKNIFTY':    {'ticker': '^NSEBANK',    'expiry_day': 2}, # Wednesday
    'FINNIFTY':     {'ticker': 'NIFTY_FIN_SERVICE.NS', 'expiry_day': 1}, # Tuesday
    'MIDCPNIFTY':   {'ticker': 'NIFTY_MID_SELECT.NS',  'expiry_day': 0}  # Monday
}

TIMEFRAMES = ['5m', '15m', '1h']

# Strategy Config
NUM_ITERATIONS = 50000       # High iterations for finding the "perfect" fit per chart
COST_PCT = 0.0005            # 0.05% per trade (Futures/Options slippage + comms)
EXIT_TIME_MINUTES = 920      # 15:20 PM (15 * 60 + 20)

if not os.path.exists("data_indices"):
    os.makedirs("data_indices")

# ==========================================
# 2. ROBUST DATA FETCHING (Max History)
# ==========================================
def fetch_index_data(index_name, timeframe):
    """
    Fetches maximum allowed intraday data from Yahoo Finance.
    5m/15m = Max 60 days (approx 4000-5000 candles).
    1h = Max 730 days.
    """
    ticker = INDICES[index_name]['ticker']
    file_path = f"data_indices/{index_name}_{timeframe}.csv"
    
    # Determine max period based on Yahoo limits
    period = "60d" if timeframe in ['5m', '15m'] else "730d"
    
    # Try downloading fresh data first (Priority: Freshness)
    try:
        tqdm.write(f"Downloading {index_name} {timeframe} ({period})...")
        df = yf.download(ticker, period=period, interval=timeframe, progress=False, multi_level_index=False)
        
        # Clean Data
        df = df.dropna()
        if len(df) > 500:
            df.to_csv(file_path)
            return df
    except Exception as e:
        pass
        
    # Fallback to local cache
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            return df
        except: return None
        
    return None

# ==========================================
# 3. NUMBA INDICATORS & PRE-PROCESSING
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
# 4. BACKTEST LOGIC (With Expiry Exit)
# ==========================================
@njit(fastmath=True)
def backtest_intraday(opens, highs, lows, closes, 
                      weekdays, times_min, expiry_day_idx,
                      short_ma, long_ma, super_ma, 
                      trail_pct, target_pct): 
    
    n = len(closes)
    
    # Capital Tracking
    initial_capital = 100000.0
    equity = initial_capital
    
    in_pos = False # False: None, 1: Long, -1: Short (If you want bi-directional)
    # NOTE: This logic is currently LONG ONLY for simplicity, can be adapted for Short.
    
    entry_price = 0.0
    high_since_entry = 0.0
    trades = 0
    wins = 0
    gross_win = 0.0
    gross_loss = 0.0
    
    start_idx = 200 # Warmup for indicators
    
    for i in range(start_idx, n - 1):
        
        # Current Market State
        curr_open = opens[i+1]
        curr_high = highs[i+1]
        curr_low = lows[i+1]
        curr_close = closes[i+1]
        curr_time = times_min[i+1]
        curr_day = weekdays[i+1]
        
        # ---------------------------
        # CHECK EXIT CONDITIONS
        # ---------------------------
        if in_pos:
            high_since_entry = max(high_since_entry, curr_high)
            should_exit = False
            exit_price = 0.0
            
            # 1. EXPIRY DAY HARD EXIT (15:20 rule)
            if curr_day == expiry_day_idx and curr_time >= EXIT_TIME_MINUTES:
                should_exit = True
                exit_price = curr_close # Force close at market close of candle
            
            # 2. Trailing Stop
            if not should_exit:
                stop_price = high_since_entry * (1.0 - trail_pct / 100.0)
                if curr_low <= stop_price:
                    should_exit = True
                    exit_price = curr_open if curr_open < stop_price else stop_price
            
            # 3. Target (Take Profit)
            if not should_exit:
                target_price = entry_price * (1.0 + target_pct / 100.0)
                if curr_high >= target_price:
                    should_exit = True
                    exit_price = target_price
            
            # 4. Indicator Reversal Exit
            if not should_exit:
                if short_ma[i-1] >= long_ma[i-1] and short_ma[i] < long_ma[i]:
                    should_exit = True
                    exit_price = curr_open
            
            # EXECUTE EXIT
            if should_exit:
                # PnL Calculation
                pnl_pct = (exit_price - entry_price) / entry_price - COST_PCT
                pnl_abs = equity * pnl_pct
                equity += pnl_abs
                
                if pnl_pct > 0: 
                    wins += 1
                    gross_win += pnl_abs
                else:
                    gross_loss += abs(pnl_abs)
                    
                trades += 1
                in_pos = False
                high_since_entry = 0.0

        # ---------------------------
        # CHECK ENTRY CONDITIONS
        # ---------------------------
        elif not in_pos:
            # Don't enter after 3:00 PM
            if curr_time < 900: 
                
                # Crossover
                crossover = short_ma[i-1] <= long_ma[i-1] and short_ma[i] > long_ma[i]
                
                # Trend Filter (Price > SuperMA)
                trend_ok = closes[i] > super_ma[i]
                
                if crossover and trend_ok:
                    in_pos = True
                    entry_price = curr_open
                    high_since_entry = curr_open

    return equity, trades, wins, gross_win, gross_loss

# ==========================================
# 5. OPTIMIZATION RUNNER
# ==========================================
def optimize_chart(index_name, timeframe):
    
    # 1. Prepare Data
    df = fetch_index_data(index_name, timeframe)
    if df is None or len(df) < 1000:
        print(f"Skipping {index_name} {timeframe}: Insufficient Data ({len(df) if df is not None else 0})")
        return

    # 2. Extract Numpy Arrays
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    # Time Pre-processing (Minutes from midnight)
    times_min = df.index.hour.values * 60 + df.index.minute.values
    weekdays = df.index.weekday.values # 0=Mon, 1=Tue...
    
    expiry_day = INDICES[index_name]['expiry_day']
    
    print(f"\n⚡ OPTIMIZING: {index_name} [{timeframe}] | Candles: {len(df)}")
    print(f"   Expiry Day Index: {expiry_day} (Auto-Exit enabled @ 15:20)")
    
    best_score = -99999
    best_p = {}
    best_stats = {}
    
    # 3. Random Search Loop
    # Pre-generate random params for speed
    r_short = np.random.randint(5, 50, NUM_ITERATIONS)
    r_long = np.random.randint(10, 100, NUM_ITERATIONS)
    r_super = np.random.randint(50, 300, NUM_ITERATIONS)
    r_trail = np.random.uniform(0.1, 3.0, NUM_ITERATIONS) # Tighter trails for intraday
    r_target = np.random.uniform(0.5, 5.0, NUM_ITERATIONS)

    for i in tqdm(range(NUM_ITERATIONS), desc="Simulating"):
        s_ma = r_short[i]
        l_ma = r_long[i]
        sl_ma = r_super[i]
        
        if s_ma >= l_ma: continue
        
        # Calculate Indicators (Re-calc per iter is costly but necessary for varying params)
        # Optimization: You could pre-calc standard EMAs, but with random periods, on-the-fly is needed.
        # Numba makes this fast enough.
        arr_s = calc_ema(closes, s_ma)
        arr_l = calc_ema(closes, l_ma)
        arr_sl = calc_ema(closes, sl_ma)
        
        final_eq, trades, wins, g_win, g_loss = backtest_intraday(
            opens, highs, lows, closes, 
            weekdays, times_min, expiry_day,
            arr_s, arr_l, arr_sl, 
            r_trail[i], r_target[i]
        )
        
        if trades > 30: # Minimum sample size
            pf = g_win / g_loss if g_loss > 0 else 99.9
            roi = (final_eq - 100000) / 100000 * 100
            
            # Score = ROI * PF (Simple yet effective)
            score = roi * pf
            
            if score > best_score:
                best_score = score
                best_p = {
                    'SMA': s_ma, 'LMA': l_ma, 'Super': sl_ma, 
                    'Trail%': r_trail[i], 'Tgt%': r_target[i]
                }
                best_stats = {
                    'ROI%': roi, 'PF': pf, 'Trades': trades, 'Win%': (wins/trades)*100
                }

    # 4. Report Results
    print(f"🏆 BEST PARAMETERS FOR {index_name} {timeframe}")
    print(f"   MA Cross: {best_p.get('SMA')}/{best_p.get('LMA')} | Filter: {best_p.get('Super')}")
    print(f"   Risk: Trail {best_p.get('Trail%'):.2f}% | Target {best_p.get('Tgt%'):.2f}%")
    print(f"   Stats: ROI {best_stats.get('ROI%'):.1f}% | PF {best_stats.get('PF'):.2f} | Trades {best_stats.get('Trades')}")
    print("-" * 60)

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=== NIFTY MULTI-INDEX INTRADAY OPTIMIZER ===")
    print("Mode: Long Only Trend Following with Expiry Exit")
    
    for idx_name in INDICES.keys():
        for tf in TIMEFRAMES:
            optimize_chart(idx_name, tf)