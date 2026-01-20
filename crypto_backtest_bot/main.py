import ccxt
import pandas as pd
import numpy as np
import random
import os
import time
from tqdm import tqdm
from numba import njit
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================
if not os.path.exists('data'):
    os.makedirs('data')

# Expanded Universe (Top Liquid Pairs)
COINS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 
    'ADA/USDT', 'LINK/USDT', 'LTC/USDT', 'BCH/USDT', 'SUI/USDT', 
    'RUNE/USDT', 'XLM/USDT', 'SHIB/USDT', 'RENDER/USDT', 'JUP/USDT', 
    'ONDO/USDT', 'OM/USDT', 'BAT/USDT', 'GLM/USDT', 'AUCTION/USDT',
    'AVAX/USDT', 'DOGE/USDT', 'DOT/USDT', 'TRX/USDT', 'MATIC/USDT',
    'UNI/USDT', 'NEAR/USDT', 'APT/USDT', 'FIL/USDT', 'ATOM/USDT',
    'IMX/USDT', 'ARB/USDT', 'OP/USDT', 'INJ/USDT', 'STX/USDT',
    'ETC/USDT', 'VET/USDT', 'TIA/USDT', 'LDO/USDT', 'GRT/USDT'
]

TIMEFRAME = '1d'
START_DATE = "2020-01-01 00:00:00"

# Optimization Settings
TOTAL_TESTS = 3000000 # High count, but handled by Numba speed
TRADING_FEE = 0.002 
FIXED_TRADE_SIZE = 1000.0 

# Weighted Scoring Configuration
# We value recent years more. 
# Format: (Year, Weight)
YEAR_WEIGHTS = {
    2020: 0.5,
    2021: 0.8,
    2022: 1.0, # Bear market survival is key
    2023: 1.2,
    2024: 1.5,
    2025: 1.5
}

RANGES = {
    'fast_rsi_len': (5, 60),       
    'fast_smooth': (1, 50),        
    'slow_rsi_len': (10, 90),     
    'slow_smooth': (2, 70),        
    'trend_ma': (20, 250),        
    'atr_period': (14, 14),       
    'atr_stop_mult': (1.5, 5.0),   
    'atr_tp_mult': (15.0, 50.0),    
    'atr_trail_mult': (1.5, 15.0),
    'btc_filter_ma': (50, 250) # BTC Trend Filter MA length
}

# ==========================================
# 2. NUMBA INDICATORS
# ==========================================
@njit(fastmath=True)
def numba_sma(arr, length):
    out = np.empty_like(arr)
    out[:] = np.nan
    cumsum = 0.0
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            cumsum += arr[i]
            if i >= length:
                cumsum -= arr[i - length]
                out[i] = cumsum / length
            elif i == length - 1:
                out[i] = cumsum / length
    return out

@njit(fastmath=True)
def numba_rma(arr, length):
    out = np.empty_like(arr)
    out[:] = np.nan
    alpha = 1.0 / length
    avg = 0.0
    count = 0
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            if count < length:
                avg += arr[i]
                count += 1
                if count == length:
                    out[i] = avg / length
            else:
                out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out

@njit(fastmath=True)
def numba_rsi(close, length):
    delta = np.empty_like(close)
    delta[0] = 0
    delta[1:] = close[1:] - close[:-1]
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = numba_rma(gain, length)
    avg_loss = numba_rma(loss, length)
    out = np.empty_like(close)
    out[:] = np.nan
    for i in range(len(close)):
        if avg_loss[i] == 0:
            out[i] = 100 if avg_gain[i] != 0 else 50
        else:
            rs = avg_gain[i] / avg_loss[i]
            out[i] = 100 - (100 / (1 + rs))
    return out

@njit(fastmath=True)
def numba_atr(high, low, close, length):
    tr = np.empty_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
    return numba_rma(tr, length)

# ==========================================
# 3. BACKTEST LOGIC (YEARLY TRACKING)
# ==========================================
@njit(fastmath=True)
def backtest_yearly_runner(opens, highs, lows, closes, years, 
                           rsi_fast, rsi_slow, ma_trend, atr, btc_bullish,
                           tp_mult, sl_mult, trail_mult, trading_fee, start_idx):
    
    # Yearly PnL Tracking (Year 2020 -> Index 2020)
    # Using a fixed array size for years 2020-2030
    yearly_pnl = np.zeros(3000) 
    
    # Metrics
    total_trades = 0
    winning_trades = 0
    
    # State
    in_pos = False
    half_sold = False 
    entry_price = 0.0
    stop_loss_price = 0.0
    tp_trigger_price = 0.0 
    units = 0.0 
    
    n = len(closes)
    
    for i in range(start_idx, n-1):
        curr_year = int(years[i+1]) # Log PnL to the year the trade CLOSES or is active
        
        curr_open = opens[i+1]
        curr_high = highs[i+1]
        curr_low = lows[i+1]
        curr_close = closes[i+1]
        curr_atr = atr[i]
        
        # -----------------------------
        # 1. MANAGE POSITION
        # -----------------------------
        if in_pos:
            net_pnl_event = 0.0
            
            # --- A. TRAILING STOP UPDATE ---
            if half_sold:
                potential_new_sl = curr_high - (curr_atr * trail_mult)
                if potential_new_sl > stop_loss_price:
                    stop_loss_price = potential_new_sl
            
            # --- B. CONSERVATIVE EXIT LOGIC ---
            # If Low hits SL, we are OUT. Even if High hit TP in same candle.
            # We assume worst case for manual trading safety.
            sl_hit = curr_low <= stop_loss_price
            tp_hit = (not half_sold) and (curr_high >= tp_trigger_price)
            
            if sl_hit:
                # Execution Price (Slippage logic: SL or Open)
                exit_p = stop_loss_price
                if curr_open < stop_loss_price: exit_p = curr_open 
                
                revenue = units * exit_p
                cost_of_chunk = units * entry_price
                gross_pnl = revenue - cost_of_chunk
                fees = (cost_of_chunk + revenue) * trading_fee
                
                net_pnl_event = gross_pnl - fees
                yearly_pnl[curr_year] += net_pnl_event
                
                in_pos = False
                half_sold = False
                units = 0.0
                
            elif tp_hit and not sl_hit:
                # Take 50% Profit
                sell_p = tp_trigger_price
                if curr_open > tp_trigger_price: sell_p = curr_open 
                
                units_to_sell = units * 0.5
                revenue = units_to_sell * sell_p
                cost_of_chunk = units_to_sell * entry_price
                
                gross_pnl = revenue - cost_of_chunk
                fees = (cost_of_chunk + revenue) * trading_fee
                
                net_pnl_event = gross_pnl - fees
                yearly_pnl[curr_year] += net_pnl_event
                
                # Update State
                units -= units_to_sell 
                half_sold = True
                # Move SL to Breakeven
                if stop_loss_price < entry_price:
                    stop_loss_price = entry_price
                
                winning_trades += 1 # Count as win once we bank profit
            
            # --- C. STRATEGY EXIT (Indicator Crossunder) ---
            else:
                crossunder = (rsi_fast[i-1] >= rsi_slow[i-1]) and (rsi_fast[i] < rsi_slow[i])
                if crossunder:
                    revenue = units * curr_open
                    cost_of_chunk = units * entry_price
                    gross_pnl = revenue - cost_of_chunk
                    fees = (cost_of_chunk + revenue) * trading_fee
                    
                    net_pnl_event = gross_pnl - fees
                    yearly_pnl[curr_year] += net_pnl_event
                    
                    if gross_pnl > 0: winning_trades += 1 # Only count win if net positive

                    in_pos = False
                    half_sold = False
                    units = 0.0

        # -----------------------------
        # 2. CHECK ENTRY
        # -----------------------------
        # Only enter if:
        # 1. Not in Position
        # 2. Signal Exists (RSI Cross)
        # 3. Trend is Bullish (Price > MA)
        # 4. MARKET FILTER: BTC is Bullish (Passed in array)
        
        if not in_pos:
            crossover = (rsi_fast[i-1] <= rsi_slow[i-1]) and (rsi_fast[i] > rsi_slow[i])
            trend_bullish = closes[i] > ma_trend[i]
            
            # BTC Filter Check
            market_bullish = btc_bullish[i] == 1.0
            
            if crossover and trend_bullish and market_bullish:
                entry_price = curr_open
                units = FIXED_TRADE_SIZE / entry_price
                
                stop_loss_price = entry_price - (curr_atr * sl_mult)
                tp_trigger_price = entry_price + (curr_atr * tp_mult)
                
                in_pos = True
                half_sold = False
                total_trades += 1
            
    return yearly_pnl, total_trades, winning_trades

# ==========================================
# 4. DATA FETCHING & PROCESSING
# ==========================================
def fetch_full_history(symbol, start_date_str):
    safe_sym = symbol.replace('/', '_')
    fname = f"data/{safe_sym}_{TIMEFRAME}_full.csv"
    
    # Check if exists
    if os.path.exists(fname):
        return pd.read_csv(fname)
    
    exchange = ccxt.binance()
    since = exchange.parse8601(start_date_str)
    all_ohlcv = []
    
    print(f"Downloading {symbol}...")
    try:
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
            if not ohlcv: break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000: break
            time.sleep(0.1)
            
        if not all_ohlcv: return None
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df.to_csv(fname, index=False)
        return df
    except Exception as e:
        print(f"Error {symbol}: {e}")
        return None

def prepare_btc_filter():
    # Ensures we have BTC data to act as the global filter
    df_btc = fetch_full_history('BTC/USDT', START_DATE)
    df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'], unit='ms')
    df_btc = df_btc.set_index('timestamp').resample('1D').ffill()
    return df_btc['close'] # Return Series

# ==========================================
# 5. OPTIMIZER CORE (ROBUSTNESS FOCUSED)
# ==========================================
def optimize():
    # 1. PREPARE DATA
    print("\n--- 1. LOADING & ALIGNING DATA ---")
    
    # Load BTC for Market Filter
    btc_series = prepare_btc_filter()
    
    market_data = {}
    valid_coins = []
    
    for coin in COINS:
        df = fetch_full_history(coin, START_DATE)
        if df is None or len(df) < 500: continue
        
        # Pre-calculate Years for indexing
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['year'] = df['datetime'].dt.year
        
        # Align BTC Filter to this coin's timeline
        # We need to look up BTC close for every timestamp in this coin's DF
        # Efficient way: Map timestamps
        
        # Convert to arrays
        d = {
            'o': df['open'].values.astype(np.float64),
            'h': df['high'].values.astype(np.float64),
            'l': df['low'].values.astype(np.float64),
            'c': df['close'].values.astype(np.float64),
            'y': df['year'].values.astype(np.int32),
            'ts': df['timestamp'].values
        }
        
        # Reindex BTC to match this coin exactly
        coin_dates = df['datetime']
        aligned_btc = btc_series.reindex(coin_dates).fillna(method='ffill').values
        d['btc_close'] = aligned_btc.astype(np.float64)
        
        market_data[coin] = d
        valid_coins.append(coin)

    print(f"Loaded {len(valid_coins)} coins for testing.")
    print("Optimization Strategy: Consistency Weighted Scoring")
    
    best_score = -999999
    best_params = None
    
    # 2. OPTIMIZATION LOOP
    for _ in tqdm(range(TOTAL_TESTS)):
        
        # A. Generate Random Params
        p = {
            'f_len': random.randint(*RANGES['fast_rsi_len']),
            'f_smt': random.randint(*RANGES['fast_smooth']),
            's_len': random.randint(*RANGES['slow_rsi_len']),
            's_smt': random.randint(*RANGES['slow_smooth']),
            'ma': random.randint(*RANGES['trend_ma']),
            'atr_per': 14,
            'sl_m': random.uniform(*RANGES['atr_stop_mult']),
            'tp_m': random.uniform(*RANGES['atr_tp_mult']),
            'tr_m': random.uniform(*RANGES['atr_trail_mult']),
            'btc_ma': random.randint(*RANGES['btc_filter_ma'])
        }
        
        if p['f_len'] >= p['s_len']: continue
        
        # Global Aggregators
        global_yearly_pnl = {} # {2020: 0.0, 2021: 0.0 ...}
        total_trades_all = 0
        total_wins_all = 0
        
        valid_runs = 0
        
        # B. Test Across Universe
        for coin in valid_coins:
            d = market_data[coin]
            
            # Warmup check
            warmup = max(p['s_len'] + p['s_smt'], p['ma'], p['btc_ma']) + 2
            if len(d['c']) <= warmup: continue
            
            # Calculate Indicators
            rsi_f = numba_sma(numba_rsi(d['c'], p['f_len']), p['f_smt'])
            rsi_s = numba_sma(numba_rsi(d['c'], p['s_len']), p['s_smt'])
            ma = numba_sma(d['c'], p['ma'])
            atr = numba_atr(d['h'], d['l'], d['c'], p['atr_per'])
            
            # BTC Filter Calculation
            btc_ma = numba_sma(d['btc_close'], p['btc_ma'])
            # Boolean array: 1.0 if BTC > MA, else 0.0
            # Handle NaNs in BTC MA
            btc_bullish = np.where(d['btc_close'] > btc_ma, 1.0, 0.0)
            
            # Run Backtest
            yearly_res, trades, wins = backtest_yearly_runner(
                d['o'], d['h'], d['l'], d['c'], d['y'],
                rsi_f, rsi_s, ma, atr, btc_bullish,
                p['tp_m'], p['sl_m'], p['tr_m'], 
                TRADING_FEE, int(warmup)
            )
            
            total_trades_all += trades
            total_wins_all += wins
            valid_runs += 1
            
            # Aggregate Yearly PnL
            # yearly_res is an array index 0-3000. 2020 is at index 2020.
            for y in range(2020, 2026):
                if y not in global_yearly_pnl: global_yearly_pnl[y] = 0.0
                global_yearly_pnl[y] += yearly_res[y]

        if valid_runs == 0: continue
        if total_trades_all < 50: continue # Must trade enough to be significant
        
        # C. SCORING MECHANISM (The "Window" Logic)
        
        # 1. Winrate Check
        wr = total_wins_all / total_trades_all
        if wr < 0.40: continue # Hard filter: If winrate < 40%, discard (user wants high winrate)
        
        # 2. Consistency Check (The "No Loser Year" Rule)
        # We allow small losses, but massive drawdown years disqualify the strat.
        # Actually, let's enforce STRICT consistency: 
        # Weighted Score = Sum(YearPnL * YearWeight)
        # PENALTY: If any year is negative, divide score by 10.
        
        weighted_score = 0.0
        has_negative_year = False
        
        years_active = [y for y in global_yearly_pnl if global_yearly_pnl[y] != 0]
        if len(years_active) < 3: continue # Must have traded in at least 3 years
        
        for y in range(2020, 2026):
            pnl = global_yearly_pnl.get(y, 0.0)
            weight = YEAR_WEIGHTS.get(y, 1.0)
            
            if pnl < -500: # Allow small drawback, but punish losses > $500
                has_negative_year = True
            
            weighted_score += (pnl * weight)
            
        if has_negative_year:
            weighted_score = -1000 # Disqualify immediately
            
        # 3. Final Score Adjustment
        # Prefer higher winrate.
        # Score = Weighted PnL * WinRate
        final_score = weighted_score * wr
        
        if final_score > best_score:
            best_score = final_score
            best_params = p
            
            # Create a pretty string for yearly breakdown
            breakdown = " | ".join([f"{y}:${int(global_yearly_pnl.get(y,0))}" for y in range(2020, 2025)])
            
            print(f"\nNEW BEST! Score: {final_score:.0f} | WR: {wr*100:.1f}% | Trades: {total_trades_all}")
            print(f"Yearly: {breakdown}")
            print(f"Params: RSI {p['f_len']}/{p['s_len']} ({p['f_smt']}/{p['s_smt']}) | MA {p['ma']} | BTC_MA {p['btc_ma']}")
            print(f"Risk: TP {p['tp_m']:.1f} | SL {p['sl_m']:.1f} | Trail {p['tr_m']:.1f}")

    print("\nOPTIMIZATION COMPLETE")
    print("Best Parameters:", best_params)

if __name__ == "__main__":
    optimize()