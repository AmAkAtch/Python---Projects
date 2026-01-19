import ccxt
import pandas as pd
import numpy as np
import os
from numba import njit
from datetime import datetime

# ==========================================
# 1. CONFIGURATION: RESULT #4 (BALANCED CHAMPION)
# ==========================================
PARAMS = {
    'f_len': 38, 'f_smt': 5,
    's_len': 54, 's_smt': 47,
    'ma': 55,
    'atr_per': 14,
    'sl_m': 3.65,   # Tighter Hard Stop
    'tp_m': 14.98,  # 15x ATR Target
    'tr_m': 6.72    # Tighter Trail (6.7 vs 9.0)
}

COINS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 
    'ADA/USDT', 'LINK/USDT', 'LTC/USDT', 'BCH/USDT', 'SUI/USDT', 
    'RUNE/USDT', 'XLM/USDT', 'SHIB/USDT', 'RENDER/USDT', 'JUP/USDT', 
    'ONDO/USDT', 'OM/USDT', 'BAT/USDT', 'GLM/USDT', 'AUCTION/USDT'
]

TIMEFRAME = '1d'
START_DATE = "2017-01-01 00:00:00" 
FIXED_TRADE_SIZE = 1000.0
TRADING_FEE = 0.002

# ==========================================
# 2. INDICATORS (Kept Fast)
# ==========================================
@njit(fastmath=True)
def numba_sma(arr, length):
    out = np.empty_like(arr); out[:] = np.nan; cumsum = 0.0
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            cumsum += arr[i]
            if i >= length: cumsum -= arr[i - length]; out[i] = cumsum / length
            elif i == length - 1: out[i] = cumsum / length
    return out

@njit(fastmath=True)
def numba_rma(arr, length):
    out = np.empty_like(arr); out[:] = np.nan; alpha = 1.0/length; avg=0.0; count=0
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            if count<length: avg+=arr[i]; count+=1; out[i]=avg/length if count==length else np.nan
            else: out[i] = alpha*arr[i] + (1-alpha)*out[i-1]
    return out

@njit(fastmath=True)
def numba_rsi(close, length):
    delta = np.empty_like(close); delta[0]=0; delta[1:]=close[1:]-close[:-1]
    gain = np.empty_like(delta); loss=np.empty_like(delta)
    for i in range(len(delta)):
        if delta[i]>0: gain[i]=delta[i]; loss[i]=0
        else: gain[i]=0; loss[i]=-delta[i]
    avg_gain=numba_rma(gain, length); avg_loss=numba_rma(loss, length)
    out=np.empty_like(close); out[:]=np.nan
    for i in range(len(close)):
        if avg_loss[i]==0: out[i]=100
        else: rs=avg_gain[i]/avg_loss[i]; out[i]=100-(100/(1+rs))
    return out

@njit(fastmath=True)
def numba_atr(h, l, c, length):
    tr=np.empty_like(c); tr[0]=h[0]-l[0]
    for i in range(1, len(c)):
        hl=h[i]-l[i]; hc=abs(h[i]-c[i-1]); lc=abs(l[i]-c[i-1])
        tr[i]=max(hl, max(hc, lc))
    return numba_rma(tr, length)

# ==========================================
# 3. BACKTEST LOGIC (Pure Python - Stable)
# ==========================================
def backtest_yearly_stable(opens, highs, lows, closes, times, p):
    # Calculate Indicators
    rsi_f = numba_sma(numba_rsi(closes, p['f_len']), p['f_smt'])
    rsi_s = numba_sma(numba_rsi(closes, p['s_len']), p['s_smt'])
    ma = numba_sma(closes, p['ma'])
    atr = numba_atr(highs, lows, closes, p['atr_per'])
    
    yearly_pnl = {}
    
    in_pos = False; half_sold = False
    entry_price = 0.0; stop_loss = 0.0; tp_price = 0.0
    units = 0.0
    
    warmup = max(p['s_len']+p['s_smt'], p['ma']) + 50
    
    for i in range(warmup, len(closes)-1):
        # 1. Get Year from timestamp (ms)
        ts = times[i+1]
        dt = datetime.fromtimestamp(ts / 1000.0)
        year = dt.year
        
        if year not in yearly_pnl: yearly_pnl[year] = 0.0
        
        curr_open=opens[i+1]; curr_high=highs[i+1]; curr_low=lows[i+1]; curr_atr=atr[i]

        if in_pos:
            pnl_event = 0.0
            
            # Trail Logic
            if half_sold:
                 new_sl = curr_high - (curr_atr * p['tr_m'])
                 if new_sl > stop_loss: stop_loss = new_sl

            # Exits
            exit_price = 0.0
            full_exit = False
            
            # SL
            if curr_low <= stop_loss:
                exit_price = curr_open if curr_open < stop_loss else stop_loss
                full_exit = True
            # Crossunder
            elif (rsi_f[i-1] >= rsi_s[i-1]) and (rsi_f[i] < rsi_s[i]):
                exit_price = curr_open
                full_exit = True
                
            if full_exit:
                rev = units * exit_price
                cost = units * entry_price
                gross = rev - cost
                fee = (rev+cost)*TRADING_FEE
                yearly_pnl[year] += (gross-fee)
                
                in_pos=False; half_sold=False; units=0.0
                continue
            
            # TP (Target)
            if not half_sold and curr_high >= tp_price:
                 sell_p = curr_open if curr_open > tp_price else tp_price
                 sell_u = units * 0.5
                 rev = sell_u * sell_p
                 cost = sell_u * entry_price
                 gross = rev - cost
                 fee = (rev+cost)*TRADING_FEE
                 yearly_pnl[year] += (gross-fee)
                 
                 units -= sell_u
                 half_sold = True
                 if stop_loss < entry_price: stop_loss = entry_price
        
        # Entry
        crossover = (rsi_f[i-1] <= rsi_s[i-1]) and (rsi_f[i] > rsi_s[i])
        trend_ok = closes[i] > ma[i]
        
        if not in_pos and crossover and trend_ok:
            entry_price = curr_open
            units = FIXED_TRADE_SIZE / entry_price
            stop_loss = entry_price - (curr_atr * p['sl_m'])
            tp_price = entry_price + (curr_atr * p['tp_m'])
            in_pos=True; half_sold=False
            
    return yearly_pnl

# ==========================================
# 4. RUNNER
# ==========================================
def run():
    print("Testing 'Balanced Champion' (Result #4) - Fixed Script...")
    exchange = ccxt.binance()
    full_yearly = {}
    
    # Pre-fetch check
    if not os.path.exists("data"): os.makedirs("data")

    for coin in COINS:
        safe_sym = coin.replace('/', '_')
        fname = f"data/{safe_sym}_{TIMEFRAME}_full.csv"
        
        # Load or Skip
        if os.path.exists(fname):
            df = pd.read_csv(fname)
        else:
            continue
            
        # Prep Data
        opens = df['open'].values.astype(np.float64)
        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        closes = df['close'].values.astype(np.float64)
        times = df['timestamp'].values.astype(np.int64)
        
        # Run Backtest
        coin_pnl = backtest_yearly_stable(opens, highs, lows, closes, times, PARAMS)
        
        # Aggregate
        for y, pnl in coin_pnl.items():
            if y not in full_yearly: full_yearly[y] = 0.0
            full_yearly[y] += pnl

    print("\n" + "="*40)
    print("YEARLY PNL (Result #4 - The 3 Pointer)")
    print("="*40)
    
    for y in sorted(full_yearly.keys()):
        print(f"{y}: ${full_yearly[y]:,.2f}")

if __name__ == "__main__":
    run()