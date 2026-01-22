import ccxt
import pandas as pd
import numpy as np
import random
import os
import time
from tqdm import tqdm
from numba import njit
from numba.typed import List
import multiprocessing

# ==========================================
# 1. CONFIGURATION
# ==========================================
COINS = [
    'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 
    'XRP/USDT', 'DOGE/USDT', 'XLM/USDT'
]

TIMEFRAME = '15m'
START_DATE = "2025-05-15 00:00:00" 
TOTAL_TESTS = 500000 
TRADING_FEE = 0.0006 
FIXED_MARGIN = 1000.0 
DECAY_RATE = 0.05 

# RANGES (Expanded with Toggles)
RANGES = {
    'short_ma': (5, 300),
    'long_ma': (20, 600),
    'super_ma': (100, 1000), 
    
    # 0=SMA, 1=EMA
    'type_s': (0, 1),
    'type_l': (0, 1),
    'type_su': (0, 1),
    
    'atr_period': (14, 14),
    
    'atr_stop_mult': (1.0, 10.0),   
    'atr_tp_mult': (2.0, 50.0),    
    'atr_trail_mult': (1.5, 10.0),
    
    'btc_filter_ma': (100, 800),
    'leverage': (1, 15),
    
    # --- NEW TOGGLES (0=OFF, 1=ON) ---
    'use_trend': (0, 1),
    'use_btc': (0, 1),
    'use_sl': (0, 1),
    'use_tp': (0, 1),
    'use_trail': (0, 1)
}

# ==========================================
# 2. NUMBA INDICATORS
# ==========================================
@njit(fastmath=True, cache=True)
def numba_sma(arr, length):
    out = np.empty_like(arr)
    out[:] = np.nan
    cumsum = 0.0
    count = 0
    for i in range(len(arr)):
        val = arr[i]
        if not np.isnan(val):
            cumsum += val
            count += 1
            if count > length:
                cumsum -= arr[i - length] 
                out[i] = cumsum / length
            elif count == length:
                out[i] = cumsum / length
    return out

@njit(fastmath=True, cache=True)
def numba_ema(arr, length):
    out = np.empty_like(arr)
    out[:] = np.nan
    if len(arr) < length: return out
    alpha = 2.0 / (length + 1)
    
    sma_sum = 0.0
    cnt = 0
    start_idx = -1
    
    for i in range(len(arr)):
        if not np.isnan(arr[i]):
            sma_sum += arr[i]
            cnt += 1
            if cnt == length:
                out[i] = sma_sum / length
                start_idx = i + 1
                break
                
    if start_idx != -1:
        for i in range(start_idx, len(arr)):
            val = arr[i]
            prev = out[i-1]
            if np.isnan(val):
                out[i] = prev
            else:
                out[i] = val * alpha + prev * (1.0 - alpha)
    return out

@njit(fastmath=True, cache=True)
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

@njit(fastmath=True, cache=True)
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
# 3. BACKTEST CORE (TOGGLE EDITION)
# ==========================================
@njit(fastmath=True, cache=True)
def run_backtest_core(opens, highs, lows, closes, years, btc_close,
                      s_len, l_len, su_len, t_s, t_l, t_su,
                      atr_per, sl_m, tp_m, tr_m, btc_ma_len, lev,
                      use_trend, use_btc, use_sl, use_tp, use_trail,
                      trading_fee, fixed_margin, min_yr, max_yr, yearly_weights):
    
    # 1. Indicators
    if t_s == 0: short_ma = numba_sma(closes, s_len)
    else: short_ma = numba_ema(closes, s_len)
        
    if t_l == 0: long_ma = numba_sma(closes, l_len)
    else: long_ma = numba_ema(closes, l_len)
        
    if t_su == 0: super_ma = numba_sma(closes, su_len)
    else: super_ma = numba_ema(closes, su_len)
    
    atr = numba_atr(highs, lows, closes, atr_per)
    btc_ma = numba_sma(btc_close, btc_ma_len)
    
    yearly_pnl = np.zeros(150, dtype=np.float32)
    n = len(closes)
    warmup = max(l_len, su_len, btc_ma_len) + 100
    
    if n <= warmup: return 0.0, 0.0, 0.0, 0, 0, 0
    
    in_pos = False
    half_sold = False
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    liq_price = 0.0
    units = 0.0
    
    gross_win = 0.0; gross_loss = 0.0
    trades = 0; wins = 0; liqs = 0
    
    for i in range(warmup, n - 1):
        curr_open = opens[i+1]
        curr_high = highs[i+1]
        curr_low = lows[i+1]
        yr_idx = years[i+1] - 2000
        
        if in_pos:
            # A. LIQUIDATION (Always Active)
            if curr_low <= liq_price:
                loss = fixed_margin
                yearly_pnl[yr_idx] -= loss
                gross_loss += loss
                liqs += 1
                in_pos = False; half_sold = False; units = 0.0
                continue
            
            # B. TRAILING STOP (If Enabled)
            if use_trail == 1:
                # Trail only activates after half sold OR if enabled globally (Logic: Standard Trail)
                # Python Logic: Trail when half_sold is true
                if half_sold:
                    new_sl = curr_high - (atr[i] * tr_m)
                    if new_sl > stop_loss: stop_loss = new_sl
            
            # C. EXITS
            sl_hit = False
            if use_sl == 1:
                sl_hit = curr_low <= stop_loss
            
            tp_hit = False
            if use_tp == 1:
                tp_hit = (not half_sold) and (curr_high >= take_profit)
                
            x_under = (short_ma[i-1] >= long_ma[i-1]) and (short_ma[i] < long_ma[i])
            
            if sl_hit:
                exit_p = stop_loss if curr_open > stop_loss else curr_open
                if exit_p < liq_price: exit_p = liq_price 
                
                rev = units * exit_p; cost = units * entry_price
                pnl = rev - cost
                fee = (rev + cost) * trading_fee
                net = pnl - fee
                
                yearly_pnl[yr_idx] += net
                if net > 0: 
                    gross_win += net; wins += 1
                else: 
                    gross_loss += abs(net)
                in_pos = False; half_sold = False; units = 0.0
                
            elif tp_hit:
                sell_p = take_profit if curr_open < take_profit else curr_open
                u_sell = units * 0.5
                
                rev = u_sell * sell_p; cost = u_sell * entry_price
                pnl = rev - cost
                fee = (rev + cost) * trading_fee
                net = pnl - fee
                
                yearly_pnl[yr_idx] += net
                if net > 0: gross_win += net
                else: gross_loss += abs(net)
                
                units -= u_sell
                half_sold = True
                # BreakEven logic (Always active if TP hit)
                if stop_loss < entry_price: stop_loss = entry_price
            
            elif x_under:
                rev = units * curr_open; cost = units * entry_price
                pnl = rev - cost
                fee = (rev + cost) * trading_fee
                net = pnl - fee
                
                yearly_pnl[yr_idx] += net
                if net > 0: 
                    gross_win += net; wins += 1
                else: 
                    gross_loss += abs(net)
                in_pos = False; half_sold = False; units = 0.0

        if not in_pos:
            x_over = (short_ma[i-1] <= long_ma[i-1]) and (short_ma[i] > long_ma[i])
            
            # Toggled Filters
            trend_ok = True
            if use_trend == 1:
                trend_ok = closes[i] > super_ma[i]
                
            btc_ok = True
            if use_btc == 1:
                btc_ok = btc_close[i] > btc_ma[i]
            
            if x_over and trend_ok and btc_ok:
                entry_price = curr_open
                notional = fixed_margin * lev
                units = notional / entry_price
                
                # Setup Exits
                stop_loss = 0.0
                if use_sl == 1:
                    stop_loss = entry_price - (atr[i] * sl_m)
                    
                take_profit = 99999999.0
                if use_tp == 1:
                    take_profit = entry_price + (atr[i] * tp_m)
                
                # Liquidation Price (Always calculated)
                liq_price = entry_price * (1.0 - (1.0/lev) + 0.005)
                
                # SL Safety Clamp
                if use_sl == 1 and stop_loss <= liq_price: 
                    stop_loss = liq_price * 1.002
                    
                in_pos = True
                trades += 1
                
    score = 0.0
    for y in range(min_yr, max_yr + 1):
        idx = y - 2000
        if idx >= 0 and idx < 150:
            score += yearly_pnl[idx] * yearly_weights[idx]
            
    return score, gross_win, gross_loss, trades, wins, liqs

@njit(fastmath=True, cache=True)
def evaluate_all_coins(p_arr, data_opens, data_highs, data_lows, data_closes, data_years, data_btc,
                       yearly_weights, min_yr, max_yr, trading_fee, fixed_margin):
    
    total_win_d = 0.0; total_loss_d = 0.0
    total_trades = 0; total_wins = 0; total_liqs = 0
    coin_scores = []
    
    for i in range(len(data_closes)):
        score, gw, gl, tr, w, lq = run_backtest_core(
            data_opens[i], data_highs[i], data_lows[i], data_closes[i], data_years[i], data_btc[i],
            int(p_arr[0]), int(p_arr[1]), int(p_arr[2]), int(p_arr[3]), int(p_arr[4]), int(p_arr[5]),
            int(p_arr[6]), p_arr[7], p_arr[8], p_arr[9], int(p_arr[10]), p_arr[11],
            int(p_arr[12]), int(p_arr[13]), int(p_arr[14]), int(p_arr[15]), int(p_arr[16]), # Toggles
            trading_fee, fixed_margin, min_yr, max_yr, yearly_weights
        )
        
        if tr > 0:
            total_win_d += gw
            total_loss_d += gl
            total_trades += tr
            total_wins += w
            total_liqs += lq
            coin_scores.append(score)
            
    if len(coin_scores) == 0 or total_trades < 10: return -999.0, 0.0, 0.0, 0, 0
    
    pf = total_win_d / total_loss_d if total_loss_d > 0 else 10.0
    win_rate = total_wins / total_trades
    
    # Punishment
    if total_liqs > 0:
        penalty_factor = 0.5 ** total_liqs
    else:
        penalty_factor = 1.0

    # Power Law Scoring
    n = len(coin_scores)
    for i in range(n):
        for j in range(0, n-i-1):
            if coin_scores[j] > coin_scores[j+1]:
                temp = coin_scores[j]; coin_scores[j] = coin_scores[j+1]; coin_scores[j+1] = temp
    if n % 2 == 1: med = coin_scores[n // 2]
    else: med = (coin_scores[n // 2 - 1] + coin_scores[n // 2]) * 0.5
    
    final_score = med * (pf ** 3) * (win_rate ** 2) * penalty_factor
    
    return final_score, pf, win_rate, total_trades, total_liqs

# ==========================================
# 4. DATA LOADING
# ==========================================
def fetch_data():
    if not os.path.exists('data'): os.makedirs('data')
    print("Loading BTC 15m Data...")
    btc_path = f"data/BTC_USDT_{TIMEFRAME}_perp.csv"
    
    if not os.path.exists(btc_path):
        print("Downloading BTC...")
        ex = ccxt.binance(); since = ex.parse8601(START_DATE); all_ohlcv = []
        try:
            while True:
                d = ex.fetch_ohlcv('BTC/USDT', TIMEFRAME, since, 1000)
                if not d: break
                all_ohlcv += d; since = d[-1][0] + 1; time.sleep(0.1)
                if len(d) < 1000: break
            df_btc = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','vol'])
            df_btc.to_csv(btc_path, index=False)
        except: return None
    else: df_btc = pd.read_csv(btc_path)
    df_btc['datetime'] = pd.to_datetime(df_btc['timestamp'], unit='ms')
    btc_series = df_btc.set_index('datetime')['close']

    d_o, d_h, d_l, d_c, d_y, d_btc = [], [], [], [], [], []
    valid_coins = []
    min_yr, max_yr = 2100, 0
    print("Loading Altcoins...")
    for coin in tqdm(COINS):
        safe = coin.replace('/', '_')
        path = f"data/{safe}_{TIMEFRAME}_perp.csv"
        if not os.path.exists(path):
            ex = ccxt.binance(); since = ex.parse8601(START_DATE); all_ohlcv = []
            try:
                while True:
                    d = ex.fetch_ohlcv(coin, TIMEFRAME, since, 1000)
                    if not d: break
                    all_ohlcv += d; since = d[-1][0] + 1; time.sleep(0.1)
                    if len(d) < 1000: break
                df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','vol'])
                df.to_csv(path, index=False)
            except: continue
        else: df = pd.read_csv(path)
        
        if len(df) < 2000: continue
        df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['year'] = df['dt'].dt.year
        aligned_btc = btc_series.reindex(df['dt']).fillna(method='ffill').values
        
        d_o.append(df['open'].values.astype(np.float32))
        d_h.append(df['high'].values.astype(np.float32))
        d_l.append(df['low'].values.astype(np.float32))
        d_c.append(df['close'].values.astype(np.float32))
        d_y.append(df['year'].values.astype(np.int32))
        d_btc.append(aligned_btc.astype(np.float32))
        if df['year'].min() < min_yr: min_yr = df['year'].min()
        if df['year'].max() > max_yr: max_yr = df['year'].max()
        valid_coins.append(coin)

    return d_o, d_h, d_l, d_c, d_y, d_btc, min_yr, max_yr

g_data = None
def init_worker(o, h, l, c, y, b, w, min_y, max_y):
    global g_data
    g_data = (List(o), List(h), List(l), List(c), List(y), List(b), w, min_y, max_y)

def worker_task(params):
    o, h, l, c, y, b, weights, min_y, max_y = g_data
    p_arr = np.array([
        params['s_len'], params['l_len'], params['super_ma'],
        params['type_s'], params['type_l'], params['type_su'],
        params['atr_period'], params['atr_stop_mult'], params['atr_tp_mult'],
        params['atr_trail_mult'], params['btc_filter_ma'], params['leverage'],
        params['use_trend'], params['use_btc'], params['use_sl'], params['use_tp'], params['use_trail']
    ], dtype=np.float64)
    
    score, pf, wr, trades, liqs = evaluate_all_coins(
        p_arr, o, h, l, c, y, b, weights, min_y, max_y, TRADING_FEE, FIXED_MARGIN
    )
    return score, pf, wr, trades, liqs, params

# ==========================================
# 5. MAIN OPTIMIZER
# ==========================================
def optimize():
    print("Initializing HIGH-PRECISION Optimizer (With Toggles)...")
    
    data_pack = fetch_data()
    if not data_pack: return
    nl_o, nl_h, nl_l, nl_c, nl_y, nl_b, min_yr, max_yr = data_pack
    
    weights = np.zeros(150, dtype=np.float32)
    for y in range(min_yr, max_yr + 1):
        weights[y - 2000] = 1.0 / ((1 + DECAY_RATE) ** (max_yr - y))
        
    param_list = []
    for _ in range(TOTAL_TESTS):
        p = {
            's_len': random.randint(*RANGES['short_ma']),
            'l_len': random.randint(*RANGES['long_ma']),
            'super_ma': random.randint(*RANGES['super_ma']),
            'type_s': random.randint(*RANGES['type_s']),
            'type_l': random.randint(*RANGES['type_l']),
            'type_su': random.randint(*RANGES['type_su']),
            'atr_period': 14,
            'atr_stop_mult': random.uniform(*RANGES['atr_stop_mult']),
            'atr_tp_mult': random.uniform(*RANGES['atr_tp_mult']),
            'atr_trail_mult': random.uniform(*RANGES['atr_trail_mult']),
            'btc_filter_ma': random.randint(*RANGES['btc_filter_ma']),
            'leverage': random.randint(*RANGES['leverage']),
            # TOGGLES
            'use_trend': random.randint(*RANGES['use_trend']),
            'use_btc': random.randint(*RANGES['use_btc']),
            'use_sl': random.randint(*RANGES['use_sl']),
            'use_tp': random.randint(*RANGES['use_tp']),
            'use_trail': random.randint(*RANGES['use_trail']),
        }
        if p['s_len'] >= p['l_len']: continue
        param_list.append(p)
        
    best_score = -99999.0
    
    print("Starting Optimization Pool...")
    with multiprocessing.Pool(initializer=init_worker, initargs=(nl_o, nl_h, nl_l, nl_c, nl_y, nl_b, weights, min_yr, max_yr)) as pool:
        results = pool.imap_unordered(worker_task, param_list, chunksize=100)
        
        with tqdm(total=len(param_list)) as pbar:
            for score, pf, wr, trades, liqs, p in results:
                pbar.update()
                
                if score > best_score:
                    best_score = score
                    
                    s_type = "EMA" if p['type_s'] else "SMA"
                    l_type = "EMA" if p['type_l'] else "SMA"
                    su_type = "EMA" if p['type_su'] else "SMA"
                    
                    # Formatting booleans
                    str_trend = f"ENABLED ({su_type} {p['super_ma']})" if p['use_trend'] else "DISABLED"
                    str_btc = f"ENABLED (SMA {p['btc_filter_ma']})" if p['use_btc'] else "DISABLED"
                    str_sl = f"{p['atr_stop_mult']:.2f}x ATR" if p['use_sl'] else "DISABLED"
                    str_tp = f"{p['atr_tp_mult']:.2f}x ATR" if p['use_tp'] else "DISABLED"
                    str_trail = f"{p['atr_trail_mult']:.2f}x ATR" if p['use_trail'] else "DISABLED"
                    
                    tqdm.write("\n" + "="*50)
                    tqdm.write(f"🚀 NEW ARCHITECTURE FOUND!")
                    tqdm.write(f"Score: {score:.0f} | PF: {pf:.2f} | WinRate: {wr*100:.1f}%")
                    tqdm.write(f"Trades: {trades} | Liqs: {liqs}")
                    tqdm.write("-" * 20)
                    tqdm.write(f"Short MA:  {s_type} {p['s_len']}")
                    tqdm.write(f"Long MA:   {l_type} {p['l_len']}")
                    tqdm.write(f"Trend Filter: {str_trend}")
                    tqdm.write(f"BTC Filter:   {str_btc}")
                    tqdm.write("-" * 20)
                    tqdm.write(f"Leverage:  {p['leverage']}x")
                    tqdm.write(f"Stop Loss: {str_sl}")
                    tqdm.write(f"Take Prof: {str_tp}")
                    tqdm.write(f"Trailing:  {str_trail}")
                    tqdm.write("="*50 + "\n")

    print("\nOPTIMIZATION COMPLETE.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    optimize()