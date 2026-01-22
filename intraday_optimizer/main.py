import yfinance as yf
import pandas as pd
import numpy as np
import os
from numba import njit
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION (2026 STANDARDS)
# ==========================================
INDICES = {
    'NIFTY':        {'ticker': '^NSEI',               'expiry_day': 1}, 
    'BANKNIFTY':    {'ticker': '^NSEBANK',            'expiry_day': 1}, 
    'FINNIFTY':     {'ticker': 'NIFTY_FIN_SERVICE.NS', 'expiry_day': 1}, 
    'MIDCPNIFTY':   {'ticker': 'NIFTY_MID_SELECT.NS',  'expiry_day': 1}, 
    'SENSEX':       {'ticker': '^BSESN',              'expiry_day': 3}  
}

TIMEFRAMES = ['5m'] 
MAX_YAHOO_PERIOD = "59d"  # STRICT LIMIT: 5m data only exists for last 60 days
NUM_ITERATIONS = 100000       
MIN_TRADES_TOTAL = 50        
EXIT_TIME_MINUTES = 920      

OPTION_COST_PCT = 0.0012     
DELTA = 0.55                 

if not os.path.exists("data_universal"):
    os.makedirs("data_universal")

# ==========================================
# 2. ROBUST DATA LOADING (FIXED)
# ==========================================
def load_csv_safely(file_path):
    """
    Attempts to load CSV. If it fails or is corrupt, returns None.
    Uses index_col=0 to avoid 'Missing column Date' errors.
    """
    try:
        # Load using first column as index, regardless of name
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        # Ensure the index is actually Datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')
        
        # Drop rows where index didn't parse correctly
        df = df[df.index.notna()]
        
        # Standardize index name
        df.index.name = 'Date'
        return df
    except Exception as e:
        return None

def fetch_data_universe():
    data_store = {}
    print(f"--- 1. Syncing Data (Auto-Repair Mode) ---")
    
    for idx_name, info in INDICES.items():
        ticker = info['ticker']
        file_path = f"data_universal/{idx_name}_5m.csv"
        
        # 1. Download Newest Data (Last 59 days)
        try:
            new_data = yf.download(ticker, period=MAX_YAHOO_PERIOD, interval="5m", progress=False, multi_level_index=False)
            
            # Clean MultiIndex headers if present
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = new_data.columns.droplevel(1)
            
            # Ensure standard columns exist
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in new_data.columns for col in required_cols):
                print(f"⚠️ {idx_name}: Downloaded data missing columns. Skipping update.")
                new_data = pd.DataFrame() # Empty
            else:
                new_data.dropna(inplace=True)
                new_data.index.name = 'Date'

        except Exception as e:
            print(f"❌ {idx_name}: Download error ({e})")
            new_data = pd.DataFrame()

        # 2. Handle Local File Storage
        if os.path.exists(file_path):
            old_data = load_csv_safely(file_path)
            
            if old_data is None or old_data.empty:
                print(f"⚠️ {idx_name}: Local file corrupted. Deleting and starting fresh.")
                os.remove(file_path)
                final_df = new_data
            else:
                if not new_data.empty:
                    # Append and Deduplicate
                    final_df = pd.concat([old_data, new_data])
                    final_df = final_df[~final_df.index.duplicated(keep='last')].sort_index()
                else:
                    final_df = old_data
        else:
            final_df = new_data

        # 3. Save if valid
        if not final_df.empty:
            final_df.to_csv(file_path)
            
            # 4. Load into Memory for Backtest
            if len(final_df) > 200:
                print(f"✅ {idx_name}: Ready ({len(final_df)} candles)")
                data_store[idx_name] = {
                    'opens': np.ascontiguousarray(final_df['Open'].values, dtype=np.float64),
                    'highs': np.ascontiguousarray(final_df['High'].values, dtype=np.float64),
                    'lows': np.ascontiguousarray(final_df['Low'].values, dtype=np.float64),
                    'closes': np.ascontiguousarray(final_df['Close'].values, dtype=np.float64),
                    'times': np.ascontiguousarray(final_df.index.hour.values * 60 + final_df.index.minute.values, dtype=np.int64),
                    'weekdays': np.ascontiguousarray(final_df.index.weekday.values, dtype=np.int64),
                    'months': np.ascontiguousarray(final_df.index.month.values, dtype=np.int64),
                    'expiry_day': info['expiry_day'],
                }
        else:
            print(f"❌ {idx_name}: No valid data available.")

    return data_store

# ==========================================
# 3. NUMBA ENGINES (Optimized)
# ==========================================
@njit(fastmath=True)
def calc_ma(prices, period, ma_type):
    n = len(prices)
    res = np.empty(n); res[:] = np.nan
    if n < period: return res
    if ma_type == 1: # EMA
        res[period-1] = np.mean(prices[:period])
        mult = 2 / (period + 1)
        for i in range(period, n):
            res[i] = (prices[i] - res[i-1]) * mult + res[i-1]
    else: # SMA
        w_sum = np.sum(prices[:period])
        res[period-1] = w_sum / period
        for i in range(period, n):
            w_sum = w_sum - prices[i-period] + prices[i]
            res[i] = w_sum / period
    return res

@njit(fastmath=True)
def backtest_options_monthly(opens, highs, lows, closes, weekdays, times, months, expiry_day,
                             s_ma, l_ma, sl_ma, trail, tgt, f_mode):
    n = len(closes)
    total_roi = 0.0
    m_roi = np.zeros(13, dtype=np.float64)
    in_pos = False 
    entry_px, high_px, prem, dte_entry, entry_idx = 0.0, 0.0, 0.0, 0, 0
    trades, wins, gw, gl = 0, 0, 0.0, 0.0
    
    for i in range(300, n - 1):
        if in_pos:
            high_px = max(high_px, highs[i+1])
            stop = high_px * (1.0 - trail/100.0)
            target = entry_px * (1.0 + tgt/100.0)
            exit_px = 0.0
            
            # Exits
            if (weekdays[i+1] == expiry_day and times[i+1] >= EXIT_TIME_MINUTES): exit_px = closes[i+1]
            elif lows[i+1] <= stop: exit_px = stop
            elif highs[i+1] >= target: exit_px = target
            elif s_ma[i-1] >= l_ma[i-1] and s_ma[i] < l_ma[i]: exit_px = opens[i+1]
            
            if exit_px > 0:
                # Option PnL Sim
                roi = (((exit_px - entry_px) * DELTA - (prem * (0.0001 * (3.0 - dte_entry*0.4)) * ((i - entry_idx) * 5))) / prem) - OPTION_COST_PCT
                total_roi += roi; m_roi[months[i+1]] += roi
                if roi > 0: wins += 1; gw += roi
                else: gl += abs(roi)
                trades += 1; in_pos = False

        elif 555 < times[i] < 900: # Entry Window
            if s_ma[i-1] <= l_ma[i-1] and s_ma[i] > l_ma[i]:
                t_ok = True
                if f_mode == 1 and closes[i] <= sl_ma[i]: t_ok = False
                if f_mode == 2 and closes[i] >= sl_ma[i]: t_ok = False
                if t_ok:
                    in_pos = True; entry_px = opens[i+1]; high_px = entry_px; entry_idx = i
                    dte = (expiry_day - weekdays[i]) % 7
                    dte_entry = dte; prem = entry_px * (0.003 + (0.002 * dte))
    
    return total_roi, m_roi, trades, wins, gw, gl

# ==========================================
# 4. EVALUATION & OPTIMIZATION
# ==========================================
def evaluate(p, data):
    if p['s_p'] >= p['l_p']: return -999, None
    port_m_roi = np.zeros(13)
    idx_scores, res_map = [], {}
    t_wins, t_gw, t_gl, t_tr = 0, 0.0, 0.0, 0
    
    for name, d in data.items():
        s_ma = calc_ma(d['closes'], p['s_p'], p['s_t'])
        l_ma = calc_ma(d['closes'], p['l_p'], p['l_t'])
        sl_ma = calc_ma(d['closes'], p['sl_p'], p['sl_t'])
        
        roi, m_roi, tr, w, gw, gl = backtest_options_monthly(d['opens'], d['highs'], d['lows'], d['closes'], d['weekdays'], d['times'], d['months'], d['expiry_day'], s_ma, l_ma, sl_ma, p['trail'], p['tgt'], p['f_m'])
        
        if roi < -0.50: return -999, None
        idx_scores.append(roi); port_m_roi += m_roi; t_tr += tr; t_gw += gw; t_gl += gl; res_map[name] = roi

    if t_tr < MIN_TRADES_TOTAL: return -999, None
    pf = t_gw / t_gl if t_gl > 0 else 1.5
    if pf < 1.1: return -999, None
    
    active_months = port_m_roi[port_m_roi != 0]
    med_month = np.median(active_months) if len(active_months) > 0 else 0
    
    return np.median(idx_scores) * pf, {'pf': pf, 'tr': t_tr, 'm_roi': med_month, 'res': res_map}

def run_universal_optimization():
    data = fetch_data_universe()
    if not data: 
        print("❌ No valid data found after sync. Please check your internet or try again later.")
        return

    best_score = -9999
    print(f"\n⚡ Starting Optimization ({NUM_ITERATIONS} iterations)...")
    
    for _ in tqdm(range(NUM_ITERATIONS)):
        p = {'s_p': np.random.randint(5, 45), 'l_p': np.random.randint(15, 95), 'sl_p': np.random.randint(100, 300), 's_t': np.random.randint(0, 2), 'l_t': np.random.randint(0, 2), 'sl_t': np.random.randint(0, 2), 'trail': np.random.uniform(0.1, 1.3), 'tgt': np.random.uniform(1.0, 5.0), 'f_m': np.random.randint(0, 3)}
        
        score, met = evaluate(p, data)
        
        if met is not None and score > best_score:
            best_score = score
            st, lt, slt = ["SMA", "EMA"][p['s_t']], ["SMA", "EMA"][p['l_t']], ["SMA", "EMA"][p['sl_t']]
            fm = ["OFF", "Price > Super", "Price < Super"][p['f_m']]
            brk = " | ".join([f"{k}:{v*100:.1f}%" for k, v in met['res'].items()])
            
            tqdm.write(f"\n🚀 NEW BEST: {score:.4f} (PF: {met['pf']:.2f})")
            tqdm.write(f"   MA: {st}{p['s_p']} x {lt}{p['l_p']} | Filter: {fm} ({slt}{p['sl_p']})")
            tqdm.write(f"   Exits: Trail {p['trail']:.2f}% | Target {p['tgt']:.2f}%")
            tqdm.write(f"   Monthly Med: {met['m_roi']*100:.2f}% | Trades: {met['tr']}")
            tqdm.write(f"   Breakdown: {brk}\n")

if __name__ == "__main__":
    run_universal_optimization()