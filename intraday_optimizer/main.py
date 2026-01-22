import yfinance as yf
import pandas as pd
import numpy as np
import os
import math
from numba import njit
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION (SAFETY FIRST)
# ==========================================
INDICES = {
    'NIFTY':        {'ticker': '^NSEI',               'expiry_day': 1, 'lot': 65,  'step': 50,  'weight': 1.0},  
    'BANKNIFTY':    {'ticker': '^NSEBANK',            'expiry_day': 1, 'lot': 30,  'step': 100, 'weight': 1.0}, 
    'FINNIFTY':     {'ticker': 'NIFTY_FIN_SERVICE.NS', 'expiry_day': 1, 'lot': 60,  'step': 50,  'weight': 1.0},  
    'MIDCPNIFTY':   {'ticker': 'NIFTY_MID_SELECT.NS',  'expiry_day': 1, 'lot': 120, 'step': 25,  'weight': 0.5}, # Low Weight
    'SENSEX':       {'ticker': '^BSESN',              'expiry_day': 3, 'lot': 10,  'step': 100, 'weight': 1.0}  
}

TIMEFRAMES = ['5m'] 
MAX_YAHOO_PERIOD = "59d"
BUDGET_PER_TRADE = 3000.0   
RISK_FREE_RATE = 0.065      

NUM_ITERATIONS = 50000       
MIN_TRADES_TOTAL = 40
EXIT_TIME_MINUTES = 920      

TRADE_COST_PCT = 0.002      

# --- SAFETY BARRIERS ---
MAX_SINGLE_TRADE_LOSS_PCT = 0.70  # If any trade loses > 70%, discard strategy.
MIN_PROFIT_FACTOR = 1.2           # Minimum PF required to be considered

if not os.path.exists("data_universal"):
    os.makedirs("data_universal")

# ==========================================
# 2. DATA ENGINE
# ==========================================
def load_vix_data():
    vix_path = "data_universal/INDIAVIX.csv"
    try:
        vix = yf.download("^INDIAVIX", period="1y", progress=False)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.droplevel(1)
        vix = vix[['Close']].rename(columns={'Close': 'VIX'})
        vix.to_csv(vix_path)
        return vix
    except:
        if os.path.exists(vix_path):
            return pd.read_csv(vix_path, index_col=0, parse_dates=True)
        return None

def fetch_data_universe():
    data_store = {}
    print(f"--- 1. Syncing Market Data & Volatility ---")
    vix_df = load_vix_data()
    
    for idx_name, info in INDICES.items():
        ticker = info['ticker']
        file_path = f"data_universal/{idx_name}_5m.csv"
        try:
            new_data = yf.download(ticker, period=MAX_YAHOO_PERIOD, interval="5m", progress=False)
            if isinstance(new_data.columns, pd.MultiIndex): new_data.columns = new_data.columns.droplevel(1)
            new_data.dropna(inplace=True)
            if os.path.exists(file_path):
                old_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                final_df = pd.concat([old_data, new_data])
                final_df = final_df[~final_df.index.duplicated(keep='last')].sort_index()
            else: final_df = new_data
            final_df.to_csv(file_path)
        except:
            if os.path.exists(file_path): final_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            else: continue

        if len(final_df) > 200:
            final_df.index.name = 'Date' 
            final_df['DateOnly'] = final_df.index.date
            if vix_df is not None:
                vix_df.index.name = 'Date'; vix_df['DateOnly'] = vix_df.index.date
                merged = final_df.reset_index().merge(vix_df[['DateOnly', 'VIX']], on='DateOnly', how='left')
                merged.set_index('Date', inplace=True)
                merged['VIX'] = merged['VIX'].fillna(15.0)
            else:
                merged = final_df; merged['VIX'] = 15.0

            data_store[idx_name] = {
                'opens': np.ascontiguousarray(merged['Open'].values, dtype=np.float64),
                'highs': np.ascontiguousarray(merged['High'].values, dtype=np.float64),
                'lows': np.ascontiguousarray(merged['Low'].values, dtype=np.float64),
                'closes': np.ascontiguousarray(merged['Close'].values, dtype=np.float64),
                'vix': np.ascontiguousarray(merged['VIX'].values, dtype=np.float64),
                'times': np.ascontiguousarray(merged.index.hour.values * 60 + merged.index.minute.values, dtype=np.int64),
                'weekdays': np.ascontiguousarray(merged.index.weekday.values, dtype=np.int64),
                'months': np.ascontiguousarray(merged.index.month.values, dtype=np.int64),
                'expiry_day': info['expiry_day'],
                'lot_size': info['lot'],
                'strike_step': info['step'],
                'weight': info['weight']
            }
            print(f"✅ {idx_name}: Loaded {len(merged)} candles.")
    return data_store

# ==========================================
# 3. BLACK-SCHOLES (SKEWED)
# ==========================================
@njit(fastmath=True)
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / 1.41421356))

@njit(fastmath=True)
def get_skewed_sigma(sigma_atm, spot, strike):
    if spot == 0: return sigma_atm
    moneyness = math.log(strike / spot)
    skew_factor = -1.5 * moneyness if moneyness > 0.0 else -2.0 * moneyness
    return max(0.05, sigma_atm * (1.0 + max(-0.3, min(0.3, skew_factor))))

@njit(fastmath=True)
def black_scholes_call(S, K, T, r, base_sigma, use_skew):
    if T <= 0: return max(0.0, S - K)
    sigma = get_skewed_sigma(base_sigma, S, K) if use_skew else base_sigma
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return max(0.0, S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2))

@njit(fastmath=True)
def find_quality_strike(spot, budget, lot_size, T, r, sigma, step):
    atm_strike = round(spot / step) * step
    atm_prem = black_scholes_call(spot, atm_strike, T, r, sigma, True)
    if (atm_prem * lot_size) <= budget: return atm_strike, atm_prem

    otm_strike = atm_strike + step
    otm_prem = black_scholes_call(spot, otm_strike, T, r, sigma, True)
    if (otm_prem * lot_size) <= budget and otm_prem > 5.0: return otm_strike, otm_prem
    return 0.0, 0.0

# ==========================================
# 4. TRADING ENGINE (WITH SAFETY BARRIERS)
# ==========================================
@njit(fastmath=True)
def calc_ma(prices, period, ma_type):
    n = len(prices)
    res = np.empty(n); res[:] = np.nan
    if n < period: return res
    if ma_type == 1: # EMA
        res[period-1] = np.mean(prices[:period])
        mult = 2 / (period + 1)
        for i in range(period, n): res[i] = (prices[i] - res[i-1]) * mult + res[i-1]
    else: # SMA
        w_sum = np.sum(prices[:period])
        res[period-1] = w_sum / period
        for i in range(period, n): w_sum = w_sum - prices[i-period] + prices[i]; res[i] = w_sum / period
    return res

@njit(fastmath=True)
def backtest_safe_strategy(opens, highs, lows, closes, vix, weekdays, times, months, 
                           expiry_day, lot_size, strike_step,
                           s_ma, l_ma, sl_ma, trail_pct, tgt_pct, f_mode):
    n = len(closes)
    
    # PnL Tracking
    total_pnl = 0.0
    monthly_pnl = np.zeros(13, dtype=np.float64)
    trades = 0; wins = 0; gross_win = 0.0; gross_loss = 0.0
    
    # SAFETY TRACKING
    max_single_trade_loss_pct = 0.0
    peak_equity = 0.0
    current_equity = 0.0
    max_drawdown = 0.0
    
    in_pos = False; entry_strike = 0.0; entry_premium = 0.0; entry_time_idx = 0; days_to_expiry_entry = 0.0
    high_spot_since_entry = 0.0
    
    for i in range(300, n - 1):
        if in_pos:
            curr_spot = closes[i+1]; curr_high = highs[i+1]; curr_low = lows[i+1]
            high_spot_since_entry = max(high_spot_since_entry, curr_high)
            should_exit = False
            
            stop_level = high_spot_since_entry * (1.0 - trail_pct/100.0)
            target_level = opens[entry_time_idx+1] * (1.0 + tgt_pct/100.0)
            
            if curr_low <= stop_level: should_exit = True
            elif curr_high >= target_level: should_exit = True
            elif s_ma[i-1] >= l_ma[i-1] and s_ma[i] < l_ma[i]: should_exit = True 
            
            is_expiry = (weekdays[i+1] == expiry_day)
            if is_expiry and times[i+1] >= EXIT_TIME_MINUTES: should_exit = True
            
            if should_exit:
                mins_elapsed = (i - entry_time_idx) * 5
                new_dte = max(0.0001, days_to_expiry_entry - (mins_elapsed / (24*60)))
                T_new = new_dte / 365.0
                sigma = vix[i+1] / 100.0
                
                # Exit Calculation
                exit_premium = black_scholes_call(curr_spot, entry_strike, T_new, RISK_FREE_RATE, sigma, True)
                
                # --- PNL MATH ---
                invested_capital = entry_premium * lot_size
                gross_pnl = (exit_premium - entry_premium) * lot_size
                costs = (invested_capital * TRADE_COST_PCT) + (exit_premium * lot_size * TRADE_COST_PCT)
                net_pnl = gross_pnl - costs
                
                # --- SAFETY BARRIER 1: BLOW-UP CHECK ---
                if net_pnl < 0:
                    loss_pct = abs(net_pnl) / invested_capital
                    if loss_pct > max_single_trade_loss_pct:
                        max_single_trade_loss_pct = loss_pct
                
                # --- SAFETY BARRIER 2: DRAWDOWN TRACKING ---
                total_pnl += net_pnl
                current_equity += net_pnl
                
                if current_equity > peak_equity:
                    peak_equity = current_equity
                
                dd = peak_equity - current_equity
                if dd > max_drawdown:
                    max_drawdown = dd
                    
                monthly_pnl[months[i+1]] += net_pnl
                if net_pnl > 0: wins += 1; gross_win += net_pnl
                else: gross_loss += abs(net_pnl)
                trades += 1; in_pos = False

        elif 555 < times[i] < 900:
            if s_ma[i-1] <= l_ma[i-1] and s_ma[i] > l_ma[i]:
                trend_ok = True
                if f_mode == 1 and closes[i] <= sl_ma[i]: trend_ok = False
                if f_mode == 2 and closes[i] >= sl_ma[i]: trend_ok = False
                
                if trend_ok:
                    curr_spot = opens[i+1]
                    days_diff = (expiry_day - weekdays[i])
                    if days_diff < 0: days_diff += 7
                    dte = float(days_diff) if float(days_diff) > 0 else 0.2
                    
                    T = dte / 365.0; sigma = vix[i] / 100.0
                    
                    k, prem = find_quality_strike(curr_spot, BUDGET_PER_TRADE, lot_size, T, RISK_FREE_RATE, sigma, strike_step)
                    
                    if k > 0:
                        in_pos = True; entry_strike = k; entry_premium = prem; entry_time_idx = i; days_to_expiry_entry = dte
                        high_spot_since_entry = curr_spot

    return total_pnl, monthly_pnl, trades, wins, gross_win, gross_loss, max_single_trade_loss_pct, max_drawdown

# ==========================================
# 5. OPTIMIZATION LOOP
# ==========================================
def evaluate(p, data):
    # Rule 1: Separation Force
    if p['l_p'] < (p['s_p'] + 10): return -99999, None
    
    port_m_pnl = np.zeros(13); idx_pnl = {}; total_trades = 0; g_win = 0.0; g_loss = 0.0
    weighted_pnl_score = 0.0
    max_blowup_pct = 0.0
    total_drawdown = 0.0
    
    for name, d in data.items():
        s_ma = calc_ma(d['closes'], p['s_p'], p['s_t'])
        l_ma = calc_ma(d['closes'], p['l_p'], p['l_t'])
        sl_ma = calc_ma(d['closes'], p['sl_p'], p['sl_t'])
        
        pnl, m_pnl, tr, w, gw, gl, max_loss_pct, max_dd = backtest_safe_strategy(
            d['opens'], d['highs'], d['lows'], d['closes'], d['vix'],
            d['weekdays'], d['times'], d['months'], 
            d['expiry_day'], d['lot_size'], d['strike_step'],
            s_ma, l_ma, sl_ma, p['trail'], p['tgt'], p['f_m']
        )
        
        # --- KILL SWITCH ---
        if max_loss_pct > MAX_SINGLE_TRADE_LOSS_PCT: 
            return -99999, None # Immediate Disqualification
            
        if pnl < -10000: return -99999, None
        
        weighted_pnl_score += (pnl * d['weight'])
        idx_pnl[name] = pnl; port_m_pnl += m_pnl; total_trades += tr
        g_win += gw; g_loss += gl
        max_blowup_pct = max(max_blowup_pct, max_loss_pct)
        total_drawdown += max_dd # Sum of DD across indices
        
    if total_trades < MIN_TRADES_TOTAL: return -99999, None
    pf = g_win / g_loss if g_loss > 0 else 1.5
    if pf < MIN_PROFIT_FACTOR: return -99999, None
    
    # --- SCORING FORMULA WITH DD PENALTY ---
    # Score = (Profit * PF) / (1 + Drawdown_Penalty)
    # This rewards high profit, high consistency, but penalizes deep valleys.
    
    # Normalize DD for scoring (e.g. 5000 DD adds 0.5 penalty)
    dd_penalty = total_drawdown / 10000.0 
    
    score = (weighted_pnl_score * pf) / (1.0 + dd_penalty)
    
    active_months = port_m_pnl[port_m_pnl != 0]
    med_month = np.median(active_months) if len(active_months) > 0 else 0
    return score, {'pf': pf, 'tr': total_trades, 'm_pnl': med_month, 'res': idx_pnl, 'max_loss_pct': max_blowup_pct, 'dd': total_drawdown}

def run_budget_optimization():
    data = fetch_data_universe()
    if not data: return
    print(f"\n⚡ Optimizing for Budget: ₹{BUDGET_PER_TRADE} | Safety: ON")
    print(f"🛑 Kill Switch: >{int(MAX_SINGLE_TRADE_LOSS_PCT*100)}% Loss in 1 Trade")
    print(f"⚖️  Scoring: Profit vs Drawdown Weighted")
    
    best_score = -999999
    
    for _ in tqdm(range(NUM_ITERATIONS)):
        s_p = np.random.randint(5, 50)
        l_p = np.random.randint(s_p + 10, 100) # Force gap
        
        p = {
             's_p': s_p, 
             'l_p': l_p, 
             'sl_p': np.random.randint(100, 300),
             's_t': np.random.randint(0, 2), 
             'l_t': np.random.randint(0, 2), 
             'sl_t': np.random.randint(0, 2),
             'trail': np.random.uniform(0.1, 2.0), 
             'tgt': np.random.uniform(1.0, 6.0), 
             'f_m': np.random.randint(0, 3)
        }
        
        score, met = evaluate(p, data)
        if met is not None and score > best_score:
            best_score = score
            st, lt, slt = ["SMA", "EMA"][p['s_t']], ["SMA", "EMA"][p['l_t']], ["SMA", "EMA"][p['sl_t']]
            fm = ["OFF", "Price > Super", "Price < Super"][p['f_m']]
            brk = " | ".join([f"{k}:₹{int(v)}" for k, v in met['res'].items()])
            
            tqdm.write(f"\n🛡️  NEW SAFEST BEST (Score: {int(score)})")
            tqdm.write(f"   📈 {st}{p['s_p']} x {lt}{p['l_p']} | Filter: {fm}")
            tqdm.write(f"   🎯 Tgt: +{p['tgt']:.2f}% | Trail: -{p['trail']:.2f}%")
            tqdm.write(f"   💰 PF: {met['pf']:.2f} | Med Monthly: ₹{int(met['m_pnl'])}")
            tqdm.write(f"   ⚠️ Max Single Loss: {met['max_loss_pct']*100:.1f}% | Total DD: ₹{int(met['dd'])}")
            tqdm.write(f"   📊 {brk}")

if __name__ == "__main__":
    run_budget_optimization()