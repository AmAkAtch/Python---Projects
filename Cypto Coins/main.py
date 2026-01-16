import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
import random  # Needed for random number generation
from tqdm import tqdm
import time

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Coins to test (Universal Robustness)
COINS = [
    # --- Majors ---
    'BTC/USDT',
    'ETH/USDT',
    'BNB/USDT',
    'SOL/USDT',
    'XRP/USDT',
    'ADA/USDT',
    
    # --- Strong Alts ---
    'LINK/USDT',   # Chainlink
    'LTC/USDT',    # Litecoin
    'BCH/USDT',    # Bitcoin Cash
    'SUI/USDT',    # Sui
    'RUNE/USDT',   # Thorchain
    'XLM/USDT',    # Stellar
    'SHIB/USDT',   # Shiba Inu
    'RENDER/USDT', # Render (Note: might be RNDR on some older data)
    'JUP/USDT',    # Jupiter
    'ONDO/USDT',   # Ondo
    'OM/USDT',     # Mantra
    'KAS/USDT',    # Kaspa (Note: Check if available on your exchange's spot market)
    'BAT/USDT',    # Basic Attention Token
    'GLM/USDT',    # Golem
    'AUCTION/USDT' # Bounce
]
TIMEFRAME = '1d'
LIMIT_CANDLES = 1000

# How many random combinations do you want to try?
# 1,000 takes about 1-2 minutes.
# 10,000 takes about 15-20 minutes (Better results).
TOTAL_TESTS = 10000 

# Define the Ranges (Min, Max) to pick from
RANGES = {
    'rsi14_min': 2, 'rsi14_max': 100,
    'smooth14_min': 1, 'smooth14_max': 50, # Smoothing > 50 is usually too laggy
    'rsi21_min': 2, 'rsi21_max': 100,
    'smooth21_min': 1, 'smooth21_max': 50,
    'ma_options': [50, 100, 150, 200] # We stick to standard MA lists usually
}

# ==========================================
# 2. DATA LOADING
# ==========================================
def fetch_data(symbol, timeframe, limit):
    exchange = ccxt.binance()
    # print(f"Fetching {symbol}...") # Commented out to reduce clutter
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# ==========================================
# 3. STRATEGY ENGINE
# ==========================================
class BacktestEngine:
    def __init__(self, df, params):
        self.df = df.copy()
        self.params = params
        
    def run(self):
        p = self.params
        
        # --- CALCULATE INDICATORS ---
        # We calculate only what is needed for this specific loop to save memory
        try:
            # RSI 1
            rsi1 = ta.rsi(self.df['close'], length=p['rsi14Period'])
            if rsi1 is None: return -100 # Invalid params check
            smooth_rsi1 = ta.sma(rsi1, length=p['smoothingPeriod14'])
            
            # RSI 2
            rsi2 = ta.rsi(self.df['close'], length=p['rsi21Period'])
            if rsi2 is None: return -100
            smooth_rsi2 = ta.sma(rsi2, length=p['smoothingPeriod21'])
            
            # Trend MA
            trend_ma = ta.sma(self.df['close'], length=p['trendMaPeriod'])
            
            # Clean Data
            # If smoothing is huge (e.g. 100), we lose the first 100 candles.
            # We must handle NaNs safely.
            start_index = max(p['rsi14Period'], p['rsi21Period'], p['trendMaPeriod']) + \
                          max(p['smoothingPeriod14'], p['smoothingPeriod21']) + 2
            
            if start_index >= len(self.df):
                return -100 # Parameters too big for data size
            
        except Exception:
            return -100

        # Convert to numpy arrays for speed
        closes = self.df['close'].values
        # Fill NaNs with 0 to prevent crashes, though we skip them via start_index
        s_rsi1 = smooth_rsi1.fillna(0).values 
        s_rsi2 = smooth_rsi2.fillna(0).values
        t_ma = trend_ma.fillna(0).values
        
        # Strategy State
        in_position = False
        entry_price = 0.0
        highest_price = 0.0
        position_size = 0.0
        take_profit_triggered = False
        balance = 1000.0 
        
        # --- THE LOOP ---
        for i in range(start_index, len(self.df)):
            current_close = closes[i]
            
            # Update Highest Price
            if in_position and current_close > highest_price:
                highest_price = current_close

            # Entry Logic
            # smoothedRsi14 > smoothedRsi21 AND crossover AND close > trendMa
            cross_over = (s_rsi1[i-1] <= s_rsi2[i-1]) and (s_rsi1[i] > s_rsi2[i])
            trend_ok = current_close > t_ma[i]
            
            if not in_position and cross_over and trend_ok:
                in_position = True
                entry_price = current_close
                highest_price = current_close
                position_size = 1.0
                take_profit_triggered = False
                
            # Take Profit Logic (50% exit at 1.3x)
            if in_position and not take_profit_triggered:
                if current_close >= (entry_price * 1.30):
                    balance += (balance * 0.50 * 0.30) # Add profit from 50%
                    position_size = 0.5
                    take_profit_triggered = True

            # Exit Logic
            # smoothedRsi14 < smoothedRsi21 AND crossunder AND close < highestPrice
            cross_under = (s_rsi1[i-1] >= s_rsi2[i-1]) and (s_rsi1[i] < s_rsi2[i])
            
            if in_position and cross_under and (current_close < highest_price):
                pnl = (current_close - entry_price) / entry_price
                balance += (balance * position_size * pnl)
                in_position = False
                position_size = 0.0
        
        # Return Net Profit %
        return ((balance - 1000.0) / 1000.0) * 100

# ==========================================
# 4. RANDOM OPTIMIZER
# ==========================================
def optimize():
    print("Loading Market Data...")
    market_data = {}
    for coin in COINS:
        df = fetch_data(coin, TIMEFRAME, LIMIT_CANDLES)
        if df is not None:
            market_data[coin] = df
        time.sleep(0.5)

    print(f"Data Loaded. Starting {TOTAL_TESTS} random tests...")
    
    best_score = -999999
    best_params = None
    
    # We use tqdm to show a progress bar
    for _ in tqdm(range(TOTAL_TESTS)):
        
        # 1. Generate Random Parameters
        params = {
            'rsi14Period': random.randint(RANGES['rsi14_min'], RANGES['rsi14_max']),
            'smoothingPeriod14': random.randint(RANGES['smooth14_min'], RANGES['smooth14_max']),
            'rsi21Period': random.randint(RANGES['rsi21_min'], RANGES['rsi21_max']),
            'smoothingPeriod21': random.randint(RANGES['smooth21_min'], RANGES['smooth21_max']),
            'trendMaPeriod': random.choice(1, 200)
        }
        
        # 2. Test on ALL coins
        total_score = 0
        valid_coin_count = 0
        
        for coin, df in market_data.items():
            engine = BacktestEngine(df, params)
            result = engine.run()
            
            # If result is -100, it meant the indicators crashed (invalid params), so we skip this param set
            if result != -100:
                total_score += result
                valid_coin_count += 1
        
        if valid_coin_count == len(market_data):
            avg_score = total_score / valid_coin_count
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params

    # ==========================================
    # 5. RESULTS
    # ==========================================
    print("\n" + "="*40)
    print("       OPTIMIZATION COMPLETE")
    print("="*40)
    print(f"Best Average Return: {best_score:.2f}%")
    print("-" * 40)
    print("BEST PARAMETERS FOUND:")
    if best_params:
        for k, v in best_params.items():
            print(f"{k}: {v}")
    else:
        print("No profitable parameters found.")
    print("-" * 40)

if __name__ == "__main__":
    optimize()