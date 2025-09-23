import websocket
import json
import threading
import time
import pandas as pd
import numpy as np
import requests
import telegram
import asyncio
from datetime import datetime
import logging
from collections import deque
import os
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bybit_screener.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BybitScreener")

# Configuration from environment variables
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '100'))
SCAN_INTERVAL = int(os.getenv('SCAN_INTERVAL', '60'))  # 60 seconds
INSTANCE_ID = os.getenv('INSTANCE_ID', 'default')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

class EfficientSymbolData:
    def __init__(self, symbol: str):
        self.symbol = symbol
        # Optimized buffer sizes for 60-second scanning
        self.price_buffers = {
            '5': deque(maxlen=80),    # 5 minutes
            '15': deque(maxlen=80),   # 15 minutes
            '60': deque(maxlen=80),   # 1 hour
            '240': deque(maxlen=80),  # 4 hours
            'D': deque(maxlen=80)     # 1 day
        }
        self.timestamp_buffers = {
            '5': deque(maxlen=80),
            '15': deque(maxlen=80),
            '60': deque(maxlen=80),
            '240': deque(maxlen=80),
            'D': deque(maxlen=80)
        }
        self.current_price = 0
        self.last_signal_time = 0
        self.signal_cooldown = SCAN_INTERVAL
        self.last_update_time = 0
        self.price_change_24h = 0
        self.volume_24h = 0
        
        # Cache for indicator values
        self.indicator_cache = {}
        
    def update_price(self, timeframe: str, price: float, timestamp: int):
        """Update price for a specific timeframe efficiently"""
        if timeframe not in self.price_buffers:
            return
            
        self.price_buffers[timeframe].append(price)
        self.timestamp_buffers[timeframe].append(timestamp)
        
        # Invalidate cache for this timeframe
        if timeframe in self.indicator_cache:
            del self.indicator_cache[timeframe]
    
    def get_prices_series(self, timeframe: str) -> pd.Series:
        """Convert deque to pandas Series efficiently"""
        if not self.price_buffers[timeframe]:
            return pd.Series()
            
        return pd.Series(
            list(self.price_buffers[timeframe]),
            index=pd.to_datetime(list(self.timestamp_buffers[timeframe]), unit='ms')
        )
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return pd.Series([np.nan] * len(prices), index=prices.index)
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_macd(self, prices: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
        """Calculate MACD indicator"""
        if len(prices) < slow_period + signal_period:
            return pd.Series(), pd.Series(), pd.Series()
            
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_stochrsi(self, prices: pd.Series, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> tuple:
        """Calculate Stochastic RSI indicator"""
        if len(prices) < period + smooth_k + smooth_d:
            return pd.Series(), pd.Series()
        
        # Calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate Stochastic RSI
        min_val = rsi.rolling(period).min()
        max_val = rsi.rolling(period).max()
        stochrsi = (rsi - min_val) / (max_val - min_val)
        stochrsi_k = stochrsi.rolling(smooth_k).mean() * 100
        stochrsi_d = stochrsi_k.rolling(smooth_d).mean()
        
        return stochrsi_k, stochrsi_d
    
    def check_stochrsi_bullish_cross_below_85(self, stochrsi_k: pd.Series, stochrsi_d: pd.Series) -> bool:
        """Check for StochRSI bullish cross below 85 level"""
        if len(stochrsi_k) < 2 or len(stochrsi_d) < 2:
            return False
        
        current_k = stochrsi_k.iloc[-1]
        current_d = stochrsi_d.iloc[-1]
        prev_k = stochrsi_k.iloc[-2]
        prev_d = stochrsi_d.iloc[-2]
        
        # Check if both current values are below 85
        if current_k >= 85 or current_d >= 85:
            return False
        
        # Check for bullish cross (K crosses above D)
        return prev_k <= prev_d and current_k > current_d
    
    def check_4h_conditions(self) -> bool:
        """Check 4h conditions: MACD positive flip and StochRSI bullish cross below 85"""
        prices = self.get_prices_series('240')  # Bybit uses '240' for 4h
        if len(prices) < 40:
            return False
        
        # Check cache first
        cache_key = f"4h_conditions_{hash(tuple(prices[-40:]))}"
        if cache_key in self.indicator_cache:
            return self.indicator_cache[cache_key]
        
        # Calculate MACD
        _, _, histogram = self.calculate_macd(prices)
        
        # Check for MACD positive flip
        if len(histogram) < 2:
            result = False
        else:
            current_histogram = histogram.iloc[-1]
            previous_histogram = histogram.iloc[-2]
            macd_flip = previous_histogram <= 0 and current_histogram > 0
            
            # Calculate StochRSI
            stochrsi_k, stochrsi_d = self.calculate_stochrsi(prices)
            
            # Check for StochRSI bullish cross below 85
            stochrsi_condition = self.check_stochrsi_bullish_cross_below_85(stochrsi_k, stochrsi_d)
            
            result = macd_flip and stochrsi_condition
        
        # Cache result
        self.indicator_cache[cache_key] = result
        return result
    
    def check_1d_conditions(self) -> bool:
        """Check 1D conditions: at least one of the three conditions"""
        prices = self.get_prices_series('D')  # Bybit uses 'D' for 1 day
        if len(prices) < 40:
            return False
        
        # Check cache first
        cache_key = f"1d_conditions_{hash(tuple(prices[-40:]))}"
        if cache_key in self.indicator_cache:
            return self.indicator_cache[cache_key]
        
        # Condition 1: Positive MACD histogram
        _, _, histogram = self.calculate_macd(prices)
        if len(histogram) > 0 and histogram.iloc[-1] > 0:
            result = True
        else:
            # Condition 2: StochRSI bullish cross below 85
            stochrsi_k, stochrsi_d = self.calculate_stochrsi(prices)
            stochrsi_condition = self.check_stochrsi_bullish_cross_below_85(stochrsi_k, stochrsi_d)
            
            if stochrsi_condition:
                result = True
            else:
                # Condition 3: Price above 7MA, 7MA > 14MA, 14MA > 28MA
                if len(prices) >= 28:
                    ma7 = prices.rolling(7).mean().iloc[-1]
                    ma14 = prices.rolling(14).mean().iloc[-1]
                    ma28 = prices.rolling(28).mean().iloc[-1]
                    current_price = prices.iloc[-1]
                    
                    result = (current_price > ma7 and ma7 > ma14 and ma14 > ma28)
                else:
                    result = False
        
        # Cache result
        self.indicator_cache[cache_key] = result
        return result
    
    def check_macd_flip(self, timeframe: str) -> bool:
        """Check for MACD positive flip on a specific timeframe"""
        prices = self.get_prices_series(timeframe)
        if len(prices) < 25:
            return False
        
        # Check cache first
        cache_key = f"macd_flip_{timeframe}_{hash(tuple(prices[-25:]))}"
        if cache_key in self.indicator_cache:
            return self.indicator_cache[cache_key]
        
        _, _, histogram = self.calculate_macd(prices)
        
        if len(histogram) < 2:
            result = False
        else:
            current_histogram = histogram.iloc[-1]
            previous_histogram = histogram.iloc[-2]
            result = previous_histogram <= 0 and current_histogram > 0
        
        # Cache result
        self.indicator_cache[cache_key] = result
        return result
    
    def can_send_signal(self) -> bool:
        """Check if we can send a signal (cooldown period)"""
        current_time = time.time()
        return current_time - self.last_signal_time >= self.signal_cooldown
    
    def update_signal_time(self):
        """Update the last signal time"""
        self.last_signal_time = time.time()

class BybitWebSocket:
    def __init__(self, symbols: list, on_message_callback, instance_id: str = "default"):
        self.symbols = symbols
        self.on_message_callback = on_message_callback
        self.instance_id = instance_id
        self.ws = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 8
        self.last_message_time = time.time()
        
    def get_ws_url(self) -> str:
        """Get Bybit WebSocket URL"""
        return "wss://stream.bybit.com/v5/public/linear"
    
    def on_open(self, ws):
        """WebSocket on_open event handler"""
        logger.info(f"[{self.instance_id}] Connected to Bybit WebSocket")
        self.connected = True
        self.reconnect_attempts = 0
        
        # Subscribe to kline streams for all symbols
        subscription_msg = {
            "op": "subscribe",
            "args": [f"kline.1.{symbol}" for symbol in self.symbols[:50]]  # Limit to 50 symbols per connection
        }
        ws.send(json.dumps(subscription_msg))
    
    def on_message(self, ws, message):
        """WebSocket on_message event handler"""
        try:
            self.last_message_time = time.time()
            data = json.loads(message)
            self.on_message_callback(data)
        except Exception as e:
            logger.error(f"[{self.instance_id}] Error processing message: {e}")
    
    def on_error(self, ws, error):
        """WebSocket on_error event handler"""
        logger.error(f"[{self.instance_id}] WebSocket error: {error}")
        self.connected = False
    
    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket on_close event handler"""
        logger.info(f"[{self.instance_id}] WebSocket connection closed: {close_status_code} - {close_msg}")
        self.connected = False
        self.try_reconnect()
    
    def try_reconnect(self):
        """Attempt to reconnect with exponential backoff"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"[{self.instance_id}] Max reconnection attempts reached")
            return
            
        self.reconnect_attempts += 1
        delay = min(45, 2 ** self.reconnect_attempts)
        
        logger.info(f"[{self.instance_id}] Reconnecting in {delay} seconds (attempt {self.reconnect_attempts})")
        time.sleep(delay)
        self.connect()
    
    def connect(self):
        """Connect to WebSocket"""
        try:
            ws_url = self.get_ws_url()
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
        except Exception as e:
            logger.error(f"[{self.instance_id}] Error connecting: {e}")
            self.try_reconnect()
    
    def disconnect(self):
        """Disconnect from WebSocket"""
        if self.ws:
            self.ws.close()

class BybitScreener60s:
    def __init__(self, telegram_token: str, telegram_chat_id: str):
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.bot = telegram.Bot(token=telegram_token)
        
        # Store symbol data
        self.symbols_data = {}
        
        # Signal cooldown to avoid duplicates
        self.signal_cooldown = SCAN_INTERVAL
        self.recent_signals = deque(maxlen=100)
        
        # Load USDT pairs from Bybit
        self.all_usdt_symbols = self.load_usdt_symbols()
        
        # Batch processing with 60-second rotation
        self.batch_size = BATCH_SIZE
        self.total_batches = math.ceil(len(self.all_usdt_symbols) / self.batch_size)
        self.current_batch_index = 0
        self.current_batch = []
        
        # WebSocket connection
        self.ws = None
        
        # 24h data cache
        self.ticker_data_24h = {}
        self.last_24h_update = 0
        
        # Performance tracking
        self.scan_count = 0
        self.start_time = time.time()
        
        # Initialize with first batch
        self.rotate_batch()
    
    def load_usdt_symbols(self):
        """Load USDT perpetual futures symbols from Bybit"""
        symbols = set()
        
        try:
            response = requests.get("https://api.bybit.com/v5/market/instruments-info?category=linear", timeout=10)
            data = response.json()
            for symbol_info in data['result']['list']:
                if (symbol_info['quoteCoin'] == 'USDT' and 
                    symbol_info['status'] == 'Trading'):
                    symbols.add(symbol_info['symbol'])
        except Exception as e:
            logger.error(f"Error loading Bybit symbols: {e}")
        
        return list(symbols)
    
    def get_next_batch(self):
        """Get the next batch of symbols for rotation"""
        start_index = self.current_batch_index * self.batch_size
        end_index = start_index + self.batch_size
        batch = self.all_usdt_symbols[start_index:end_index]
        
        # Update batch index for next rotation
        self.current_batch_index = (self.current_batch_index + 1) % self.total_batches
        
        return batch
    
    def update_24h_data(self):
        """Update 24h price and volume data for all symbols"""
        try:
            response = requests.get("https://api.bybit.com/v5/market/tickers?category=linear", timeout=10)
            data = response.json()
            for item in data['result']['list']:
                symbol = item['symbol']
                if symbol in self.all_usdt_symbols:
                    # Bybit returns percentage as decimal (0.05 for 5%)
                    price_change_pcnt = float(item['price24hPcnt']) * 100
                    self.ticker_data_24h[symbol] = {
                        'priceChangePercent': price_change_pcnt,
                        'volume': float(item['volume24h']),
                        'turnover': float(item['turnover24h'])
                    }
        except Exception as e:
            logger.error(f"Error updating Bybit 24h data: {e}")
        
        self.last_24h_update = time.time()
        logger.info(f"[{INSTANCE_ID}] Updated 24h data for {len(self.ticker_data_24h)} symbols")
    
    def rotate_batch(self):
        """Rotate to the next batch of symbols"""
        self.current_batch = self.get_next_batch()
        
        logger.info(f"[{INSTANCE_ID}] Rotating to batch {self.current_batch_index}/{self.total_batches} with {len(self.current_batch)} symbols")
        
        # Reinitialize WebSocket with new batch
        if self.ws:
            self.ws.disconnect()
        
        self.ws = BybitWebSocket(self.current_batch, self.on_websocket_message, INSTANCE_ID)
        self.ws.connect()
    
    def on_websocket_message(self, data):
        """Process WebSocket message from Bybit"""
        try:
            if 'topic' in data and 'kline' in data['topic']:
                topic_parts = data['topic'].split('.')
                symbol = topic_parts[2]
                interval = topic_parts[1]
                
                candle_data = data['data'][0]
                price = float(candle_data['close'])
                timestamp = candle_data['start']
                
                if symbol in self.current_batch:
                    if symbol not in self.symbols_data:
                        self.symbols_data[symbol] = EfficientSymbolData(symbol)
                    
                    # Map Bybit interval to our timeframe format
                    timeframe_map = {'1': '5', '3': '15', '60': '60', '240': '240', 'D': 'D'}
                    if interval in timeframe_map:
                        mapped_timeframe = timeframe_map[interval]
                        self.symbols_data[symbol].update_price(mapped_timeframe, price, timestamp)
                    
                    # Update current price
                    self.symbols_data[symbol].current_price = price
                    self.symbols_data[symbol].last_update_time = time.time()
                    
                    # Update 24h data if available
                    if symbol in self.ticker_data_24h:
                        ticker_data = self.ticker_data_24h[symbol]
                        self.symbols_data[symbol].price_change_24h = ticker_data['priceChangePercent']
                        self.symbols_data[symbol].volume_24h = ticker_data['volume']
                    
        except Exception as e:
            logger.error(f"[{INSTANCE_ID}] Error processing Bybit message: {e}")
    
    def scan_batch(self):
        """Scan the current batch for signals"""
        signals_found = 0
        for symbol in self.current_batch:
            if symbol in self.symbols_data:
                if self.check_signals(symbol):
                    signals_found += 1
        
        self.scan_count += 1
        elapsed = time.time() - self.start_time
        scan_rate = self.scan_count / elapsed if elapsed > 0 else 0
        
        logger.info(f"[{INSTANCE_ID}] Scan {self.scan_count}: Checked {len(self.current_batch)} symbols, found {signals_found} signals. Rate: {scan_rate:.2f} scans/sec")
    
    def check_signals(self, symbol: str) -> bool:
        """Check for trading signals for a symbol"""
        if symbol not in self.symbols_data:
            return False
            
        symbol_data = self.symbols_data[symbol]
        
        # Check if we can send a signal (cooldown period)
        if not symbol_data.can_send_signal():
            return False
        
        # Check 4h conditions (Bybit uses '240' for 4h)
        condition_4h = symbol_data.check_4h_conditions()
        
        # Check 1d conditions (Bybit uses 'D' for 1 day)
        condition_1d = symbol_data.check_1d_conditions()
        
        # Check smaller timeframes
        condition_1h = symbol_data.check_macd_flip('60')    # 1 hour
        condition_15m = symbol_data.check_macd_flip('15')   # 15 minutes
        condition_5m = symbol_data.check_macd_flip('5')     # 5 minutes
        
        signal_sent = False
        
        # Priority 0: 4h and 1d conditions met
        if condition_4h and condition_1d:
            message = (f"🚨 PRIORITY 0 ALERT 🚨\n"
                      f"Symbol: {symbol}\n"
                      f"Exchange: Bybit\n"
                      f"Price: ${symbol_data.current_price:,.2f}\n"
                      f"24h Change: {symbol_data.price_change_24h:+.2f}%\n"
                      f"24h Volume: {symbol_data.volume_24h:,.0f}\n"
                      f"4H: MACD Flip + StochRSI Bullish Cross <85\n"
                      f"1D: Condition Met\n"
                      f"Instance: {INSTANCE_ID}")
            
            if not self.is_duplicate_signal(symbol, "P0"):
                self.send_telegram_alert(message)
                symbol_data.update_signal_time()
                self.record_signal(symbol, "P0")
                signal_sent = True
        
        # Priority 1: 4h condition met and smaller timeframes aligned
        elif condition_4h and condition_1h and condition_15m and condition_5m:
            message = (f"🚨 PRIORITY 1 ALERT 🚨\n"
                      f"Symbol: {symbol}\n"
                      f"Exchange: Bybit\n"
                      f"Price: ${symbol_data.current_price:,.2f}\n"
                      f"24h Change: {symbol_data.price_change_24h:+.2f}%\n"
                      f"24h Volume: {symbol_data.volume_24h:,.0f}\n"
                      f"4H: MACD Flip + StochRSI Bullish Cross <85\n"
                      f"1H/15M/5M: MACD Flips Aligned\n"
                      f"Instance: {INSTANCE_ID}")
            
            if not self.is_duplicate_signal(symbol, "P1"):
                self.send_telegram_alert(message)
                symbol_data.update_signal_time()
                self.record_signal(symbol, "P1")
                signal_sent = True
        
        return signal_sent
    
    def is_duplicate_signal(self, symbol: str, signal_type: str) -> bool:
        """Check if this signal was recently sent"""
        signal_id = f"{symbol}_{signal_type}"
        for recent_signal in self.recent_signals:
            if recent_signal[0] == signal_id and time.time() - recent_signal[1] < self.signal_cooldown:
                return True
        return False
    
    def record_signal(self, symbol: str, signal_type: str):
        """Record a signal to prevent duplicates"""
        signal_id = f"{symbol}_{signal_type}"
        self.recent_signals.append((signal_id, time.time()))
    
    async def send_telegram_alert_async(self, message: str):
        """Send alert to Telegram asynchronously"""
        try:
            await self.bot.send_message(chat_id=self.telegram_chat_id, text=message)
            logger.info(f"[{INSTANCE_ID}] Telegram alert sent: {message}")
        except Exception as e:
            logger.error(f"[{INSTANCE_ID}] Error sending Telegram message: {e}")
    
    def send_telegram_alert(self, message: str):
        """Send alert to Telegram"""
        threading.Thread(target=asyncio.run, args=(self.send_telegram_alert_async(message),)).start()
    
    def start(self):
        """Start monitoring"""
        logger.info(f"[{INSTANCE_ID}] Bybit Screener started with 60-second intervals")
        logger.info(f"[{INSTANCE_ID}] Total symbols: {len(self.all_usdt_symbols)}, Batch size: {self.batch_size}, Total batches: {self.total_batches}")
        
        # Update 24h data initially
        self.update_24h_data()
        
        # Start the main scanning loop
        self.scan_loop()
    
    def scan_loop(self):
        """Main scanning loop with 60-second intervals"""
        while True:
            try:
                scan_start = time.time()
                
                # Perform the scan
                self.scan_batch()
                
                # Calculate processing time and adjust sleep time
                processing_time = time.time() - scan_start
                sleep_time = max(1, SCAN_INTERVAL - processing_time)
                
                # Sleep until next scan
                logger.debug(f"[{INSTANCE_ID}] Processing took {processing_time:.2f}s, sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
                
                # Rotate to next batch after scan
                self.rotate_batch()
                
            except Exception as e:
                logger.error(f"[{INSTANCE_ID}] Error in scan loop: {e}")
                time.sleep(30)  # Wait 30 seconds before retrying
    
    def stop(self):
        """Stop monitoring"""
        if self.ws:
            self.ws.disconnect()
        logger.info(f"[{INSTANCE_ID}] Bybit Screener stopped")

def main():
    # Initialize the screener
    screener = BybitScreener60s(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    
    # Start monitoring
    screener.start()

if __name__ == "__main__":
    main()