import json
import os
import numpy as np
import pandas as pd
import ta
import gym
from gym import spaces
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention
from transformers import GPT2Model, GPT2Config
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import asyncio
import time
import ccxt.async_support as ccxt_async
import logging
import logging.handlers
from tenacity import retry, wait_exponential, stop_after_attempt
import pickle
import websockets
from collections import deque
from bayes_opt import BayesianOptimization
import gc
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import multiprocessing
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from darts import TimeSeries
from darts.models import TFTModel
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import torch.cuda.amp
import torch.optim as optim
import zlib
import keyboard


# คอนฟิกหลักของระบบ (กำหนดค่าพื้นฐานสำหรับการเทรดและโมเดล)
# การใช้งาน: ปรับค่าในนี้เพื่อควบคุมพฤติกรรมของระบบ เช่น API key, ระดับความเสี่ยง, หรือโหมดจำลอง
CONFIG = {
    'binance_api_key': 'YJS7F8A7GiK64lEbcqREcVfWzMPOsEtvPnjlq8YWNVKsw1ClCiGg1hD8AHJozZnZ',  # คีย์ API สำหรับเชื่อมต่อ Binance Futures
    'binance_api_secret': '6qstSbNbveryYMDGjy64SqboFrrphYxapTh1ufZ0yBaZp2Uhn0SHokfpRqUzPQOo',  # รหัสลับ API สำหรับ Binance Futures
    'dry_run': False,  # True = จำลองการเทรด, False = เทรดจริง
    'profit_lock_percentage': 0.05,  # เปอร์เซ็นต์กำไรที่ล็อกเมื่อถึงเป้า
    'loss_strategy': 'dynamic',  # กลยุทธ์ตัดขาดทุน ('dynamic' = ปรับตาม ATR)
    'stop_loss_percentage': 0.005,  # เปอร์เซ็นต์ขาดทุนเริ่มต้น (ปรับได้ตามกลยุทธ์)พ
    'cut_loss_threshold': 0.2,  # ขีดจำกัดขาดทุนสูงสุด (เป็น fraction เช่น 20%)
    'risk_per_trade': 0.2,  # ความเสี่ยงต่อการเทรดแต่ละครั้ง (ปรับตาม KPI)
    'initial_balance': 100,  # ยอดเงินเริ่มต้นในบัญชี (หน่วย USDT)
    'reinvest_profits': True,  # True = นำกำไรมา reinvest, False = ไม่ reinvest
    'log_level': 'INFO',  # ระดับการบันทึก log ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    'max_api_retries': 10,  # จำนวนครั้งสูงสุดที่พยายามเรียก API ใหม่เมื่อล้มเหลว
    'api_timeout': 30,  # วินาทีที่รอการตอบกลับจาก API ก่อน timeout
    'futures_weight': 0.9,  # น้ำหนักการใช้งาน API สำหรับ Futures
    'rate_limit_per_minute': 6000,  # จำนวนคำขอ API ต่อนาที (อัพเดทตาม Binance 2025)
    'max_drawdown': 0.2,  # เปอร์เซ็นต์ drawdown สูงสุดที่ยอมรับได้ (20%)
    'sim_volatility': 0.02,  # ความผันผวนในโหมดจำลอง
    'sim_trend': 0.001,  # แนวโน้มราคาในโหมดจำลอง
    'sim_spike_chance': 0.05,  # โอกาสเกิดการพุ่งของราคาในโหมดจำลอง
    'auto_ml_interval': 500,  # จำนวน steps ก่อนฝึก ML อัตโนมัติ
    'rl_train_interval': 200,  # จำนวน steps ก่อนฝึก Reinforcement Learning
    'target_kpi_daily': 100000.0,  # เป้าหมายกำไรรายวัน (หน่วย USDT)
    'min_daily_kpi': 50000.0,  # เป้าหมายกำไรขั้นต่ำระหว่างวัน (50% ของ target)
    'checkpoint_interval': 360,  # จำนวน steps ก่อนบันทึก checkpoint (ทุก 6 ชม.)
    'auto_bug_fix': True,  # True = แก้บั๊กอัตโนมัติเมื่อเกิดข้อผิดพลาด
    'bug_fix_attempts': 5,  # จำนวนครั้งสูงสุดที่พยายามแก้บั๊ก
    'resource_adaptive': True,  # True = ปรับการใช้ทรัพยากร (CPU/RAM) อัตโนมัติ
    'min_ram_reserve_mb': 1024,  # จำนวน MB ของ RAM ที่สำรองไว้
    'min_cpu_idle_percent': 20,  # เปอร์เซ็นต์ CPU ว่างขั้นต่ำ
    'min_volatility_threshold': 0.001,    # จาก 0.005 ลดลง
    'liquidity_threshold': 50,
    'tft_forecast_horizon': 60,  # จำนวน timesteps ที่ TFT คาดการณ์ (นาที)
    'gnn_update_interval': 900,  # วินาทีที่อัพเดท GNN graph (15 นาที)
    'madrl_agent_count': 50,  # จำนวน agent สูงสุดใน MADRL (ตามจำนวนเหรียญ)
    'bayes_opt_steps': 10,  # จำนวน iterations สำหรับ Bayesian Optimization
    'max_leverage_per_symbol': {},  # เก็บค่า leverage สูงสุดของแต่ละเหรียญ (จาก API)
    'max_coins_per_trade': 15,  # จำนวนเหรียญสูงสุดที่เทรดพร้อมกัน
    'min_coins_per_trade': 6,  # จำนวนเหรียญขั้นต่ำที่เทรดพร้อมกัน
    'maml_lr_inner': 0.01,  # Learning rate ภายในสำหรับ MAML
    'maml_lr_outer': 0.001,  # Learning rate ภายนอกสำหรับ MAML
    'maml_steps': 5,  # จำนวน steps สำหรับ fine-tuning ใน MAML
    'multi_tf_list': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d'],  # รายการ timeframe (อัพเดท 13 timeframe)
    'historical_years': 5,  # จำนวนปีของข้อมูลย้อนหลัง
    'system_running': False,  # สถานะการรันระบบ
    'trade_log_file': 'trade_log.xlsx',  # ไฟล์บันทึกการเทรด
    'nas_iterations': 100  # จำนวน iterations สำหรับ Neural Architecture Search
}

# คำนวณ input_dim อัตโนมัติ
num_timeframes = len(CONFIG['multi_tf_list'])
features_per_tf = 10 + 10 + 5  # 10 base indicators, 10 synthetic, 5 GNN correlations
CONFIG['input_dim'] = num_timeframes * features_per_tf

INITIAL_BALANCE = CONFIG['initial_balance']
REINVEST_PROFITS = CONFIG['reinvest_profits']
TIMESTEPS = 10
MAX_RETRIES = CONFIG['max_api_retries']
RETRY_DELAY = 5
RATE_LIMIT = CONFIG['rate_limit_per_minute']
MAX_HISTORY_SIZE = 1000
SLIPPAGE_DEFAULT = 0.0005
TAKER_FEE = 0.0004
LATENCY_DEFAULT = 0.001
API_CALL_INTERVAL = 60 / RATE_LIMIT

# การตั้งค่าระบบบันทึก log
logging.basicConfig(
    level=getattr(logging, CONFIG['log_level']),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler('trading.log', maxBytes=10*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
error_logger = logging.getLogger('error_logger')
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler('error.log')
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
error_logger.addHandler(error_handler)

# การตั้งค่า exchange สำหรับ Binance Futures
exchange = ccxt_async.binance({
    'apiKey': CONFIG['binance_api_key'],
    'secret': CONFIG['binance_api_secret'],
    'enableRateLimit': True,
    'adjustForTimeDifference': True,
    'timeout': CONFIG['api_timeout'] * 1000,
    'rateLimit': int(60000 / RATE_LIMIT),
    'options':{
        'recvWindow': 15000
    } 
})

api_call_timestamps = deque(maxlen=RATE_LIMIT)
last_api_call = 0
executor = ThreadPoolExecutor(max_workers=5)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
ARCHIVE_DIR = os.path.join(BASE_DIR, 'archives')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# คลาส APIManager: จัดการการเรียก API และควบคุม rate limit
class APIManager:
    def __init__(self):
        self.weight_used = 0
        self.weight_limit = 6000  # อัพเดทตาม Binance 2025
        self.last_reset = time.time()
        self.is_rate_limited = False
        self.ban_until = 0
        self.request_count = 0
        self.rate_limit_status = {}
        self.kpi_priority_weight = 0

    async def update_weight(self, response):
        if 'x-mbx-used-weight-1m' in response.headers:
            self.weight_used = int(response.headers['x-mbx-used-weight-1m'])
            self.rate_limit_status['used_weight'] = self.weight_used
        if 'x-mbx-order-count-1m' in response.headers:
            self.rate_limit_status['order_count'] = int(response.headers['x-mbx-order-count-1m'])
        if time.time() - self.last_reset >= 60:
            self.weight_used = 0
            self.rate_limit_status = {}
            self.last_reset = time.time()

    async def rate_limit_control(self):
        now = time.time()
        if self.is_rate_limited and now < self.ban_until:
            wait_time = self.ban_until - now
            logging.warning(f"ถูกจำกัด IP รอ {wait_time:.2f} วินาที")
            await asyncio.sleep(wait_time)
            self.is_rate_limited = False
        if self.weight_used >= self.weight_limit * 0.9:
            wait_time = 60 - (now - self.last_reset)
            if wait_time > 0:
                logging.warning(f"น้ำหนัก API ใกล้เต็ม รอ {wait_time:.2f} วินาที")
                await asyncio.sleep(wait_time)
            self.weight_used = 0
            self.last_reset = now
        api_call_timestamps.append(now)
        self.request_count += 1
        if len(api_call_timestamps) >= RATE_LIMIT:
            wait_time = API_CALL_INTERVAL - (now - api_call_timestamps[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)

    async def optimize_api_usage(self, kpi_tracker=None):
        if self.weight_used > self.weight_limit * 0.5:
            global API_CALL_INTERVAL
            API_CALL_INTERVAL = min(API_CALL_INTERVAL * 1.2, 0.1)
            logging.info(f"ปรับ API Call Interval เป็น {API_CALL_INTERVAL:.4f} วินาที")
        if kpi_tracker and kpi_tracker.total_profit < CONFIG['min_daily_kpi']:
            self.kpi_priority_weight += 0.1
            logging.info(f"เพิ่มน้ำหนัก API priority เป็น {self.kpi_priority_weight:.2f} เพื่อเกิน KPI")
        self.request_count = 0

    async def fetch_max_leverage(self):
        await self.rate_limit_control()
        try:
            info = await exchange.fapiPublic_get_exchangeInfo()
            max_leverage = {s['symbol']: s['leverage']['max'] for s in info['symbols'] if s['symbol'].endswith('USDT')}
            CONFIG['max_leverage_per_symbol'] = max_leverage
            logging.info(f"ดึง max_leverage สำเร็จ: {len(max_leverage)} เหรียญ")
            return max_leverage
        except Exception as e:
            logging.error(f"ดึง max_leverage ล้มเหลว: {e}")
            return {}

    async def cancel_order_async(self, symbol, order_id):
        endpoint = '/fapi/v1/order'
        params = {'symbol': symbol, 'orderId': order_id}
        await self.rate_limit_control()
        try:
            response = await exchange.fapiPrivate_delete_order(params)
            logging.info(f"ยกเลิกออเดอร์ {order_id} สำหรับ {symbol} สำเร็จ")
            return response
        except Exception as e:
            logging.error(f"ยกเลิกออเดอร์ {order_id} ล้มเหลว: {e}")
            return None

    async def predict_usage(self):
        # ทำนายการใช้งาน API ในอนาคต (60 วินาทีข้างหน้า)
        usage_history = [self.weight_used] * 60
        return np.mean(usage_history)  # คาดการณ์จากค่าเฉลี่ย

api_manager = APIManager()

# คลาส WebSocketManager: จัดการ WebSocket เพื่อรับข้อมูลเรียลไทม์จาก Binance
class WebSocketManager:
    def __init__(self):
        self.url = 'wss://fstream.binance.com/ws'
        self.backup_url = 'wss://stream.binance.com:9443/ws'
        self.data = {}
        self.running = False
        self.subscribed_symbols = set()
        self.reconnect_attempts = 0
        self.max_reconnects = 10
        self.cache = {}
        self.all_usdt_pairs = []
        self.db_conn = sqlite3.connect(os.path.join(BASE_DIR, 'ws_backup.db'), timeout=10)
        self.db_conn.execute("CREATE TABLE IF NOT EXISTS ws_data (symbol TEXT, timestamp REAL, close REAL, volume REAL, funding_rate REAL, depth REAL)")
        self.balance_data = {'free': 0, 'locked': 0}  # เพิ่มเพื่อเก็บยอดเงินเรียลไทม์
        self.position_data = {}  # เพิ่มเพื่อเก็บข้อมูลตำแหน่ง

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
    async def fetch_all_usdt_pairs(self):
        await api_manager.rate_limit_control()
        markets = await exchange.load_markets()
        self.all_usdt_pairs = [s for s in markets.keys() if s.endswith('/USDT')]
        logging.info(f"ดึงคู่ USDT ทั้งหมด: {len(self.all_usdt_pairs)} เหรียญ")

    async def update_symbols(self, symbols):
        new_symbols = [s.lower().replace('/', '') + '@ticker' for s in symbols]
        if set(new_symbols) != self.subscribed_symbols:
            self.subscribed_symbols = set(new_symbols[:1024])
            if self.running:
                await self.resubscribe()

    async def resubscribe(self, websocket=None):
        await api_manager.rate_limit_control()
        if websocket:
            # เพิ่มการสมัครรับข้อมูลยอดเงินและตำแหน่งเรียลไทม์
            await websocket.send(json.dumps({
                'method': 'SUBSCRIBE',
                'params': ['!userData@balance', '!userData@position'] + list(self.subscribed_symbols),
                'id': 1
            }))

    async def start(self, symbols):
        if not self.all_usdt_pairs:
            await self.fetch_all_usdt_pairs()
        await self.update_symbols(symbols)
        self.running = True
        urls = [self.url, self.backup_url]
        current_url_idx = 0
        while self.running and CONFIG['system_running']:
            try:
                async with websockets.connect(urls[current_url_idx]) as websocket:
                    await self.resubscribe(websocket)
                    self.reconnect_attempts = 0
                    while self.running and CONFIG['system_running']:
                        message = await asyncio.wait_for(websocket.recv(), timeout=10)
                        data = json.loads(message)
                        await self._handle_message(data, websocket)
            except (websockets.ConnectionClosed, asyncio.TimeoutError) as e:
                self.reconnect_attempts += 1
                logging.warning(f"WebSocket ล้มเหลว: {e}, พยายามใหม่ครั้งที่ {self.reconnect_attempts}")
                if self.reconnect_attempts > self.max_reconnects:
                    logging.error("WebSocket ล้มเหลวเกินจำกัด ใช้ข้อมูลสำรอง")
                    await self.use_fallback_data(symbols)
                await asyncio.sleep(min(5 * self.reconnect_attempts, 60))
                current_url_idx = (current_url_idx + 1) % len(urls)

    async def stop(self):
        self.running = False
        self.db_conn.close()

    async def _handle_message(self, data, websocket):
        if 'ping' in data:
            await websocket.send(json.dumps({'pong': data['ping']}))
            logging.info("ส่ง pong ตอบ ping จาก Binance")
        elif 'e' in data and data['e'] == 'balanceUpdate':
            self._update_balance(data)
        elif 'e' in data and data['e'] == 'position':
            self._update_position(data)
        elif 's' in data:
            self._update_data(data)
            await self.save_to_sqlite(data)

    def _update_balance(self, data):
        # อัพเดทยอดเงินเรียลไทม์จาก WebSocket
        asset = data.get('a', 'USDT')
        if asset == 'USDT':
            self.balance_data = {
                'free': float(data.get('f', 0)),
                'locked': float(data.get('l', 0))
            }
            logging.debug(f"อัพเดทยอดเงิน USDT: free={self.balance_data['free']}, locked={self.balance_data['locked']}")

    def _update_position(self, data):
        # อัพเดทข้อมูลตำแหน่งเรียลไทม์
        symbol = data.get('s')
        if symbol:
            self.position_data[symbol] = {
                'size': float(data.get('ps', 0)),
                'entryPrice': float(data.get('ep', 0)),
                'leverage': float(data.get('lev', 1))
            }

    def _update_data(self, data):
        symbol = data['s']
        self.data[symbol] = {
            'close': float(data['c']),
            'volume': float(data['v']),
            'timestamp': datetime.utcnow(),
            'funding_rate': float(data.get('r', 0.0001)),
            'depth': float(data.get('b', 0)) - float(data.get('a', 0))
        }
        self.cache[symbol] = self.data[symbol]
        if len(self.cache) > 1000:
            self.cache.pop(next(iter(self.cache)))

    async def save_to_sqlite(self, data):
        symbol = data['s']
        with self.db_conn:
            self.db_conn.execute(
                "INSERT INTO ws_data (symbol, timestamp, close, volume, funding_rate, depth) VALUES (?, ?, ?, ?, ?, ?)",
                (symbol, datetime.utcnow().timestamp(), float(data['c']), float(data['v']),
                 float(data.get('r', 0.0001)), float(data.get('b', 0)) - float(data.get('a', 0)))
            )
            self.db_conn.commit()

    async def fetch_backup_data(self, symbol):
        with self.db_conn:
            cursor = self.db_conn.execute(
                "SELECT close, volume, funding_rate, depth FROM ws_data WHERE symbol=? ORDER BY timestamp DESC LIMIT 100",
                (symbol,)
            )
            data = cursor.fetchall()
        if data:
            df = pd.DataFrame(data, columns=['close', 'volume', 'funding_rate', 'depth'])
            logging.info(f"ดึงข้อมูลสำรองสำหรับ {symbol}: {len(df)} แถว")
            return df
        return pd.DataFrame()

    async def use_fallback_data(self, symbols):
        for symbol in symbols:
            if symbol not in self.data or not self.data[symbol]:
                df = await self.fetch_backup_data(symbol)
                if not df.empty:
                    self.data[symbol] = {
                        'close': df['close'].iloc[-1],
                        'volume': df['volume'].iloc[-1],
                        'timestamp': datetime.utcnow(),
                        'funding_rate': df['funding_rate'].iloc[-1],
                        'depth': df['depth'].iloc[-1]
                    }
                else:
                    last_price = await exchange.fetch_ticker(symbol)['last']
                    self.data[symbol] = {
                        'close': last_price,
                        'volume': 100,
                        'timestamp': datetime.utcnow(),
                        'funding_rate': 0.0001,
                        'depth': 0
                    }
                logging.info(f"ใช้ข้อมูลสำรองสำหรับ {symbol}: ราคา {self.data[symbol]['close']}")

    def get_latest_price(self, symbol):
        default_price = 10000
        return self.cache.get(symbol.upper(), self.data.get(symbol.upper(), {})).get('close', default_price)

    def get_latest_balance(self):
        # เมธอดสำหรับดึงยอดเงินล่าสุดจาก WebSocket
        return self.balance_data.get('free', 0)

    async def prefetch_data(self, symbols, timeframes):
        for symbol in symbols:
            for tf in timeframes:
                await api_manager.rate_limit_control()
                try:
                    klines = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=100)
                    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    logging.debug(f"ดึงข้อมูลล่วงหน้าสำหรับ {symbol} ใน {tf} สำเร็จ")
                except Exception as e:
                    logging.error(f"ดึงข้อมูลล่วงหน้าสำหรับ {symbol} ใน {tf} ล้มเหลว: {e}")

ws_manager = WebSocketManager()

# คลาส SSD: โมเดล State Space Dynamics สำหรับทำนายสถานะถัดไป
class SSD(nn.Module):
    def __init__(self, input_dim):
        super(SSD, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU()
        ).to(self.device)
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.ReLU()
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.01)
        self.adaptive_lr_factor = 1.0

    def forward(self, x):
        x = x.to(self.device)
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def train(self, state_batch, next_state_batch, volatility=None):
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        reconstructed = self.forward(state_batch)
        loss = nn.MSELoss()(reconstructed, next_state_batch)
        if volatility:
            self.adaptive_lr_factor = min(1.0, max(0.1, volatility / CONFIG['min_volatility_threshold']))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = 0.0001 * self.adaptive_lr_factor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# คลาส QNetworkTransformer: โมเดล Q-Learning ที่ใช้ Transformer
class QNetworkTransformer(nn.Module):
    def __init__(self, input_dim, action_space_size):
        super(QNetworkTransformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = GPT2Config(n_embd=128, n_layer=4, n_head=8, n_positions=TIMESTEPS)
        self.transformer = GPT2Model(config)
        self.fc1 = nn.Linear(input_dim * TIMESTEPS, 256)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(256 + config.n_embd, 128)
        self.dropout2 = nn.Dropout(0.1)
        self.q_output = nn.Linear(128, action_space_size)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.01)
        self.confidence = deque(maxlen=50)
        self.to(self.device)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.to(self.device)
        flat_x = x.view(batch_size, -1)
        fc1_out = torch.relu(self.fc1(flat_x))
        fc1_out = self.dropout1(fc1_out)
        transformer_out = self.transformer(inputs_embeds=x).last_hidden_state[:, -1, :]
        combined = torch.cat((fc1_out, transformer_out), dim=1)
        fc2_out = torch.relu(self.fc2(combined))
        fc2_out = self.dropout2(fc2_out)
        q_values = self.q_output(fc2_out)
        return q_values

    def train(self, states, actions, rewards, next_states):
        q_values = self.forward(torch.FloatTensor(states).to(self.device))
        loss = nn.MSELoss()(q_values, torch.FloatTensor(actions).to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        confidence = 1 / (loss.item() + 1e-6)
        self.confidence.append(confidence)
        return loss.item()

# คลาส EvoGAN: โมเดล GAN ที่มีวิวัฒนาการสำหรับสร้างกลยุทธ์
class EvoGAN:
    def __init__(self, input_dim, action_space_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = nn.Sequential(
            nn.Linear(input_dim * TIMESTEPS, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_space_size),
            nn.Softmax(dim=-1)
        ).to(self.device)
        self.discriminator = nn.Sequential(
            nn.Linear(action_space_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0001, weight_decay=0.01)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, weight_decay=0.01)
        self.evo_population = []
        self.strategy_confidence = {}

    def generate(self, state):
        state = torch.FloatTensor(state).to(self.device)
        strategy = self.generator(state.view(-1, TIMESTEPS * state.shape[-1]))
        confidence = self.discriminator(strategy).mean().item()
        self.strategy_confidence[tuple(strategy.cpu().detach().numpy()[0])] = confidence
        return strategy

    def train(self, real_strategies, fake_strategies):
        real_strategies = torch.FloatTensor(real_strategies).to(self.device)
        fake_strategies = torch.FloatTensor(fake_strategies).to(self.device)
        real_labels = torch.ones(real_strategies.size(0), 1).to(self.device)
        fake_labels = torch.zeros(fake_strategies.size(0), 1).to(self.device)
        self.disc_optimizer.zero_grad()
        real_loss = nn.BCELoss()(self.discriminator(real_strategies), real_labels)
        fake_loss = nn.BCELoss()(self.discriminator(fake_strategies.detach()), fake_labels)
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        self.disc_optimizer.step()
        self.gen_optimizer.zero_grad()
        gen_loss = nn.BCELoss()(self.discriminator(fake_strategies), real_labels)
        gen_loss.backward()
        self.gen_optimizer.step()
        return disc_loss.item(), gen_loss.item()

    def evolve(self, strategies, rewards):
        self.evo_population = sorted(zip(strategies, rewards), key=lambda x: x[1], reverse=True)[:10]
        for _ in range(5):
            parent1, parent2 = np.random.choice(self.evo_population[:5], 2, replace=False)
            child = (parent1[0] + parent2[0]) / 2 + np.random.normal(0, 0.01, size=parent1[0].shape)
            self.evo_population.append((child, 0))
        return [p[0] for p in self.evo_population[:10]]

    def search_architecture(self, state_dim, action_dim):
        # ค้นหาสถาปัตยกรรมโมเดลที่ดีที่สุดด้วย Neural Architecture Search
        population = self._generate_initial_population()
        for _ in range(CONFIG['nas_iterations']):
            fitness = self._evaluate_population(population)
            population = self._evolve_population(population, fitness)
        return population[0]  # โมเดลที่ดีที่สุด

    def _generate_initial_population(self):
        # สร้างประชากรเริ่มต้น (placeholder)
        return [self.generator.state_dict() for _ in range(10)]

    def _evaluate_population(self, population):
        # ประเมินความฟิตของแต่ละโมเดล (placeholder)
        return [random.random() for _ in population]

    def _evolve_population(self, population, fitness):
        # วิวัฒนาการประชากร (placeholder)
        sorted_pop = [p for _, p in sorted(zip(fitness, population), reverse=True)]
        return sorted_pop[:10]

# คลาส DDPG: Deep Deterministic Policy Gradient สำหรับการกระทำต่อเนื่อง
class DDPG:
    def __init__(self, state_dim, action_dim, symbol):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_action = CONFIG['max_leverage_per_symbol'].get(symbol, 125)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim),
            nn.Tanh()
        ).to(self.device)
        self.actor_target = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim),
            nn.Tanh()
        ).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        ).to(self.device)
        self.critic_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001, weight_decay=0.01)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001, weight_decay=0.01)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.confidence_history = deque(maxlen=50)

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().detach().numpy() * self.max_action
        return np.clip(action, 5, self.max_action)

    def train(self, state_batch, action_batch, reward_batch, next_state_batch):
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        next_actions = self.actor_target(next_state_batch)
        target_q = self.critic_target(torch.cat([next_state_batch, next_actions], dim=1))
        target_q = reward_batch + 0.99 * target_q.detach()
        current_q = self.critic(torch.cat([state_batch, action_batch], dim=1))
        critic_loss = nn.MSELoss()(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actions_pred = self.actor(state_batch)
        actor_loss = -self.critic(torch.cat([state_batch, actions_pred], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(0.001 * param.data + 0.999 * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(0.001 * param.data + 0.999 * target_param.data)
        self.confidence_history.append(1 / (critic_loss.item() + 1e-6))

# คลาส TFTWrapper: Temporal Fusion Transformer สำหรับการพยากรณ์
class TFTWrapper:
    def __init__(self, input_dim, forecast_horizon=CONFIG['tft_forecast_horizon']):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TFTModel(
            input_chunk_length=TIMESTEPS,
            output_chunk_length=forecast_horizon,
            hidden_size=64,
            lstm_layers=2,
            num_attention_heads=4,
            dropout=0.1,
            batch_size=32,
            n_epochs=10,
            pl_trainer_kwargs={"accelerator": "gpu", "devices": 1} if torch.cuda.is_available() else {}
        )
        self.input_dim = input_dim
        self.scaler = RobustScaler()
        self.confidence_scores = deque(maxlen=50)
        self.multi_tf_data = {tf: None for tf in CONFIG['multi_tf_list']}

    def preprocess(self, data):
        df = pd.DataFrame(data, columns=['close', 'volume', 'RSI', 'MACD', 'RV', 'funding_rate', 'depth'])
        scaled_data = self.scaler.fit_transform(df)
        series = TimeSeries.from_dataframe(pd.DataFrame(scaled_data, columns=df.columns))
        return series

    def preprocess_multi_tf(self, multi_tf_data):
        all_series = []
        for tf in CONFIG['multi_tf_list']:
            if tf in multi_tf_data and multi_tf_data[tf] is not None and not multi_tf_data[tf].empty:
                df = pd.DataFrame(multi_tf_data[tf], columns=['close', 'volume', 'RSI', 'MACD', 'RV', 'funding_rate', 'depth'])
                scaled_data = self.scaler.fit_transform(df)
                series = TimeSeries.from_dataframe(pd.DataFrame(scaled_data, columns=df.columns))
                all_series.append(series)
        if all_series:
            combined_series = all_series[0]
            for s in all_series[1:]:
                combined_series = combined_series.stack(s)
            return combined_series
        return self.preprocess(np.zeros((TIMESTEPS, 7)))

    def predict(self, state_data):
        if isinstance(state_data, dict):
            series = self.preprocess_multi_tf(state_data)
        else:
            series = self.preprocess(state_data)
        with torch.no_grad():
            pred = self.model.predict(n=CONFIG['tft_forecast_horizon'], series=series)
        pred_values = pred.values()
        confidence = np.mean([1 / (np.std(pred_values[:, i]) + 1e-6) for i in range(pred_values.shape[1])])
        self.confidence_scores.append(confidence)
        return pred_values

    def train(self, state_batch, target_batch):
        if isinstance(state_batch, dict):
            series = self.preprocess_multi_tf(state_batch)
            target = self.preprocess_multi_tf(target_batch)
        else:
            series = self.preprocess(state_batch)
            target = self.preprocess(target_batch)
        self.model.fit(series=series, future_covariates=target, verbose=False)

    def update_realtime(self, ws_data, symbol):
        if symbol in ws_data:
            # แก้ไข Syntax Error: เปลี่ยน gri0 เป็น 0
            latest_data = np.array([[ws_data[symbol]['close'], ws_data[symbol]['volume'], 0, 0, 0,
                                   ws_data[symbol]['funding_rate'], ws_data[symbol]['depth']]])
            series = self.preprocess(latest_data)
            self.model.fit(series=series, verbose=False)
            logging.debug(f"อัพเดท TFT เรียลไทม์สำหรับ {symbol}")

# คลาส GNN: Graph Neural Network สำหรับวิเคราะห์ความสัมพันธ์ระหว่างเหรียญ
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(GNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.confidence = deque(maxlen=50)
        self.to(self.device)

    def forward(self, x, edge_index):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x

    def train(self, graph_data):
        self.optimizer.zero_grad()
        out = self.forward(graph_data.x, graph_data.edge_index)
        loss = nn.MSELoss()(out, graph_data.y)
        loss.backward()
        self.optimizer.step()
        self.confidence.append(1 / (loss.item() + 1e-6))
        return loss.item()

# คลาส MADRL: Multi-Agent Deep Reinforcement Learning สำหรับหลายเหรียญ
class MADRL:
    def __init__(self, state_dim, action_dim, num_agents=CONFIG['madrl_agent_count'], symbols=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_agents = min(num_agents, CONFIG['madrl_agent_count'])
        self.symbols = symbols or []
        self.max_actions = [CONFIG['max_leverage_per_symbol'].get(s, 125) for s in self.symbols]
        self.actors = [nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        ).to(self.device) for _ in range(self.num_agents)]
        self.actor_targets = [nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        ).to(self.device) for _ in range(self.num_agents)]
        self.critic = nn.Sequential(
            nn.Linear(state_dim * self.num_agents + action_dim * self.num_agents, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
        self.critic_target = nn.Sequential(
            nn.Linear(state_dim * self.num_agents + action_dim * self.num_agents, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ).to(self.device)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=0.0001) for actor in self.actors]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        for i in range(self.num_agents):
            self.actor_targets[i].load_state_dict(self.actors[i].state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.confidence_history = deque(maxlen=50)

    def act(self, states):
        states = torch.FloatTensor(states).to(self.device)
        actions = [actor(states[i]).cpu().detach().numpy() * self.max_actions[i]
                   for i, actor in enumerate(self.actors)]
        return np.clip(np.array(actions), 5, self.max_actions)

    def train(self, state_batch, action_batch, reward_batch, next_state_batch):
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        next_actions = torch.stack([actor(next_state_batch[i]) for i, actor in enumerate(self.actor_targets)])
        critic_input = torch.cat([state_batch.view(-1), action_batch.view(-1)], dim=0)
        next_critic_input = torch.cat([next_state_batch.view(-1), next_actions.view(-1)], dim=0)
        target_q = self.critic_target(next_critic_input) + 0.99 * reward_batch
        current_q = self.critic(critic_input)
        critic_loss = nn.MSELoss()(current_q, target_q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        for i, actor in enumerate(self.actors):
            actions_pred = actor(state_batch[i])
            actor_loss = -self.critic(torch.cat([state_batch.view(-1), actions_pred.view(-1)], dim=0)).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
        for i, (target, actor) in enumerate(zip(self.actor_targets, self.actors)):
            for t_param, param in zip(target.parameters(), actor.parameters()):
                t_param.data.copy_(0.001 * param.data + 0.999 * t_param.data)
        for t_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            t_param.data.copy_(0.001 * param.data + 0.999 * t_param.data)
        self.confidence_history.append(1 / (critic_loss.item() + 1e-6))

    def update_symbols(self, new_symbols):
        self.symbols = new_symbols
        self.num_agents = min(len(new_symbols), CONFIG['madrl_agent_count'])
        self.max_actions = [CONFIG['max_leverage_per_symbol'].get(s, 125) for s in self.symbols]
        self.actors = self.actors[:self.num_agents]
        self.actor_targets = self.actor_targets[:self.num_agents]
        self.actor_optimizers = self.actor_optimizers[:self.num_agents]
        logging.info(f"อัพเดท MADRL symbols: {self.num_agents} agents")

# คลาส MetaSelector: โมเดลเลือกเหรียญด้วย Meta-Learning
class MetaSelector(nn.Module):
    def __init__(self, input_dim):
        super(MetaSelector, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=CONFIG['maml_lr_outer'])
        self.confidence = {}

    def forward(self, x):
        x = x.to(self.device)
        return self.model(x)

    def predict(self, state):
        with torch.no_grad():
            score = self.forward(torch.FloatTensor(state).to(self.device)).cpu().numpy()
            symbol_key = tuple(state)
            confidence = self.confidence.get(symbol_key, 1.0)
        return score[0] * confidence

    def train_few_shot(self, state_batch, reward_batch):
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        fast_weights = [p.clone() for p in self.parameters()]
        for _ in range(CONFIG['maml_steps']):
            pred = self.model(state_batch)
            loss = nn.MSELoss()(pred, reward_batch)
            grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            fast_weights = [w - CONFIG['maml_lr_inner'] * g for w, g in zip(fast_weights, grads)]
        for i, state in enumerate(state_batch):
            symbol_key = tuple(state.cpu().numpy())
            self.confidence[symbol_key] = 1 / (loss.item() + 1e-6)
        return fast_weights, loss.item()

    def train_meta(self, task_batch):
        meta_loss = 0
        for states, rewards in task_batch:
            fast_weights, loss = self.train_few_shot(states, rewards)
            pred = nn.Sequential(*self.model)(torch.FloatTensor(states).to(self.device))
            meta_loss += nn.MSELoss()(pred, torch.FloatTensor(rewards).to(self.device))
        self.optimizer.zero_grad()
        meta_loss.backward()
        self.optimizer.step()
        return meta_loss.item()

# คลาส UnifiedQuantumTrader: รวมโมเดลทั้งหมดสำหรับการเทรด
class UnifiedQuantumTrader:
    def __init__(self, input_dim, discrete_action_size=3, continuous_action_dim=2, num_symbols=CONFIG['madrl_agent_count']):
        self.input_dim = input_dim
        self.discrete_action_size = discrete_action_size
        self.continuous_action_dim = continuous_action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.qnt = QNetworkTransformer(input_dim, discrete_action_size)
        self.evogan = EvoGAN(input_dim, discrete_action_size)
        self.ddpg = DDPG(input_dim, continuous_action_dim, 'BTC/USDT')
        self.ssd = SSD(input_dim)
        self.tft = TFTWrapper(input_dim)
        self.gnn = GNN(input_dim)
        self.madrl = MADRL(input_dim, continuous_action_dim, num_symbols, symbols=['BTC/USDT']*num_symbols)
        self.meta_selector = MetaSelector(input_dim)
        self.replay_buffer = None
        self.strategy_memory = []
        self.loss_history = {
            'qnt': [], 'ssd': [], 'evogan_disc': [], 'evogan_gen': [],
            'ddpg_actor': [], 'ddpg_critic': [], 'tft': [], 'gnn': [], 'madrl': [], 'meta': []
        }
        self.bayes_opt = BayesianOptimization(
            f=self._bayes_objective,
            pbounds={'qnt_w': (0, 1), 'evogan_w': (0, 1), 'tft_w': (0, 1), 'gnn_w': (0, 1), 'madrl_w': (0, 1)},
            random_state=42
        )
        self.model_weights = {'qnt_w': 0.2, 'evogan_w': 0.2, 'tft_w': 0.2, 'gnn_w': 0.2, 'madrl_w': 0.2}
        self.resource_manager = IntelligentResourceManager()
        self.overfit_detector = {'val_loss': deque(maxlen=50), 'train_loss': deque(maxlen=50)}
        self.best_loss = float('inf')
        self.patience = 10
        self.wait = 0
        self.adaptive_symbol_selector = {}
        self.current_symbols = ['BTC/USDT'] * num_symbols
        self.multi_tf_data = {tf: deque(maxlen=MAX_HISTORY_SIZE) for tf in CONFIG['multi_tf_list']}
        self.scaler = torch.amp.GradScaler()
        self.risk_guardian = RiskGuardian()  # เพิ่ม RiskGuardian เข้ามาใน trader

    def _bayes_objective(self, qnt_w, evogan_w, tft_w, gnn_w, madrl_w):
        total = qnt_w + evogan_w + tft_w + gnn_w + madrl_w
        if total == 0:
            return 0
        weights = {k: v/total for k, v in locals().items() if k.endswith('_w')}
        if self.replay_buffer and self.replay_buffer.buffer:
            batch = self.replay_buffer.sample(32)
            if batch:
                states, discrete_actions, continuous_actions, rewards, _, _, _, _, _ = batch
                pred_discrete, pred_continuous = self._combine_predictions(states, weights)
                discrete_loss = np.mean((pred_discrete - discrete_actions) ** 2)
                continuous_loss = np.mean((pred_continuous - continuous_actions) ** 2)
                return -(discrete_loss + continuous_loss)
        return 0

    def _combine_predictions(self, state_data, weights):
        with torch.cuda.amp.autocast():
            q_values = self.qnt(torch.FloatTensor(state_data).to(self.device)).cpu().detach().numpy()
            evogan_strategies = self.evogan.generate(state_data).cpu().detach().numpy()
            tft_pred = self.tft.predict(state_data)
            gnn_pred = self.gnn.forward(torch.FloatTensor(state_data).to(self.device),
                                      self._create_graph(len(state_data))).cpu().detach().numpy()
            madrl_actions = self.madrl.act(state_data)
        model_confidences = {
            'qnt': np.mean(self.qnt.confidence) if self.qnt.confidence else 1.0,
            'evogan': np.mean(list(self.evogan.strategy_confidence.values())) if self.evogan.strategy_confidence else 1.0,
            'tft': np.mean(self.tft.confidence_scores) if self.tft.confidence_scores else 1.0,
            'gnn': np.mean(self.gnn.confidence) if self.gnn.confidence else 1.0,
            'madrl': np.mean(self.madrl.confidence_history) if self.madrl.confidence_history else 1.0
        }
        total_confidence = sum(model_confidences.values())
        dynamic_weights = {k: v / total_confidence * weights[k] for k, v in model_confidences.items()}
        discrete_pred = (dynamic_weights['qnt'] * q_values +
                        dynamic_weights['evogan'] * evogan_strategies +
                        dynamic_weights['tft'] * tft_pred[:, :self.discrete_action_size])
        continuous_pred = (dynamic_weights['madrl'] * madrl_actions +
                          dynamic_weights['gnn'] * gnn_pred[:, :self.continuous_action_dim])
        return discrete_pred, continuous_pred

    def _create_graph(self, num_nodes):
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                symbol_i = self.current_symbols[i % len(self.current_symbols)]
                symbol_j = self.current_symbols[j % len(self.current_symbols)]
                if symbol_i in ws_manager.data and symbol_j in ws_manager.data:
                    corr = np.corrcoef(
                        [ws_manager.data[symbol_i]['close']] * TIMESTEPS,
                        [ws_manager.data[symbol_j]['close']] * TIMESTEPS
                    )[0, 1] if ws_manager.data[symbol_i]['close'] and ws_manager.data[symbol_j]['close'] else 0
                    if abs(corr) > 0.5:
                        edges.append([i, j])
                        edges.append([j, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().to(self.device) if edges else torch.tensor([[0, 0]], dtype=torch.long).t().to(self.device)
        return Data(x=torch.ones((num_nodes, self.input_dim)), edge_index=edge_index)

    def select_top_coins(self, all_symbols, ws_data, kpi_tracker):
        scores = {}
        multi_tf_states = self._aggregate_multi_tf_data(ws_data)
        for symbol in all_symbols:
            if symbol in ws_data:
                state = np.array([ws_data[symbol]['close'], ws_data[symbol]['volume'], 0, 0, 0,
                                ws_data[symbol]['funding_rate'], ws_data[symbol]['depth']])
                meta_score = self.meta_selector.predict(state)
                reward_pred = self.tft.predict(multi_tf_states.get(symbol, state.reshape(1, -1)))
                volatility = np.std(reward_pred)
                liquidity = ws_data[symbol]['volume']
                logging.info(f"{symbol} volume = {liquidity}")
                if liquidity >= CONFIG['liquidity_threshold']:
                    scores[symbol] = (meta_score * reward_pred.mean() * volatility * liquidity) / (1 + ws_data[symbol]['funding_rate'])
        sorted_symbols = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if not scores:
            logging.warning("⚠️ ไม่มีเหรียญผ่านเกณฑ์ ใช้ BTC/USDT ชั่วคราว")
            scores['BTC/USDT'] = 1.0  # ใช้ BTC/USDT เป็น default
    
        num_coins = max(CONFIG['min_coins_per_trade'], min(CONFIG['max_coins_per_trade'],
                        int(len(sorted_symbols) * (kpi_tracker.total_profit / CONFIG['target_kpi_daily'] + 0.5))))
        top_symbols = [s[0] for s in sorted_symbols[:num_coins]]
        self.current_symbols = top_symbols
        self.madrl.update_symbols(top_symbols)
        self.adaptive_symbol_selector = scores
        logging.info(f"เลือก {len(top_symbols)} เหรียญ: {top_symbols[:5]}...")
        return top_symbols

    def _aggregate_multi_tf_data(self, ws_data):
        multi_tf_states = {}
        for symbol in ws_data:
            state = np.array([ws_data[symbol]['close'], ws_data[symbol]['volume'], 0, 0, 0,
                            ws_data[symbol]['funding_rate'], ws_data[symbol]['depth']])
            for tf in CONFIG['multi_tf_list']:
                self.multi_tf_data[tf].append(state)
            multi_tf_states[symbol] = {tf: np.array(list(self.multi_tf_data[tf])[-TIMESTEPS:]) for tf in CONFIG['multi_tf_list']}
        return multi_tf_states

    def predict(self, state_data):
        self.resource_manager.adjust_resources(self)
        discrete_pred, continuous_pred = self._combine_predictions(state_data, self.model_weights)
        return discrete_pred, continuous_pred

    async def train(self, states, discrete_actions, continuous_actions, rewards, next_states):
        batch_size = min(len(states), self.resource_manager.model_batch_sizes['qnt'])
        if batch_size < 1:
            return
        states = np.array(states[:batch_size])
        discrete_actions = np.array(discrete_actions[:batch_size])
        continuous_actions = np.array(continuous_actions[:batch_size])
        rewards = np.array(rewards[:batch_size])
        next_states = np.array(next_states[:batch_size])
        volatility = np.mean([self.replay_buffer.buffer[-1][-2] for _ in range(min(10, len(self.replay_buffer.buffer)))
                            if self.replay_buffer.buffer[-1][-2] is not None]) if self.replay_buffer.buffer else 0.01
        lr_factor = min(1.0, max(0.1, volatility / CONFIG['min_volatility_threshold']))
        for opt in [self.qnt.optimizer, self.ddpg.actor_optimizer, self.ddpg.critic_optimizer]:
            for param_group in opt.param_groups:
                param_group['lr'] = 0.0001 * lr_factor
        val_size = int(batch_size * 0.2)
        train_size = batch_size - val_size
        idx = np.random.permutation(batch_size)
        train_idx, val_idx = idx[:train_size], idx[train_size:]
        train_states, val_states = states[train_idx], states[val_idx]
        train_discrete, val_discrete = discrete_actions[train_idx], discrete_actions[val_idx]
        train_continuous, val_continuous = continuous_actions[train_idx], continuous_actions[val_idx]
        train_rewards, val_rewards = rewards[train_idx], rewards[val_idx]
        train_next_states, val_next_states = next_states[train_idx], next_states[val_idx]

        with torch.cuda.amp.autocast():
            qnt_loss = self.qnt.train(train_states, train_discrete, train_rewards, train_next_states)
            self.ddpg.train(train_states, train_continuous, train_rewards, train_next_states)
            ssd_loss = self.ssd.train(train_states, train_next_states, volatility)
            disc_loss, gen_loss = self.evogan.train(train_discrete, self.evogan.generate(train_states).cpu().detach().numpy())
            self.tft.train(train_states, train_next_states)
            self.madrl.train(train_states, train_continuous, train_rewards, train_next_states)
            meta_loss = self.meta_selector.train_meta([(train_states, train_rewards)])

        val_q_values = self.qnt(torch.FloatTensor(val_states).to(self.device)).cpu().detach().numpy()
        val_loss = np.mean((val_q_values - val_discrete) ** 2)
        train_loss = np.mean((self.qnt(torch.FloatTensor(train_states).to(self.device)).cpu().detach().numpy() - train_discrete) ** 2)
        self.overfit_detector['val_loss'].append(val_loss)
        self.overfit_detector['train_loss'].append(train_loss)

        if len(self.overfit_detector['val_loss']) > 10 and np.mean(self.overfit_detector['val_loss']) > np.mean(self.overfit_detector['train_loss']) * 1.5:
            logging.warning(f"ตรวจพบ overfitting: val_loss={np.mean(self.overfit_detector['val_loss']):.4f}, train_loss={np.mean(self.overfit_detector['train_loss']):.4f}")
            for model in [self.qnt, self.ddpg.actor, self.ddpg.critic, self.ssd]:
                for param_group in model.optimizer.param_groups:
                    param_group['weight_decay'] *= 1.2
            for layer in [self.qnt.dropout1, self.qnt.dropout2]:
                layer.p = min(layer.p + 0.1, 0.5)
            logging.info("ปรับ regularization เพื่อแก้ overfitting")

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logging.info("Early stopping ทำงาน หยุดฝึกเพื่อป้องกัน overfitting")
                return

        self.loss_history['qnt'].append(train_loss)
        self.loss_history['ssd'].append(ssd_loss)
        self.loss_history['evogan_disc'].append(disc_loss)
        self.loss_history['evogan_gen'].append(gen_loss)
        self.loss_history['ddpg_critic'].append(np.mean((self.ddpg.critic(torch.FloatTensor(np.concatenate([train_states, train_continuous], axis=1)).to(self.device)).cpu().detach().numpy() - train_rewards) ** 2))
        self.loss_history['tft'].append(np.mean((self.tft.predict(train_states) - train_next_states) ** 2))
        self.loss_history['madrl'].append(np.mean((self.madrl.critic(torch.FloatTensor(np.concatenate([train_states.flatten(), train_continuous.flatten()])).to(self.device)).cpu().detach().numpy() - train_rewards) ** 2))
        self.loss_history['meta'].append(meta_loss)

    async def evolve(self, state_data, reward, volatility):
        strategies = self.evogan.generate(state_data).cpu().detach().numpy()
        self.strategy_memory.extend(strategies)
        if len(self.strategy_memory) >= 10:
            evolved_strategies = self.evogan.evolve(self.strategy_memory, [reward] * len(self.strategy_memory))
            self.strategy_memory = evolved_strategies[:10]

    async def adversarial_train(self, states):
        # ฝึกโมเดลแบบ Adversarial เพื่อเพิ่มความทนทาน
        noise = np.random.normal(0, 0.1, states.shape)
        adv_states = states + noise
        adv_q_values = self.qnt(torch.FloatTensor(adv_states).to(self.device)).cpu().detach().numpy()
        adv_continuous = self.ddpg.act(adv_states)
        batch = self.replay_buffer.sample(32)
        if batch:
            orig_states, discrete_actions, continuous_actions, rewards, next_states, _, _, _, _ = batch
            await self.train(np.concatenate([orig_states, adv_states]),
                           np.concatenate([discrete_actions, discrete_actions]),
                           np.concatenate([continuous_actions, adv_continuous]),
                           np.concatenate([rewards, rewards * 0.5]),
                           np.concatenate([next_states, next_states]))



# คลาส ReplayBuffer: เก็บข้อมูลประสบการณ์สำหรับการฝึกโมเดล
class ReplayBuffer:
    def __init__(self, db_path=None, capacity=10000):
        self.db_path = db_path if db_path else os.path.join(BASE_DIR, 'replay_buffer.db')
        self.buffer = deque(maxlen=capacity)
        self.db_conn = sqlite3.connect(self.db_path, timeout=10)
        self.db_conn.execute("CREATE TABLE IF NOT EXISTS experiences (id INTEGER PRIMARY KEY, state BLOB, discrete_action INTEGER, continuous_action BLOB, reward REAL, next_state BLOB, gnn_embedding BLOB, tft_pred BLOB, atr REAL, multi_tf_data BLOB)")
        self.db_size = 0
        self.anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
        self.load_from_db()

    def add(self, state, discrete_action, continuous_action, reward, next_state, gnn_embedding=None, tft_pred=None, atr=None, multi_tf_data=None):
        state_blob = zlib.compress(pickle.dumps(state))
        continuous_action_blob = zlib.compress(pickle.dumps(continuous_action))
        next_state_blob = zlib.compress(pickle.dumps(next_state))
        gnn_embedding_blob = zlib.compress(pickle.dumps(gnn_embedding)) if gnn_embedding is not None else None
        tft_pred_blob = zlib.compress(pickle.dumps(tft_pred)) if tft_pred is not None else None
        multi_tf_data_blob = zlib.compress(pickle.dumps(multi_tf_data)) if multi_tf_data is not None else None
        features = np.concatenate([state.flatten(), [discrete_action], continuous_action, [reward]])
        if len(self.buffer) > 50 and self.anomaly_detector.predict([features])[0] == -1:
            logging.warning(f"ตรวจพบ anomaly ในข้อมูล: reward={reward}, ปรับลดน้ำหนัก")
            reward *= 0.5
        self.buffer.append((state, discrete_action, continuous_action, reward, next_state, gnn_embedding, tft_pred, atr, multi_tf_data))
        with self.db_conn:
            self.db_conn.execute("INSERT INTO experiences (state, discrete_action, continuous_action, reward, next_state, gnn_embedding, tft_pred, atr, multi_tf_data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                 (state_blob, discrete_action, continuous_action_blob, reward, next_state_blob, gnn_embedding_blob, tft_pred_blob, atr, multi_tf_data_blob))
            self.db_size += 1
        if self.db_size % 100 == 0:
            self.fit_anomaly_detector()

    def sample(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return None
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([x[0] for x in batch])
        discrete_actions = np.array([x[1] for x in batch])
        continuous_actions = np.array([x[2] for x in batch])
        rewards = np.array([x[3] for x in batch])
        next_states = np.array([x[4] for x in batch])
        gnn_embeddings = np.array([x[5] for x in batch if x[5] is not None], dtype=object)
        tft_preds = np.array([x[6] for x in batch if x[6] is not None], dtype=object)
        atrs = np.array([x[7] for x in batch if x[7] is not None])
        multi_tf_data = np.array([x[8] for x in batch if x[8] is not None], dtype=object)
        return states, discrete_actions, continuous_actions, rewards, next_states, gnn_embeddings, tft_preds, atrs, multi_tf_data

    def load_from_db(self):
        with self.db_conn:
            cursor = self.db_conn.execute("SELECT state, discrete_action, continuous_action, reward, next_state, gnn_embedding, tft_pred, atr, multi_tf_data FROM experiences ORDER BY id DESC LIMIT 10000")
            data = cursor.fetchall()
            for state_blob, discrete_action, continuous_action_blob, reward, next_state_blob, gnn_embedding_blob, tft_pred_blob, atr, multi_tf_data_blob in data[::-1]:
                state = pickle.loads(zlib.decompress(state_blob))
                continuous_action = pickle.loads(zlib.decompress(continuous_action_blob))
                next_state = pickle.loads(zlib.decompress(next_state_blob))
                gnn_embedding = pickle.loads(zlib.decompress(gnn_embedding_blob)) if gnn_embedding_blob else None
                tft_pred = pickle.loads(zlib.decompress(tft_pred_blob)) if tft_pred_blob else None
                multi_tf_data = pickle.loads(zlib.decompress(multi_tf_data_blob)) if multi_tf_data_blob else None
                self.buffer.append((state, discrete_action, continuous_action, reward, next_state, gnn_embedding, tft_pred, atr, multi_tf_data))
        logging.info(f"โหลด {len(data)} ข้อมูลจาก SQLite")
        if len(self.buffer) > 50:
            self.fit_anomaly_detector()

    def fit_anomaly_detector(self):
        # แก้ไขการใช้ list comprehension และ np.concatenate
        data = [np.concatenate([e[0].flatten(), np.array([e[1]]), e[2], np.array([e[3]])]) for e in self.buffer]
        self.anomaly_detector.fit(data)
        logging.debug("อัพเดท anomaly detector")

    def __del__(self):
        self.db_conn.close()

# คลาส IntelligentResourceManager: จัดการทรัพยากร CPU และ RAM
class IntelligentResourceManager:
    def __init__(self):
        self.cpu_usage = deque(maxlen=60)
        self.ram_usage = deque(maxlen=60)
        self.model_batch_sizes = {'qnt': 32, 'ddpg': 32, 'ssd': 32, 'evogan': 32, 'tft': 32, 'gnn': 32, 'madrl': 32, 'meta': 32}
        self.task_priorities = {'train': 0.7, 'predict': 0.2, 'data_fetch': 0.1}
        self.resource_lock = asyncio.Lock()

    async def monitor_resources(self):
        process = psutil.Process()
        while CONFIG['system_running']:
            self.cpu_usage.append(process.cpu_percent(interval=1))
            self.ram_usage.append(process.memory_info().rss / (1024 * 1024))
            if CONFIG['resource_adaptive']:
                await self.adjust_resources()
            await asyncio.sleep(60)

    async def adjust_resources(self, trader=None):
        async with self.resource_lock:
            avg_cpu = np.mean(self.cpu_usage) if self.cpu_usage else 50
            avg_ram = np.mean(self.ram_usage) if self.ram_usage else 1024
            if avg_cpu > (100 - CONFIG['min_cpu_idle_percent']) or avg_ram > (psutil.virtual_memory().total / (1024 * 1024) - CONFIG['min_ram_reserve_mb']):
                for model in self.model_batch_sizes:
                    self.model_batch_sizes[model] = max(1, int(self.model_batch_sizes[model] * 0.8))
                logging.info(f"ลด batch size เนื่องจาก CPU={avg_cpu:.1f}%, RAM={avg_ram:.1f}MB")
            elif avg_cpu < 50 and avg_ram < (psutil.virtual_memory().total / (1024 * 1024) * 0.5):
                for model in self.model_batch_sizes:
                    self.model_batch_sizes[model] = min(128, int(self.model_batch_sizes[model] * 1.2))
                logging.info(f"เพิ่ม batch size เนื่องจาก CPU={avg_cpu:.1f}%, RAM={avg_ram:.1f}MB")
            if trader:
                trader.resource_manager.model_batch_sizes = self.model_batch_sizes

# คลาส MultiMarketEnv: สภาพแวดล้อมสำหรับการเทรดหลายเหรียญ
class MultiMarketEnv(gym.Env):
    def __init__(self, account_balance=INITIAL_BALANCE, risk_per_trade=CONFIG['risk_per_trade'], dry_run=CONFIG['dry_run']):
        super().__init__()
        self.account_balance = account_balance
        self.available_balance = account_balance  # เพิ่มสำหรับยอดเงินที่พร้อมใช้
        self.reinvest_cap = account_balance * 2  # ขีดจำกัดการทบต้นเริ่มต้น
        self.initial_balance = INITIAL_BALANCE
        self.risk_per_trade = risk_per_trade
        self.dry_run = dry_run
        self.symbols = []
        self.positions = {}
        self.current_step = 0
        self.day_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        self.ws_manager = ws_manager
        self.simulator = RealTimeSimulator(self.symbols)
        self.data = {}
        self.raw_data = {}
        self.scalers = {}
        self.trader = None
        self.trade_log_file = os.path.join(BASE_DIR, 'trade_log.xlsx')
        self.db_conn = sqlite3.connect(os.path.join(BASE_DIR, 'env_history.db'), timeout=10)
        self.db_conn.execute("CREATE TABLE IF NOT EXISTS returns (id INTEGER PRIMARY KEY, step INT, return REAL, timestamp REAL)")
        self.db_conn.execute("CREATE TABLE IF NOT EXISTS historical_data (id INTEGER PRIMARY KEY, symbol TEXT, timestamp REAL, close REAL, volume REAL, funding_rate REAL, depth REAL)")
        if not os.path.exists(self.trade_log_file):
            pd.DataFrame(columns=['DateTime', 'Symbol', 'TradeType', 'BuyPrice', 'SellPrice', 'Quantity', 'Capital', 'ProfitLoss']).to_excel(self.trade_log_file, index=False)
        self.min_kpi_threshold = CONFIG['min_daily_kpi']
        self.multi_tf_data = {tf: {} for tf in CONFIG['multi_tf_list']}
        self.kpi_tracker = KPITracker()  # เพิ่ม KPITracker
        self.kpi_optimizer = KPIOptimizer()  # เพิ่ม KPIOptimizer
        self.balance_last_updated = 0  # เวลาอัพเดทยอดเงินล่าสุด

    async def load_historical_data(self, symbol, years=CONFIG['historical_years']):
        years_ago = datetime.utcnow() - timedelta(days=365 * years)
        with self.db_conn:
            cursor = self.db_conn.execute("SELECT timestamp, close, volume, funding_rate, depth FROM historical_data WHERE symbol=? AND timestamp>=? ORDER BY timestamp ASC",
                                         (symbol, years_ago.timestamp()))
            data = cursor.fetchall()
            if len(data) < TIMESTEPS and not self.dry_run:
                await api_manager.rate_limit_control()
                try:
                    klines = await exchange.fetch_ohlcv(symbol, timeframe='1h', since=int(years_ago.timestamp() * 1000), limit=17520)
                    for kline in klines:
                        timestamp, _, _, _, close, volume = kline
                        self.db_conn.execute("INSERT INTO historical_data (symbol, timestamp, close, volume, funding_rate, depth) VALUES (?, ?, ?, ?, ?, ?)",
                                            (symbol, timestamp / 1000, close, volume, 0.0001, 0))
                    self.db_conn.commit()
                    logging.info(f"โหลดข้อมูลย้อนหลัง {years} ปีสำหรับ {symbol}")
                    cursor = self.db_conn.execute("SELECT timestamp, close, volume, funding_rate, depth FROM historical_data WHERE symbol=? AND timestamp>=? ORDER BY timestamp ASC",
                                                 (symbol, years_ago.timestamp()))
                    data = cursor.fetchall()
                except Exception as e:
                    logging.error(f"ดึงข้อมูลย้อนหลังสำหรับ {symbol} ล้มเหลว: {e}")
        return pd.DataFrame(data, columns=['timestamp', 'close', 'volume', 'funding_rate', 'depth']) if data else pd.DataFrame()

    async def transfer_from_spot_to_futures(self):
        if self.dry_run:
            logging.info("โหมดจำลอง: จำลองการโอนเงินจาก Spot ไป Futures")
            shortfall = max(0, CONFIG['initial_balance'] - self.account_balance)
            self.account_balance += shortfall
            self.available_balance += shortfall
            return shortfall
        await api_manager.rate_limit_control()
        try:
            spot_balance = await exchange.fetch_balance(params={'type': 'spot'})['USDT']['free']
            shortfall = max(0, CONFIG['initial_balance'] - self.account_balance)
            if shortfall > 0 and spot_balance >= shortfall:
                await exchange.transfer('USDT', shortfall, 'spot', 'futures')
                self.account_balance += shortfall
                self.available_balance += shortfall
                logging.info(f"โอน {shortfall:.2f} USDT จาก Spot ไป Futures")
                return shortfall
            else:
                logging.warning(f"ยอด Spot ไม่เพียงพอ: มี {spot_balance:.2f}, ต้องการ {shortfall:.2f}")
                return 0
        except Exception as e:
            logging.error(f"การโอนเงินจาก Spot ไป Futures ล้มเหลว: {e}")
            return 0

    async def execute_trade_async(self, symbol, side, size, leverage, stop_loss, take_profit, trailing_stop=None, trailing_take_profit=None):
        # อัพเดทยอดเงินจาก WebSocket ทุก 60 วินาที
        current_time = time.time()
        if current_time - self.balance_last_updated > 60:
            self.account_balance = self.ws_manager.get_latest_balance()
            self.available_balance = self.account_balance
            if REINVEST_PROFITS:
                self.available_balance += self.kpi_tracker.total_profit * 0.5
            self.balance_last_updated = current_time

        price = self.ws_manager.get_latest_price(symbol) if not self.dry_run else self.raw_data[symbol]['close'].iloc[-1]
        # คำนวณ margin โดยรวมค่าธรรมเนียมและ slippage
        required_margin = (size * price) / leverage * (1 + TAKER_FEE + SLIPPAGE_DEFAULT)
        if required_margin > self.available_balance:
            logging.warning(f"Margin ไม่พอสำหรับ {symbol}: ต้องการ {required_margin:.2f}, มี {self.available_balance:.2f}")
            return 0

        # ตรวจสอบความเสี่ยงก่อนเทรด
        if not self.trader.risk_guardian.evaluate_position(symbol, price, price, size, leverage, side):
            logging.warning(f"ตำแหน่ง {symbol} ไม่ผ่านการประเมินความเสี่ยง")
            return 0

        if self.dry_run:
            future_step = min(self.current_step + TIMESTEPS, len(self.raw_data[symbol]) - 1)
            future_price = self.raw_data[symbol]['close'].iloc[future_step]
            profit = (future_price - price) * size * leverage * (1 - TAKER_FEE - SLIPPAGE_DEFAULT) * (-1 if side == 'SELL' else 1)
        else:
            await api_manager.rate_limit_control()
            try:
                order = await exchange.create_order(symbol=symbol, type='market', side=side.lower(), amount=size, 
                                                  params={'leverage': int(leverage)})
                order_info = await exchange.fetch_order(order['id'], symbol)
                profit = (order_info['filled'] * (order_info['price'] - price) * leverage * (1 - TAKER_FEE) * 
                         (-1 if side == 'SELL' else 1))
            except Exception as e:
                logging.error(f"การเทรด {symbol} ล้มเหลว: {e}")
                return 0

        self.positions[symbol] = {
            'size': size, 'entry': price, 'leverage': leverage, 
            'stop_loss': price * (1 - stop_loss if side == 'BUY' else 1 + stop_loss),
            'take_profit': price * (1 + take_profit if side == 'BUY' else 1 - take_profit), 
            'side': side, 'trailing_stop': trailing_stop, 'trailing_take_profit': trailing_take_profit,
            'highest_price': price if side == 'BUY' else float('inf'),
            'lowest_price': price if side == 'SELL' else float('-inf')
        }
        self.account_balance -= required_margin
        self.available_balance -= required_margin
        self.account_balance += profit
        if REINVEST_PROFITS and profit > 0:
            reinvest_amount = min(profit * 0.5, self.reinvest_cap - self.available_balance)
            self.available_balance += reinvest_amount
            logging.info(f"Reinvest กำไร {reinvest_amount:.2f} USDT")
        trade_log = pd.DataFrame([{
            'วันที่': datetime.utcnow(), 
            'เหรียญ': symbol, 
            'ประเภทการเทรด': side, 
            'ราคาซื้อ': price if side == 'BUY' else 0, 
            'ราคาขาย': price if side == 'SELL' else 0, 
            'ปริมาณ': size, 
            'ทุน': self.account_balance, 
            'กำไร/ขาดทุน': profit,
            'โหมด': 'จำลอง' if self.dry_run else 'จริง'
        }])
        try:
            with pd.ExcelWriter(self.trade_log_file, mode='a', if_sheet_exists='overlay') as writer:
                trade_log.to_excel(writer, index=False, header=False)
            logging.info(f"บันทึกการเทรดลง {self.trade_log_file} สำเร็จ")
        except Exception as e:
            logging.error(f"บันทึกการเทรดล้มเหลว: {e}")
        return profit

    async def close_position_async(self, symbol, current_price):
        if self.positions.get(symbol):
            size = self.positions[symbol]['size']
            leverage = self.positions[symbol]['leverage']
            profit = (current_price - self.positions[symbol]['entry']) * size * leverage * (1 - TAKER_FEE) * \
                     (-1 if self.positions[symbol]['side'] == 'SELL' else 1)
            self.account_balance += profit + (size * self.positions[symbol]['entry'] / leverage)
            self.available_balance += (size * self.positions[symbol]['entry'] / leverage)
            del self.positions[symbol]
            return profit
        return 0

    async def process_symbol(self, symbol):
        current_price = self.ws_manager.get_latest_price(symbol) if not self.dry_run else self.raw_data[symbol]['close'].iloc[-1]
        position = self.positions.get(symbol)
        profit = 0
        if position:
            if position['trailing_stop']:
                if position['side'] == 'BUY' and current_price > position['highest_price']:
                    position['highest_price'] = current_price
                    position['stop_loss'] = current_price - position['trailing_stop']
                elif position['side'] == 'SELL' and current_price < position['lowest_price']:
                    position['lowest_price'] = current_price
                    position['stop_loss'] = current_price + position['trailing_stop']
            if position['trailing_take_profit']:
                if position['side'] == 'BUY' and current_price > position['highest_price']:
                    position['highest_price'] = current_price
                    position['take_profit'] = current_price - position['trailing_take_profit']
                elif position['side'] == 'SELL' and current_price < position['lowest_price']:
                    position['lowest_price'] = current_price
                    position['take_profit'] = current_price + position['trailing_take_profit']
            if (position['side'] == 'BUY' and (current_price <= position['stop_loss'] or current_price >= position['take_profit'])) or \
               (position['side'] == 'SELL' and (current_price >= position['stop_loss'] or current_price <= position['take_profit'])):
                profit = await self.close_position_async(symbol, current_price)
        reward = profit / self.initial_balance if profit != 0 else 0
        return reward, profit

    async def fetch_multi_tf_data(self, symbol):
        for tf in CONFIG['multi_tf_list']:
            if self.dry_run:
                df = self.simulator.data[symbol]
                if not df.empty:
                    self.multi_tf_data[tf][symbol] = df.resample(tf).last().tail(TIMESTEPS)
            else:
                await api_manager.rate_limit_control()
                try:
                    klines = await exchange.fetch_ohlcv(symbol, timeframe=tf, limit=TIMESTEPS)
                    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    self.multi_tf_data[tf][symbol] = df
                except Exception as e:
                    logging.error(f"ดึงข้อมูล {tf} สำหรับ {symbol} ล้มเหลว: {e}")
                    self.multi_tf_data[tf][symbol] = pd.DataFrame()

    async def step(self):
        if self.dry_run:
            self.simulator.simulate_step()
            for symbol in self.symbols:
                state_lstm, _, state_ensemble, _, scaler, raw = self.simulator.get_data(symbol)
                self.data[symbol] = {'lstm': state_lstm, 'ensemble': state_ensemble}
                self.scalers[symbol] = scaler
                self.raw_data[symbol] = raw
        await asyncio.gather(*(self.fetch_multi_tf_data(symbol) for symbol in self.symbols))
        if self.account_balance < CONFIG['initial_balance'] * 0.9:
            await self.transfer_from_spot_to_futures()
        await self.check_new_day()
        rewards = []
        total_profit = 0
        tasks = [self.process_symbol(symbol) for symbol in self.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Error processing {self.symbols[i]}: {result}")
                rewards.append(0)
                total_profit += 0
            else:
                reward, profit = result
                rewards.append(reward)
                total_profit += profit
        self.current_step += 1
        done = self.current_step >= 1440 or self.account_balance < self.initial_balance * (1 - CONFIG['max_drawdown']) or not CONFIG['system_running']
        with self.db_conn:
            self.db_conn.execute("INSERT INTO returns (step, return, timestamp) VALUES (?, ?, ?)",
                               (self.current_step, total_profit, time.time()))
        if done:
            self.reset()
        return self.get_observation(), rewards, done, {'profit': total_profit}

    async def check_new_day(self):
        now = datetime.utcnow()
        if now >= self.day_start + timedelta(days=1):
            excess = max(0, self.account_balance - self.initial_balance)
            if excess > 0 and not self.dry_run:
                await exchange.fapiPrivate_post_transfer({'asset': 'USDT', 'amount': excess, 'type': 2})
                self.account_balance -= excess
                logging.info(f"โอนกำไรส่วนเกิน {excess:.2f} USDT ไป Spot")
            self.day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    def get_observation(self):
        dyn_gen = DynamicIndicatorGenerator(self.trader.evogan, self.trader.gnn, self.trader.tft.multi_tf_data)
        observations = []
        for symbol in self.symbols:
            ind = dyn_gen.generate_synthetic_indicators(symbol)
            obs = []
            for tf in CONFIG['multi_tf_list']:
                tf_ind = ind.get(tf, {
                    'base': {k: 0 for k in ['ATR', 'RSI', 'MACD', 'EMA', 'BB_upper', 'BB_lower', 'SMA', 'Stoch_RSI', 'OBV', 'Volume']},
                    'synthetic': np.zeros(10),
                    'gnn_correlations': np.zeros(5)
                })
                base_values = list(tf_ind['base'].values())
                synthetic_values = list(tf_ind['synthetic'].flatten()[:10])  # จำกัดขนาด synthetic features
                gnn_values = list(tf_ind['gnn_correlations'].flatten()[:5])  # จำกัดขนาด gnn correlations
                obs.extend(base_values + synthetic_values + gnn_values)
            observations.append(np.array(obs))
        return np.array(observations)

    def reset(self):
        self.account_balance = self.initial_balance
        self.available_balance = self.initial_balance
        self.positions = {s: None for s in self.symbols}
        self.current_step = 0
        if self.dry_run:
            self.simulator.reset()

# คลาส RiskGuardian: ควบคุมความเสี่ยงในการเทรด
class RiskGuardian:
    def __init__(self, max_drawdown=CONFIG['max_drawdown'], cut_loss_threshold=CONFIG['cut_loss_threshold']):
        self.max_drawdown = max_drawdown
        self.cut_loss_threshold = cut_loss_threshold
        self.drawdown_history = deque(maxlen=1440)
        self.positions = {}
        self.total_trades = 0
        self.failed_trades = 0
        self.env = None
        self.dynamic_risk_factor = 1.0
        self.volatility_history = deque(maxlen=60)

    def assess_risk(self, balance, initial_balance):
        current_drawdown = (initial_balance - balance) / initial_balance
        self.drawdown_history.append(current_drawdown)
        if current_drawdown > self.max_drawdown * self.dynamic_risk_factor:
            logging.warning(f"Drawdown เกินขีดจำกัด: {current_drawdown:.2%} > {self.max_drawdown * self.dynamic_risk_factor:.2%}")
            return False
        return True

    def evaluate_position(self, symbol, current_price, entry_price, size, leverage, side):
        unrealized_pnl = (current_price - entry_price) * size * leverage * (1 if side == 'BUY' else -1)
        position_value = size * entry_price / leverage
        loss_ratio = -unrealized_pnl / position_value
        adjusted_threshold = self.cut_loss_threshold * self.dynamic_risk_factor
        if loss_ratio > adjusted_threshold:
            logging.warning(f"ตำแหน่ง {symbol} ขาดทุนเกิน {adjusted_threshold:.2%}: {loss_ratio:.2%}")
            return False
        return True

    async def update_dynamic_risk(self, ws_data):
        volatilities = []
        for symbol in ws_data:
            if 'close' in ws_data[symbol]:
                pct_change = (ws_data[symbol]['close'] - ws_data[symbol].get('prev_close', ws_data[symbol]['close'])) / ws_data[symbol]['close']
                volatilities.append(pct_change)
                ws_data[symbol]['prev_close'] = ws_data[symbol]['close']
        if volatilities:
            avg_volatility = np.std(volatilities)
            self.volatility_history.append(avg_volatility)
            avg_vol_history = np.mean(self.volatility_history) if self.volatility_history else CONFIG['min_volatility_threshold']
            self.dynamic_risk_factor = min(2.0, max(0.5, avg_vol_history / CONFIG['min_volatility_threshold']))
            logging.debug(f"อัพเดท dynamic risk factor: {self.dynamic_risk_factor:.2f}")

    async def emergency_stop(self):
        if not self.env:
            return
        for symbol in list(self.positions.keys()):
            current_price = ws_manager.get_latest_price(symbol)
            await self.env.close_position_async(symbol, current_price)
        logging.critical("หยุดฉุกเฉิน: ปิดทุกตำแหน่ง")

# คลาส StrategyGenerator: สร้างและดำเนินกลยุทธ์การเทรด
class StrategyGenerator:
    def __init__(self, trader, env, risk_guardian):
        self.trader = trader
        self.env = env
        self.risk_guardian = risk_guardian
        self.action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}

    async def generate_strategy(self, state, symbol, volatility):
        discrete_pred, continuous_pred = self.trader.predict(state)
        action_idx = np.argmax(discrete_pred[0])
        action = self.action_map[action_idx]
        leverage, size = continuous_pred[0]
        stop_loss = CONFIG['stop_loss_percentage'] * (1 + volatility / CONFIG['min_volatility_threshold'])
        take_profit = stop_loss * 2
        return {
            'action': action,
            'symbol': symbol,
            'size': size,
            'leverage': min(leverage, CONFIG['max_leverage_per_symbol'].get(symbol, 125)),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': stop_loss if volatility > CONFIG['min_volatility_threshold'] else None,
            'trailing_take_profit': take_profit if volatility > CONFIG['min_volatility_threshold'] else None
        }

    async def execute_strategy(self, strategy):
        if strategy['action'] == 'HOLD' or not self.risk_guardian.assess_risk(self.env.account_balance, self.env.initial_balance):
            return 0
        profit = await self.env.execute_trade_async(
            strategy['symbol'], strategy['action'], strategy['size'], strategy['leverage'],
            strategy['stop_loss'], strategy['take_profit'], strategy['trailing_stop'], strategy['trailing_take_profit']
        )
        self.risk_guardian.total_trades += 1
        if profit < 0:
            self.risk_guardian.failed_trades += 1
        return profit

# คลาส KPIOptimizer: ปรับปรุง KPI การเทรด
class KPIOptimizer:
    def __init__(self):
        self.target_kpi = CONFIG['target_kpi_daily']
        self.min_kpi = CONFIG['min_daily_kpi']

    def optimize(self, current_kpi):
        # ปรับ reinvest_cap ตาม KPI รายวัน
        if current_kpi >= self.target_kpi:
            kpi_factor = min(2.0, current_kpi / self.target_kpi)
        elif current_kpi < self.min_kpi:
            kpi_factor = max(0.5, current_kpi / self.min_kpi)
        else:
            kpi_factor = 1.0
        return kpi_factor

# คลาส DynamicRiskAllocator: จัดสรรความเสี่ยงแบบไดนามิก
class DynamicRiskAllocator:
    def __init__(self):
        self.base_risk = CONFIG['risk_per_trade']

    async def allocate_risk(self, symbols, ws_data, kpi_factor):
        risk_weights = {}
        for symbol in symbols:
            if symbol in ws_data:
                volatility = np.std([ws_data[symbol]['close']] if 'close' in ws_data[symbol] else [0]) or 0.01
                liquidity = ws_data[symbol].get('volume', 0)
                risk_weights[symbol] = self.base_risk * (1 / (volatility + 1e-6)) * (liquidity / CONFIG['liquidity_threshold']) * kpi_factor
            else:
                risk_weights[symbol] = self.base_risk
        return risk_weights

# คลาส KPITracker: ติดตาม KPI การเทรด
class KPITracker:
    def __init__(self):
        self.total_profit = 0
        self.daily_profit = 0
        self.day_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    async def update(self, profit):
        self.total_profit += profit
        now = datetime.utcnow()
        if now >= self.day_start + timedelta(days=1):
            self.daily_profit = profit
            self.day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            self.daily_profit += profit
        logging.info(f"KPI อัพเดท: กำไรวันนี้={self.daily_profit:.2f}, กำไรรวม={self.total_profit:.2f}")

# คลาส AutomaticBugFixer: แก้ไขบั๊กอัตโนมัติ
class AutomaticBugFixer:
    def __init__(self):
        self.attempts_left = CONFIG['bug_fix_attempts']

    async def analyze_and_fix(self, error, trader, env):
        if not CONFIG['auto_bug_fix'] or self.attempts_left <= 0:
            return False
        error_str = str(error)
        if "CUDA out of memory" in error_str:
            trader.resource_manager.model_batch_sizes = {k: max(1, v // 2) for k, v in trader.resource_manager.model_batch_sizes.items()}
            gc.collect()
            torch.cuda.empty_cache()
            logging.info("แก้ไข CUDA OOM: ลด batch size และเคลียร์หน่วยความจำ")
        elif "API rate limit" in error_str:
            await asyncio.sleep(60)
            logging.info("แก้ไข API rate limit: รอ 60 วินาที")
        elif "network" in error_str.lower():
            await ws_manager.stop()
            await ws_manager.start(env.symbols)
            logging.info("แก้ไข network error: รีสตาร์ท WebSocket")
        else:
            logging.warning(f"ไม่สามารถแก้ไขบั๊ก: {error_str}")
            self.attempts_left -= 1
            return False
        self.attempts_left -= 1
        return True

# คลาส RealTimeSimulator: จำลองข้อมูลเรียลไทม์ในโหมด dry_run
class RealTimeSimulator:
    def __init__(self, symbols):
        self.symbols = symbols
        self.data = {symbol: pd.DataFrame() for symbol in symbols}
        self.step = 0

    def update_symbols(self, symbols):
        self.symbols = symbols
        self.data = {symbol: pd.DataFrame() for symbol in symbols if symbol not in self.data}

    def simulate_step(self):
        for symbol in self.symbols:
            if self.data[symbol].empty:
                df = pd.DataFrame(index=range(1440), columns=['close', 'volume'])
                df['close'] = 10000 * (1 + np.random.normal(CONFIG['sim_trend'], CONFIG['sim_volatility'], 1440))
                df['volume'] = np.random.uniform(50, 500, 1440)
                if np.random.random() < CONFIG['sim_spike_chance']:
                    spike_idx = np.random.randint(0, 1440)
                    df.loc[spike_idx, 'close'] *= 1.1
                self.data[symbol] = df
            else:
                new_price = self.data[symbol]['close'].iloc[-1] * (1 + np.random.normal(CONFIG['sim_trend'], CONFIG['sim_volatility']))
                new_volume = np.random.uniform(50, 500)
                self.data[symbol].loc[len(self.data[symbol])] = [new_price, new_volume]
        self.step += 1

    def get_data(self, symbol):
        df = self.data[symbol]
        if len(df) < TIMESTEPS:
            return np.zeros((TIMESTEPS, 7)), None, np.zeros(7), None, None, df
        window = df.tail(TIMESTEPS)
        close = window['close'].values
        volume = window['volume'].values
        rsi = ta.momentum.RSIIndicator(close).rsi().values[-1] or 50
        macd = ta.trend.MACD(close).macd().values[-1] or 0
        rv = np.std(close) / np.mean(close) if np.mean(close) != 0 else 0
        funding_rate = 0.0001
        depth = 0
        state_lstm = np.array([close, volume, [rsi]*TIMESTEPS, [macd]*TIMESTEPS, [rv]*TIMESTEPS, [funding_rate]*TIMESTEPS, [depth]*TIMESTEPS]).T
        state_ensemble = np.array([close[-1], volume[-1], rsi, macd, rv, funding_rate, depth])
        scaler = RobustScaler()
        scaled_lstm = scaler.fit_transform(state_lstm)
        return scaled_lstm, None, state_ensemble, None, scaler, df

    def reset(self):
        self.step = 0
        self.data = {symbol: pd.DataFrame() for symbol in self.symbols}

# คลาส IndicatorCalculator: คำนวณ Indicator พื้นฐาน
class IndicatorCalculator:
    def __init__(self, multi_tf_data):
        self.multi_tf_data = multi_tf_data

    def calculate_indicators(self, symbol):
        indicators = {}
        for tf in CONFIG['multi_tf_list']:
            df = self.multi_tf_data[tf].get(symbol, pd.DataFrame())
            if df.empty:
                continue
            close = df['close'].values
            high = df['high'].values if 'high' in df else close
            low = df['low'].values if 'low' in df else close
            volume = df['volume'].values
            indicators[tf] = {
                'ATR': ta.volatility.AverageTrueRange(high, low, close).average_true_range().iloc[-1] if len(close) >= 14 else 0,
                'RSI': ta.momentum.RSIIndicator(close).rsi().iloc[-1] if len(close) >= 14 else 50,
                'MACD': ta.trend.MACD(close).macd().iloc[-1] if len(close) >= 26 else 0,
                'EMA': ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1] if len(close) >= 20 else close[-1],
                'BB_upper': ta.volatility.BollingerBands(close).bollinger_hband().iloc[-1] if len(close) >= 20 else close[-1],
                'BB_lower': ta.volatility.BollingerBands(close).bollinger_lband().iloc[-1] if len(close) >= 20 else close[-1],
                'SMA': ta.trend.SMAIndicator(close, window=20).sma_indicator().iloc[-1] if len(close) >= 20 else close[-1],
                'Stoch_RSI': ta.momentum.StochasticRSIIndicator(close).stochrsi().iloc[-1] if len(close) >= 14 else 0.5,
                'OBV': ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume().iloc[-1] if len(close) > 1 else volume[-1],
                'Volume': volume[-1]
            }
        return indicators

# คลาส DynamicIndicatorGenerator: สร้าง Indicator แบบไดนามิก
class DynamicIndicatorGenerator:
    def __init__(self, evogan, gnn, multi_tf_data):
        self.evogan = evogan
        self.gnn = gnn
        self.multi_tf_data = multi_tf_data
        self.base_calc = IndicatorCalculator(multi_tf_data)
        self.feature_weights = torch.tensor([
            1.0,  # ATR
            0.5,  # RSI
            1.0,  # MACD
            1.0,  # EMA
            0.5,  # BB_upper
            0.5,  # BB_lower
            0.5,  # SMA
            0.5,  # Stoch_RSI
            1.0,  # OBV
            1.0   # Volume
        ]).to(self.evogan.device)

    def generate_synthetic_indicators(self, symbol):
        synthetic = {}
        base_ind = self.base_calc.calculate_indicators(symbol)
        for tf in CONFIG['multi_tf_list']:
            if tf not in base_ind:
                continue
            base_features = np.array([
                base_ind[tf]['ATR'], base_ind[tf]['MACD'], base_ind[tf]['EMA'],
                base_ind[tf]['OBV'], base_ind[tf]['Volume'],
                base_ind[tf]['RSI'], base_ind[tf]['BB_upper'], base_ind[tf]['BB_lower'],
                base_ind[tf]['SMA'], base_ind[tf]['Stoch_RSI']
            ])
            weighted_features = base_features * self.feature_weights.cpu().numpy()
            synthetic_features = self.evogan.generate(
                torch.FloatTensor(weighted_features).to(self.evogan.device)
            ).cpu().numpy()
            graph = self._create_asset_graph(self.multi_tf_data)
            gnn_features = self.gnn.forward(
                torch.FloatTensor(synthetic_features).to(self.gnn.device), graph
            ).cpu().numpy()
            synthetic[tf] = {
                'base': base_ind[tf],
                'synthetic': synthetic_features,
                'gnn_correlations': gnn_features
            }
        return synthetic

    def _create_asset_graph(self, multi_tf_data):
        num_assets = len(multi_tf_data[CONFIG['multi_tf_list'][0]])
        edge_index = torch.tensor([[i, j] for i in range(num_assets) for j in range(num_assets) if i != j], 
                                  dtype=torch.long).t()
        return Data(edge_index=edge_index.to(self.gnn.device))

# ฟังก์ชันควบคุมการรันและหยุดระบบ
async def control_loop():
    while True:
        if keyboard.is_pressed('r'):
            CONFIG['system_running'] = True
            logging.info("ระบบเริ่มทำงาน (กด 'r')")
        elif keyboard.is_pressed('q'):
            CONFIG['system_running'] = False
            logging.info("ระบบหยุดทำงาน (กด 'q')")
        await asyncio.sleep(0.1)


# ฟังก์ชัน main: รวมทุกส่วนและรันระบบ
async def main():
    CONFIG['system_running'] = False
    trader = UnifiedQuantumTrader(input_dim=CONFIG['input_dim'])
    risk_guardian = RiskGuardian()
    env = MultiMarketEnv()
    env.trader = trader
    trader.replay_buffer = ReplayBuffer()
    risk_guardian.env = env
    strategy_gen = StrategyGenerator(trader, env, risk_guardian)
    resource_manager = IntelligentResourceManager()
    kpi_optimizer = KPIOptimizer()
    risk_allocator = DynamicRiskAllocator()
    kpi_tracker = KPITracker()
    bug_fixer = AutomaticBugFixer()
    await ws_manager.start(['BTC/USDT'])
    await ws_manager.fetch_all_usdt_pairs()
    trader.current_symbols = ws_manager.all_usdt_pairs[:CONFIG['madrl_agent_count']]
    env.symbols = trader.current_symbols
    env.simulator.update_symbols(env.symbols)
    trader.madrl.update_symbols(env.symbols)
    asyncio.create_task(resource_manager.monitor_resources())
    step_count = 0

    for symbol in env.symbols:
        await env.load_historical_data(symbol)

    control_task = asyncio.create_task(control_loop())

    while True:
        try:
            if not CONFIG['system_running']:
                logging.info("ระบบหยุดรอคำสั่ง 'r' เพื่อเริ่ม หรือ 'q' เพื่อออก")
                await asyncio.sleep(1)
                continue

            if step_count % CONFIG['checkpoint_interval'] == 0:
                torch.save(trader.qnt.state_dict(), os.path.join(MODEL_DIR, 'qnt_checkpoint.pth'))
                logging.info(f"บันทึก checkpoint ที่ step {step_count}")
            top_symbols = trader.select_top_coins(ws_manager.all_usdt_pairs, ws_manager.data, kpi_tracker)
            if set(top_symbols) != set(env.symbols):
                env.symbols = top_symbols
                env.simulator.update_symbols(top_symbols)
                trader.madrl.update_symbols(top_symbols)
                await ws_manager.update_symbols(top_symbols)
            observation, rewards, done, info = await env.step()
            total_profit = info['profit']
            await kpi_tracker.update(total_profit)
            kpi_factor = kpi_optimizer.optimize(kpi_tracker.daily_profit)
            env.reinvest_cap = env.initial_balance * kpi_factor  # ปรับ reinvest_cap ตาม KPI
            await risk_guardian.update_dynamic_risk(ws_manager.data)
            risk_weights = await risk_allocator.allocate_risk(env.symbols, ws_manager.data, kpi_factor)
            for i, symbol in enumerate(env.symbols):
                state = observation[i]
                volatility = np.std([ws_manager.data[symbol]['close']] if symbol in ws_manager.data else [0]) or 0.01
                strategy = await strategy_gen.generate_strategy(state.reshape(1, -1), symbol, volatility)
                strategy['size'] *= risk_weights.get(symbol, CONFIG['risk_per_trade'])
                profit = await strategy_gen.execute_strategy(strategy)
                multi_tf_data = {tf: env.multi_tf_data[tf].get(symbol, pd.DataFrame()).to_dict() for tf in CONFIG['multi_tf_list']}
                trader.replay_buffer.add(state, np.argmax(trader.predict(state.reshape(1, -1))[0]), 
                                       trader.predict(state.reshape(1, -1))[1][0], profit, observation[i], None, None, volatility, multi_tf_data)
                await trader.evolve(state.reshape(1, -1), profit, volatility)
            if step_count % CONFIG['rl_train_interval'] == 0 and trader.replay_buffer.buffer:
                batch = trader.replay_buffer.sample(32)
                if batch:
                    states, discrete_actions, continuous_actions, rewards, next_states, _, _, atrs, multi_tf_data = batch
                    await trader.train(states, discrete_actions, continuous_actions, rewards, next_states)
                    await trader.adversarial_train(states)
            if step_count % CONFIG['auto_ml_interval'] == 0:
                trader.bayes_opt.maximize(init_points=2, n_iter=CONFIG['bayes_opt_steps'])
                trader.model_weights = trader.bayes_opt.max['params']
                logging.info(f"ปรับน้ำหนักโมเดล: {trader.model_weights}")
            step_count += 1
            await asyncio.sleep(60)
        except Exception as e:
            if await bug_fixer.analyze_and_fix(e, trader, env):
                logging.info("แก้ไขบั๊กสำเร็จ ทำงานต่อ")
                continue
            logging.critical(f"ข้อผิดพลาดร้ายแรง: {e}")
            traceback.print_exc()
            await risk_guardian.emergency_stop()
            break
    
    control_task.cancel()
    await ws_manager.stop()
    await exchange.close()
    logging.info("ระบบปิดสมบูรณ์")

if __name__ == "__main__":
    
    asyncio.run(main())