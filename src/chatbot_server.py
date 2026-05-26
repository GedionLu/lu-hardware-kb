#!/usr/bin/env python3
"""
AI服务助手 — 生产级服务 (AI KB ChatBot)
端口: 10054
特性: 并发 / 图片HTTP / GZIP / 限流 / 健康检查 / 结构化日志

用法: python3 chatbot_server.py [--port 10054]
"""

import sys, os, re, json, time, gzip, io, hashlib, logging, urllib.parse
from datetime import datetime

# 确保 src/ 在 sys.path 中，支持项目结构
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from functools import wraps
from threading import Lock

# ── 配置 ──
PORT = 10054
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
STATIC_DIR = os.path.join(ROOT_DIR, 'static')
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_BASE = os.path.join(PROJECT_ROOT, 'kb-images')
RATE_LIMIT = 30          # 每 IP 每分钟
RATE_WINDOW = 60         # 窗口秒
QUERY_TIMEOUT = 30       # 秒
MAX_QUERY_LEN = 500

# ── 日志 ──
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'chatbot_server.log')),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger('chatbot')

# ── 频率限制 ──
rate_store = {}   # ip → [(timestamp, ...)]
rate_lock = Lock()

def rate_check(ip):
    with rate_lock:
        now = time.time()
        reqs = [t for t in rate_store.get(ip, []) if t > now - RATE_WINDOW]
        if len(reqs) >= RATE_LIMIT:
            return False
        reqs.append(now)
        rate_store[ip] = reqs
        return True

# ── GZIP ──
def gzip_compress(data):
    if len(data) < 500:
        return data, False
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=6) as f:
        f.write(data)
    compressed = buf.getvalue()
    if len(compressed) < len(data):
        return compressed, True
    return data, False

# ── 图片解析 ──
def to_local(url):
    """将外部图片URL映射到本地文件"""
    if '/kb/' not in url:
        return None
    rel = urllib.parse.unquote(url.split('/kb/')[1])
    p = os.path.join(IMG_BASE, rel)
    if os.path.exists(p):
        return p
    # fallback: 用 hash 后缀扫描
    parts = os.path.basename(p).rsplit('_', 1)
    d = os.path.dirname(p)
    if len(parts) > 1 and os.path.exists(d):
        for f in os.listdir(d):
            if parts[-1] in f:
                return os.path.join(d, f)
    return None

def extract_img_path(url):
    """将外链转为本地 /img/... 路径"""
    if '/kb/' not in url:
        return None
    rel = url.split('/kb/')[1]  # category/filename.png
    if not rel or '..' in rel:
        return None
    return '/img/' + rel

def resolve_img(path):
    """安全解析 /img/<category>/<filename> → 本地文件"""
    rel = path[len('/img/'):]
    rel = urllib.parse.unquote(rel)
    if '..' in rel or rel.startswith('/'):
        return None
    p = os.path.join(IMG_BASE, rel)
    if os.path.exists(p) and os.path.isfile(p):
        return p
    # fallback: hash 扫描
    parts = os.path.basename(p).rsplit('_', 1)
    d = os.path.dirname(p)
    if len(parts) > 1 and os.path.exists(d):
        for f in os.listdir(d):
            if parts[-1] in f:
                fp = os.path.join(d, f)
                if os.path.isfile(fp):
                    return fp
    return None

# ── 查询引擎 — import 模式 ──
_engine = None
_engine_lock = Lock()

def get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                log.info('初始化查询引擎...')
                # jieba + retriever 初始化输出重定向到日志
                import io as _io
                _old_stdout = sys.stdout
                sys.stdout = _io.StringIO()
                try:
                    from query import QueryEngineV3
                    _engine = QueryEngineV3()
                finally:
                    _init_output = sys.stdout.getvalue()
                    sys.stdout = _old_stdout
                    if _init_output.strip():
                        for line in _init_output.strip().split('\n'):
                            log.debug(line.strip())
                log.info('查询引擎就绪')
    return _engine

def do_query(query):
    engine = get_engine()
    result = engine.query(query)

    response = result.get('response', '')
    segments = []
    models = result.get('models', [])
    intents = result.get('intents', [])

    for line in response.split('\n'):
        if not line.strip():
            continue
        m = re.search(r'\[图片: (.+?)\]', line)
        if m:
            text_part = line.replace(m.group(0), '').strip()
            if text_part:
                segments.append({'type': 'text', 'content': text_part})
            local_path = extract_img_path(m.group(1))
            if local_path:
                label = _extract_step_label(text_part)
                segments.append({
                    'type': 'image',
                    'url': local_path,
                    'label': label,
                })
        else:
            segments.append({'type': 'text', 'content': line.strip()})

    return {
        'segments': segments,
        'models': models,
        'intents': intents,
        'search_method': result.get('search_method', ''),
        'fulltext': result.get('fulltext_results', []),
    }

def _extract_step_label(text):
    """从文本中提取步骤标签"""
    m = re.search(r'→ 步骤 \d+: (.+)', text)
    if m:
        return m.group(1).strip()
    return None

# ── ThreadingMixIn ──
class ThreadedServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

class Handler(BaseHTTPRequestHandler):
    server_version = 'AIKBChatBot/1.0'

    def log_message(self, fmt, *args):
        log.info('%s %s %dms', self.client_address[0],
                 args[0] if args else '',
                 int((time.time() - self._start_time) * 1000) if hasattr(self, '_start_time') else 0)

    def _start_time(self):
        return time.time()

    def _supports_gzip(self):
        return 'gzip' in self.headers.get('Accept-Encoding', '')

    def _send(self, code, data, content_type='text/html; charset=utf-8'):
        body = data.encode('utf-8') if isinstance(data, str) else data
        if self._supports_gzip():
            compressed, is_gzip = gzip_compress(body)
            if is_gzip:
                body = compressed
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        if isinstance(body, bytes) and len(body) < len(data.encode('utf-8') if isinstance(data, str) else data):
            self.send_header('Content-Encoding', 'gzip')
        self.send_header('Content-Length', len(body))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        if self._supports_gzip():
            compressed, is_gzip = gzip_compress(body)
            if is_gzip:
                body = compressed
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        if self._supports_gzip() and len(body) < len(json.dumps(data, ensure_ascii=False).encode('utf-8')):
            self.send_header('Content-Encoding', 'gzip')
        self.send_header('Content-Length', len(body))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)

    def _static(self, path):
        fp = os.path.join(STATIC_DIR, path)
        if not os.path.abspath(fp).startswith(os.path.abspath(STATIC_DIR)):
            self._json(403, {'error': 'forbidden'})
            return
        if os.path.isfile(fp):
            ct = {
                '.html': 'text/html; charset=utf-8',
                '.css': 'text/css; charset=utf-8',
                '.js': 'application/javascript; charset=utf-8',
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.svg': 'image/svg+xml',
            }.get(os.path.splitext(fp)[1].lower(), 'application/octet-stream')
            with open(fp, 'rb') as f:
                data = f.read()
            if fp.endswith(('.html', '.css', '.js', '.svg')) and self._supports_gzip():
                compressed, is_gzip = gzip_compress(data)
                if is_gzip:
                    data = compressed
                    self.send_response(200)
                    self.send_header('Content-Encoding', 'gzip')
                else:
                    self.send_response(200)
            else:
                self.send_response(200)
            self.send_header('Content-Type', ct)
            etag = hashlib.md5(data).hexdigest()[:8]
            self.send_header('ETag', etag)
            self.send_header('Cache-Control', 'public, max-age=3600')
            self.send_header('Content-Length', len(data))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json(404, {'error': 'not found'})

    def _img(self, path):
        fp = resolve_img(path)
        if fp and os.path.isfile(fp):
            with open(fp, 'rb') as f:
                data = f.read()
            ct = 'image/png' if fp.endswith('.png') else 'image/jpeg'
            etag = hashlib.md5(data).hexdigest()[:8]
            if self.headers.get('If-None-Match') == etag:
                self.send_response(304)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header('Content-Type', ct)
            self.send_header('ETag', etag)
            self.send_header('Cache-Control', 'public, max-age=86400')
            self.send_header('Content-Length', len(data))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json(404, {'error': 'image not found'})

    def do_GET(self):
        self._start_time = time.time()
        path = self.path.split('?')[0]  # strip query params

        if path == '/':
            self._static('index.html')
        elif path == '/health':
            import requests as _r
            status = {'status': 'ok', 'server': 'AIKBChatBot/1.0'}
            # 检查 Qdrant
            try:
                resp = _r.get('http://localhost:6333/collections/hardware_kb', timeout=3)
                status['qdrant'] = resp.json().get('result', {}).get('points_count', 0)
            except:
                status['qdrant'] = None
            # 检查 embed_server
            try:
                resp = _r.get('http://localhost:8190/health', timeout=3)
                status['embed'] = resp.json().get('status', 'error')
            except:
                status['embed'] = None
            self._json(200, status)
        elif path.startswith('/img/'):
            self._img(path)
        elif path.startswith('/static/'):
            self._static(path[8:])
        else:
            self._json(404, {'error': 'not found'})

    def do_POST(self):
        self._start_time = time.time()
        if self.path != '/ask':
            self._json(404, {'error': 'not found'})
            return

        # 频率限制
        ip = self.client_address[0]
        if not rate_check(ip):
            self._json(429, {'error': 'rate limit exceeded', 'retry_after': 60})
            log.warning('rate limit: %s', ip)
            return

        # 读取请求体
        length = int(self.headers.get('Content-Length', 0))
        if length > 4096:
            self._json(413, {'error': 'request too large'})
            return

        body = self.rfile.read(length).decode('utf-8', errors='replace')
        try:
            query = json.loads(body).get('query', '').strip()
        except:
            query = body.strip()

        if not query:
            self._json(400, {'error': 'empty query'})
            return
        if len(query) > MAX_QUERY_LEN:
            self._json(400, {'error': 'query too long'})
            return

        t0 = time.time()
        try:
            result = do_query(query)
            result['elapsed_ms'] = int((time.time() - t0) * 1000)
            log.info('query "%s" → %dms [%s]', query[:50], result['elapsed_ms'],
                     result.get('search_method', '?'))
            self._json(200, result)
        except Exception as e:
            elapsed = int((time.time() - t0) * 1000)
            log.error('query "%s" failed in %dms: %s', query[:50], elapsed, e)
            self._json(500, {'error': 'internal error', 'elapsed_ms': elapsed})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT

    # 预热引擎
    log.info('启动 AI服务助手 (AI KB ChatBot)...')
    try:
        get_engine()
    except Exception as e:
        log.error('引擎初始化失败: %s', e)
        log.warning('服务器仍会启动，查询时将返回错误')

    server = ThreadedServer(('0.0.0.0', port), Handler)
    log.info('监听 http://0.0.0.0:%d', port)
    log.info('首页: http://47.101.48.143:%d', port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info('服务器关闭')
        server.shutdown()


if __name__ == '__main__':
    main()
