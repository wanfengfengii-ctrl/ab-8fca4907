"""编钟残音联合基频识别服务（仅使用 Python 标准库）。

路由：
  GET  /healthz          健康检查
  GET  /                 前端页面（静态文件）
  POST /api/solve        联合求解业务 API
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.solver import ValidationError, solve  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}

MAX_BODY_BYTES = 64 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "BianzhongSolver/1.0"

    def log_message(self, fmt, *args):  # 精简访问日志
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        static = STATIC_FILES.get(path)
        if path == "/healthz":
            self._send_json(HTTPStatus.OK, {"status": "ok", "service": "bianzhong-solver"})
            return
        if static is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": "资源不存在"})
            return
        name, content_type = static
        try:
            body = (STATIC_DIR / name).read_bytes()
        except OSError:
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": "静态资源缺失"})
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/solve":
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "message": "未知接口"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(HTTPStatus.BAD_REQUEST,
                            {"status": "error", "message": "请求体缺失或超出大小限制"})
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": "请求不是合法 JSON"})
            return
        if not isinstance(payload, dict):
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": "请求体必须是 JSON 对象"})
            return

        try:
            result = solve(
                payload.get("frequencies"),
                payload.get("tolerances"),
                payload.get("f0_min"),
                payload.get("f0_max"),
                payload.get("max_harmonic"),
                payload.get("max_rejected"),
            )
        except ValidationError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR,
                            {"status": "error", "message": f"服务端内部错误：{exc}"})
            return

        self._send_json(HTTPStatus.OK, result.to_dict())


def main():
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"编钟残音联合基频识别服务监听 0.0.0.0:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
