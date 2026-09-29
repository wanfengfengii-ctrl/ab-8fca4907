FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# 本项目仅使用 Python 标准库，无第三方运行时依赖（空安装，不触网）。
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY tests ./tests
COPY verify.py ./verify.py

# 字节编译作为构建期校验：语法错误直接让镜像构建失败。
RUN python -m compileall -q app verify.py

EXPOSE 8000

# 健康检查：不依赖 curl/wget，直接用标准库请求健康端点。
HEALTHCHECK --interval=5s --timeout=3s --start-period=3s --retries=10 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ.get('PORT','8000')+'/healthz', timeout=3); sys.exit(0 if r.status==200 else 1)"

CMD ["python", "-m", "app.server"]
