"""PersonalRAG 启动脚本

用法：
    python run.py                 # 默认 http://127.0.0.1:8000
    python run.py --open          # 启动后自动打开浏览器
    python run.py --port 8010     # 指定端口
    python run.py --reload        # 开发模式（改 Python 代码自动重启）

启动前会做依赖自检：缺什么直接打印 pip 命令，而不是抛一堆 traceback。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.config import config  # noqa: E402

REQUIRED = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("multipart", "python-multipart"),
    ("chromadb", "chromadb"),
    ("rank_bm25", "rank-bm25"),
    ("fitz", "PyMuPDF"),
    ("openai", "openai"),
    ("dotenv", "python-dotenv"),
]
OPTIONAL = [
    ("sentence_transformers", "sentence-transformers（本地向量模型，EMBEDDING_PROVIDER=local 时必需）"),
]


def _installed(module: str) -> bool:
    if module == "multipart":
        return any(
            importlib.util.find_spec(name) is not None
            for name in ("multipart", "python_multipart")
        )
    return importlib.util.find_spec(module) is not None


def check_dependencies() -> bool:
    missing = [package for module, package in REQUIRED if not _installed(module)]
    if missing:
        print("\n[启动失败] 缺少依赖：" + "、".join(missing))
        print("  修复：pip install -r requirements.txt\n")
        return False

    for module, package in OPTIONAL:
        if not _installed(module):
            print(f"[提示] 未安装 {package}")
    return True


def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def pick_port(preferred: int) -> int:
    if is_port_free(preferred):
        return preferred
    for candidate in range(preferred + 1, preferred + 20):
        if is_port_free(candidate):
            print(f"[提示] 端口 {preferred} 已被占用，改用 {candidate}")
            return candidate
    print(f"[启动失败] 端口 {preferred}~{preferred + 19} 都被占用，请用 --port 指定其他端口")
    raise SystemExit(1)


def open_when_ready(base_url: str, timeout: float = 300.0) -> None:
    """等 /api/health 可用后再开浏览器（首次启动要建索引，可能几十秒）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/health", timeout=1):
                webbrowser.open(base_url)
                return
        except Exception:
            time.sleep(0.6)


def main() -> None:
    parser = argparse.ArgumentParser(description="PersonalRAG 启动器")
    parser.add_argument("--host", default=config.host, help="监听地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=config.port, help="端口（默认 8000）")
    parser.add_argument("--reload", action="store_true", help="开发模式：代码变更自动重启")
    parser.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = parser.parse_args()

    if not check_dependencies():
        raise SystemExit(1)

    port = pick_port(args.port)
    display_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    base_url = f"http://{display_host}:{port}"
    has_ui = config.frontend_dist.is_dir()

    print("=" * 62)
    print("  PersonalRAG · 个人知识库问答")
    print(f"  API 文档 : {base_url}/docs")
    print(f"  聊天界面 : {base_url}/" if has_ui else "  聊天界面 : 未构建（先执行 npm run build）")
    print(f"  模型     : {config.llm.model}  向量: {config.embedding.provider}")
    print("=" * 62)

    for warning in config.validate():
        print(f"[提示] {warning}")
    print()

    if args.open:
        threading.Thread(
            target=open_when_ready,
            args=(base_url,),
            daemon=True,
        ).start()
    else:
        print("首次启动会加载向量模型并补建索引，请等日志出现 Application startup complete\n")

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
