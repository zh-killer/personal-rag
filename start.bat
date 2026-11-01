@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0backend"

echo.
echo PersonalRAG 启动中...
echo.

python run.py --open
if errorlevel 1 (
    echo.
    echo [启动失败] 请查看上面的错误信息。
    echo 如果提示找不到 python：安装 Python 3.10+ 并勾选 "Add Python to PATH"。
    echo 如果提示缺少依赖：执行 pip install -r requirements.txt
    echo.
    pause
)
