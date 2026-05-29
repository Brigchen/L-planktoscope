#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 planktoscope_segment_viewer.py
if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] 启动失败，请确认已安装 Python 和依赖："
    echo "  pip install -r requirements.txt"
    echo ""
    read -p "按回车键退出..."
fi
