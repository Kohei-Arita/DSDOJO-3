# Python 3.11ベースイメージ
FROM python:3.11-slim

# 作業ディレクトリ
WORKDIR /workspace

# システムパッケージの更新とビルドツールのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 日本語フォントのインストール（japanize-matplotlib用）
RUN apt-get update && apt-get install -y \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Jupyter設定
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py

# ポート公開
EXPOSE 8888

# デフォルトコマンド
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
