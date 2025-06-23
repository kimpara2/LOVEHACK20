# Pythonの安定版をベースに
FROM python:3.11-slim

# 作業ディレクトリ
WORKDIR /app

# 必要ファイルをコピー
COPY . .

# 依存パッケージをインストール
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ポートを公開
EXPOSE 5000

# アプリ起動
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
