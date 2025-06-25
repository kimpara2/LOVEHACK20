# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()
import sqlite3
import stripe
import requests
import zipfile
import json
from functools import lru_cache

app = Flask(__name__)

def notify_gas_payment_success(user_id):
    GAS_URL = os.getenv("GAS_NOTIFY_URL")  # .env に記述しておく
    try:
        res = requests.post(GAS_URL, json={"userId": user_id, "paid": True})
        print("✅ GAS通知送信済み:", res.status_code, res.text)
    except Exception as e:
        print("❌ GAS通知エラー:", str(e))

# 🔐 OpenAI・Stripe・LINE設定
openai_api_key = os.getenv("OPENAI_API_KEY")
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
LINE_WEBHOOK_URL = os.getenv("LINE_WEBHOOK_URL")

# 🧳 chroma_db.zip を展開（初回起動時）
if not os.path.exists("./chroma_db") and os.path.exists("./chroma_db.zip"):
    with zipfile.ZipFile("./chroma_db.zip", 'r') as zip_ref:
        zip_ref.extractall("./chroma_db")

# 📖 MBTIアドバイス読み込み
with open("mbti_advice.json", "r", encoding="utf-8") as f:
    mbti_detailed_advice = json.load(f)

MBTI_NICKNAME = {
    "INTJ": "静かなる愛の地雷処理班",
    "ENTP": "恋のジェットコースター",
    # 必要に応じて他も追加
}

# 🧠 ベクトルDBを構成
VECTOR_BASE = "./chroma_db"
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)

# 💾 SQLite初期化

def init_db():
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            mbti TEXT,
            gender TEXT,
            target_mbti TEXT,
            is_paid BOOLEAN DEFAULT 0
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stripe_customers (
            customer_id TEXT PRIMARY KEY,
            user_id TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# 📦 ベクトルDB読み込み関数

@lru_cache(maxsize=64)
def load_retriever(path: str):
    return Chroma(persist_directory=path, embedding_function=embedding).as_retriever()

def get_retrievers(user_profile):
    sub_paths = [
        f"self/{user_profile['mbti']}",
        f"partner/{user_profile['target_mbti']}",
        user_profile['gender'],
        "common"
    ]

    retrievers = []

    for sub in sub_paths:
        path = os.path.join(VECTOR_BASE, sub)
        if os.path.exists(path):
            retrievers.append(load_retriever(path))

    return retrievers

# 🔄 複数Retrieverを結合（AND）
def get_qa_chain(user_profile):
    from langchain.retrievers import EnsembleRetriever
    retrievers = get_retrievers(user_profile)
    if not retrievers:
        raise ValueError("該当するベクトルDBが見つかりません")
    combined = EnsembleRetriever(retrievers=retrievers)
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    return RetrievalQA.from_chain_type(llm=llm, retriever=combined), llm

# 📬 LINE通知

def send_line_notification(user_id, message):
    try:
        requests.post(
            LINE_WEBHOOK_URL,
            json={"userId": user_id, "message": message},
            timeout=5
        )
    except Exception as e:
        print("LINE通知失敗:", str(e))

# 💾 DB操作

def get_user_profile(user_id):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT mbti, gender, target_mbti, is_paid FROM users WHERE user_id=?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return {
        "mbti": row[0] if row else "不明",
        "gender": row[1] if row else "不明",
        "target_mbti": row[2] if row else "不明",
        "is_paid": bool(row[3]) if row else False
    }

def save_message(user_id, role, content):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
    conn.commit()
    conn.close()

def get_recent_history(user_id, limit=5):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages WHERE user_id=? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [f"{row[0]}: {row[1]}" for row in reversed(rows)]

# 📍 MBTI診断登録
@app.route("/mbti_collect", methods=["POST"])
def mbti_collect():
    data = request.get_json()
    user_id = data.get("userId")
    gender = data.get("gender")
    target_mbti = data.get("targetMbti", "不明")
    answers = data.get("answers", [])

    if not user_id or len(answers) != 10:
        return jsonify({"error": "userIdと10個の回答が必要です"}), 400

    score = {"E":0, "I":0, "S":0, "N":0, "T":0, "F":0, "J":0, "P":0}
    mapping = [
        ("E", "I"), ("F", "T"), ("S", "N"), ("J", "P"), ("J", "P"),
        ("I", "E"), ("T", "F"), ("J", "P"), ("F", "T"), ("E", "I")
    ]
    for i, (yes_key, no_key) in enumerate(mapping):
        if answers[i]:
            score[yes_key] += 1
        else:
            score[no_key] += 1

    mbti = ""
    mbti += "E" if score["E"] >= score["I"] else "I"
    mbti += "S" if score["S"] >= score["N"] else "N"
    mbti += "T" if score["T"] >= score["F"] else "F"
    mbti += "J" if score["J"] >= score["P"] else "P"

    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute('''
        REPLACE INTO users (user_id, mbti, gender, target_mbti, is_paid)
        VALUES (?, ?, ?, ?, 0)
    ''', (user_id, mbti, gender, target_mbti))
    conn.commit()
    conn.close()

    return jsonify({"advice": "診断完了。詳細アドバイスは有料プランで見れるよ！"})

# 🔍 MBTI詳細アドバイス
@app.route("/mbti_detail", methods=["POST"])
def mbti_detail():
    data = request.get_json()
    user_id = data.get("userId")
    profile = get_user_profile(user_id)
    if not profile["is_paid"]:
        return jsonify({"error": "この機能は有料ユーザー限定です。"}), 403

    advice = mbti_detailed_advice.get(profile["mbti"], "準備中です")
    return jsonify({"detailed_advice": advice})

# ❓ 質問受付
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_id = data.get("userId")
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "質問が空です"}), 400

    profile = get_user_profile(user_id)
    if not profile["is_paid"]:
        return "", 204

    history = get_recent_history(user_id)

    try:
        qa_chain, llm = get_qa_chain(profile)

        answer = qa_chain.run(question)
        if any(x in answer for x in ["申し訳", "お答えできません", "確認できません"]):
            prompt = (
                f"ユーザー: {profile['gender']}の方（あなたの性格タイプ） / "
                f"相手の性格タイプあり\n"
                f"履歴:\n" + "\n".join(history) + "\n"
                f"質問: {question}\n"
                f"あなたはMBTI診断ベースの恋愛アドバイザーです。\n"
                f"性格タイプ名は出さず、親しみやすくタメ口で絵文字なども使ってわかりやすくアドバイスしてください。"
            )
            answer = llm.invoke(prompt).content

        save_message(user_id, "user", question)
        save_message(user_id, "bot", answer)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 💰 Stripe Webhook
@app.route("/stripe_webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("stripe-signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except Exception as e:
        return "Webhook error", 400

    if event["type"] in ["invoice.payment_succeeded", "checkout.session.completed"]:
    customer_id = event["data"]["object"].get("customer")

    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM stripe_customers WHERE customer_id=?", (customer_id,))
    row = cursor.fetchone()
    if row:
        user_id = row[0]
        cursor.execute("UPDATE users SET is_paid=1 WHERE user_id=?", (user_id,))
        conn.commit()

        text = mbti_detailed_advice.get(get_user_profile(user_id)["mbti"], "準備中")
        send_line_notification(user_id, text)
        notify_gas_payment_success(user_id)  # ←これがちゃんと動くようになる！
    conn.close()


    # 🔔 LINEに詳細アドバイスを送信
            text = mbti_detailed_advice.get(get_user_profile(user_id)["mbti"], "準備中")
            send_line_notification(user_id, text)

    # ✅ GASにも通知（← ここ追加！）
            notify_gas_payment_success(user_id)
            print("受信イベントタイプ:", event["type"])


        else:
            print(f"⚠️ customer_id に紐づく user_id が見つかりません: {customer_id}")

        conn.close()

    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

