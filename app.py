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

# GASへの決済成功通知関数
def notify_gas_payment_success(user_id):
    GAS_URL = os.getenv("GAS_NOTIFY_URL")
    if not GAS_URL:
        print("⚠️ GAS_NOTIFY_URLが設定されていません。通知をスキップします。")
        return
    
    try:
        res = requests.post(GAS_URL, json={"userId": user_id, "paid": True})
        print("✅ GAS通知送信済み:", res.status_code, res.text)
    except Exception as e:
        print("❌ GAS通知エラー:", str(e))

# GASへの詳細アドバイス送信関数
def send_detailed_advice_to_gas(user_id, mbti):
    GAS_URL = os.getenv("GAS_NOTIFY_URL")
    if not GAS_URL:
        print("⚠️ GAS_NOTIFY_URLが設定されていません。詳細アドバイス送信をスキップします。")
        return
    
    try:
        # GASのgetDetailedAdvice関数を呼び出すためのリクエスト
        res = requests.post(GAS_URL, json={
            "action": "send_detailed_advice",
            "userId": user_id,
            "mbti": mbti
        })
        print("✅ 詳細アドバイス送信済み:", res.status_code, res.text)
    except Exception as e:
        print("❌ 詳細アドバイス送信エラー:", str(e))

# 🔐 OpenAI・Stripe・LINE設定
openai_api_key = os.getenv("OPENAI_API_KEY")
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
stripe_price_id = os.getenv("STRIPE_PRICE_ID")
LINE_WEBHOOK_URL = os.getenv("LINE_WEBHOOK_URL")

# 🧳 chroma_db.zip を展開（初回起動時）
# chroma_dbディレクトリが存在しない、かつchroma_db.zipが存在する場合に展開
if not os.path.exists("./chroma_db") and os.path.exists("./chroma_db.zip"):
    print("chroma_db.zipを展開中...")
    with zipfile.ZipFile("./chroma_db.zip", 'r') as zip_ref:
        zip_ref.extractall("./") # カレントディレクトリに展開
    print("chroma_db.zipの展開が完了しました。")

# 📖 MBTIアドバイス読み込み
# mbti_advice.jsonが存在するか確認
if not os.path.exists("mbti_advice.json"):
    print("エラー: mbti_advice.jsonが見つかりません。")
    mbti_detailed_advice = {} # ファイルがない場合は空の辞書を設定
else:
    with open("mbti_advice.json", "r", encoding="utf-8") as f:
        mbti_detailed_advice = json.load(f)
    print("mbti_advice.jsonを読み込みました。")


# MBTIニックネームの定義 (GASと同期させることを推奨)
MBTI_NICKNAME = {
    "INTJ": "静かなる愛の地雷処理班",
    "INTP": "こじらせ知能型ラブロボ",
    "ENTJ": "恋も主導権ガチ勢",
    "ENTP": "恋のジェットコースター",
    "INFJ": "重ためラブポエマー📜",
    "INFP": "愛されたいモンスター ",
    "ENFJ": "ご奉仕マネージャー📋",
    "ENFP": "かまってフェニックス🔥",
    "ISTJ": "恋愛ルールブック📘",
    "ISFJ": "感情しみしみおでん🍢",
    "ESTJ": "正論ぶん回し侍⚔️",
    "ESFJ": "愛の押し売り百貨店🛍️",
    "ISTP": "甘え方わからん星人🪐",
    "ISFP": "ぬくもり中毒者🔥",
    "ESTP": "勢い重視族📶",
    "ESFP": "ハイテン・ラブ・ジェット🚀"
}

# 🧠 ベクトルDBを構成
VECTOR_BASE = "./chroma_db"
# OpenAIEmbeddingsの初期化（APIキーが設定されていない場合はエラーになるので注意）
try:
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    print("OpenAIEmbeddingsを初期化しました。")
except Exception as e:
    print(f"OpenAIEmbeddingsの初期化に失敗しました: {e}")
    embedding = None # エラー時はNoneを設定し、後続処理でハンドリング


# 💾 SQLite初期化
def init_db():
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    # usersテーブルの作成
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            mbti TEXT,
            gender TEXT,
            target_mbti TEXT,
            is_paid BOOLEAN DEFAULT 0
        )
    ''')
    # stripe_customersテーブルの作成
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stripe_customers (
            customer_id TEXT PRIMARY KEY,
            user_id TEXT
        )
    ''')
    # messagesテーブルの作成（会話履歴保存用）
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
    print("SQLiteデータベースを初期化しました。")

init_db() # アプリケーション起動時にDBを初期化

# 📦 ベクトルDB読み込み関数 (Lruキャッシュでパフォーマンス向上)
@lru_cache(maxsize=64)
def load_retriever(path: str):
    if embedding is None:
        raise ValueError("Embedding function is not initialized. Cannot load retriever.")
    if not os.path.exists(path):
        print(f"警告: ベクトルDBのパスが見つかりません: {path}")
        return None # パスが存在しない場合はNoneを返す
    try:
        return Chroma(persist_directory=path, embedding_function=embedding).as_retriever()
    except Exception as e:
        print(f"ベクトルDBの読み込みに失敗しました ({path}): {e}")
        return None

# ユーザープロファイルに基づいたRetrieverの取得
def get_retrievers(user_profile):
    sub_paths = []
    # 自分のMBTIに基づくパス
    if user_profile['mbti'] and user_profile['mbti'] != "不明":
        sub_paths.append(f"self/{user_profile['mbti']}")
    # 相手のMBTIに基づくパス
    if user_profile['target_mbti'] and user_profile['target_mbti'] != "不明":
        sub_paths.append(f"partner/{user_profile['target_mbti']}")
    # 性別に基づくパス
    if user_profile['gender'] and user_profile['gender'] != "不明":
        sub_paths.append(user_profile['gender'])
    # 共通パスは常に含める
    sub_paths.append("common")

    retrievers = []
    for sub in sub_paths:
        path = os.path.join(VECTOR_BASE, sub)
        ret = load_retriever(path)
        if ret:
            retrievers.append(ret)
    return retrievers

# 🔄 複数Retrieverを結合（EnsembleRetrieverを使用）
def get_qa_chain(user_profile):
    from langchain.retrievers import EnsembleRetriever
    retrievers = get_retrievers(user_profile)
    if not retrievers:
        # どのRetrieverも見つからなかった場合、エラーではなく、デフォルトのLLMを返すなどの対応も検討
        print("警告: 該当するベクトルDBが見つかりません。デフォルトのLLMを使用します。")
        llm = ChatOpenAI(openai_api_key=openai_api_key)
        return None, llm # Retrieverがない場合はqa_chainをNoneとして返す

    # weightsは全てのRetrieverに均等に設定（必要に応じて調整）
    weights = [1.0 / len(retrievers)] * len(retrievers)
    combined = EnsembleRetriever(retrievers=retrievers, weights=weights)
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    return RetrievalQA.from_chain_type(llm=llm, retriever=combined), llm

# 📬 LINE通知（GASからHTTP POSTで呼び出される想定）
# この関数はFlaskアプリケーション自体からLINEに直接通知を送るもので、
# GASからLINEへのリプライとは異なります。
def send_line_notification(user_id, message):
    # LINE Messaging APIへの直接リクエスト
    # FlaskからはWebhook URLではなく、Messaging APIのエンドポイントを叩く必要があります
    # LINE Developersで発行したChannel Access Tokenが必要
    LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
    if not LINE_CHANNEL_ACCESS_TOKEN:
        print("エラー: LINE_CHANNEL_ACCESS_TOKENが設定されていません。LINE通知はスキップされます。")
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    payload = {
        "to": user_id,
        "messages": [
            {
                "type": "text",
                "text": message
            }
        ]
    }
    try:
        res = requests.post("https://api.line.me/v2/bot/message/push", headers=headers, json=payload, timeout=5)
        res.raise_for_status() # HTTPエラーがあれば例外を発生
        print(f"LINEプッシュ通知成功: {res.status_code}")
    except requests.exceptions.Timeout as e:
        print("LINE通知タイムアウト:", str(e))
    except requests.exceptions.RequestException as e:
        print("LINE通知失敗:", str(e))

# 💾 DB操作

# ユーザープロファイルの取得
def get_user_profile(user_id):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT mbti, gender, target_mbti, is_paid FROM users WHERE user_id=?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return {
        "mbti": row[0] if row and row[0] else "不明", # Noneや空文字の場合も"不明"に
        "gender": row[1] if row and row[1] else "不明",
        "target_mbti": row[2] if row and row[2] else "不明",
        "is_paid": bool(row[3]) if row else False
    }

# メッセージ履歴の保存
def save_message(user_id, role, content):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
    conn.commit()
    conn.close()

# 最新の会話履歴の取得
def get_recent_history(user_id, limit=5):
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages WHERE user_id=? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    # 履歴を古い順に並べ替える
    return [f"{row[0]}: {row[1]}" for row in reversed(rows)]

# GASと完全一致のMBTI集計ロジック
def calc_mbti(answers):
    score = {'E': 0, 'I': 0, 'S': 0, 'N': 0, 'T': 0, 'F': 0, 'J': 0, 'P': 0}
    mapping = [
        ('E', 'I'),
        ('P', 'J'),
        ('S', 'N'),
        ('T', 'F'),
        ('E', 'I'),
        ('J', 'P'),
        ('N', 'S'),
        ('I', 'E'),
        ('F', 'T'),
        ('P', 'J')
    ]
    for i, ans in enumerate(answers):
        yes, no = mapping[i]
        if ans:
            score[yes] += 1
        else:
            score[no] += 1
    mbti = (
        ('E' if score['E'] >= score['I'] else 'I') +
        ('S' if score['S'] >= score['N'] else 'N') +
        ('T' if score['T'] >= score['F'] else 'F') +
        ('J' if score['J'] >= score['P'] else 'P')
    )
    return mbti

# 📍 MBTI診断結果登録エンドポイント
# GASから診断結果が送信されることを想定
@app.route("/mbti_collect", methods=["POST"])
def mbti_collect():
    data = request.get_json()
    user_id = data.get("userId")
    gender = data.get("gender", "不明")
    target_mbti = data.get("targetMbti", "不明")
    answers = data.get("answers", [])
    if not user_id or not isinstance(answers, list) or len(answers) != 10:
        return jsonify({"error": "userIdと10個のanswersが必要です"}), 400
    mbti = calc_mbti(answers)
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT is_paid FROM users WHERE user_id=?", (user_id,))
    row = cursor.fetchone()
    is_paid = bool(row[0]) if row else False
    cursor.execute('''
        INSERT OR REPLACE INTO users (user_id, mbti, gender, target_mbti, is_paid)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, mbti, gender, target_mbti, is_paid))
    conn.commit()
    conn.close()
    return jsonify({"mbti": mbti})

# Checkoutセッション作成エンドポイント
@app.route("/create_checkout_session", methods=["POST"])
def create_checkout_session():
    data = request.get_json()
    user_id = data.get("userId")
    if not user_id:
        return jsonify({"error": "userIdが必要です"}), 400

    # デバッグ用ログ
    print(f"DEBUG: stripe_price_id = {stripe_price_id}")
    print(f"DEBUG: stripe.api_key = {'SET' if stripe.api_key else 'NOT SET'}")
    
    # 一時的にデフォルト値を設定（テスト用）
    price_id = stripe_price_id or "price_1RYfUgGEUGCv0Pohu7xYJzlJ"
    
    if not price_id:
        return jsonify({"error": "Stripe Price IDが設定されていません"}), 500
    
    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price": price_id,
                "quantity": 1,
            }],
            mode="payment",
            success_url="https://lovehack20.onrender.com/success",
            cancel_url="https://lovehack20.onrender.com/cancel",
            metadata={"userId": user_id}
        )
        print(f"DEBUG: Created session URL = {session.url}")
        return jsonify({"checkout_url": session.url})
    except Exception as e:
        print(f"DEBUG: Stripe error = {str(e)}")
        return jsonify({"error": f"Stripe error: {str(e)}"}), 500

# 🔍 MBTI詳細アドバイス取得エンドポイント
# 有料ユーザー向けの詳細アドバイスを返す
@app.route("/mbti_detail", methods=["POST"])
def mbti_detail():
    data = request.get_json()
    user_id = data.get("userId")
    if not user_id:
        return jsonify({"error": "userIdが必要です"}), 400

    profile = get_user_profile(user_id)
    if not profile["is_paid"]:
        return jsonify({"error": "この機能は有料ユーザー限定です。"}), 403

    advice = mbti_detailed_advice.get(profile["mbti"], "詳細アドバイスは現在準備中です。")
    return jsonify({"detailed_advice": advice})

# ❓ 質問受付エンドポイント（AI相談機能）
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_id = data.get("userId")
    question = data.get("question")
    if not user_id or not question:
        return jsonify({"error": "userIdとquestionが必要です"}), 400
    user_profile = get_user_profile(user_id)
    if not user_profile["is_paid"]:
        return jsonify({"error": "有料会員のみ利用可能です"}), 403
    history = get_recent_history(user_id) # 会話履歴を取得

    try:
        qa_chain, llm = get_qa_chain(user_profile)
        answer = "質問の答えが見つかりませんでした。" # デフォルトの回答

        # Retrieverが存在する場合のみRetrievalQAを実行
        if qa_chain:
            qa_result = qa_chain.invoke({"query": question})
            answer = qa_result.get("result", answer)
            print(f"RetrievalQAの回答: {answer}")

        # 回答が不十分な場合や特定のキーワードが含まれる場合にLLMに直接質問
        if not qa_chain or any(x in answer for x in ["申し訳", "お答えできません", "確認できません", "見つかりません", "提供できません"]):
            # LLMに直接質問するためのプロンプト
            prompt = (
                f"あなたはMBTI診断ベースの恋愛アドバイザーです。\n"
                f"ユーザーは{user_profile['gender']}の方で、性格タイプは{MBTI_NICKNAME.get(user_profile['mbti'], '不明')}です。\n"
                f"相手の性格タイプは{MBTI_NICKNAME.get(user_profile['target_mbti'], '不明')}です。\n"
                f"会話履歴:\n" + "\n".join(history) + "\n"
                f"質問: {question}\n\n"
                f"性格タイプ名は出さず、ユーザーに寄り添い、親しみやすくタメ口で絵文字なども使ってわかりやすくアドバイスしてください。\n"
                f"ただし、ユーザーの性別や相手のMBTIタイプを踏まえた上で回答してください。"
            )
            print("RetrievalQAの回答が不十分だったため、LLMに直接質問します。")
            llm_response = llm.invoke(prompt)
            answer = llm_response.content if llm_response.content else answer
            print(f"LLM直接回答: {answer}")


        save_message(user_id, "user", question)
        save_message(user_id, "bot", answer)
        return jsonify({"answer": answer})

    except Exception as e:
        print(f"AI質問処理中にエラーが発生しました: {e}")
        return jsonify({"error": "AIの応答中にエラーが発生しました。時間を置いて再度お試しください。"}), 500


# 💰 Stripe Webhookエンドポイント
# Stripeからのイベント通知を受け取り、決済状況をDBに反映
@app.route("/stripe_webhook", methods=["POST"])
def stripe_webhook():
    data = request.get_json()
    user_id = data.get('userId')
    if not user_id:
        return '', 400
    
    try:
        conn = sqlite3.connect("user_data.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET is_paid=1 WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
        
        # 課金完了通知をGASに送信
        notify_gas_payment_success(user_id)
        
        # ユーザーのMBTIを取得して詳細アドバイスを送信
        user_profile = get_user_profile(user_id)
        if user_profile and user_profile.get("mbti"):
            send_detailed_advice_to_gas(user_id, user_profile["mbti"])
            print(f"✅ 課金完了: ユーザー{user_id}の詳細アドバイスを送信しました（MBTI: {user_profile['mbti']}）")
        else:
            print(f"⚠️ 課金完了: ユーザー{user_id}のMBTIが見つかりませんでした")
        
        return '', 200
    except Exception as e:
        print(f"Stripe webhook処理エラー: {e}")
        return '', 500

# 決済URL作成エンドポイント（GASから呼び出される）
# GASのcreatePaymentUrl関数がこのエンドポイントを呼び出し、
# ユーザーをStripeのCheckoutページにリダイレクトさせるURLを返す
@app.route("/create_payment_url", methods=["POST"])
def create_payment_url():
    try:
        data = request.get_json()
        print(f"DEBUG: create_payment_url received data: {data}")
        
        user_id = data.get("userId")
        print(f"DEBUG: userId extracted: {user_id}")

        if not user_id:
            print("ERROR: userId is missing or empty")
            return jsonify({"error": "userIdが必要です"}), 400

        # 環境変数の確認
        print(f"DEBUG: stripe.api_key = {'SET' if stripe.api_key else 'NOT SET'}")
        print(f"DEBUG: stripe_price_id = {stripe_price_id}")
        
        # Stripe APIキーが設定されているか確認
        if not stripe.api_key:
            print("ERROR: Stripe API key is not set")
            return jsonify({"error": "Stripe API key is not configured"}), 500

        # 直接Stripeのチェックアウトセッションを作成
        price_id = stripe_price_id or "price_1RYfUgGEUGCv0Pohu7xYJzlJ"
        print(f"DEBUG: Using price_id: {price_id}")
        
        if not price_id:
            print("ERROR: No valid price ID found")
            return jsonify({"error": "Stripe Price IDが設定されていません"}), 500
        
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price": price_id,
                "quantity": 1,
            }],
            mode="payment",
            success_url="https://lovehack20.onrender.com/success",
            cancel_url="https://lovehack20.onrender.com/cancel",
            metadata={"userId": user_id}
        )
        
        original_url = session.url
        print(f"DEBUG: Original Stripe URL length: {len(original_url)}")
        print(f"DEBUG: Original Stripe URL: {original_url}")
        
        # URL短縮を試行
        try:
            shortened_url = shorten_url(original_url)
            print(f"DEBUG: Shortened URL length: {len(shortened_url)}")
            print(f"DEBUG: Shortened URL: {shortened_url}")
            final_url = shortened_url
        except Exception as e:
            print(f"WARNING: URL shortening failed: {str(e)}, using original URL")
            final_url = original_url
        
        return jsonify({"url": final_url})
        
    except stripe.error.StripeError as e:
        print(f"ERROR: Stripe API error: {str(e)}")
        return jsonify({"error": f"Stripe API error: {str(e)}"}), 500
    except Exception as e:
        print(f"ERROR: Unexpected error in create_payment_url: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# URL短縮関数
def shorten_url(long_url):
    """URL短縮サービスを使用してURLを短縮する"""
    try:
        # TinyURL APIを使用（無料で利用可能）
        response = requests.post(
            "https://tinyurl.com/api-create.php",
            data={"url": long_url},
            timeout=10
        )
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"TinyURL API error: {response.status_code}")
    except Exception as e:
        print(f"URL shortening error: {str(e)}")
        # 短縮に失敗した場合は元のURLを返す
        return long_url

# 成功ページ
@app.route("/success", methods=["GET"])
def success_page():
    return "<h1>決済が完了しました🎉 LINEに戻ってください！</h1>"

# キャンセルページ
@app.route("/cancel", methods=["GET"])
def cancel_page():
    return "<h1>決済をキャンセルしました。</h1>"

# ルートエンドポイント（ヘルスチェック用）
@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "message": "LoveHack API is running"})

# LINE Webhookエンドポイント（LINEプラットフォームからのPOSTリクエストを受け取る）
@app.route("/webhook", methods=["POST"])
def line_webhook():
    try:
        # LINEプラットフォームからのリクエストを受け取る
        data = request.get_json()
        print(f"LINE Webhook received: {data}")
        
        # ここでLINEからのメッセージを処理する
        # 現在はGASがメッセージ処理を担当しているため、
        # このエンドポイントは単純に200 OKを返すだけ
        
        return '', 200
    except Exception as e:
        print(f"LINE Webhook error: {e}")
        return '', 200  # エラーが発生しても200 OKを返す（LINEの要件）

# LINE Webhookの代替パス（LINEプラットフォームが使用する可能性のあるパス）
@app.route("/callback", methods=["POST"])
def line_callback():
    try:
        data = request.get_json()
        print(f"LINE Callback received: {data}")
        return '', 200
    except Exception as e:
        print(f"LINE Callback error: {e}")
        return '', 200

# LINE Messaging API Webhook（標準的なパス）
@app.route("/messaging-api/webhook", methods=["POST"])
def messaging_api_webhook():
    try:
        data = request.get_json()
        print(f"LINE Messaging API Webhook received: {data}")
        return '', 200
    except Exception as e:
        print(f"LINE Messaging API Webhook error: {e}")
        return '', 200

# テスト用エンドポイント（GASからのリクエスト確認用）
@app.route("/test", methods=["POST"])
def test_endpoint():
    try:
        data = request.get_json()
        print(f"TEST: Received data: {data}")
        return jsonify({"status": "success", "received_data": data})
    except Exception as e:
        print(f"TEST ERROR: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 環境変数が設定されているか確認
    print("=== 環境変数チェック ===")
    required_env_vars = ["OPENAI_API_KEY", "STRIPE_SECRET_KEY", "STRIPE_PRICE_ID", "STRIPE_WEBHOOK_SECRET"]
    for var in required_env_vars:
        value = os.getenv(var)
        if not value:
            print(f"⚠️ 警告: 環境変数 {var} が設定されていません。関連機能が動作しない可能性があります。")
        else:
            print(f"✅ {var}: {'SET' if value else 'NOT SET'}")
    
    print(f"=== Stripe設定確認 ===")
    print(f"Stripe API Key: {'SET' if stripe.api_key else 'NOT SET'}")
    print(f"Stripe Price ID: {stripe_price_id}")
    print(f"GAS Notify URL: {os.getenv('GAS_NOTIFY_URL', 'NOT SET')}")
    print("========================")

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000))) # PORT環境変数を使用
 
 
 