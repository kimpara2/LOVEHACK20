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
import re

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
            is_paid BOOLEAN DEFAULT 0,
            mode TEXT DEFAULT '',
            mbti_answers TEXT DEFAULT '[]'
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
    cursor.execute("SELECT mbti, gender, target_mbti, is_paid, mode FROM users WHERE user_id=?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    return {
        "mbti": row[0] if row and row[0] else "不明",
        "gender": row[1] if row and row[1] else "不明",
        "target_mbti": row[2] if row and row[2] else "不明",
        "is_paid": bool(row[3]) if row else False,
        "mode": row[4] if row and row[4] else ""
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
        INSERT OR REPLACE INTO users (user_id, mbti, gender, target_mbti, is_paid, mode)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, mbti, gender, target_mbti, is_paid, ""))
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
        
        # LINE Webhookの検証（LINEプラットフォームからの検証リクエスト）
        if 'events' not in data:
            return '', 200
        
        # イベントを処理
        for event in data['events']:
            if event['type'] == 'message' and event['message']['type'] == 'text':
                user_id = event['source']['userId']
                user_message = event['message']['text'].strip()
                reply_token = event['replyToken']
                
                # ユーザープロファイルを取得
                user_profile = get_user_profile(user_id)
                
                # メッセージ処理
                response_message = process_user_message(user_id, user_message, user_profile)
                
                # LINEにリプライを送信
                send_line_reply(reply_token, response_message)
        
        return '', 200
    except Exception as e:
        print(f"LINE Webhook error: {e}")
        return '', 200  # エラーが発生しても200 OKを返す（LINEの要件）

# ユーザーメッセージ処理関数
def process_user_message(user_id, message, user_profile):
    """ユーザーメッセージを処理して適切な応答を返す"""
    
    # 性別登録モードの処理
    if user_profile.get('mode') == 'register_gender':
        if message in ['男', '女']:
            # 性別を保存
            conn = sqlite3.connect("user_data.db")
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET gender=? WHERE user_id=?", (message, user_id))
            cursor.execute("UPDATE users SET mode='' WHERE user_id=?", (user_id,))
            conn.commit()
            conn.close()
            return f"性別【{message}】を登録したよ！"
        else:
            return "【男】か【女】で答えてね！"
    
    # 相手のMBTI登録モードの処理
    if user_profile.get('mode') == 'register_partner_mbti':
        if re.match(r'^[EI][NS][FT][JP]$', message.upper()):
            mbti = message.upper()
            conn = sqlite3.connect("user_data.db")
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET target_mbti=? WHERE user_id=?", (mbti, user_id))
            cursor.execute("UPDATE users SET mode='' WHERE user_id=?", (user_id,))
            conn.commit()
            conn.close()
            return f"お相手のMBTI【{mbti}】を登録したよ！"
        else:
            return "正しいMBTI形式（例：INTJ、ENFP）で入力してね！"
    
    # MBTI診断モードの処理
    if user_profile.get('mode') == 'mbti_diagnosis':
        if message in ['はい', 'いいえ']:
            return process_mbti_answer(user_id, message, user_profile)
        else:
            return "【はい】か【いいえ】で答えてね！"
    
    # 通常のメッセージ処理
    if message == "診断開始":
        return start_mbti_diagnosis(user_id)
    
    elif message == "性別登録":
        conn = sqlite3.connect("user_data.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET mode='register_gender' WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
        return "性別を教えてね！\n【男】か【女】で答えてください。"
    
    elif message == "相手MBTI登録":
        conn = sqlite3.connect("user_data.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET mode='register_partner_mbti' WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
        return "お相手のMBTIを教えてね！\n（例：INTJ、ENFP、ISFJなど）"
    
    elif message == "チャット相談":
        if user_profile.get('is_paid'):
            return "チャット相談を開始します！\n恋愛の悩みを何でも相談してくださいね✨"
        else:
            return "チャット相談は有料機能です。\nまずは診断を完了して、詳細アドバイスをご購入ください！"
    
    else:
        # その他のメッセージはAIチャットで処理
        if user_profile.get('is_paid'):
            return process_ai_chat(user_id, message, user_profile)
        else:
            return "有料チャット相談をご利用いただくには、まず詳細アドバイスをご購入ください！"

# LINEリプライ送信関数
def send_line_reply(reply_token, message):
    """LINEにリプライメッセージを送信"""
    try:
        line_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
        if not line_token:
            print("⚠️ LINE_CHANNEL_ACCESS_TOKENが設定されていません")
            return
        
        url = "https://api.line.me/v2/bot/message/reply"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {line_token}"
        }
        data = {
            "replyToken": reply_token,
            "messages": [{"type": "text", "text": message}]
        }
        
        response = requests.post(url, headers=headers, json=data)
        print(f"LINEリプライ送信結果: {response.status_code}")
        
    except Exception as e:
        print(f"LINEリプライ送信エラー: {e}")

# AIチャット処理関数
def process_ai_chat(user_id, message, user_profile):
    """AIチャットの処理"""
    try:
        # 既存のask関数のロジックを使用
        qa_chain, llm = get_qa_chain(user_profile)
        
        # 会話履歴を取得
        history = get_recent_history(user_id, limit=5)
        history_text = "\n".join(history) if history else ""
        
        # プロンプトを作成
        prompt = (
            f"あなたはMBTI診断ベースの恋愛アドバイザーです。\n"
            f"ユーザーは{user_profile.get('gender', '不明')}の方で、性格タイプは{MBTI_NICKNAME.get(user_profile.get('mbti', ''), '不明')}です。\n"
            f"相手の性格タイプは{MBTI_NICKNAME.get(user_profile.get('target_mbti', ''), '不明')}です。\n"
            f"会話履歴:\n{history_text}\n"
            f"質問: {message}\n\n"
            f"性格タイプ名は出さず、ユーザーに寄り添い、親しみやすくタメ口で絵文字なども使ってわかりやすくアドバイスしてください。"
        )
        
        # LLMに質問
        if llm:
            response = llm.invoke(prompt)
            answer = response.content if response.content else "申し訳ありません。回答を生成できませんでした。"
        else:
            answer = "申し訳ありません。AIサービスが利用できません。"
        
        # 会話履歴を保存
        save_message(user_id, "user", message)
        save_message(user_id, "bot", answer)
        
        return answer
        
    except Exception as e:
        print(f"AIチャット処理エラー: {e}")
        return "申し訳ありません。エラーが発生しました。時間を置いて再度お試しください。"

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

# MBTI診断開始関数
def start_mbti_diagnosis(user_id):
    """MBTI診断を開始する"""
    # 診断状態を初期化
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET mode='mbti_diagnosis' WHERE user_id=?", (user_id,))
    cursor.execute("UPDATE users SET mbti_answers='[]' WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()
    
    # 最初の質問を送信
    return send_mbti_question(user_id, 0)

# MBTI質問送信関数
def send_mbti_question(user_id, question_index):
    """MBTI診断の質問を送信"""
    questions = [
        "好きな人とは、毎日LINEしたいほう？🥺",
        "デートの計画よりも、その時の気分で動くのが好き😳",
        "恋人のちょっとした変化にもすぐ気づくほうだ😊",
        "恋人の相談には、共感よりもアドバイスを優先しがち？📱",
        "初対面でも気になる人には自分から話しかけるほうだ？📅",
        "好きな人との関係がハッキリしないのは苦手？☕️",
        "デートは、思い出に残るようなロマンチックな演出が好き？💬➡️ ",
        "気になる人がいても、自分の気持ちはなかなか伝えられない？👫🔮",
        "恋愛には、価値観の一致が何より大事だと思う？💌",
        "相手の好みに合わせて、自分のキャラを柔軟に変えられる？😅"
    ]
    
    if question_index >= len(questions):
        return "診断が完了しました！"
    
    return f"質問{question_index + 1}/10\n\n{questions[question_index]}\n\n【はい】か【いいえ】で答えてね！"

# MBTI回答処理関数
def process_mbti_answer(user_id, answer, user_profile):
    """MBTI診断の回答を処理"""
    try:
        # 現在の回答を取得
        conn = sqlite3.connect("user_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT mbti_answers FROM users WHERE user_id=?", (user_id,))
        row = cursor.fetchone()
        
        if row and row[0]:
            answers = json.loads(row[0])
        else:
            answers = []
        
        # 新しい回答を追加
        answers.append(1 if answer == "はい" else 0)
        
        # 回答を保存
        cursor.execute("UPDATE users SET mbti_answers=? WHERE user_id=?", (json.dumps(answers), user_id))
        conn.commit()
        conn.close()
        
        # 次の質問を送信
        next_question_index = len(answers)
        if next_question_index < 10:
            return send_mbti_question(user_id, next_question_index)
        else:
            # 診断完了
            return complete_mbti_diagnosis(user_id, answers)
            
    except Exception as e:
        print(f"MBTI回答処理エラー: {e}")
        return "エラーが発生しました。もう一度診断を開始してください。"

# MBTI診断完了関数
def complete_mbti_diagnosis(user_id, answers):
    """MBTI診断を完了し、結果を送信"""
    try:
        # MBTI計算
        mbti = calc_mbti(answers)
        
        # 結果を保存
        conn = sqlite3.connect("user_data.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET mbti=?, mode='' WHERE user_id=?", (mbti, user_id))
        conn.commit()
        conn.close()
        
        # 診断結果メッセージを作成（簡潔版）
        result_message = f"🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️{MBTI_NICKNAME.get(mbti, mbti)}❤️\n\n{get_mbti_description(mbti)}"
        
        # 決済誘導メッセージ
        payment_message = ""──────────────\n💡もっと詳しく知りたい？💘\n\nどんな異性も落とせるようになるあなただけの詳しい恋愛攻略法\n『あなただけの専属の恋愛AI相談』が解放されます✨\n\n👉今すぐ登録して、完全版アドバイスと専属恋愛AIを試してみよう！\n" + checkoutUrl + "\n──────────────""
        
        # GASに詳細アドバイス送信を依頼（課金後に送信される）
        send_detailed_advice_to_gas(user_id, mbti)
        
        return f"{result_message}\n\n{payment_message}"
        
    except Exception as e:
        print(f"MBTI診断完了エラー: {e}")
        return "診断結果の処理中にエラーが発生しました。"

# MBTI説明取得関数
def get_mbti_description(mbti):
    """MBTIタイプの説明を取得"""
    descriptions = {
        "INTJ": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️静かなる愛の地雷処理班❤️\n\n一見クールで無関心そうなのに、実は「本命だけには一途」なあなた。\n感情よりも論理で動く慎重派で、恋も戦略的に進めがち。\nでも、心を許した相手には不器用ながらもちゃんと弱さを見せられる、ギャップが魅力💭\n「この人にだけは見せる顔」があるあなたに、落ちる人は多いはず。\n\n───\n🌙夜の顔は…【知的ドS教官🧠】\n普段は理性的なのに、夜は完全主導型。\n相手の反応を観察して流れをコントロールする、【静かな支配】タイプ。\n感情よりも満足度を重視するロジカルな夜、だけど…\n本気で心許した相手には、独占欲がちらっと出ることも🔥",
        "INTP": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️こじらせ知能型ラブロボ❤️\n\n感情よりも思考が先に立っちゃうあなた。\n「なんでそうなるの？」って考えすぎて、\n素直な一言がなかなか出てこないことも多いよね💭\n\nでも、興味を持った相手にはめちゃくちゃ深掘りするタイプで、\n一度ハマると他が見えなくなる「愛のオタク気質」も持ってる💘\nその分、自分でも気づかないうちに距離を取っちゃって、\n「冷たい…？」って誤解されることもあるかも😢\n\nでも大丈夫。あなたの魅力は、\n『知的さ×ピュアさ』という最強コンボだから🌟\n素直な一言で世界が変わる恋、あるかもしれないね。\n\n🌙夜の顔は…【知識プレイ職人📚】\n静かに、でも的確に。\n思考型ならではの『知っててやる』スイッチが入ると、\nテンポもタッチも計算され尽くしてる。\n無言なのにドキッとする、知的なゾクゾク感を演出するタイプ。",
        "ENTJ": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️恋も主導権ガチ勢❤️\n\n頼れるしっかり者で、自分の意思がはっきりしてるあなた。\n恋愛でも「こうしたい」「こうあるべき」って理想が明確で、ついつい主導権を握っちゃうことが多いよね💼\n\nでも本当は、恋にはちょっぴり不器用。\n甘えたいのにうまく出せなかったり、「好かれてから動きたい」って慎重になりがち。\n\n惹かれた相手には、誠実で計画的なアプローチで距離を縮めるタイプ。\n感情を素直に見せられた瞬間、一気に関係が進展するはず💘\n\n🌙夜の顔は…【支配のカリスマ指揮官🎩】\nベッドでも主導権を握りたい派。\n自分で雰囲気を組み立て、じっくり攻めてくる『理性と支配』のハイブリッド。\nでも実は、相手の気持ちや様子にもめちゃくちゃ敏感。\n一度任せたら、全部委ねたくなる人。",
        "ENTP": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️恋のジェットコースター❤️\n\n明るくてノリがよく、恋も勢い重視なあなた。\n好きになったら全力ダッシュで距離を詰めるけど、ちょっとでも冷めたら急ブレーキ…そんなアップダウンが魅力でもあるんだよね🎢\n自由で楽しい恋愛が好きだから、束縛やルールはちょっと苦手。\nでも、心から惹かれた相手にはちゃんと本気で向き合うよ💘\n一緒にいて飽きない、刺激的な存在になれるかが恋の鍵！\n\n🌙夜の顔は…【カオスな快楽実験者🧪】\n「これもアリ？」ってテンションで、毎回違うムードを演出。\nルール無用の自由プレイ派で、相手の反応を楽しみながら変化球を投げてくる。\n刺激と笑いに満ちた『予測不能な夜』を求めるタイプ。",
        "INFJ": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️重ためラブポエマー📜❤️\n\n見た目はおだやかでも、心の中は感情でぎゅうぎゅうなあなた。\n一度「この人だ」と思ったら、誰よりも深く、まっすぐ愛し抜くタイプ。\nその愛情は尊くて、美しいけど…ちょっと重ためなのもご愛敬📜\n\n駆け引きよりも共鳴を求めて、相手の気持ちを読みすぎて疲れちゃうこともあるよね。\nでも大丈夫。そんな繊細さこそが、あなたの魅力✨\n\n「この人なら分かってくれる」って心から思える相手に出会えたら、\nあなたの愛は最強の癒しになる。\n\n🌙夜の顔は…【妄想系エモスキンシップ魔🫂】\n妄想と理想が混ざりあったような甘くて深い世界観で、全身で『想い』を伝えるタイプ。\nゆっくりと抱きしめて、感情とぬくもりをじわじわ注ぎ込んでくる。\n目線や呼吸、全部に意味があるような繊細なリードが特徴。\n静かなのに、記憶に残る余韻系。",
        "INFP": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️愛されたいモンスター🧸❤️\n\n人一倍感受性が強くて、頭の中ではいつも恋の妄想がぐるぐる…\nでも実際はちょっぴり人見知りで、なかなか踏み込めなかったりするよね。\n\n理想の恋を大切にするロマンチストで、「本当に大切にしてくれる人じゃないとムリ」って気持ちが強め。\n裏切りや雑な扱いには超敏感で、心を開くのに時間がかかるぶん、一度許すと超献身的。\n\n本気になった時の『溺れ方』はピカイチで、相手のために尽くしたくなる愛の重さが最大の魅力。\n\n🌙夜の顔は…【妄想スイッチ爆走モンスター🧸】\n静かに見えて頭の中は常に全開モード。\nふとした瞬間にスイッチが入ると、想像を超える大胆さを見せてくるギャップ系。\n気持ちが乗った瞬間の甘え方がえぐい。",
        "ENFJ": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️ご奉仕マネージャー📋❤️\n\n人の気持ちに敏感で、つい周りを優先しちゃうあなた。\n恋愛でも「相手のために何ができるか」を考えて行動する、思いやりのプロ。\nただ、その優しさが『重い』って言われないか気にして、遠慮しすぎる一面もあるかも。\nでも本当は、愛されたい気持ちもめちゃくちゃ強いタイプ💘\nちゃんと「求めていいんだよ」って受け入れてくれる人に出会えたら、最強のパートナーになれるはず。\n\n🌙夜の顔は…【ご奉仕カスタマーサポート📞】\n優しさ100%で、相手の「気持ちよさ第一」に寄り添う奉仕型。\nどうしたら喜ぶか、何を求めてるかを察して自然に動けるから、安心感と快感のバランスが絶妙。\nエスコート力が高く、どんな要望にも『丁寧に対応』してくれるタイプ♡",
        "ENFP": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️かまってフェニックス🔥❤️\n\n感情豊かでノリがよくて、恋に全力なあなた。\n好奇心とテンションで距離を縮めるのが得意だけど、\nちょっとでも不安を感じると一気にテンションが下がっちゃう繊細さも💭\n\n気持ちの波が激しい分、喜怒哀楽を素直に出せるのが魅力。\n相手を楽しませようと頑張るけど、\nほんとは「自分が楽しませてもらいたい」気持ちも強いタイプかも。\n\n恋が続くかどうかのカギは、テンションじゃなくて『安心感』\nそのままの自分でいられる相手を選べば、長く愛せる人になるよ🌱\n\n🌙夜の顔は…【夜型テンションクラッシャー🌙】\n盛り上がるとスイッチが入って止まらないタイプ。\nテンションの爆発力で主導権を握りつつも、\nその場のノリと感情で流れをつくる「エモ速攻型」。\n終わったあとに急に静かになるギャップも魅力。",
        "ISTJ": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️恋愛ルールブック📘❤️\n\n真面目で誠実、計画的に恋を進めたいあなた。\n勢いやノリの恋よりも、安心感や信頼を大事にするタイプ。\n相手に振り回されるのは苦手で、自分のペースを崩さずに進めたい派。\nそのぶん、付き合ったあとの安定感は抜群！\nただ、ちょっと堅すぎたり、柔軟さに欠けると思われがちかも💭\nでも、ルールの中で見せるあなたの優しさや誠実さが、\n「ちゃんと向き合いたい」って人には最高の安心材料になるよ。\n\n🌙夜の顔は…【真面目な快楽マニュアル持参人📘】\nふざけたノリは少なめ、でもその分『確実に気持ちいいやつ』を用意してくる職人肌。\n静かに、でも丁寧に。\n頭の中には快楽のマニュアルが入っていて、一つひとつ手順を確認しながら進める感じ。\nムードより実行、でもその慎重さが逆に刺さるタイプ。",
        "ISFJ": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️感情しみしみおでん🍢❤️\n\n優しくて思いやりにあふれるあなた。\n相手の気持ちに敏感で、ちょっとした変化にもすぐ気づく『感情レーダー』タイプ。\n恋愛でも相手を最優先に考えて、つい自分のことは後回しにしがちかも。\nでもその「支えたい」「癒したい」気持ちが、相手の心をとろけさせる最大の魅力🫶\n安心感の塊みたいな存在だから、一緒にいるだけでホッとされるよ☕️\n\n🌙夜の顔は…【癒しの密着療法士👐】\n心も体も包み込む『触れる安心感』の持ち主。\nリードよりも寄り添い重視で、手を繋ぐだけでも気持ちを伝えるタイプ。\nスキンシップがゆっくり丁寧だから、気づけば深く安心してる…そんな「ぬくもり×共感」の夜に。",
        "ESTJ": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️正論ぶん回し侍⚔️❤️\n\n物事を理屈で整理して、白黒ハッキリつけたがるあなた。\n恋愛においても「あるべき関係像」が明確で、\n曖昧な態度や気まぐれな言動にはイライラしがち⚡️\nでも実は、とっても誠実で、責任感のある相手に弱い一面も。\n信頼できる相手には、一途に尽くす堅実派💎\nちょっとぶっきらぼうだけど、行動で愛を示すタイプ。\n感情表現は不器用でも、「守りたい」という気持ちは本物。\n\n🌙夜の顔は…【命令型快感プロデューサー🎬】\n理屈と段取りを駆使して、流れを完璧に組み立てるタイプ。\nムードは演出するもの、快楽はプロデュースするものというスタンスで、\n責めも褒めも計算済み。\nでも相手の満足を最優先に動く『戦略的優しさ』がある。\n冷静に見えて内心は熱く、期待を超える演出で魅せてくれる人。",
        "ESFJ": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️愛の押し売り百貨店🛍️❤️\n\n相手の喜ぶ顔が何よりのご褒美なあなた。\n「困ってない？」ってすぐ手を差し伸べたくなっちゃう、おせっかいな優しさが魅力💐\n\n恋愛でも相手目線で動こうとするから、つい無理してしまうことも。\nでも、その気配りがグッとくる人も多くて、いつの間にか好かれてることが多いタイプ。\n\nただ、自分の気持ちは後回しにしがちだから、たまにはワガママになっても大丈夫。\n『ありがとう』のひと言で、もっと自信持っていいんだよ✨\n\n🌙夜の顔は…【おせっかい夜間シフト係🌙】\n夜になっても気配りは止まらない。\n体温や呼吸まで気にしながら、とにかく『相手が心地いいか』を最優先。\n自分の欲よりも「満たしてあげたい」が先に来る、究極のホスピタリティ型。\nその優しさ、逆にクセになります。",
        "ISTP": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️甘え方わからん星人🪐❤️\n\nサバサバしてて一匹狼感あるあなた。\n感情を言葉で伝えるのはちょっと苦手だけど、\n態度や行動で誠実さを見せるタイプだよね🛠️\n\n恋愛においても無理にテンションを上げず、\n自然体でいられる関係を求めるクール派。\nでも、心を許した相手にはじんわりと優しさが伝わる…そんな不器用な一面も魅力💭\n\n言葉じゃない『空気』でつながるタイプだからこそ、\n無理せず自分らしくいられる相手と出会えたら強い✨\n\n🌙夜の顔は…【無言の手さばきマスター🖐️】\n会話少なめ、でも手は語る。\n相手の反応を静かに読み取りながら、\n淡々と、でも的確にツボを突いてくる。\nテクニカルなのにどこか素朴で、\n気づけば夢中にさせられてる…そんな静かなる支配系。",
        "ISFP": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️ぬくもり中毒者🔥❤️\n\n感情に正直で、あったかい空気感を大切にするあなた。\n人に優しく寄り添えるけど、傷つきやすさもあって、\n恋愛では「本当に信じられる人」じゃないと踏み込めない慎重派。\nでも、ひとたび心を開いた相手には、驚くほど深い愛情を注ぐ『本能型』。\nちょっと天然に見えて、実は直感で相手の心を読んでる…そんな魅力があるよ💫\nあなたの抱える『やさしさ』は、言葉よりも触れ方や表情で伝わるもの。\nそのぬくもりで、相手の心をほどく力があるよ。\n\n🌙夜の顔は…【密着とろけ職人🛏️】\nとにかく距離ゼロ。\n肌を合わせるたびに安心感が溢れ出す。\nリードは控えめでも、自然と求められる存在になる。\nその密着スキル、破壊力高め。",
        "ESTP": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️勢い重視族📶❤️\n\n思い立ったら即行動！\n恋も人生もノリと勢いで切り開いていくタイプ。\n興味を持った相手にはストレートにアプローチして、\n一気に距離を縮めるのが得意💥\n\nでも、熱しやすく冷めやすいところもあって、\n退屈になったらふと離れちゃうことも。\nドキドキ感を保つのが恋愛長続きのカギかも。\n感情をガマンするのが苦手だから、\nちゃんと本音で向き合ってくれる相手が◎\n\n🌙夜の顔は…【ハイテンポ破壊王🎮】\nムード？前戯？考えるよりまず行動！\n勢いと本能で押し切るタイプで、\nスピードと刺激を求める『ノンストップアタッカー』。\nでも本当は、相手の反応にめっちゃ敏感で、\n「楽しませたい」って気持ちが強いサービス精神旺盛タイプ。\n熱量MAXなぶつかり合いで、気づけば夢中になってるかも🔥",
        "ESFP": "🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️ハイテン・ラブ・ジェット🚀❤️\n\nいつでも元気でポジティブなあなたは、恋愛でもノリと勢いで飛び込んじゃうタイプ！\n感情表現が豊かで、一緒にいる人を自然と笑顔にしちゃう天性のムードメーカー🎉\n好きな人にはとことん尽くすし、ちょっとした変化にも敏感。\nでも、その分感情に振り回されたり、「気分屋？」って誤解されることも。\nそれでもあなたの魅力は、『楽しさの中にある本気さ』。\n軽そうに見えて、実はちゃんと想ってる…そのギャップが刺さるよ💘\n\n🌙夜の顔は…【快感ジャングルジム🛝】\nひとたびスイッチが入れば、テンションと好奇心で攻め続ける快楽マシーン。\n予測不能なタッチとノリの連続で、まるで遊園地みたいな時間を演出。\n「楽しませたい」気持ちがそのまま表れるから、一緒にいるとずっと飽きない。\n感情のままに動くようでいて、ちゃんと『相手を見てる』のがすごいところ。"
    }
    
    return descriptions.get(mbti, f"{mbti}タイプのあなたは、独特な魅力を持った恋愛タイプです。")

if __name__ == "__main__":
    # 環境変数が設定されているか確認
    print("=== 環境変数チェック ===")
    required_env_vars = ["OPENAI_API_KEY", "STRIPE_SECRET_KEY", "STRIPE_PRICE_ID", "STRIPE_WEBHOOK_SECRET", "LINE_CHANNEL_ACCESS_TOKEN"]
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