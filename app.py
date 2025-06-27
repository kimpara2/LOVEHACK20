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
    except requests.exceptions.Timeout:
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


# 📍 MBTI診断結果登録エンドポイント
# GASから診断結果が送信されることを想定
@app.route("/mbti_collect", methods=["POST"])
def mbti_collect():
    data = request.get_json()
    user_id = data.get("userId")
    gender = data.get("gender")
    target_mbti = data.get("targetMbti", "不明")
    answers = data.get("answers", []) # GASから送られてくるanswersを使用

    if not user_id:
        return jsonify({"error": "userIdが必要です"}), 400

    # 既存のユーザープロファイルを取得
    existing_profile = get_user_profile(user_id)

    # MBTIの再計算（GASとFlaskでMAPPINGが完全に一致していることを確認してください）
    # 今回の修正ではGAS側のMAPPINGに合わせるために、Flask側のMAPPINGも更新することを推奨します。
    # ここでは既存のFlask側のMAPPINGを維持しますが、注意が必要です。
    score = {"E":0, "I":0, "S":0, "N":0, "T":0, "F":0, "J":0, "P":0}
    # ユーザーが提供したanswersに基づいてスコアを計算
    # 修正: MAPPINGのインデックスと順番をGASのMAPPINGに合わせて修正
    # このmappingはGASのMAPPINGと一致させる必要があります
    mapping_flask = [
        ("E", "I"), # 好きな人とは、毎日LINEしたいほう？🥺
        ("P", "J"), # デートの計画よりも、その時の気分で動くのが好き😳
        ("S", "N"), # 恋人のちょっとした変化にもすぐ気づくほうだ😊
        ("T", "F"), # 恋人の相談には、共感よりもアドバイスを優先しがち？📱
        ("E", "I"), # 初対面でも気になる人には自分から話しかけるほうだ？📅
        ("J", "P"), # 好きな人との関係がハッキリしないのは苦手？☕️
        ("N", "S"), # デートは、思い出に残るようなロマンチックな演出が好き？💬➡️🤝
        ("I", "E"), # 気になる人がいても、自分の気持ちはなかなか伝えられない？👫🔮
        ("F", "T"), # 恋愛には、価値観の一致が何より大事だと思う？💌
        ("P", "J")  # 相手の好みに合わせて、自分のキャラを柔軟に変えられる？😅
    ]

    # answersはboolのリストとしてGASから送られてくる想定
    if len(answers) != len(mapping_flask):
        return jsonify({"error": "answersの数が不正です"}), 400

    for i, (yes_key, no_key) in enumerate(mapping_flask):
        if answers[i]: # answers[i]がTrueならyes_key、Falseならno_key
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
    # user_idが存在しない場合はINSERT、存在する場合はUPDATE
    cursor.execute('''
        INSERT OR REPLACE INTO users (user_id, mbti, gender, target_mbti, is_paid)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, mbti, gender, target_mbti, existing_profile["is_paid"])) # 既存のis_paid値を引き継ぐ

    conn.commit()
    conn.close()
    print(f"ユーザーID: {user_id} のMBTI診断結果を保存しました: {mbti}")

    return jsonify({"message": "MBTI診断結果を正常に受け取りました"}), 200

# Checkoutセッション作成エンドポイント（Stripe決済への誘導）
@app.route("/create_checkout_session", methods=["POST"])
def create_checkout_session():
    data = request.get_json()
    user_id = data.get("userId")

    if not user_id:
        return jsonify({"error": "userIdが必要です"}), 400

    try:
        # Stripeカスタマーの検索または作成
        customer_id = None
        conn = sqlite3.connect("user_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT customer_id FROM stripe_customers WHERE user_id=?", (user_id,))
        row = cursor.fetchone()
        if row:
            customer_id = row[0]
            print(f"既存のStripeカスタマーIDを使用: {customer_id} (user_id: {user_id})")
        else:
            # Stripeカスタマー作成
            customer = stripe.Customer.create(
                metadata={"user_id": user_id} # user_idをメタデータとして保存
            )
            customer_id = customer.id
            # customer_id と user_id を紐付けてDBに保存
            cursor.execute("INSERT INTO stripe_customers (customer_id, user_id) VALUES (?, ?)", (customer_id, user_id))
            conn.commit()
            print(f"新しいStripeカスタマーを作成: {customer_id} (user_id: {user_id})")
        conn.close()

        if not customer_id:
            raise Exception("StripeカスタマーIDの取得または作成に失敗しました。")

        # Checkoutセッション作成
        # Stripe Price IDが設定されていることを確認
        stripe_price_id = os.getenv("STRIPE_PRICE_ID")
        if not stripe_price_id:
            raise ValueError("STRIPE_PRICE_IDが設定されていません。")

        session = stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=["card"],
            line_items=[{
                "price": stripe_price_id,  # 環境変数から取得
                "quantity": 1,
            }],
            mode="subscription", # サブスクリプションモード
            success_url=f"{request.url_root}success", # アプリケーションのルートURLを使用
            cancel_url=f"{request.url_root}cancel"   # アプリケーションのルートURLを使用
        )
        print(f"Stripe Checkout Sessionを作成しました: {session.url}")
        return jsonify({"checkout_url": session.url})

    except stripe.error.StripeError as e:
        print(f"Stripeエラー: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"Checkout Session作成中にエラーが発生しました: {e}")
        return jsonify({"error": str(e)}), 500

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
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "質問が空です"}), 400

    profile = get_user_profile(user_id)
    # 有料ユーザーでない場合は、空のレスポンス（LINEに通知しない）を返す
    if not profile["is_paid"]:
        print(f"ユーザー {user_id} は有料ユーザーではありません。AI質問をスキップします。")
        return "", 204 # No Content

    history = get_recent_history(user_id) # 会話履歴を取得

    try:
        qa_chain, llm = get_qa_chain(profile)
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
                f"ユーザーは{profile['gender']}の方で、性格タイプは{MBTI_NICKNAME.get(profile['mbti'], '不明')}です。\n"
                f"相手の性格タイプは{MBTI_NICKNAME.get(profile['target_mbti'], '不明')}です。\n"
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
    payload = request.data
    sig_header = request.headers.get("stripe-signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    event = None
    try:
        # Webhookイベントの検証
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        print("🧾 Stripe イベントタイプ:", event["type"])
    except ValueError as e:
        # Invalid payload
        print("ValueError: Invalid payload", str(e))
        return "Invalid payload", 400
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        print("SignatureVerificationError: Invalid signature", str(e))
        return "Invalid signature", 400
    except Exception as e:
        print("Webhook error:", str(e))
        return "Webhook error", 400

    # 支払い成功イベントの処理
    if event["type"] in ["invoice.payment_succeeded", "checkout.session.completed"]:
        customer_id = event["data"]["object"].get("customer")
        if not customer_id:
            print("customer_idがイベントデータに含まれていません。")
            return "OK", 200

        conn = sqlite3.connect("user_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM stripe_customers WHERE customer_id=?", (customer_id,))
        row = cursor.fetchone()

        if row:
            user_id = row[0]
            # ユーザーの支払い状態を更新
            cursor.execute("UPDATE users SET is_paid=1 WHERE user_id=?", (user_id,))
            conn.commit()

            # ユーザープロファイルを再取得（最新のis_paid状態を反映）
            updated_profile = get_user_profile(user_id)
            user_mbti = updated_profile["mbti"]

            # 🔔 LINEに詳細アドバイスをプッシュ通知
            # mbti_detailed_adviceから詳細アドバイスを取得
            text_to_send = mbti_detailed_advice.get(user_mbti, "お支払いありがとうございます！詳細アドバイスは現在準備中です。")
            send_line_notification(user_id, text_to_send)

            # ✅ GASにも通知（支払い状況を同期するため）
            notify_gas_payment_success(user_id)

            print(f"ユーザーID: {user_id} の支払い状況を更新し、通知を送信しました。")
        else:
            print(f"⚠️ customer_id に紐づく user_id が見つかりません: {customer_id}")

        conn.close()

    return "OK", 200

# 決済URL作成エンドポイント（GASから呼び出される）
# GASのcreatePaymentUrl関数がこのエンドポイントを呼び出し、
# ユーザーをStripeのCheckoutページにリダイレクトさせるURLを返す
@app.route("/create_payment_url", methods=["POST"])
def create_payment_url():
    data = request.get_json()
    user_id = data.get("userId")

    if not user_id:
        return jsonify({"error": "userIdが必要です"}), 400

    # FlaskアプリケーションのURLと、checkoutセッション作成エンドポイントを組み合わせて返す
    # GAS側がこのURLをユーザーに提示し、ユーザーがブラウザで開くとStripeの決済ページに遷移する
    checkout_session_url = f"{request.url_root}create_checkout_session" # 修正: create_checkout_sessionを呼び出す
    print(f"create_payment_urlが呼び出されました。checkoutセッションURL: {checkout_session_url}")
    return jsonify({
        # 修正: ここで直接StripeのセッションURLを生成するのではなく、
        # /create_checkout_sessionを呼び出すためのURLを返す
        "url": f"{request.url_root}checkout?uid={user_id}" # このURLにアクセスするとcreate_checkout_sessionが呼び出されるようにする
    })


# 成功ページ
@app.route("/success", methods=["GET"])
def success_page():
    return "<h1>決済が完了しました🎉 LINEに戻ってください！</h1>"

# キャンセルページ
@app.route("/cancel", methods=["GET"])
def cancel_page():
    return "<h1>決済をキャンセルしました。</h1>"

# /checkoutエンドポイントの追加 (GETリクエストでStripe Checkout Sessionを呼び出すためのリダイレクト)
@app.route("/checkout", methods=["GET"])
def checkout_redirect():
    user_id = request.args.get("uid")
    if not user_id:
        return "エラー: userIdが指定されていません。", 400

    # Flask内部で/create_checkout_sessionを呼び出す
    with app.test_request_context(method='POST', path='/create_checkout_session', json={"userId": user_id}):
        response = create_checkout_session()
        data = json.loads(response[0].get_data(as_text=True)) # レスポンスからJSONデータを解析
        checkout_url = data.get("checkout_url")
        if checkout_url:
            from flask import redirect
            return redirect(checkout_url, code=302) # StripeのCheckoutページへリダイレクト
        else:
            return "決済URLの生成に失敗しました。", 500


if __name__ == "__main__":
    # 環境変数が設定されているか確認
    required_env_vars = ["OPENAI_API_KEY", "STRIPE_SECRET_KEY", "STRIPE_PRICE_ID", "STRIPE_WEBHOOK_SECRET"]
    for var in required_env_vars:
        if not os.getenv(var):
            print(f"⚠️ 警告: 環境変数 {var} が設定されていません。関連機能が動作しない可能性があります。")

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000))) # PORT環境変数を使用
 