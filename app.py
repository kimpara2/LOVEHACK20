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
import re
import traceback
import random

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
        res = requests.post(GAS_URL, json={
            "action": "send_detailed_advice",
            "userId": user_id,
            "mbti": mbti
        })
        print("✅ 詳細アドバイス送信済み:", res.status_code, res.text)
    except Exception as e:
        print("❌ 詳細アドバイス送信エラー:", str(e))

# GASへのチャットメッセージ送信関数
def send_chat_message_to_gas(user_id, mbti):
    GAS_URL = os.getenv("GAS_NOTIFY_URL")
    if not GAS_URL:
        print("⚠️ GAS_NOTIFY_URLが設定されていません。チャットメッセージ送信をスキップします。")
        return
    
    try:
        res = requests.post(GAS_URL, json={
            "action": "send_chat_message",
            "userId": user_id,
            "mbti": mbti
        })
        print("✅ チャットメッセージ送信済み:", res.status_code, res.text)
    except Exception as e:
        print("❌ チャットメッセージ送信エラー:", str(e))

# 🔐 OpenAI・Stripe・LINE設定
openai_api_key = os.getenv("OPENAI_API_KEY")
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
stripe_price_id = os.getenv("STRIPE_PRICE_ID")
stripe_webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

# 💾 データベースパス設定（環境に応じて切り替え）
DB_PATH = os.getenv("DB_PATH", "/data/user_data.db")  # 本番環境では永続ディスクを使用

# 🧳 chroma_db.zip を展開（初回起動時）
if not os.path.exists("./chroma_db") and os.path.exists("./chroma_db.zip"):
    print("chroma_db.zipを展開中...")
    with zipfile.ZipFile("./chroma_db.zip", 'r') as zip_ref:
        zip_ref.extractall("./")
    print("chroma_db.zipの展開が完了しました。")

# 📖 MBTIアドバイス読み込み
if not os.path.exists("mbti_advice.json"):
    print("エラー: mbti_advice.jsonが見つかりません。")
    mbti_detailed_advice = {}
else:
    with open("mbti_advice.json", "r", encoding="utf-8") as f:
        mbti_detailed_advice = json.load(f)
    print("mbti_advice.jsonを読み込みました。")

# MBTIニックネームの定義
MBTI_NICKNAME = {
    "INTJ": "静かなる愛の地雷処理班",
    "INTP": "こじらせ知能型ラブロボ",
    "ENTJ": "恋も主導権ガチ勢",
    "ENTP": "恋のジェットコースター",
    "INFJ": "重ためラブポエマー📜",
    "INFP": "愛されたいモンスター🧸",
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

# MBTI別の詳細な性格特徴と恋愛傾向
MBTI_PERSONALITY = {
    "INTJ": {
        "traits": ["戦略的思考", "独立心が強い", "完璧主義", "論理的", "洞察力", "効率性重視", "プライバシー重視"],
        "love_style": "深い絆を求めるが、プライバシーも大切にする知的パートナーシップ。長期的なビジョンを共有し、お互いの成長をサポートし合う関係を重視する。感情表現は控えめだが、誠実さと一貫性で相手を支える。効率的で戦略的なアプローチを好み、無駄のない愛情表現を心がける。",
        "strengths": ["長期的な視点", "誠実さ", "知的刺激", "戦略的アプローチ", "一貫性"],
        "challenges": ["感情表現が苦手", "完璧を求める", "相手の感情を理解しにくい", "柔軟性不足"],
        "advice_style": "論理的で構造化されたアドバイス",
        "likes_in_partner": ["知的刺激", "独立心", "効率性", "誠実さ", "長期的ビジョン", "プライバシーを尊重"],
        "dislikes_in_partner": ["感情的な要求", "無計画", "表面的な関係", "依存心", "非効率"],
        "common_concerns": ["相手が自分の感情を理解してくれない", "完璧主義で相手を追い詰めてしまう", "感情表現が下手"],
        "my_approaches": ["相手の知的好奇心を刺激する", "効率的なデートプランを提案する", "プライバシーを尊重する"],
        "partner_approaches": ["相手の戦略的思考を活かした会話をする", "相手の効率性を重視したデートプランを立てる", "相手のプライバシーを尊重した距離感を保つ", "相手の論理的思考に合わせた説明をする", "相手の長期的ビジョンに共感を示す"],
        "ng_behaviors": ["感情的な要求を押し付ける", "無計画な行動を強要する", "表面的な会話ばかりする"],
        "partner_ng_behaviors": ["相手のプライバシーを侵害する", "相手の効率性を無視した無計画な提案をする", "相手の論理的思考を否定する", "相手の完璧主義を批判する", "相手の独立心を束縛しようとする"],
        "partner_ng_behaviors": ["相手のプライバシーを侵害する", "相手の効率性を無視した無計画な提案をする", "相手の論理的思考を否定する", "相手の完璧主義を批判する", "相手の独立心を束縛しようとする"]
    },
    "INTP": {
        "traits": ["分析的思考", "好奇心旺盛", "独創的", "柔軟性", "論理的", "創造的", "独立心"],
        "love_style": "知的パートナーシップを重視し、自由な関係を求める。深い会話と新しい発見を共有し、お互いの好奇心を刺激し合う関係を大切にする。感情表現は控えめだが、相手の個性を尊重し、束縛のない自由な愛情を表現する。創造的で革新的なアプローチを好み、伝統的な恋愛観にとらわれない。",
        "strengths": ["創造性", "オープンマインド", "深い洞察力", "問題解決力", "独創的アイデア"],
        "challenges": ["日常的な感情表現", "決断力", "相手の感情への配慮", "実践性"],
        "advice_style": "分析的で多角的な視点からのアドバイス",
        "likes_in_partner": ["知的刺激", "創造性", "独立心", "オープンマインド", "深い会話", "自由な関係"],
        "dislikes_in_partner": ["感情的な要求", "伝統的な価値観", "表面的な関係", "束縛", "非論理的"],
        "common_concerns": ["感情表現が苦手", "決断ができない", "相手の感情を理解しにくい"],
        "my_approaches": ["知的会話で興味を引く", "創造的なデートを提案する", "自由な関係を尊重する"],
        "partner_approaches": ["相手の創造性を刺激する話題を提供する", "相手の好奇心を満たす新しい体験を提案する", "相手の独立心を尊重した自由な関係を築く", "相手の論理的思考に合わせた深い会話をする", "相手の柔軟性を活かしたアプローチを心がける"],
        "ng_behaviors": ["感情的な要求を押し付ける", "伝統的な価値観を強要する", "束縛する"],
        "partner_ng_behaviors": ["相手の独立心を束縛しようとする", "相手の論理的思考を否定する", "相手の創造性を制限する", "相手の好奇心を無視する", "相手の自由を制限しようとする"]
    },
    "ENTJ": {
        "traits": ["リーダーシップ", "決断力", "効率性重視", "自信家", "戦略的", "実行力", "目標志向"],
        "love_style": "パートナーシップを戦略的に構築し、目標を共有する。明確なビジョンを持ち、お互いの成長と成功をサポートし合う関係を重視する。リーダーシップを発揮し、効率的で実践的な愛情表現を心がける。相手の可能性を信じ、具体的なステップで関係を発展させる。",
        "strengths": ["明確なビジョン", "実行力", "相手の成長をサポート", "リーダーシップ", "効率性"],
        "challenges": ["感情的な柔軟性", "相手のペースを尊重", "支配的になりがち", "共感力"],
        "advice_style": "具体的で実行可能なステップを重視",
        "likes_in_partner": ["目標志向", "効率性", "成長意欲", "独立心", "知的刺激", "実行力"],
        "dislikes_in_partner": ["無計画", "依存心", "非効率", "感情的な要求", "消極的"],
        "common_concerns": ["相手が自分のペースについていけない", "支配的になってしまう", "感情表現が苦手"],
        "my_approaches": ["相手の成長をサポートする", "効率的なデートプランを提案する", "目標を共有する"],
        "partner_approaches": ["相手のリーダーシップを尊重し、サポートする", "相手の効率性に合わせた行動を心がける", "相手の目標に共感し、協力する", "相手の決断力に信頼を置く", "相手の実行力に合わせたペースで進める"],
        "ng_behaviors": ["感情的な要求を押し付ける", "無計画な行動を強要する", "依存心を強める"],
        "partner_ng_behaviors": ["相手のリーダーシップを否定する", "相手の効率性を無視した無計画な行動をする", "相手の目標を軽視する", "相手の決断力に疑問を投げかける", "相手の実行力を阻害する"]
    },
    "ENTP": {
        "traits": ["創造的思考", "適応力", "議論好き", "冒険心", "機知", "柔軟性", "社交性"],
        "love_style": "刺激的で変化に富んだ関係を求める。新しい体験と冒険を共有し、お互いを楽しませ合う関係を大切にする。ユーモアと創造性を武器に、常に新鮮で魅力的な愛情表現を心がける。柔軟性があり、相手の興味や関心に合わせてアプローチを変化させる。",
        "strengths": ["ユーモア", "新しいアイデア", "相手を楽しませる", "適応力", "創造性"],
        "challenges": ["一貫性", "深い感情の共有", "長期的なコミットメント", "感情の安定"],
        "advice_style": "創造的で革新的なアプローチ",
        "likes_in_partner": ["刺激", "創造性", "冒険心", "知的刺激", "柔軟性", "ユーモア"],
        "dislikes_in_partner": ["退屈", "伝統的な価値観", "束縛", "感情的な要求", "非論理的"],
        "common_concerns": ["一貫性がない", "感情が安定しない", "長期的な関係が苦手"],
        "my_approaches": ["刺激的なデートを提案する", "創造的なアプローチをする", "柔軟な関係を築く"],
        "partner_approaches": ["相手の創造性を刺激する新しい体験を提案する", "相手のユーモアに合わせた楽しい会話をする", "相手の柔軟性を活かした変化に富んだ関係を築く", "相手の冒険心を満たす刺激的なデートを計画する", "相手の適応力に合わせたアプローチを心がける"],
        "ng_behaviors": ["束縛する", "伝統的な価値観を強要する", "感情的な要求を押し付ける"],
        "partner_ng_behaviors": ["相手の創造性を制限する", "相手の柔軟性を無視した固定観念を押し付ける", "相手の冒険心を阻害する", "相手のユーモアを否定する", "相手の変化を嫌う"]
    },
    "INFJ": {
        "traits": ["共感力", "理想主義", "洞察力", "創造性", "献身性", "深い愛情", "神秘性"],
        "love_style": "深い精神的な絆と真の理解を求める。相手の心の奥底まで理解し、無条件の愛情で包み込む関係を重視する。理想主義的で、完璧な愛を追求するが、現実とのバランスも大切にする。相手の成長をサポートし、深い洞察力で関係を導く。",
        "strengths": ["相手の感情を理解", "深い愛情", "成長をサポート", "洞察力", "創造性"],
        "challenges": ["完璧主義", "相手の期待に応えすぎる", "感情的な疲労", "現実との折り合い"],
        "advice_style": "感情的で共感的なアドバイス",
        "likes_in_partner": ["深い理解", "成長意欲", "理想主義", "創造性", "誠実さ", "精神的な絆"],
        "dislikes_in_partner": ["表面的な関係", "非誠実", "成長意欲のない", "現実的すぎる", "感情的な理解不足"],
        "common_concerns": ["相手が自分の深い感情を理解してくれない", "完璧主義で相手を追い詰めてしまう", "感情的な疲労"],
        "my_approaches": ["深い会話で心を通わせる", "相手の成長をサポートする", "精神的な絆を築く"],
        "partner_approaches": ["相手の深い感情を理解し、共感を示す", "相手の理想主義に寄り添い、サポートする", "相手の洞察力に敬意を払い、深い会話を心がける", "相手の創造性を刺激し、精神的な絆を深める", "相手の献身性に感謝し、同じ愛情で返す"],
        "ng_behaviors": ["表面的な関係を求める", "非誠実な態度", "感情的な理解を怠る"],
        "partner_ng_behaviors": ["相手の深い感情を軽視する", "相手の理想主義を否定する", "相手の洞察力を無視する", "相手の献身性を当たり前と思う", "相手の精神的な絆を軽視する"]
    },
    "INFP": {
        "traits": ["理想主義", "共感力", "創造性", "価値観重視", "深い感情", "柔軟性", "芸術性"],
        "love_style": "真実の愛と深い感情的な絆を求める。相手の個性を無条件で受け入れ、詩的で美しい愛情表現を心がける。価値観の一致を重視し、お互いの心の奥底まで理解し合う関係を大切にする。創造的で芸術的なアプローチを好み、感情の起伏が豊か。",
        "strengths": ["無条件の愛情", "創造性", "相手の個性を尊重", "共感力", "深い感情"],
        "challenges": ["現実との折り合い", "感情の起伏", "相手の期待に応えすぎる", "決断力"],
        "advice_style": "詩的で感情的なアドバイス",
        "likes_in_partner": ["深い感情", "創造性", "個性", "理想主義", "共感力", "価値観の一致"],
        "dislikes_in_partner": ["表面的な関係", "非創造的", "個性を尊重しない", "現実的すぎる", "感情的な理解不足"],
        "common_concerns": ["現実との折り合いがつかない", "感情の起伏が激しい", "相手の期待に応えすぎる"],
        "my_approaches": ["深い感情を共有する", "創造的なアプローチをする", "個性を尊重する"],
        "partner_approaches": ["相手の深い感情を受け入れ、共感を示す", "相手の創造性を刺激し、芸術的な体験を共有する", "相手の個性を無条件で受け入れる", "相手の理想主義に理解を示し、サポートする", "相手の価値観を尊重し、一致点を見つける"],
        "ng_behaviors": ["表面的な関係を求める", "個性を尊重しない", "感情的な理解を怠る"],
        "partner_ng_behaviors": ["相手の深い感情を軽視する", "相手の創造性を否定する", "相手の個性を否定する", "相手の理想主義を現実的すぎる態度で否定する", "相手の価値観を軽視する"]
    },
    "ENFJ": {
        "traits": ["共感力", "リーダーシップ", "協調性", "献身性", "社交性", "成長サポート", "調和重視"],
        "love_style": "パートナーの成長と幸福を最優先にする。相手をサポートし、調和のある関係を築くことに情熱を注ぐ。コミュニケーションを重視し、お互いの気持ちを理解し合う関係を大切にする。相手の可能性を信じ、具体的なサポートで関係を発展させる。",
        "strengths": ["相手をサポート", "コミュニケーション", "調和を創る", "成長サポート", "共感力"],
        "challenges": ["自分の感情を後回し", "相手に依存されすぎる", "完璧主義", "感情的な疲労"],
        "advice_style": "サポート的で励ましのアドバイス",
        "likes_in_partner": ["成長意欲", "協調性", "感謝の気持ち", "コミュニケーション", "調和", "誠実さ"],
        "dislikes_in_partner": ["成長意欲のない", "非協調的", "感謝の気持ちがない", "コミュニケーション不足", "非誠実"],
        "common_concerns": ["相手に依存されすぎる", "自分の感情を後回しにしてしまう", "完璧主義"],
        "my_approaches": ["相手の成長をサポートする", "調和を創る", "コミュニケーションを大切にする"],
        "partner_approaches": ["相手のサポートに感謝し、同じ愛情で返す", "相手の調和を重視した行動を心がける", "相手のコミュニケーションに積極的に応える", "相手の成長サポートに協力し、自分も成長する", "相手の共感力に信頼を置き、心を開く"],
        "ng_behaviors": ["成長意欲を阻害する", "非協調的な態度", "感謝の気持ちを表さない"],
        "partner_ng_behaviors": ["相手のサポートを当たり前と思う", "相手の調和を乱す", "相手のコミュニケーションを無視する", "相手の成長サポートを軽視する", "相手の共感力を否定する"]
    },
    "ENFP": {
        "traits": ["熱意", "創造性", "適応力", "共感力", "社交性", "冒険心", "楽観性"],
        "love_style": "情熱的で冒険的な愛を求める。新しい体験と感情を共有し、お互いを楽しませ合う関係を大切にする。楽観的で前向きなアプローチを心がけ、相手の可能性を信じてサポートする。感情表現が豊かで、創造的で魅力的な愛情表現を得意とする。",
        "strengths": ["相手を楽しませる", "深い愛情", "新しい体験", "創造性", "共感力"],
        "challenges": ["一貫性", "感情の起伏", "長期的な計画", "現実との折り合い"],
        "advice_style": "情熱的で創造的なアドバイス",
        "likes_in_partner": ["冒険心", "創造性", "楽観性", "新しい体験", "深い感情", "柔軟性"],
        "dislikes_in_partner": ["退屈", "非創造的", "悲観的", "変化を嫌う", "感情的な理解不足"],
        "common_concerns": ["一貫性がない", "感情の起伏が激しい", "長期的な計画が苦手"],
        "my_approaches": ["冒険的なデートを提案する", "創造的なアプローチをする", "新しい体験を共有する"],
        "partner_approaches": ["相手の楽観性に合わせた前向きな態度を心がける", "相手の創造性を刺激し、新しい体験を提案する", "相手の冒険心を満たす刺激的なデートを計画する", "相手の感情表現に共感し、同じ熱意で返す", "相手の柔軟性を活かしたアプローチを心がける"],
        "ng_behaviors": ["退屈な関係を求める", "変化を嫌う", "感情的な理解を怠る"],
        "partner_ng_behaviors": ["相手の楽観性を否定する", "相手の創造性を制限する", "相手の冒険心を阻害する", "相手の感情表現を軽視する", "相手の柔軟性を無視する"]
    },
    "ISTJ": {
        "traits": ["責任感", "実用性", "信頼性", "伝統重視", "組織力", "一貫性", "効率性"],
        "love_style": "安定した信頼できる関係を求める。伝統的な価値観を重視し、実用的で具体的な愛情表現を心がける。責任感が強く、相手を守り支える関係を大切にする。一貫性があり、長期的で安定した関係を構築することを重視する。",
        "strengths": ["信頼性", "実用的なサポート", "一貫性", "責任感", "効率性"],
        "challenges": ["感情表現", "柔軟性", "自発性", "変化への対応"],
        "advice_style": "実用的で具体的なアドバイス",
        "likes_in_partner": ["信頼性", "責任感", "効率性", "伝統的な価値観", "一貫性", "実用性"],
        "dislikes_in_partner": ["非効率", "非責任的", "変化を好む", "感情的な要求", "非実用的"],
        "common_concerns": ["感情表現が苦手", "柔軟性がない", "自発性がない"],
        "my_approaches": ["信頼できる関係を築く", "実用的なサポートをする", "一貫性を保つ"],
        "partner_approaches": ["相手の信頼性に信頼を置き、安心感を与える", "相手の責任感に敬意を払い、協力する", "相手の効率性に合わせた行動を心がける", "相手の伝統的な価値観を尊重する", "相手の一貫性に合わせた安定した関係を築く"],
        "ng_behaviors": ["非効率な行動を強要する", "非責任的な態度", "変化を強要する"],
        "partner_ng_behaviors": ["相手の信頼性を疑う", "相手の責任感を軽視する", "相手の効率性を無視する", "相手の伝統的な価値観を否定する", "相手の一貫性を乱す"]
    },
    "ISFJ": {
        "traits": ["献身性", "実用性", "共感力", "責任感", "思いやり", "調和重視", "伝統重視"],
        "love_style": "相手を大切にし、安定した関係を築く。思いやりと献身性を武器に、相手の幸せを最優先にする関係を重視する。実用的なサポートを提供し、調和のある関係を心がける。伝統的な価値観を大切にし、長期的で安定した愛情を表現する。",
        "strengths": ["思いやり", "実用的なサポート", "信頼性", "献身性", "調和"],
        "challenges": ["自分の感情を表現", "相手に依存されすぎる", "変化への対応", "自己主張"],
        "advice_style": "思いやりのある実践的なアドバイス",
        "likes_in_partner": ["感謝の気持ち", "調和", "伝統的な価値観", "思いやり", "責任感", "安定"],
        "dislikes_in_partner": ["感謝の気持ちがない", "非調和的", "伝統を軽視する", "非思いやり", "非責任的"],
        "common_concerns": ["相手に依存されすぎる", "自分の感情を表現できない", "自己主張ができない"],
        "my_approaches": ["思いやりのあるサポートをする", "調和を創る", "感謝の気持ちを表す"],
        "partner_approaches": ["相手の思いやりに感謝し、同じ愛情で返す", "相手の調和を重視した行動を心がける", "相手の献身性に敬意を払い、協力する", "相手の伝統的な価値観を尊重する", "相手の実用的なサポートに感謝し、協力する"],
        "ng_behaviors": ["感謝の気持ちを表さない", "非調和的な態度", "伝統を軽視する"],
        "partner_ng_behaviors": ["相手の思いやりを当たり前と思う", "相手の調和を乱す", "相手の献身性を軽視する", "相手の伝統的な価値観を否定する", "相手の実用的なサポートを軽視する"]
    },
    "ESTJ": {
        "traits": ["組織力", "決断力", "実用性", "責任感", "効率性", "リーダーシップ", "伝統重視"],
        "love_style": "明確な役割分担と効率的な関係を求める。リーダーシップを発揮し、実用的で構造化された愛情表現を心がける。伝統的な価値観を重視し、責任感と効率性で関係を発展させる。相手の成長をサポートし、明確な目標を持った関係を構築する。",
        "strengths": ["リーダーシップ", "実用的なサポート", "信頼性", "効率性", "組織力"],
        "challenges": ["感情的な柔軟性", "相手の感情を理解", "支配的になりがち", "共感力"],
        "advice_style": "構造化された実践的なアドバイス",
        "likes_in_partner": ["効率性", "責任感", "組織力", "伝統的な価値観", "明確な役割分担", "実用性"],
        "dislikes_in_partner": ["非効率", "非責任的", "無計画", "感情的な要求", "非実用的"],
        "common_concerns": ["相手が自分のペースについていけない", "支配的になってしまう", "感情表現が苦手"],
        "my_approaches": ["効率的な関係を築く", "明確な役割分担をする", "実用的なサポートをする"],
        "partner_approaches": ["相手のリーダーシップを尊重し、サポートする", "相手の効率性に合わせた行動を心がける", "相手の組織力に協力し、役割分担を明確にする", "相手の責任感に信頼を置き、協力する", "相手の伝統的な価値観を尊重する"],
        "ng_behaviors": ["非効率な行動を強要する", "非責任的な態度", "無計画な行動を強要する"],
        "partner_ng_behaviors": ["相手のリーダーシップを否定する", "相手の効率性を無視する", "相手の組織力を軽視する", "相手の責任感を疑う", "相手の伝統的な価値観を否定する"]
    },
    "ESFJ": {
        "traits": ["協調性", "共感力", "責任感", "社交性", "思いやり", "調和重視", "献身性"],
        "love_style": "調和のある関係と相手の幸福を重視する。社交的で思いやりがあり、お互いを支え合う関係を大切にする。コミュニケーションを重視し、調和のある関係を心がける。相手の気持ちを理解し、具体的なサポートで関係を発展させる。",
        "strengths": ["思いやり", "コミュニケーション", "調和を創る", "献身性", "社交性"],
        "challenges": ["相手の期待に応えすぎる", "自分の感情を後回し", "変化への対応", "感情的な疲労"],
        "advice_style": "調和的でサポート的なアドバイス",
        "likes_in_partner": ["感謝の気持ち", "調和", "社交性", "思いやり", "協調性", "責任感"],
        "dislikes_in_partner": ["感謝の気持ちがない", "非調和的", "非社交的", "非思いやり", "非協調的"],
        "common_concerns": ["相手の期待に応えすぎる", "自分の感情を後回しにしてしまう", "感情的な疲労"],
        "my_approaches": ["調和を創る", "感謝の気持ちを表す", "コミュニケーションを大切にする"],
        "partner_approaches": ["相手の思いやりに感謝し、同じ愛情で返す", "相手の調和を重視した行動を心がける", "相手の社交性に合わせた楽しい時間を過ごす", "相手の協調性に協力し、調和を保つ", "相手の献身性に敬意を払い、協力する"],
        "ng_behaviors": ["感謝の気持ちを表さない", "非調和的な態度", "非社交的な態度"],
        "partner_ng_behaviors": ["相手の思いやりを当たり前と思う", "相手の調和を乱す", "相手の社交性を無視する", "相手の協調性を軽視する", "相手の献身性を軽視する"]
    },
    "ISTP": {
        "traits": ["実用性", "適応力", "独立心", "論理的", "問題解決力", "柔軟性", "冷静さ"],
        "love_style": "自由な関係と実用的なサポートを求める。独立心が強く、相手の個性を尊重する関係を重視する。実用的で柔軟なアプローチを心がけ、問題解決能力を活かした愛情表現をする。束縛を嫌い、お互いの自由を尊重し合う関係を大切にする。",
        "strengths": ["実用的なサポート", "柔軟性", "問題解決", "独立心", "適応力"],
        "challenges": ["感情表現", "長期的なコミットメント", "相手の感情を理解", "感情的な深さ"],
        "advice_style": "実用的で柔軟なアドバイス",
        "likes_in_partner": ["独立心", "実用性", "柔軟性", "問題解決力", "自由", "冷静さ"],
        "dislikes_in_partner": ["依存心", "非実用的", "非柔軟", "感情的な要求", "束縛"],
        "common_concerns": ["感情表現が苦手", "長期的な関係が苦手", "相手の感情を理解しにくい"],
        "my_approaches": ["実用的なサポートをする", "自由な関係を尊重する", "問題解決をサポートする"],
        "partner_approaches": ["相手の独立心を尊重し、自由な関係を築く", "相手の実用性に合わせた行動を心がける", "相手の柔軟性を活かしたアプローチを心がける", "相手の問題解決能力に信頼を置き、協力する", "相手の冷静さに敬意を払い、感情的な要求を控える"],
        "ng_behaviors": ["依存心を強める", "非実用的な要求をする", "束縛する"],
        "partner_ng_behaviors": ["相手の独立心を束縛しようとする", "相手の実用性を無視する", "相手の柔軟性を制限する", "相手の問題解決能力を否定する", "相手の冷静さを無視した感情的な要求をする"]
    },
    "ISFP": {
        "traits": ["芸術性", "共感力", "実用性", "柔軟性", "美意識", "独立心", "創造性"],
        "love_style": "美しい関係と深い感情的な絆を求める。芸術的で美意識が高く、創造的な愛情表現を心がける。相手の個性を尊重し、深い感情を共有する関係を大切にする。実用的なサポートも提供し、美しく調和の取れた関係を構築する。",
        "strengths": ["創造性", "思いやり", "実用的なサポート", "美意識", "柔軟性"],
        "challenges": ["感情の起伏", "長期的な計画", "相手の期待に応えすぎる", "自己主張"],
        "advice_style": "芸術的で感情的なアドバイス",
        "likes_in_partner": ["美意識", "創造性", "独立心", "思いやり", "柔軟性", "深い感情"],
        "dislikes_in_partner": ["非美的", "非創造的", "依存心", "非思いやり", "非柔軟"],
        "common_concerns": ["感情の起伏が激しい", "長期的な計画が苦手", "自己主張ができない"],
        "my_approaches": ["美しい関係を築く", "創造的なアプローチをする", "深い感情を共有する"],
        "partner_approaches": ["相手の美意識を理解し、美しい体験を共有する", "相手の創造性を刺激し、芸術的な体験を提案する", "相手の独立心を尊重し、自由な関係を築く", "相手の思いやりに感謝し、同じ愛情で返す", "相手の柔軟性を活かしたアプローチを心がける"],
        "ng_behaviors": ["非美的な態度", "非創造的な要求", "依存心を強める"],
        "partner_ng_behaviors": ["相手の美意識を否定する", "相手の創造性を制限する", "相手の独立心を束縛しようとする", "相手の思いやりを軽視する", "相手の柔軟性を無視する"]
    },
    "ESTP": {
        "traits": ["実用性", "適応力", "冒険心", "社交性", "問題解決力", "楽観性", "行動力"],
        "love_style": "刺激的で実用的な関係を求める。冒険心と行動力があり、新しい体験を共有する関係を大切にする。実用的で柔軟なアプローチを心がけ、相手を楽しませる愛情表現を得意とする。社交的で楽観的、問題解決能力を活かした関係を構築する。",
        "strengths": ["問題解決", "相手を楽しませる", "実用的なサポート", "冒険心", "適応力"],
        "challenges": ["長期的な計画", "感情的な深さ", "一貫性", "感情の安定"],
        "advice_style": "実用的で刺激的なアドバイス",
        "likes_in_partner": ["冒険心", "実用性", "楽観性", "社交性", "行動力", "柔軟性"],
        "dislikes_in_partner": ["退屈", "非実用的", "悲観的", "非社交的", "消極的"],
        "common_concerns": ["長期的な計画が苦手", "感情的な深さがない", "一貫性がない"],
        "my_approaches": ["刺激的なデートを提案する", "実用的なサポートをする", "冒険的な体験を共有する"],
        "partner_approaches": ["相手の冒険心を満たす刺激的な体験を提案する", "相手の実用性に合わせた行動を心がける", "相手の楽観性に合わせた前向きな態度を心がける", "相手の社交性に合わせた楽しい時間を過ごす", "相手の行動力に合わせた積極的なアプローチを心がける"],
        "ng_behaviors": ["退屈な関係を求める", "非実用的な要求をする", "消極的な態度"],
        "partner_ng_behaviors": ["相手の冒険心を阻害する", "相手の実用性を無視する", "相手の楽観性を否定する", "相手の社交性を制限する", "相手の行動力を阻害する"]
    },
    "ESFP": {
        "traits": ["熱意", "社交性", "実用性", "適応力", "楽観性", "思いやり", "行動力"],
        "love_style": "楽しく刺激的な関係を求める。社交的で楽観的、お互いを楽しませ合う関係を大切にする。実用的なサポートも提供し、思いやりと行動力で関係を発展させる。感情表現が豊かで、相手を幸せにする愛情表現を心がける。",
        "strengths": ["相手を楽しませる", "コミュニケーション", "実用的なサポート", "楽観性", "社交性"],
        "challenges": ["長期的な計画", "感情の起伏", "一貫性", "感情の安定"],
        "advice_style": "楽しく実践的なアドバイス",
        "likes_in_partner": ["楽観性", "社交性", "実用性", "思いやり", "行動力", "柔軟性"],
        "dislikes_in_partner": ["悲観的", "非社交的", "非実用的", "非思いやり", "消極的"],
        "common_concerns": ["長期的な計画が苦手", "感情の起伏が激しい", "一貫性がない"],
        "my_approaches": ["楽しいデートを提案する", "実用的なサポートをする", "社交的な体験を共有する"],
        "partner_approaches": ["相手の楽観性に合わせた前向きな態度を心がける", "相手の社交性に合わせた楽しい時間を過ごす", "相手の実用性に合わせた行動を心がける", "相手の思いやりに感謝し、同じ愛情で返す", "相手の行動力に合わせた積極的なアプローチを心がける"],
        "ng_behaviors": ["悲観的な態度", "非社交的な態度", "消極的な態度"],
        "partner_ng_behaviors": ["相手の楽観性を否定する", "相手の社交性を制限する", "相手の実用性を無視する", "相手の思いやりを軽視する", "相手の行動力を阻害する"]
    }
}



# MBTI診断用の質問とマッピング（グローバル定義）
questions = [
    "好きな人とは毎日LINEしたい？🥺",                # E
    "初対面でも気になった人には自分から話しかける？",             # E
    "異性との沈黙は気にならない？💑",           # I
    "一人の時間がないと疲れてしまう？🌙",             # I
    "恋人の小さな変化にすぐ気づく？😊",             # S
    "過去の出来事や細かい記憶をよく覚えてる方？",           # S
    "恋愛はフィーリングが大事？💡",           # N
    "恋人の気持ちをすぐ察する自信がある？🔮",       # N
    "恋人と未来を考える時、まず『現実的な条件』が気になる🧠",             # T
    "恋人の相談には共感よりもアドバイスを優先しがち？📱",     # T
    "恋愛中、相手が理不尽でも『でも好きだから…』って思っちゃうことがある？💓",             # F
    "好きな人のためなら、自分が少し無理しても構わない",           # F
    "デートは計画を立ててから動きたい？📆",         # J
    "先のことが見えない関係はちょっと苦手",           # J
    "恋愛でも『ノリ』と『勢い』って、結構大事だと思う",   # P
    "『いつ告白してくれるの？』って言われたらプレッシャーに感じる🌈"           # P
]

mapping = [
    ("E", "I"), ("E", "I"), ("I", "E"), ("I", "E"),
    ("S", "N"), ("S", "N"), ("N", "S"), ("N", "S"),
    ("T", "F"), ("T", "F"), ("F", "T"), ("F", "T"),
    ("J", "P"), ("J", "P"), ("P", "J"), ("P", "J")
]

# 💾 SQLite初期化
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            mbti TEXT,
            gender TEXT,
            target_mbti TEXT,
            is_paid INTEGER DEFAULT 0,
            mode TEXT,
            mbti_answers TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            user_id TEXT,
            role TEXT,
            content TEXT
        )
    ''')
    # Stripe顧客テーブルを追加
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stripe_customers (
            user_id TEXT PRIMARY KEY,
            customer_id TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("SQLiteデータベースを初期化しました。")

init_db()

# ユーザープロファイルの取得
def get_user_profile(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT mbti, gender, target_mbti, is_paid, mode, mbti_answers FROM users WHERE user_id=?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    print(f"get_user_profile: user_id={user_id}, row={row}")
    return {
        "mbti": row[0] if row else "不明",
        "gender": row[1] if row else "不明",
        "target_mbti": row[2] if row else "不明",
        "is_paid": bool(row[3]) if row else False,
        "mode": row[4] if row else "",
        "mbti_answers": row[5] if row else ""
    }

# MBTI集計ロジック
def calc_mbti(answers):
    score = {'E': 0, 'I': 0, 'S': 0, 'N': 0, 'T': 0, 'F': 0, 'J': 0, 'P': 0}
    n = min(len(questions), len(answers), len(mapping))
    for i in range(n):
        ans = answers[i]
        type1, type2 = mapping[i]
        if ans == 1:
            score[type1] += 1
        else:
            score[type2] += 1
    mbti = (
        ('E' if score['E'] >= score['I'] else 'I') +
        ('S' if score['S'] >= score['N'] else 'N') +
        ('T' if score['T'] >= score['F'] else 'F') +
        ('J' if score['J'] >= score['P'] else 'P')
    )
    return mbti

# MBTI診断開始関数
def start_mbti_diagnosis(user_id):
    print(f"Starting MBTI diagnosis for user_id: {user_id}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # ユーザーがいなければINSERT
    cursor.execute(
        "INSERT OR IGNORE INTO users (user_id, mode, mbti_answers) VALUES (?, 'mbti_diagnosis', '[]')",
        (user_id,)
    )
    # 必ずmodeとmbti_answersをセット
    cursor.execute(
        "UPDATE users SET mode='mbti_diagnosis', mbti_answers='[]' WHERE user_id=?",
        (user_id,)
    )
    conn.commit()
    # ここで確認
    cursor.execute("SELECT mode FROM users WHERE user_id=?", (user_id,))
    row = cursor.fetchone()
    print(f"確認: 設定後のmode = {row[0] if row else 'None'}")
    conn.close()
    print(f"MBTI diagnosis mode set for user_id: {user_id}")
    first_question = send_mbti_question(user_id, 0)
    print(f"First question generated: {first_question}")
    return first_question

# MBTI質問送信関数（ボタン式）
def send_mbti_question(user_id, question_index):
    """MBTI診断の質問を送信（ボタン式）"""
    if question_index >= len(questions):
        return "診断が完了しました！"
    
    # ボタンテンプレートを作成（messageアクションでユーザーの吹き出しに）
    template = {
        "type": "template",
        "altText": f"質問{question_index + 1}/16: {questions[question_index]}",
        "template": {
            "type": "buttons",
            "title": f"質問{question_index + 1}/16",
            "text": questions[question_index],
            "actions": [
                {
                    "type": "message",
                    "label": "はい",
                    "text": "はい"
                },
                {
                    "type": "message",
                    "label": "いいえ",
                    "text": "いいえ"
                }
            ]
        }
    }
    
    return template

# MBTI回答処理関数
def process_mbti_answer(user_id, answer, user_profile):
    try:
        print(f"process_mbti_answer: user_id={user_id}, answer={answer}")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT mbti_answers FROM users WHERE user_id=?", (user_id,))
        row = cursor.fetchone()
        if row and row[0]:
            answers = json.loads(row[0])
        else:
            answers = []
        answers.append(1 if answer == "はい" else 0)
        print(f"=== MBTI回答ログ ===")
        print(f"ユーザーID: {user_id}")
        print(f"現在の回答数: {len(answers)}/16")
        print(f"最新の回答: {answer} (数値: {1 if answer == 'はい' else 0})")
        print(f"全回答履歴: {answers}")
        print(f"==================")
        cursor.execute("UPDATE users SET mbti_answers=? WHERE user_id=?", (json.dumps(answers), user_id))
        conn.commit()
        conn.close()
        next_question_index = len(answers)
        print(f"next_question_index: {next_question_index}")
        if next_question_index < 16:
            print(f"次の質問を送信: 質問{next_question_index + 1}/16")
            return send_mbti_question(user_id, next_question_index)
        else:
            print(f"診断完了！全回答: {answers}")
            result_message = complete_mbti_diagnosis(user_id, answers)
            payment_message = get_payment_message(user_id)
            # 診断完了メッセージ送信後にmodeをリセット
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET mode='' WHERE user_id=?", (user_id,))
            conn.commit()
            conn.close()
            return [
                {"type": "text", "text": result_message},
                {"type": "text", "text": payment_message}
            ]

    except Exception as e:
        print(f"MBTI回答処理エラー: {e}")
        return "エラーが発生しました。もう一度診断を開始してください。"

# MBTI診断完了関数
def complete_mbti_diagnosis(user_id, answers):
    """MBTI診断を完了し、結果を送信"""
    try:
        # MBTI計算
        mbti = calc_mbti(answers)

        # 結果を保存（modeは維持して、診断完了メッセージを送信後にリセット）
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET mbti=? WHERE user_id=?", (mbti, user_id))
        conn.commit()
        conn.close()
        
        # 診断結果メッセージのみ（課金誘導なし）
        result_message = f"🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️{MBTI_NICKNAME.get(mbti, mbti)}❤️\n\n{get_mbti_description(mbti)}"
        
        # GASへの詳細アドバイス送信はここでは呼ばない（決済完了時のみ）
        # send_detailed_advice_to_gas(user_id, mbti)
        
        return result_message
        
    except Exception as e:
        print(f"MBTI診断完了エラー: {e}")
        return "診断結果の処理中にエラーが発生しました。"

def get_mbti_description(mbti):
    """MBTIタイプの説明を取得"""
    descriptions = {
        "INTJ": "一見クールで無関心そうなのに、実は「本命だけには一途」なあなた。\n感情よりも論理で動く慎重派で、恋も戦略的に進めがち。\nでも、心を許した相手には不器用ながらもちゃんと弱さを見せられる、ギャップが魅力💭\n「この人にだけは見せる顔」があるあなたに、落ちる人は多いはず。\n\n───\n🌙夜の顔は…【知的ドS教官🧠】\n普段は理性的なのに、夜は完全主導型。\n相手の反応を観察して流れをコントロールする、【静かな支配】タイプ。\n感情よりも満足度を重視するロジカルな夜、だけど…\n本気で心許した相手には、独占欲がちらっと出ることも🔥",
        "INTP": "感情よりも思考が先に立っちゃうあなた。\n「なんでそうなるの？」って考えすぎて、\n素直な一言がなかなか出てこないことも多いよね💭\n\nでも、興味を持った相手にはめちゃくちゃ深掘りするタイプで、\n一度ハマると他が見えなくなる「愛のオタク気質」も持ってる💘\nその分、自分でも気づかないうちに距離を取っちゃって、\n「冷たい…？」って誤解されることもあるかも😢\n\nでも大丈夫。あなたの魅力は、\n『知的さ×ピュアさ』という最強コンボだから🌟\n素直な一言で世界が変わる恋、あるかもしれないね。\n\n🌙夜の顔は…【知識プレイ職人📚】\n静かに、でも的確に。\n思考型ならではの『知っててやる』スイッチが入ると、\nテンポもタッチも計算され尽くしてる。\n無言なのにドキッとする、知的なゾクゾク感を演出するタイプ。",
        "ENTJ": "頼れるしっかり者で、自分の意思がはっきりしてるあなた。\n恋愛でも「こうしたい」「こうあるべき」って理想が明確で、ついつい主導権を握っちゃうことが多いよね💼\n\nでも本当は、恋にはちょっぴり不器用。\n甘えたいのにうまく出せなかったり、「好かれてから動きたい」って慎重になりがち。\n\n惹かれた相手には、誠実で計画的なアプローチで距離を縮めるタイプ。\n感情を素直に見せられた瞬間、一気に関係が進展するはず💘\n\n🌙夜の顔は…【支配のカリスマ指揮官🎩】\nベッドでも主導権を握りたい派。\n自分で雰囲気を組み立て、じっくり攻めてくる『理性と支配』のハイブリッド。\nでも実は、相手の気持ちや様子にもめちゃくちゃ敏感。\n一度任せたら、全部委ねたくなる人。",
        "ENTP": "明るくてノリがよく、恋も勢い重視なあなた。\n好きになったら全力ダッシュで距離を詰めるけど、ちょっとでも冷めたら急ブレーキ…そんなアップダウンが魅力でもあるんだよね🎢\n自由で楽しい恋愛が好きだから、束縛やルールはちょっと苦手。\nでも、心から惹かれた相手にはちゃんと本気で向き合うよ💘\n一緒にいて飽きない、刺激的な存在になれるかが恋の鍵！\n\n🌙夜の顔は…【カオスな快楽実験者🧪】\n「これもアリ？」ってテンションで、毎回違うムードを演出。\nルール無用の自由プレイ派で、相手の反応を楽しみながら変化球を投げてくる。\n刺激と笑いに満ちた『予測不能な夜』を求めるタイプ。",
        "INFJ": "見た目はおだやかでも、心の中は感情でぎゅうぎゅうなあなた。\n一度「この人だ」と思ったら、誰よりも深く、まっすぐ愛し抜くタイプ。\nその愛情は尊くて、美しいけど…ちょっと重ためなのもご愛敬📜\n\n駆け引きよりも共鳴を求めて、相手の気持ちを読みすぎて疲れちゃうこともあるよね。\nでも大丈夫。そんな繊細さこそが、あなたの魅力✨\n\n「この人なら分かってくれる」って心から思える相手に出会えたら、\nあなたの愛は最強の癒しになる。\n\n🌙夜の顔は…【妄想系エモスキンシップ魔🫂】\n妄想と理想が混ざりあったような甘くて深い世界観で、全身で『想い』を伝えるタイプ。\nゆっくりと抱きしめて、感情とぬくもりをじわじわ注ぎ込んでくる。\n目線や呼吸、全部に意味があるような繊細なリードが特徴。\n静かなのに、記憶に残る余韻系。",
        "INFP": "人一倍感受性が強くて、頭の中ではいつも恋の妄想がぐるぐる…\nでも実際はちょっぴり人見知りで、なかなか踏み込めなかったりするよね。\n\n理想の恋を大切にするロマンチストで、「本当に大切にしてくれる人じゃないとムリ」って気持ちが強め。\n裏切りや雑な扱いには超敏感で、心を開くのに時間がかかるぶん、一度許すと超献身的。\n\n本気になった時の『溺れ方』はピカイチで、相手のために尽くしたくなる愛の重さが最大の魅力。\n\n🌙夜の顔は…【妄想スイッチ爆走モンスター🧸】\n静かに見えて頭の中は常に全開モード。\nふとした瞬間にスイッチが入ると、想像を超える大胆さを見せてくるギャップ系。\n気持ちが乗った瞬間の甘え方がえぐい。",
        "ENFJ": "人の気持ちに敏感で、つい周りを優先しちゃうあなた。\n恋愛でも「相手のために何ができるか」を考えて行動する、思いやりのプロ。\nただ、その優しさが『重い』って言われないか気にして、遠慮しすぎる一面もあるかも。\nでも本当は、愛されたい気持ちもめちゃくちゃ強いタイプ💘\nちゃんと「求めていいんだよ」って受け入れてくれる人に出会えたら、最強のパートナーになれるはず。\n\n🌙夜の顔は…【ご奉仕カスタマーサポート📞】\n優しさ100%で、相手の「気持ちよさ第一」に寄り添う奉仕型。\nどうしたら喜ぶか、何を求めてるかを察して自然に動けるから、安心感と快感のバランスが絶妙。\nエスコート力が高く、どんな要望にも『丁寧に対応』してくれるタイプ♡",
        "ENFP": "感情豊かでノリがよくて、恋に全力なあなた。\n好奇心とテンションで距離を縮めるのが得意だけど、\nちょっとでも不安を感じると一気にテンションが下がっちゃう繊細さも💭\n\n気持ちの波が激しい分、喜怒哀楽を素直に出せるのが魅力。\n相手を楽しませようと頑張るけど、\nほんとは「自分が楽しませてもらいたい」気持ちも強いタイプかも。\n\n恋が続くかどうかのカギは、テンションじゃなくて『安心感』\nそのままの自分でいられる相手を選べば、長く愛せる人になるよ🌱\n\n🌙夜の顔は…【夜型テンションクラッシャー🌙】\n盛り上がるとスイッチが入って止まらないタイプ。\nテンションの爆発力で主導権を握りつつも、\nその場のノリと感情で流れをつくる「エモ速攻型」。\n終わったあとに急に静かになるギャップも魅力。",
        "ISTJ": "真面目で誠実、計画的に恋を進めたいあなた。\n勢いやノリの恋よりも、安心感や信頼を大事にするタイプ。\n相手に振り回されるのは苦手で、自分のペースを崩さずに進めたい派。\nそのぶん、付き合ったあとの安定感は抜群！\nただ、ちょっと堅すぎたり、柔軟さに欠けると思われがちかも💭\nでも、ルールの中で見せるあなたの優しさや誠実さが、\n「ちゃんと向き合いたい」って人には最高の安心材料になるよ。\n\n🌙夜の顔は…【真面目な快楽マニュアル持参人📘】\nふざけたノリは少なめ、でもその分『確実に気持ちいいやつ』を用意してくる職人肌。\n静かに、でも丁寧に。\n頭の中には快楽のマニュアルが入っていて、一つひとつ手順を確認しながら進める感じ。\nムードより実行、でもその慎重さが逆に刺さるタイプ。",
        "ISFJ": "優しくて思いやりにあふれるあなた。\n相手の気持ちに敏感で、ちょっとした変化にもすぐ気づく『感情レーダー』タイプ。\n恋愛でも相手を最優先に考えて、つい自分のことは後回しにしがちかも。\nでもその「支えたい」「癒したい」気持ちが、相手の心をとろけさせる最大の魅力🫶\n安心感の塊みたいな存在だから、一緒にいるだけでホッとされるよ☕️\n\n🌙夜の顔は…【癒しの密着療法士👐】\n心も体も包み込む『触れる安心感』の持ち主。\nリードよりも寄り添い重視で、手を繋ぐだけでも気持ちを伝えるタイプ。\nスキンシップがゆっくり丁寧だから、気づけば深く安心してる…そんな「ぬくもり×共感」の夜に。",
        "ESTJ": "物事を理屈で整理して、白黒ハッキリつけたがるあなた。\n恋愛においても「あるべき関係像」が明確で、\n曖昧な態度や気まぐれな言動にはイライラしがち⚡️\nでも実は、とっても誠実で、責任感のある相手に弱い一面も。\n信頼できる相手には、一途に尽くす堅実派💎\nちょっとぶっきらぼうだけど、行動で愛を示すタイプ。\n感情表現は不器用でも、「守りたい」という気持ちは本物。\n\n🌙夜の顔は…【命令型快感プロデューサー🎬】\n理屈と段取りを駆使して、流れを完璧に組み立てるタイプ。\nムードは演出するもの、快楽はプロデュースするものというスタンスで、\n責めも褒めも計算済み。\nでも相手の満足を最優先に動く『戦略的優しさ』がある。\n冷静に見えて内心は熱く、期待を超える演出で魅せてくれる人。",
        "ESFJ": "相手の喜ぶ顔が何よりのご褒美なあなた。\n「困ってない？」ってすぐ手を差し伸べたくなっちゃう、おせっかいな優しさが魅力💐\n\n恋愛でも相手目線で動こうとするから、つい無理してしまうことも。\nでも、その気配りがグッとくる人も多くて、いつの間にか好かれてることが多いタイプ。\n\nただ、自分の気持ちは後回しにしがちだから、たまにはワガママになっても大丈夫。\n『ありがとう』のひと言で、もっと自信持っていいんだよ✨\n\n🌙夜の顔は…【おせっかい夜間シフト係🌙】\n夜になっても気配りは止まらない。\n体温や呼吸まで気にしながら、とにかく『相手が心地いいか』を最優先。\n自分の欲よりも「満たしてあげたい」が先に来る、究極のホスピタリティ型。\nその優しさ、逆にクセになります。",
        "ISTP": "サバサバしてて一匹狼感あるあなた。\n感情を言葉で伝えるのはちょっと苦手だけど、\n態度や行動で誠実さを見せるタイプだよね🛠️\n\n恋愛においても無理にテンションを上げず、\n自然体でいられる関係を求めるクール派。\nでも、心を許した相手にはじんわりと優しさが伝わる…そんな不器用な一面も魅力💭\n\n言葉じゃない『空気』でつながるタイプだからこそ、\n無理せず自分らしくいられる相手と出会えたら強い✨\n\n🌙夜の顔は…【無言の手さばきマスター🖐️】\n会話少なめ、でも手は語る。\n相手の反応を静かに読み取りながら、\n淡々と、でも的確にツボを突いてくる。\nテクニカルなのにどこか素朴で、\n気づけば夢中にさせられてる…そんな静かなる支配系。",
        "ISFP": "感情に正直で、あったかい空気感を大切にするあなた。\n人に優しく寄り添えるけど、傷つきやすさもあって、\n恋愛では「本当に信じられる人」じゃないと踏み込めない慎重派。\nでも、ひとたび心を開いた相手には、驚くほど深い愛情を注ぐ『本能型』。\nちょっと天然に見えて、実は直感で相手の心を読んでる…そんな魅力があるよ💫\nあなたの抱える『やさしさ』は、言葉よりも触れ方や表情で伝わるもの。\nそのぬくもりで、相手の心をほどく力があるよ。\n\n🌙夜の顔は…【密着とろけ職人🛏️】\nとにかく距離ゼロ。\n肌を合わせるたびに安心感が溢れ出す。\nリードは控えめでも、自然と求められる存在になる。\nその密着スキル、破壊力高め。",
        "ESTP": "思い立ったら即行動！\n恋も人生もノリと勢いで切り開いていくタイプ。\n興味を持った相手にはストレートにアプローチして、\n一気に距離を縮めるのが得意💥\n\nでも、熱しやすく冷めやすいところもあって、\n退屈になったらふと離れちゃうことも。\nドキドキ感を保つのが恋愛長続きのカギかも。\n感情をガマンするのが苦手だから、\nちゃんと本音で向き合ってくれる相手が◎\n\n🌙夜の顔は…【ハイテンポ破壊王🎮】\nムード？前戯？考えるよりまず行動！\n勢いと本能で押し切るタイプで、\nスピードと刺激を求める『ノンストップアタッカー』。\nでも本当は、相手の反応にめっちゃ敏感で、\n「楽しませたい」って気持ちが強いサービス精神旺盛タイプ。\n熱量MAXなぶつかり合いで、気づけば夢中になってるかも🔥",
        "ESFP": "いつでも元気でポジティブなあなたは、恋愛でもノリと勢いで飛び込んじゃうタイプ！\n感情表現が豊かで、一緒にいる人を自然と笑顔にしちゃう天性のムードメーカー🎉\n好きな人にはとことん尽くすし、ちょっとした変化にも敏感。\nでも、その分感情に振り回されたり、「気分屋？」って誤解されることも。\nそれでもあなたの魅力は、『楽しさの中にある本気さ』。\n軽そうに見えて、実はちゃんと想ってる…そのギャップが刺さるよ💘\n\n🌙夜の顔は…【快感ジャングルジム🛝】\nひとたびスイッチが入れば、テンションと好奇心で攻め続ける快楽マシーン。\n予測不能なタッチとノリの連続で、まるで遊園地みたいな時間を演出。\n「楽しませたい」気持ちがそのまま表れるから、一緒にいるとずっと飽きない。\n感情のままに動くようでいて、ちゃんと『相手を見てる』のがすごいところ。"
    }
    
    return descriptions.get(mbti, f"{mbti}タイプのあなたは、独特な魅力を持った恋愛タイプです。")

# payment_messageを返すだけの関数に変更
def get_payment_message(user_id):
    try:
        if stripe.api_key and stripe_price_id:
            # 本番用URL設定（環境変数から取得）
            base_url = os.getenv("BASE_URL", "https://lovehack20.onrender.com")
            success_url = f"{base_url}/success?session_id={{CHECKOUT_SESSION_ID}}&user_id={user_id}"
            cancel_url = f"{base_url}/cancel?user_id={user_id}"
            
            # Stripe Checkout Sessionを作成
            checkout_session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': stripe_price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                metadata={
                    'user_id': user_id
                }
            )
            payment_url = checkout_session.url
        else:
            payment_url = f"https://checkout.stripe.com/pay/test_{user_id}"
    except Exception as e:
        payment_url = f"https://checkout.stripe.com/pay/test_{user_id}"
        print(f"❌ Stripe決済URL生成エラー: {e}")
    payment_message = f"--------------------------------\n💡もっと詳しく知りたい？💘\n\nどんな異性も落とせるようになるあなただけの詳しい恋愛攻略法\n『あなただけの専属の恋愛AI相談』が解放されます✨\n\n👉今すぐ登録して、完全版アドバイスと専属恋愛AIを試してみよう！\n\n決済URL: {payment_url}\n--------------------------------解約時は『解約』と入力でいつでも解約できます。"
    return payment_message

# 課金完了時の処理関数
def handle_payment_completion(user_id):
    """課金完了時の処理"""
    try:
        # ユーザーを有料会員に更新
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET is_paid=1 WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
        
        # ユーザーのMBTIを取得
        user_profile = get_user_profile(user_id)
        mbti = user_profile.get('mbti', '不明') if user_profile else '不明'
        
        # GASに詳細アドバイスとチャットメッセージを送信
        send_detailed_advice_to_gas(user_id, mbti)
        send_chat_message_to_gas(user_id, mbti)
        
        print(f"✅ 課金完了処理完了: user_id={user_id}, mbti={mbti}")
        
    except Exception as e:
        print(f"❌ 課金完了処理エラー: {e}")

# ユーザーメッセージ処理関数
def process_user_message(user_id, message, user_profile):
    os.makedirs("/data/logs", exist_ok=True)
    with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
        f.write(f"[process_user_message] user_id={user_id}, message={message}, user_profile={user_profile}\n")
    try:
        # 1. 診断モード優先
        if user_profile and user_profile.get('mode') == 'mbti_diagnosis':
            if message in ['はい', 'いいえ']:
                return process_mbti_answer(user_id, message, user_profile)
            else:
                return "【はい】か【いいえ】で答えてね！"

        # 2. 解約ワード
        if message in ["解約", "キャンセル", "やめる", "退会"]:
            if not user_profile.get('is_paid', False):
                return "この機能は有料会員様限定です。"
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT customer_id FROM stripe_customers WHERE user_id=?", (user_id,))
            row = cursor.fetchone()
            customer_id = row[0] if row else None
            if not customer_id:
                conn.close()
                return "ご利用履歴が見つかりませんでした。"
            try:
                session = stripe.billing_portal.Session.create(
                    customer=customer_id,
                    return_url=os.getenv("BASE_URL", "https://lovehack20.onrender.com") + "/return"
                )
                portal_url = session.url
            except Exception as e:
                conn.close()
                print(f"❌ Customer Portal発行エラー: {e}")
                return "解約ページの発行に失敗しました。時間をおいて再度お試しください。"
            cursor.execute("UPDATE users SET is_paid=0 WHERE user_id=?", (user_id,))
            conn.commit()
            conn.close()
            return f"ご解約・お支払い管理はこちらから行えます：\n{portal_url}\n\n解約手続きが完了するとAI相談機能も停止します。"

        # 3. 初回ユーザー
        if not user_profile:
            return start_mbti_diagnosis(user_id)

        # 4. 性別登録モード
        if user_profile.get('mode') == 'register_gender':
            if message in ['男', '女']:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET gender=? WHERE user_id=?", (message, user_id))
                cursor.execute("UPDATE users SET mode='' WHERE user_id=?", (user_id,))
                conn.commit()
                conn.close()
                return f"性別【{message}】を登録したよ！"
            else:
                return "【男】か【女】で答えてね！"

        # 5. 相手MBTI登録モード
        if user_profile.get('mode') == 'register_partner_mbti':
            if re.match(r'^[EI][NS][FT][JP]$', message):
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET target_mbti=? WHERE user_id=?", (message, user_id))
                cursor.execute("UPDATE users SET mode='' WHERE user_id=?", (user_id,))
                conn.commit()
                conn.close()
                return f"相手のMBTI【{message}】を登録したよ！"
            else:
                return "正しいMBTI形式（例：INTJ、ENFP）で答えてね！"

        # 6. 無課金ユーザーの制限
        if not user_profile.get('is_paid', False):
            if message == "診断開始":
                return start_mbti_diagnosis(user_id)
            elif message == "性別登録":
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET mode='register_gender' WHERE user_id=?", (user_id,))
                conn.commit()
                conn.close()
                return "性別を教えてね！【男】か【女】で答えてください。"
            elif message == "相手MBTI登録":
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("UPDATE users SET mode='register_partner_mbti' WHERE user_id=?", (user_id,))
                conn.commit()
                conn.close()
                return "相手のMBTIを教えてね！（例：INTJ、ENFP）"
            else:
                return "📌専属恋愛AIのお喋り機能は有料会員様限定です！\n恋愛傾向診断を始めて有料会員になりたい場合は『診断開始』と送ってね✨"

        # 7. 有料ユーザーの通常処理
        if message == "診断開始":
            return start_mbti_diagnosis(user_id)
        elif message == "性別登録":
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET mode='register_gender' WHERE user_id=?", (user_id,))
            conn.commit()
            conn.close()
            return "性別を教えてね！【男】か【女】で答えてください。"
        elif message == "相手MBTI登録":
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET mode='register_partner_mbti' WHERE user_id=?", (user_id,))
            conn.commit()
            conn.close()
            return "相手のMBTIを教えてね！（例：INTJ、ENFP）"
        else:
            return process_ai_chat(user_id, message, user_profile)
    except Exception as e:
        import traceback
        print(f"process_user_message エラー: {e}")
        traceback.print_exc()
        return "エラーが発生しました。もう一度お試しください。"

# LINEリプライ送信関数
def send_line_reply(reply_token, message):
    """LINEにリプライメッセージを送信"""
    try:
        print(f"Sending LINE reply with token: {reply_token}")
        print(f"Message content: {message}")
        
        line_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
        if not line_token:
            print("⚠️ LINE_CHANNEL_ACCESS_TOKENが設定されていません")
            return
        
        url = "https://api.line.me/v2/bot/message/reply"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {line_token}"
        }
        
        # メッセージが配列の場合は複数メッセージ
        if isinstance(message, list):
            data = {
                "replyToken": reply_token,
                "messages": message
            }
        # メッセージが辞書（テンプレート）の場合はそのまま使用
        elif isinstance(message, dict):
            data = {
                "replyToken": reply_token,
                "messages": [message]
            }
        else:
            # 文字列の場合は通常のテキストメッセージ
            data = {
                "replyToken": reply_token,
                "messages": [{"type": "text", "text": message}]
            }
        
        print(f"Sending request to LINE API: {url}")
        response = requests.post(url, headers=headers, json=data)
        print(f"LINE API response status: {response.status_code}")
        print(f"LINE API response: {response.text}")
        
        if response.status_code != 200:
            print(f"⚠️ LINE API error: {response.status_code} - {response.text}")
        
    except Exception as e:
        print(f"LINE送信エラー: {e}")

def classify_intent(message):
    """メッセージの意図を分類"""
    try:
        llm = ChatOpenAI(openai_api_key=openai_api_key)
        prompt = (
            "Classify the following message into one of these categories:\n"
            "1: Greeting (hello, hi, good morning, good evening, こんにちは, こんばんは, おはよう, おやすみ, etc.)\n"
            "2: Thanks (thank you, thanks, ありがとう, どうも, etc.)\n"
            "3: Short reply (ok, yes, got it, わかった, うん, はい, 了解, etc.)\n"
            "4: Love advice (questions about love, dating, relationships, 恋愛, 相手, デート, 告白, etc.)\n"
            "5: Casual chat (weather, hobbies, daily conversation, 天気, 趣味, 日常会話, etc.)\n"
            "6: Other\n"
            "Return only the number (1-6)."
        )
        
        response = llm.invoke(f"{prompt}\n\nMessage: {message}")
        result = int(response.content.strip())
        
        # デバッグログを追加
        with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
            f.write(f"[classify_intent] message: {message}, response: {response.content}, result: {result}\n")
        
        return result
    except Exception as e:
        with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
            f.write(f"[classify_intent] error: {e}\n")
        return 6  # デフォルトは「その他」

def handle_casual_chat(user_id, message, user_profile):
    """雑談処理"""
    try:
        llm = ChatOpenAI(openai_api_key=openai_api_key)
        prompt = (
            f"あなたはMBTI診断ベースの女性の恋愛マスターの友達です。\n"
            f"ユーザー情報: あなたのMBTI: {user_profile.get('mbti', '不明')}, あなたの性別: {user_profile.get('gender', '不明')}\n"
            f"ユーザーの発言: {message}\n"
            f"これは雑談です。親しみやすくタメ口で絵文字も使って、短めに（100文字以内）返してください。\n"
            f"恋愛アドバイスではなく、日常会話として返してください。"
        )
        
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return "うん、そうだね！😊"

# AIチャット処理関数
def process_ai_chat(user_id, message, user_profile):
    os.makedirs("/data/logs", exist_ok=True)
    with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
        f.write(f"[process_ai_chat] user_id={user_id}, message={message}, user_profile={user_profile}\n")
    try:
        if user_profile.get('is_paid', False):
            with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
                f.write("[process_ai_chat] is_paid True, calling intent classification\n")
            
            # 意図分類
            intent = classify_intent(message)
            with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
                f.write(f"[process_ai_chat] intent classified as: {intent}\n")
            
            if intent == 1:  # 挨拶
                return "こんばんは！今日も気軽に話してね😊"
            elif intent == 2:  # 感謝
                return "どういたしまして！また何でも聞いてね✨"
            elif intent == 3:  # 短い返事
                return "うん、また何かあったら教えてね！"
            elif intent == 4:  # 恋愛相談
                return ask_ai_with_vector_db(user_id, message, user_profile)
            elif intent == 5:  # 雑談
                return handle_casual_chat(user_id, message, user_profile)
            else:  # その他（恋愛相談として処理）
                return ask_ai_with_vector_db(user_id, message, user_profile)
        
        with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
            f.write("[process_ai_chat] is_paid False or not found\n")
        if "こんにちは" in message or "hello" in message.lower():
            with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
                f.write("[process_ai_chat] greeting branch\n")
            return "こんにちは！恋愛の相談があるときはいつでも聞いてね💕"
        elif "ありがとう" in message:
            with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
                f.write("[process_ai_chat] thanks branch\n")
            return "どういたしまして！他にも恋愛の悩みがあれば気軽に相談してね✨"
        else:
            with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
                f.write("[process_ai_chat] default advice branch\n")
            return f"【{user_profile.get('mbti', '不明')}タイプ】のあなたへのアドバイス：\n{message}について詳しく教えてくれると、もっと具体的なアドバイスができるよ！"
    except Exception as e:
        import traceback
        with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
            f.write(f"[process_ai_chat] Exception: {e}\n")
            f.write(traceback.format_exc() + "\n")
        traceback.print_exc()
        return "申し訳ありません。エラーが発生しました。時間を置いて再度お試しください。"

# LINE Webhookエンドポイント
@app.route("/webhook", methods=["POST"])
def line_webhook():
    os.makedirs("/data/logs", exist_ok=True)
    with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
        f.write("[line_webhook] called\n")
    try:
        # LINEプラットフォームからのリクエストを受け取る
        data = request.get_json()
        print(f"LINE Webhook received: {data}")
        
        # LINE Webhookの検証（LINEプラットフォームからの検証リクエスト）
        if 'events' not in data:
            print("No events in data, returning 200")
            return '', 200
        
        print(f"Processing {len(data['events'])} events")
        
        # イベントを処理
        for event in data['events']:
            print(f"Processing event: {event}")
            
            # テキストメッセージの処理
            if event['type'] == 'message' and event['message']['type'] == 'text':
                user_id = event['source']['userId']
                user_message = event['message']['text'].strip()
                reply_token = event['replyToken']
                
                print(f"User ID: {user_id}")
                print(f"User message: {user_message}")
                
                # ユーザープロファイルを取得
                user_profile = get_user_profile(user_id)
                print(f"User profile: {user_profile}")
                
                # メッセージを処理
                response_message = process_user_message(user_id, user_message, user_profile)
                print(f"Response message: {response_message}")
                
                # LINEにリプライを送信
                send_line_reply(reply_token, response_message)
            
            # ボタンクリック（postback）の処理
            elif event['type'] == 'postback':
                user_id = event['source']['userId']
                postback_data = event['postback']['data']
                reply_token = event['replyToken']
                
                print(f"Postback from user_id: {user_id}")
                print(f"Postback data: {postback_data}")
                
                # MBTI回答の処理
                if postback_data.startswith('mbti_answer:'):
                    parts = postback_data.split(':')
                    if len(parts) == 3:
                        answer = "はい" if parts[1] == "yes" else "いいえ"
                        question_index = int(parts[2])
                        
                        # ユーザープロファイルを取得
                        user_profile = get_user_profile(user_id)
                        
                        # Bot側の吹き出しで「あなたの回答：はい/いいえ」を表示
                        bot_answer_message = f"あなたの回答：{answer}"
                        send_line_reply(reply_token, bot_answer_message)
                        
                        # 少し遅延させてから次の質問を処理（スピードアップ）
                        import threading
                        import time
                        
                        def process_next_question():
                            time.sleep(0.5)  # 0.5秒に短縮
                            # MBTI回答を処理
                            response_message = process_mbti_answer(user_id, answer, user_profile)
                            print(f"MBTI response: {response_message}")
                            
                            # 次の質問または診断完了を送信
                            line_token = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
                            if line_token:
                                url = "https://api.line.me/v2/bot/message/push"
                                headers = {
                                    "Content-Type": "application/json",
                                    "Authorization": f"Bearer {line_token}"
                                }
                                
                                # メッセージが辞書（テンプレート）の場合はそのまま使用
                                if isinstance(response_message, dict):
                                    data = {
                                        "to": user_id,
                                        "messages": [response_message]
                                    }
                                else:
                                    # 文字列の場合は通常のテキストメッセージ
                                    data = {
                                        "to": user_id,
                                        "messages": [{"type": "text", "text": response_message}]
                                    }
                                
                                response = requests.post(url, headers=headers, json=data)
                                print(f"Next question sent: {response.status_code}")
                                
                                # 診断完了の場合、課金誘導メッセージを別途送信
                                if "診断完了" in str(response_message):
                                    time.sleep(1)  # 1秒に短縮
                                    payment_message = get_payment_message(user_id)
                                    send_line_reply(reply_token, payment_message)
                        
                        threading.Thread(target=process_next_question).start()
        
        return '', 200
        
    except Exception as e:
        print(f"LINE Webhook error: {e}")
        return '', 200

# 課金完了Webhookエンドポイント
@app.route("/payment_webhook", methods=["POST"])
def payment_webhook():
    """課金完了時のWebhook"""
    try:
        data = request.get_json()
        user_id = data.get('userId')
        
        if user_id:
            handle_payment_completion(user_id)
            return jsonify({"status": "success"}), 200
        else:
            return jsonify({"error": "userId required"}), 400
            
    except Exception as e:
        print(f"Payment webhook error: {e}")
        return jsonify({"error": str(e)}), 500

# 環境変数確認用エンドポイント
@app.route("/env_test", methods=["GET"])
def env_test():
    """環境変数の設定状況を確認"""
    env_vars = {
        "OPENAI_API_KEY": "SET" if os.getenv("OPENAI_API_KEY") else "NOT SET",
        "STRIPE_SECRET_KEY": "SET" if os.getenv("STRIPE_SECRET_KEY") else "NOT SET",
        "STRIPE_PRICE_ID": "SET" if os.getenv("STRIPE_PRICE_ID") else "NOT SET",
        "STRIPE_WEBHOOK_SECRET": "SET" if os.getenv("STRIPE_WEBHOOK_SECRET") else "NOT SET",
        "LINE_CHANNEL_ACCESS_TOKEN": "SET" if os.getenv("LINE_CHANNEL_ACCESS_TOKEN") else "NOT SET",
        "LINE_CHANNEL_SECRET": "SET" if os.getenv("LINE_CHANNEL_SECRET") else "NOT SET",
        "GAS_NOTIFY_URL": "SET" if os.getenv("GAS_NOTIFY_URL") else "NOT SET"
    }
    return jsonify(env_vars)

# ルートエンドポイント
@app.route("/", methods=["GET"])
def root():
    return "LINE MBTI診断ボットが動作中です！"

@app.route("/return", methods=["GET"])
def return_page():
    return "<h1>決済が完了しました！LINEに戻ってサービスをご利用ください。</h1>"

@app.route("/success", methods=["GET"])
def success_page():
    return "<h1>決済が完了しました🎉 LINEに戻ってください！</h1>"

@app.route("/cancel", methods=["GET"])
def cancel_page():
    return "<h1>決済をキャンセルしました。</h1>"

@app.route("/mbti_collect", methods=["POST"])
def mbti_collect():
    data = request.get_json()
    user_id = data.get("userId")
    gender = data.get("gender")
    target_mbti = data.get("targetMbti", "不明")
    answers = data.get("answers", [])

    if not user_id or len(answers) != 16:
        return jsonify({"error": "userIdと16個の回答が必要です"}), 400

    score = {"E":0, "I":0, "S":0, "N":0, "T":0, "F":0, "J":0, "P":0}
    mapping = [
        ("E", "I"), ("E", "I"), ("I", "E"), ("I", "E"),
        ("S", "N"), ("S", "N"), ("N", "S"), ("N", "S"),
        ("T", "F"), ("T", "F"), ("F", "T"), ("F", "T"),
        ("J", "P"), ("J", "P"), ("P", "J"), ("P", "J")
    ]
    for i, (yes_key, no_key) in enumerate(mapping):
        ans = answers[i]
        if ans in [1, True, "1", "はい", "yes"]:
            score[yes_key] += 1
        else:
            score[no_key] += 1

    mbti = ""
    mbti += "E" if score["E"] >= score["I"] else "I"
    mbti += "S" if score["S"] >= score["N"] else "N"
    mbti += "T" if score["T"] >= score["F"] else "F"
    mbti += "J" if score["J"] >= score["P"] else "P"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        REPLACE INTO users (user_id, mbti, gender, target_mbti, is_paid)
        VALUES (?, ?, ?, ?, 0)
    ''', (user_id, mbti, gender, target_mbti))
    conn.commit()
    conn.close()

    # result_messageとpayment_messageも返す
    result_message = f"🔍診断完了っ！\n\nあなたの恋愛タイプは…\n❤️{MBTI_NICKNAME.get(mbti, mbti)}❤️\n\n{get_mbti_description(mbti)}"
    payment_message = get_payment_message(user_id)

    return jsonify({
        "mbti": mbti,
        "result_message": result_message,
        "payment_message": payment_message
    })

@app.route("/stripe_webhook", methods=["POST"])
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("stripe-signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
        print("🧾 Stripe イベントタイプ:", event["type"])
    except Exception as e:
        print(f"Webhook error: {e}")
        return "Webhook error", 400

    # 決済完了イベント時の処理
    if event["type"] in ["invoice.payment_succeeded", "checkout.session.completed"]:
        # user_idを特定
        obj = event["data"]["object"]
        user_id = None
        # checkout.session.completedの場合
        if "metadata" in obj and "user_id" in obj["metadata"]:
            user_id = obj["metadata"]["user_id"]
        # invoice.payment_succeededの場合（customer_idからuser_idを逆引き）
        elif "customer" in obj:
            customer_id = obj["customer"]
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT user_id FROM stripe_customers WHERE customer_id=?", (customer_id,))
            row = cursor.fetchone()
            if row:
                user_id = row[0]
            conn.close()
        if user_id:
            handle_payment_completion(user_id)
            print(f"✅ 決済完了処理実行: user_id={user_id}")
        else:
            print("⚠️ user_idが特定できませんでした")
    return "OK", 200

# --- PDF/LLM連携AI応答用の補助関数 ---
def save_message(user_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
    conn.commit()
    conn.close()

def get_recent_history(user_id, limit=5):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages WHERE user_id=? ORDER BY rowid DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [f"{row[0]}: {row[1]}" for row in reversed(rows)]

# PDFベクトルDBからRetrieverを取得
VECTOR_BASE = "chroma_db"
def get_retrievers(user_profile):
    import os
    sub_paths = []

    # self/MBTI
    if user_profile.get('mbti') and user_profile['mbti'] not in [None, '', '不明']:
        sub_paths.append(f"self/{user_profile['mbti']}")

    # partner/MBTI
    if user_profile.get('target_mbti') and user_profile['target_mbti'] not in [None, '', '不明']:
        sub_paths.append(f"partner/{user_profile['target_mbti']}")

    # gender
    if user_profile.get('gender') == '男':
        sub_paths.append("man")
    elif user_profile.get('gender') == '女':
        sub_paths.append("woman")

    # common（必ず）
    sub_paths.append("common")

    retrievers = []
    for sub in sub_paths:
        base_path = os.path.join(VECTOR_BASE, sub)
        if os.path.exists(base_path):
            # PDFごとの全ディレクトリを対象にする
            for pdf_dir in os.listdir(base_path):
                pdf_path = os.path.join(base_path, pdf_dir)
                if os.path.isdir(pdf_path):
                    retrievers.append(
                        Chroma(persist_directory=pdf_path, embedding_function=OpenAIEmbeddings()).as_retriever()
                    )
    return retrievers

def get_qa_chain(user_profile):
    retrievers = get_retrievers(user_profile)
    print("retrievers len:", len(retrievers))
    if not retrievers:
        raise ValueError("該当するベクトルDBが見つかりません")
    retriever = retrievers[0]
    print("=== retriever type:", type(retriever), "===")
    print("=== retriever repr:", repr(retriever), "===")
    with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
        f.write(f"retriever type: {type(retriever)}\n")
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever), llm

# --- AI質問受付エンドポイント ---
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_id = data.get("userId")
    question = data.get("question", "")
    profile = get_user_profile(user_id)
    if not question:
        return jsonify({"error": "質問が空です"}), 400
    if not profile["is_paid"]:
        return "", 204
    answer = ask_ai_with_vector_db(user_id, question, profile)
    return jsonify({"answer": answer})

@app.route("/upload_db", methods=["POST"])
def upload_db():
    """一時的なデータベースアップロードエンドポイント"""
    if 'file' not in request.files:
        return jsonify({"error": "ファイルがありません"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "ファイルが選択されていません"}), 400
    
    if file.filename != 'user_data.db':
        return jsonify({"error": "user_data.dbファイルのみアップロード可能です"}), 400
    
    try:
        # 永続ディスクに保存
        file.save('/data/user_data.db')
        return jsonify({"message": "データベースファイルが正常にアップロードされました"}), 200
    except Exception as e:
        return jsonify({"error": f"アップロードエラー: {str(e)}"}), 500

# --- AI応答ロジックを関数化 ---
def ask_ai_with_vector_db(user_id, question, user_profile):
    os.makedirs("/data/logs", exist_ok=True)
    with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
        f.write(f"[ask_ai_with_vector_db] user_id={user_id}, question={question}, user_profile={user_profile}\n")
    if not question:
        with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
            f.write("[ask_ai_with_vector_db] question is empty\n")
        return "質問が空です"
    if not user_profile.get("is_paid"):
        with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
            f.write("[ask_ai_with_vector_db] user is not paid\n")
        return "有料会員のみ利用できます"
    history = get_recent_history(user_id)
    try:
        llm = ChatOpenAI(openai_api_key=openai_api_key)
        
        # パーソナライズされたアドバイスコンテキストを生成
        personality_context = generate_personalized_advice(user_profile, question, history)
        
        # より詳細で構造化されたプロンプトを構築
        prompt = f"""
{personality_context}

【チャット履歴】
{chr(10).join(history) if history else "初回の相談です"}

【ユーザーの質問】
{question}

【回答の指示】
• ユーザーのMBTIの特徴を活かした具体的で実践的なアドバイスを提供してください
• 相手のMBTIとの相性も考慮してください
• 箇条書き、ステップ形式、物語形式など、適切な構造化を使用してください
• 親しみやすくタメ口で絵文字も多めに使ってください
• 同じ内容の繰り返しを避け、多様で魅力的な表現を使用してください
• 具体的な例やシチュエーションも含めてください
• ユーザーの強みを活かし、課題を克服する方法を提案してください

【重要】絶対にMBTI名（ENTJ、INFPなど）を回答に含めないでください。
"""
        
        answer = llm.invoke(prompt).content
        with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
            f.write(f"[ask_ai_with_vector_db] LLM only answer: {answer}\n")
        save_message(user_id, "user", question)
        save_message(user_id, "bot", answer)
        return answer
    except Exception as e:
        import traceback
        with open("/data/logs/debug.log", "a", encoding="utf-8") as f:
            f.write(f"[ask_ai_with_vector_db] Exception: {e}\n")
            f.write(traceback.format_exc() + "\n")
        traceback.print_exc()
        return "AI応答中にエラーが発生しました。"

# MBTI別のパーソナライズされたアドバイス生成関数
def generate_personalized_advice(user_profile, question, history):
    """MBTI別の性格特徴を活用したパーソナライズされたアドバイスを生成"""
    import random
    
    user_mbti = user_profile.get('mbti', '不明')
    user_gender = user_profile.get('gender', '不明')
    target_mbti = user_profile.get('target_mbti', '不明')
    
    # ユーザーと相手のMBTI情報を取得
    user_personality = MBTI_PERSONALITY.get(user_mbti, {})
    target_personality = MBTI_PERSONALITY.get(target_mbti, {})
    user_nickname = MBTI_NICKNAME.get(user_mbti, "恋愛探検家")
    target_nickname = MBTI_NICKNAME.get(target_mbti, "恋愛相手")
    
    # レスポンススタイルを決定（箇条書き、物語形式、対話形式など）
    response_styles = [
        "bullet_points",  # 箇条書き
        "story_format",   # 物語形式
        "dialogue_format", # 対話形式
        "step_by_step",   # ステップ形式
        "comparison",     # 比較形式
        "emotional",      # 感情重視
        "tips_format",    # ティップス形式
        "qa_format"       # Q&A形式
    ]
    
    # ユーザーのMBTIに基づいてレスポンススタイルを選択
    if user_mbti in ["INTJ", "INTP", "ENTJ", "ESTJ"]:
        preferred_styles = ["bullet_points", "step_by_step", "comparison", "tips_format"]
    elif user_mbti in ["INFJ", "INFP", "ENFJ", "ENFP"]:
        preferred_styles = ["story_format", "emotional", "dialogue_format", "qa_format"]
    elif user_mbti in ["ISTJ", "ISFJ", "ESFJ"]:
        preferred_styles = ["step_by_step", "bullet_points", "comparison", "tips_format"]
    else:
        preferred_styles = ["dialogue_format", "story_format", "emotional", "qa_format"]
    
    style = random.choice(preferred_styles)
    
    # 相性分析（簡単な相性判定）
    compatibility_notes = ""
    if user_mbti != '不明' and target_mbti != '不明':
        # 同じ機能（E/I, S/N, T/F, J/P）の組み合わせで相性を判定
        user_functions = [user_mbti[0], user_mbti[1], user_mbti[2], user_mbti[3]]
        target_functions = [target_mbti[0], target_mbti[1], target_mbti[2], target_mbti[3]]
        
        matching_count = sum(1 for u, t in zip(user_functions, target_functions) if u == t)
        if matching_count >= 3:
            compatibility_notes = "✨ とても相性が良い組み合わせです！共通点を活かしたアプローチが効果的です。"
        elif matching_count == 2:
            compatibility_notes = "😊 バランスの取れた相性です。お互いの違いを理解し合うことが大切です。"
        elif matching_count == 1:
            compatibility_notes = "🤝 補完し合える相性です。相手の特徴を活かしたアプローチが効果的です。"
        else:
            compatibility_notes = "💫 刺激的な相性です！お互いの違いを楽しみながら、理解を深めることが大切です。"
    
    # パーソナライズされたプロンプトを構築
    personality_context = f"""
あなたは{user_nickname}の恋愛マスターの女友達です。

【ユーザーの特徴】
• MBTI: {user_mbti}
• 性別: {user_gender}
• 性格特徴: {', '.join(user_personality.get('traits', []))}
• 恋愛スタイル: {user_personality.get('love_style', '')}
• 強み: {', '.join(user_personality.get('strengths', []))}
• 課題: {', '.join(user_personality.get('challenges', []))}
• 好きな異性のタイプ: {', '.join(user_personality.get('likes_in_partner', []))}
• 苦手な異性のタイプ: {', '.join(user_personality.get('dislikes_in_partner', []))}
• よくある悩み: {', '.join(user_personality.get('common_concerns', []))}
• 自分のアプローチ方法: {', '.join(user_personality.get('my_approaches', []))}
• NG行動: {', '.join(user_personality.get('ng_behaviors', []))}

【相手の特徴】
• MBTI: {target_mbti}
• ニックネーム: {target_nickname}
• 性格特徴: {', '.join(target_personality.get('traits', []))}
• 恋愛スタイル: {target_personality.get('love_style', '')}
• 強み: {', '.join(target_personality.get('strengths', []))}
• 課題: {', '.join(target_personality.get('challenges', []))}
• 好きな異性のタイプ: {', '.join(target_personality.get('likes_in_partner', []))}
• 苦手な異性のタイプ: {', '.join(target_personality.get('dislikes_in_partner', []))}
• 相手へのアプローチ方法: {', '.join(target_personality.get('partner_approaches', []))}
• 相手へのNGアプローチ: {', '.join(target_personality.get('partner_ng_behaviors', []))}

【相性分析】
{compatibility_notes}

【レスポンススタイル】
{style}で回答してください。

【重要指示】
1. 絶対にMBTI名（ENTJ、INFPなど）を回答に含めないでください
2. 「ユーザー」ではなく「あなた」「君」など親しみやすい呼び方を使ってください
3. 親しみやすくタメ口で絵文字も多めに使ってください
4. 改行を効果的に使って、読みやすく構造化してください
5. 難しい言葉は避けて、簡単で分かりやすい表現を使ってください
6. 実際のLINEの例文を具体的に示してください（例：「お疲れさま〜！今日はどんな一日だった？」）
7. 自分のMBTIと相手のMBTIの特徴を考慮して、相手に響くアプローチ方法を提案してください
8. 相手の気持ちに寄り添い、共感を示しながらアドバイスしてください
9. 具体的で実践できるアドバイスを提供してください
10. 友達がアドバイスしているような自然な会話の流れを心がけてください
"""
    
    return personality_context

# 豊富なレスポンスパターンの定義
RESPONSE_PATTERNS = {
    "greeting": [
        "こんにちは！あなた、今日も恋愛について相談したいことがあるの？😊",
        "やっほー！何か恋愛の悩みでもある？💕",
        "お疲れさま！恋愛について話したいことがあればいつでも聞くよ〜✨",
        "こんばんは！今日はどんな恋愛の相談があるの？🌟",
        "はーい！恋愛について何でも聞いてね〜😄"
    ],
    "thanks": [
        "どういたしまして！あなたの恋愛がうまくいくことを願ってるよ〜💖",
        "いえいえ！あなたが幸せになれるように全力でサポートするからね〜✨",
        "ありがとう！あなたの恋愛相談、いつでも聞くよ〜😊",
        "うれしい！あなたの恋愛がうまくいくといいね〜🌟",
        "こちらこそ！あなたの恋愛を応援してるよ〜💕"
    ],
    "casual": [
        "そうなんだ〜！あなたの恋愛についてもっと詳しく教えて〜😊",
        "なるほど！あなたの恋愛観、とても興味深いね〜✨",
        "へー！恋愛についてこんな風に考えてるんだ〜💕",
        "面白い！あなたの恋愛の話、もっと聞きたいな〜🌟",
        "そうなんだ！恋愛について色々考えてるんだね〜😄"
    ],
    "encouragement": [
        "大丈夫！あなたなら絶対にうまくいくよ〜💪✨",
        "頑張って！あなたの恋愛、応援してるよ〜💖",
        "きっと大丈夫！あなたの魅力、相手にも伝わるはず〜😊",
        "諦めないで！{あなたの恋愛、必ず良い方向に進むよ〜🌟",
        "信じてる！あなたの恋愛、きっと素敵なものになるよ〜💕"
    ],
    "advice_intro": [
        "よし！あなたの恋愛について、一緒に考えてみよう〜✨",
        "なるほど！あなたの恋愛の悩み、解決策を考えてみるね〜😊",
        "わかった！あなたの恋愛について、具体的なアドバイスをしてみるよ〜💕",
        "そうなんだ！あなたの恋愛について、一緒に考えてみよう〜🌟",
        "了解！あなたの恋愛について、詳しくアドバイスしてみるね〜😄"
    ]
}

# ランダムなレスポンスパターンを取得する関数
def get_random_response_pattern(pattern_type, user_profile):
    """指定されたタイプのランダムなレスポンスパターンを取得"""
    import random
    
    user_mbti = user_profile.get('mbti', '不明')
    nickname = MBTI_NICKNAME.get(user_mbti, "恋愛探検家")
    
    patterns = RESPONSE_PATTERNS.get(pattern_type, [])
    if patterns:
        return random.choice(patterns).format(nickname=nickname)
    return f"こんにちは！{nickname}のあなた、何かお手伝いできることはありますか？😊"

if __name__ == '__main__':
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

    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000))) 