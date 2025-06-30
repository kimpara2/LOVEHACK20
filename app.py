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
            is_paid BOOLEAN DEFAULT 0,
            mode TEXT DEFAULT '',
            mbti_answers TEXT DEFAULT '[]'
        )
    ''')
    conn.commit()
    conn.close()
    print("SQLiteデータベースを初期化しました。")

init_db()

# ユーザープロファイルの取得
def get_user_profile(user_id):
    print(f"Getting user profile for user_id: {user_id}")
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT mbti, gender, target_mbti, is_paid, mode FROM users WHERE user_id=?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        # ユーザーが存在しない場合は作成
        conn = sqlite3.connect("user_data.db")
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (user_id) VALUES (?)", (user_id,))
        conn.commit()
        conn.close()
        return None
    
    profile = {
        "mbti": row[0] if row[0] else "不明",
        "gender": row[1] if row[1] else "不明",
        "target_mbti": row[2] if row[2] else "不明",
        "is_paid": bool(row[3]) if row[3] else False,
        "mode": row[4] if row[4] else ""
    }
    
    print(f"User profile result: {profile}")
    return profile

# MBTI集計ロジック
def calc_mbti(answers):
    score = {'E': 0, 'I': 0, 'S': 0, 'N': 0, 'T': 0, 'F': 0, 'J': 0, 'P': 0}
    mapping = [
        ('E', 'I'), ('P', 'J'), ('S', 'N'), ('T', 'F'), ('E', 'I'),
        ('J', 'P'), ('N', 'S'), ('I', 'E'), ('F', 'T'), ('P', 'J')
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

# MBTI診断開始関数
def start_mbti_diagnosis(user_id):
    """MBTI診断を開始する"""
    print(f"Starting MBTI diagnosis for user_id: {user_id}")
    
    # 診断状態を初期化
    conn = sqlite3.connect("user_data.db")
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET mode='mbti_diagnosis' WHERE user_id=?", (user_id,))
    cursor.execute("UPDATE users SET mbti_answers='[]' WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()
    
    print(f"MBTI diagnosis mode set for user_id: {user_id}")
    
    # 最初の質問を送信
    first_question = send_mbti_question(user_id, 0)
    print(f"First question generated: {first_question}")
    
    return first_question

# MBTI質問送信関数（ボタン式）
def send_mbti_question(user_id, question_index):
    """MBTI診断の質問を送信（ボタン式）"""
    questions = [
        "好きな人とは、毎日LINEしたいほう？🥺",
        "デートの計画よりも、その時の気分で動くのが好き😳",
        "恋人のちょっとした変化にもすぐ気づくほうだ😊",
        "恋人の相談には、共感よりもアドバイスを優先しがち？📱",
        "初対面でも気になる人には自分から話しかけるほうだ？📅",
        "好きな人との関係がハッキリしないのは苦手？☕️",
        "デートは、思い出に残るようなロマンチックな演出が好き？💬",
        "気になる人がいても、自分の気持ちはなかなか伝えられない？👫",
        "恋愛には、価値観の一致が何より大事だと思う？💌",
        "相手の好みに合わせて、自分のキャラを柔軟に変えられる？😅"
    ]
    
    if question_index >= len(questions):
        return "診断が完了しました！"
    
    # ボタンテンプレートを作成（messageアクションでユーザーの吹き出しに）
    template = {
        "type": "template",
        "altText": f"質問{question_index + 1}/10: {questions[question_index]}",
        "template": {
            "type": "buttons",
            "title": f"質問{question_index + 1}/10",
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
        conn = sqlite3.connect("user_data.db")
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
        print(f"現在の回答数: {len(answers)}/10")
        print(f"最新の回答: {answer} (数値: {1 if answer == 'はい' else 0})")
        print(f"全回答履歴: {answers}")
        print(f"==================")
        cursor.execute("UPDATE users SET mbti_answers=? WHERE user_id=?", (json.dumps(answers), user_id))
        conn.commit()
        conn.close()
        next_question_index = len(answers)
        if next_question_index < 10:
            print(f"次の質問を送信: 質問{next_question_index + 1}/10")
            return send_mbti_question(user_id, next_question_index)
        else:
            print(f"診断完了！全回答: {answers}")
            result_message = complete_mbti_diagnosis(user_id, answers)
            payment_message = get_payment_message(user_id)
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
        
        # 結果を保存
        conn = sqlite3.connect("user_data.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET mbti=?, mode='' WHERE user_id=?", (mbti, user_id))
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
    payment_message = f"----------------------\n💡もっと詳しく知りたい？💘\n\nどんな異性も落とせるようになるあなただけの詳しい恋愛攻略法\n『あなただけの専属の恋愛AI相談』が解放されます✨\n\n👉今すぐ登録して、完全版アドバイスと専属恋愛AIを試してみよう！\n\n決済URL: {payment_url}\n----------------------"
    return payment_message

# 課金完了時の処理関数
def handle_payment_completion(user_id):
    """課金完了時の処理"""
    try:
        # ユーザーを有料会員に更新
        conn = sqlite3.connect("user_data.db")
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
    """ユーザーメッセージを処理して適切な応答を返す"""
    
    # 初回ユーザーの場合、自動的に診断開始
    if not user_profile:
        return start_mbti_diagnosis(user_id)
    
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
        if re.match(r'^[EI][NS][FT][JP]$', message):
            # 相手のMBTIを保存
            conn = sqlite3.connect("user_data.db")
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET target_mbti=? WHERE user_id=?", (message, user_id))
            cursor.execute("UPDATE users SET mode='' WHERE user_id=?", (user_id,))
            conn.commit()
            conn.close()
            return f"相手のMBTI【{message}】を登録したよ！"
        else:
            return "正しいMBTI形式（例：INTJ、ENFP）で答えてね！"
    
    # MBTI診断モードの処理
    if user_profile.get('mode') == 'mbti_diagnosis':
        if message in ['はい', 'いいえ']:
            return process_mbti_answer(user_id, message, user_profile)
        else:
            return "【はい】か【いいえ】で答えてね！"
    
    # 無課金ユーザーの制限（診断中以外は課金誘導）
    if not user_profile.get('is_paid', False):
        if message == "診断開始":
            return start_mbti_diagnosis(user_id)
        elif message == "性別登録":
            # 性別登録モードに設定
            conn = sqlite3.connect("user_data.db")
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET mode='register_gender' WHERE user_id=?", (user_id,))
            conn.commit()
            conn.close()
            return "性別を教えてね！【男】か【女】で答えてください。"
        elif message == "相手MBTI登録":
            # 相手MBTI登録モードに設定
            conn = sqlite3.connect("user_data.db")
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET mode='register_partner_mbti' WHERE user_id=?", (user_id,))
            conn.commit()
            conn.close()
            return "相手のMBTIを教えてね！（例：INTJ、ENFP）"
        else:
            # 無課金ユーザーは課金誘導メッセージ
            return "📌専属恋愛AIのお喋り機能は有料会員様限定です！\n恋愛傾向診断を始めて有料会員になりたい場合は『診断開始』と送ってね✨"
    
    # 有料ユーザーの通常処理
    if message == "診断開始":
        return start_mbti_diagnosis(user_id)
    elif message == "性別登録":
        # 性別登録モードに設定
        conn = sqlite3.connect("user_data.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET mode='register_gender' WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
        return "性別を教えてね！【男】か【女】で答えてください。"
    elif message == "相手MBTI登録":
        # 相手MBTI登録モードに設定
        conn = sqlite3.connect("user_data.db")
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET mode='register_partner_mbti' WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
        return "相手のMBTIを教えてね！（例：INTJ、ENFP）"
    else:
        # AIチャット処理
        return process_ai_chat(user_id, message, user_profile)

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

# AIチャット処理関数
def process_ai_chat(user_id, message, user_profile):
    """AIチャット処理"""
    try:
        # 簡単な応答（実際はLangChainを使用）
        if "こんにちは" in message or "hello" in message.lower():
            return "こんにちは！恋愛の相談があるときはいつでも聞いてね💕"
        elif "ありがとう" in message:
            return "どういたしまして！他にも恋愛の悩みがあれば気軽に相談してね✨"
        else:
            return f"【{user_profile.get('mbti', '不明')}タイプ】のあなたへのアドバイス：\n{message}について詳しく教えてくれると、もっと具体的なアドバイスができるよ！"
    except Exception as e:
        print(f"AIチャット処理エラー: {e}")
        return "申し訳ありません。エラーが発生しました。時間を置いて再度お試しください。"

# LINE Webhookエンドポイント
@app.route("/webhook", methods=["POST"])
def line_webhook():
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
                                    send_payment_message(user_id)
                        
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