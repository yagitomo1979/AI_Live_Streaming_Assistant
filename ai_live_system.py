import time
import requests
import json
import io
import os
import pygame
import pytchat
import autogen
from llama_cpp import Llama

# ==========================================
# ★追加: テキスト整形＆保存関数 (30文字改行対応)
# ==========================================
def write_output_text(text: str):
    """
    テキストをoutput.txtに保存する。
    1行が30文字を超える場合は、30文字目で自動的に改行する。
    """
    lines = []
    for line in text.splitlines():
        while len(line) > 30:
            lines.append(line[:30])
            line = line[30:]
        lines.append(line)
        
    formatted_text = "\n".join(lines)
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(formatted_text)

# ==========================================
# 0. クラス定義＆グローバル設定
# ==========================================
class Message:
    def __init__(self, content):
        self.content = content
        self.function_call = None

class Choice:
    def __init__(self, message):
        self.message = message

class Usage:
    def __init__(self, p, c, t):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = t

class Response:
    def __init__(self, choices, usage):
        self.choices = choices
        self.usage = usage
        self.model = "local-gguf"

_loaded_models = {}

# Pygameの初期化（音声再生用）
pygame.mixer.init()

# ==========================================
# 1. カスタムクライアント（キャッシュ機能付き）
# ==========================================
class LocalGGUFClient:
    def __init__(self, config, **kwargs):
        self.model_name = config.get("model")
        model_path = config.get("model_path")
        
        if model_path not in _loaded_models:
            print(f"\n⚙️ [{self.model_name}] をロード中... (初回のみ)")
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=-1, 
                n_ctx=8192,      
                verbose=False,   
                flash_attn=True  
            )
            _loaded_models[model_path] = llm
            print(f"✅ [{self.model_name}] ロード完了！")
            
        self.llm = _loaded_models[model_path]

    def create(self, params):
        messages = params.get("messages", [])
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=params.get("temperature", 0.7),
            max_tokens=params.get("max_tokens", 1024),
        )
        content = response["choices"][0]["message"]["content"]
        
        # ★追加: LLMのラリーの様子を output.txt に出力する
        role_name = "プロデューサー" if self.model_name == "Producer" else "ライター"
        write_output_text(f"【{role_name}の思考】\n{content}")
        
        usage_data = response["usage"]
        return Response([Choice(Message(content))], Usage(usage_data["prompt_tokens"], usage_data["completion_tokens"], usage_data["total_tokens"]))

    def message_retrieval(self, response):
        return [choice.message.content for choice in response.choices]

    def cost(self, response) -> float: return 0.0

    @staticmethod
    def get_usage(response):
        return {"prompt_tokens": response.usage.prompt_tokens, "completion_tokens": response.usage.completion_tokens, "total_tokens": response.usage.total_tokens, "cost": 0.0, "model": "local-gguf"}

# ==========================================
# 2. VOICEVOX 読み上げ＆テキスト保存関数
# ==========================================
def speak_and_save_text(text: str, speaker_id: int = 3):
    """最終テキストを保存し、VOICEVOXで再生する"""
    print("\n📝 最終回答を output.txt に出力します...")
    
    # ★追加: 整形関数を使って最終回答を出力
    write_output_text(f"【回答】\n{text}")
        
    print(f"🎙️ VOICEVOXで音声を生成・再生します...")
    try:
        query_payload = {"text": text, "speaker": speaker_id}
        query_response = requests.post("http://127.0.0.1:50021/audio_query", params=query_payload)
        query_response.raise_for_status()
        
        synth_payload = {"speaker": speaker_id}
        synth_response = requests.post("http://127.0.0.1:50021/synthesis", params=synth_payload, json=query_response.json())
        synth_response.raise_for_status()

        audio_stream = io.BytesIO(synth_response.content)
        pygame.mixer.music.load(audio_stream)
        pygame.mixer.music.play()
        
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
    except Exception as e:
        print(f"❌ 音声再生エラー (VOICEVOXが起動しているか確認してください): {e}")

# ==========================================
# 3. AI議論＆最終回答抽出ロジック
# ==========================================
def process_comment_with_ai(comment_text: str):
    """コメントをAIに渡し、最終的な回答テキストを返す"""
    initial_message = f"以下の相談内容に対する回答案を作成してください。\nまずはプロデューサーから、どのような回答にすべきかの指示を出してください。\n\n【視聴者からの相談内容】\n{comment_text}"
    
    # ⚠️ ここをご自身のモデルパスに書き換えてください ⚠️
    PRODUCER_MODEL_PATH = "your_model.gguf" 
    WRITER_MODEL_PATH = "your_model.gguf" 
    
    config_list_producer = [{"model": "Producer", "model_client_cls": "LocalGGUFClient", "model_path": PRODUCER_MODEL_PATH}]
    config_list_writer = [{"model": "Writer", "model_client_cls": "LocalGGUFClient", "model_path": WRITER_MODEL_PATH}]

    termination_msg = lambda msg: "[MISSION_COMPLETE]" in str(msg.get("content", "")).upper()

    producer_agent = autogen.UserProxyAgent(
        name="Producer",
        llm_config={"config_list": config_list_producer, "cache_seed": None},
        human_input_mode="NEVER",
        code_execution_config=False,
        max_consecutive_auto_reply=2, 
        is_termination_msg=termination_msg,
        system_message="あなたはライブ配信のプロデューサーです。\n視聴者からの相談に対し、ライターへ『回答の方向性とトーン』を簡潔に指示してください。\nライターが回答案を出したら、修正指示は出さず、必ず「[MISSION_COMPLETE]」と発言して完了させてください。\n英語は使用不可。"
    )

    writer_agent = autogen.AssistantAgent(
        name="Writer",
        llm_config={"config_list": config_list_writer, "cache_seed": None},
        is_termination_msg=termination_msg,
        system_message="あなたは優秀なAIカウンセラーです。プロデューサーの指示に従い、視聴者への回答を作成します。\n【重要】ライブ配信で読み上げるため、温かい口調で、必ず「300文字以内」で簡潔に回答してください。\n英語の思考プロセスは絶対に出力しないでください。\n出力する文章は、一文ごとに改行を入れるようにして下さい。"
    )

    producer_agent.register_model_client(model_client_cls=LocalGGUFClient)
    writer_agent.register_model_client(model_client_cls=LocalGGUFClient)

    chat_result = writer_agent.initiate_chat(producer_agent, message=initial_message)
    
    chat_history = producer_agent.chat_messages[writer_agent]
    final_advice = "（回答を生成できませんでした）"
    
    for message in reversed(chat_history):
        if message.get("name") == "Writer":
            final_advice = message.get("content")
            break
            
    return final_advice

# ==========================================
# 4. メインループ（YouTube監視 → AI → 音声）
# ==========================================
def main():
    # ⚠️ ここをテストしたいYouTubeライブの動画IDに変更してください ⚠️
    VIDEO_ID = "uPZeSdpLxs4" 
    
    print("\n" + "="*50)
    print(" 🚀 AIライブ配信システム 起動！（24時間稼働モード）")
    print("="*50)
    
    write_output_text("システム起動中...")

    # ★変更点1: プログラム全体を絶対に終わらせない「大外の無限ループ」
    while True:
        try:
            # 接続を確立
            chat = pytchat.create(video_id=VIDEO_ID)
            print(f"\n✅ YouTube Live [{VIDEO_ID}] に接続しました。コメントを待機します...\n")
            write_output_text("コメントを受付中です。")

            while chat.is_alive():
                # ★変更点2: 1回のコメント処理ごとにエラーをキャッチして、ループを死守する
                try:
                    items = chat.get().items
                    if items:
                        latest_comment = items[-1]
                        author = latest_comment.author.name
                        text = latest_comment.message
                        
                        print("-" * 50)
                        print(f"👤 {author} さんからの相談: {text}")
                        
                        write_output_text(f"コメントを受付ました。\n相談者: {author}さん")
                        time.sleep(1)
                        
                        final_answer = process_comment_with_ai(text)
                        
                        print(f"\n✨ AIの回答:\n{final_answer}\n")
                        speak_and_save_text(final_answer)
                        
                        print("\n次のコメントを待機しています...")
                        write_output_text("コメントを受付中です。")
                        print("-" * 50)
                        
                    time.sleep(2)
                    
                except Exception as inner_e:
                    # AI処理やVOICEVOXでエラーが起きても、ここでキャッチして無視（ループ継続）
                    print(f"⚠️ コメント処理中に軽微なエラーが発生しました（スキップします）: {inner_e}")
                    time.sleep(2)

            # chat.is_alive() が False (通信切断) になったらここに来る
            print("⚠️ YouTubeとのチャット接続が切れました。10秒後に再接続を試みます...")
            write_output_text("通信待機中...")
            time.sleep(10)

        except KeyboardInterrupt:
            # Ctrl+C (手動停止) の時だけは、本当にプログラムを終了させる
            print("\n⏹️ 管理者によってシステムが手動停止されました。お疲れ様でした！")
            break
            
        except Exception as e:
            # ネットワークが完全に無い場合など、致命的なエラー
            print(f"❌ 致命的な接続エラーが発生しました。15秒後に再起動します: {e}")
            write_output_text("システム再接続中...")
            time.sleep(15)

if __name__ == "__main__":
    main()
