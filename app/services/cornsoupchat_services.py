import openai
import os
import json

class CornSoupChatService:
    def __init__(self):
        base_path = os.path.join(os.path.dirname(__file__), "../resources")
        with open(os.path.join(base_path, "persprompt.json"), "r", encoding="utf-8") as file:
            self.prompt_data = json.load(file)

    def generate_response(self, user_input: str) -> str:
        system_prompt = self.prompt_data.get("system_prompt", "")
        examples = self.prompt_data.get("examples", [])

        # OpenAI 메시지 형식 구성
        messages = [{"role": "system", "content": system_prompt}]
        for example in examples:
            messages.append({"role": "user", "content": example["input"]})
            messages.append({"role": "assistant", "content": example["output"]})

        messages.append({"role": "user", "content": user_input})

        # OpenAI API 호출
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        return response["choices"][0]["message"]["content"]

    def __init__(self):
        # 데이터 저장 디렉토리 설정
        self.data_dir = os.path.join(os.path.dirname(__file__), "../data")
        self.chat_file = os.path.join(self.data_dir, "chat_archive.json")

        # 데이터 디렉토리 생성
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # JSON 파일 초기화
        if not os.path.exists(self.chat_file):
            with open(self.chat_file, "w", encoding="utf-8") as file:
                json.dump({"conversation": []}, file)

    def save_chat(self, user_input: str, bot_response: str):
        """
        대화 기록을 JSON 파일에 저장합니다.
        """
        try:
            # 기존 대화 기록 불러오기
            with open(self.chat_file, "r", encoding="utf-8") as file:
                chat_data = json.load(file)

            # 새로운 대화 추가
            chat_data["conversation"].append({
                "user": user_input,
                "bot": bot_response
            })

            # 파일에 다시 저장
            with open(self.chat_file, "w", encoding="utf-8") as file:
                json.dump(chat_data, file, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"대화 저장 중 오류 발생: {e}")

    def get_chat_history(self):
        """
        JSON 파일에서 대화 기록을 불러옵니다.
        """
        try:
            with open(self.chat_file, "r", encoding="utf-8") as file:
                chat_data = json.load(file)
            return chat_data.get("conversation", [])
        except Exception as e:
            print(f"대화 기록 불러오기 중 오류 발생: {e}")
            return []
