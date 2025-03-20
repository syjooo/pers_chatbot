import os
import warnings
import logging
import json
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import SequentialChain

# 환경 변수 로드
load_dotenv()

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# 로그 설정
#나중에 지울코드
logging.basicConfig(level=logging.INFO)
logging.info("성향 분석 챗봇 시작")

# Ollama LLM 설정
ollama_llm = OllamaLLM(base_url="http://210.110.103.73:11400", model="cornsoup_9b")

# 서비스 클래스 정의
class CornSoupChatService:
    def __init__(self):
        # 대화 기록 관리
        self.chat_history = {"conversation": []}
        self.recent_history = []  # 최근 5개 대화만 유지

        # JSON 파일 경로 설정
        base_path = os.path.join(os.path.dirname(__file__), "../data")
        self.chat_file = os.path.join(base_path, "chat_archive.json")

        # JSON 파일 생성 (존재하지 않으면)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        if not os.path.exists(self.chat_file):
            with open(self.chat_file, "w", encoding="utf-8") as file:
                json.dump({"conversation": []}, file)

        # 프롬프트 템플릿 로드
        self.chat_prompt = self._load_chat_prompt()

    def _load_chat_prompt(self):
        # ChatPromptTemplate 정의
        return ChatPromptTemplate.from_template(
            "사용자의 입력: {user_input}에 대해 대답해줘."
        )

    def generate_response(self, user_input: str) -> str:
        """
        사용자 입력에 대한 챗봇 응답 생성
        """
        response = self.chat_prompt.invoke({"user_input": user_input})
        return response

    def save_chat(self, user_input: str, bot_response: str):
        """
        대화 기록을 JSON 파일에 저장
        """
        try:
            with open(self.chat_file, "r", encoding="utf-8") as file:
                chat_data = json.load(file)

            chat_data["conversation"].append({
                "user": user_input,
                "bot": bot_response
            })

            with open(self.chat_file, "w", encoding="utf-8") as file:
                json.dump(chat_data, file, ensure_ascii=False, indent=4)

        except Exception as e:
            logging.error(f"대화 기록 저장 중 오류 발생: {e}")

    def get_chat_history(self):
        """
        JSON 파일에서 대화 기록 불러오기
        """
        try:
            with open(self.chat_file, "r", encoding="utf-8") as file:
                chat_data = json.load(file)
            return chat_data.get("conversation", [])
        except Exception as e:
            logging.error(f"대화 기록 불러오기 중 오류 발생: {e}")
            return []
