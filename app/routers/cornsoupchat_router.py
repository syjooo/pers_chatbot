from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.cornsoupchat_services import CornSoupChatService

# 라우터 생성
router = APIRouter()

# 요청 데이터 모델 정의
class ChatRequest(BaseModel):
    user_input: str

# 응답 데이터 모델 정의
class ChatResponse(BaseModel):
    bot_response: str

# 서비스 인스턴스 생성
chat_service = CornSoupChatService()

@router.post("/chat", response_model=ChatResponse)
async def start_chat(request: ChatRequest):
    """
    사용자 입력에 대한 챗봇 응답 생성
    """
    try:
        # 챗봇 응답 생성
        bot_response = chat_service.generate_response(request.user_input)

        # 대화 기록 저장
        chat_service.save_chat(request.user_input, bot_response)

        return ChatResponse(bot_response=bot_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"챗봇 응답 생성 중 오류 발생: {str(e)}")

@router.get("/history")
async def get_chat_history():
    """
    저장된 대화 기록 반환
    """
    try:
        return {"conversation": chat_service.get_chat_history()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"대화 기록 불러오기 중 오류 발생: {str(e)}")
