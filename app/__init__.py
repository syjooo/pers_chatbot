from fastapi import FastAPI
from app.routers.cornsoupchat_router import router as cornsoupchat_router

# FastAPI 앱 생성
app = FastAPI(
    title="CornSoup Chat API",
    description="심리 성향 분석과 팀 장점 설명을 위한 API 서비스",
    version="1.0.0",
)

# 라우터 등록
app.include_router(cornsoupchat_router, prefix="/cornsoupchat", tags=["CornSoup Chat"])
