from fastapi import FastAPI
from app.routers import cornsoupchat_router, meritdescribe_router

app = FastAPI(title="PromptPipeline API")

# 라우터 등록
app.include_router(cornsoupchat_router.router, prefix="/chat", tags=["CornSoup Chatbot"])
app.include_router(meritdescribe_router.router, prefix="/merit", tags=["MeritDescribe"])

@app.get("/")
async def root():
    return {"message": "Welcome to the PromptPipeline API"}
