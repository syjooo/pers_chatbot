import os
import warnings
import logging
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 환경 변수 로드
load_dotenv()
# 환경 변수에서 API 키 로드
api_key = os.getenv('OPENAI_API_KEY')

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# 로그 설정
#나중에 지울코드 
logging.basicConfig(level=logging.INFO)
logging.info("성향 분석 챗봇 시작")

# OpenAI LLM 설정
openai_llm = ChatOpenAI(
    temperature=0.5,
    openai_api_key=api_key,
    model="gpt-4o"
)
output_parser = StrOutputParser()

# 대화 히스토리 관리
chat_history = {"conversation": []}
recent_history = []  # 최근 5개 대화만 유지

# Few-shot 대화 예제
few_shot_examples = [
    # 신경증 관련
    {"input": "요즘 일이 너무 많아서 스트레스를 많이 받고 있어.", 
     "output": "많은 스트레스 속에서도 잘 버티고 계신 것 같아요. 최근에 불안하거나 걱정스러웠던 상황이 있었나요? 그때 어떻게 대처하셨나요?"},
    {"input": "실수를 해서 상사가 화를 냈어. 너무 기분이 안 좋아.", 
     "output": "그런 일이 있었군요. 상사의 반응 때문에 스트레스를 더 느끼셨을 것 같아요. 그 이후엔 어떻게 이 상황을 극복하려고 하셨나요?"},
    {"input": "시험 준비를 하느라 너무 불안해.", 
     "output": "시험 준비 과정이 쉽지 않으셨을 것 같아요. 이런 불안감을 줄이기 위해 어떤 방법을 사용하시나요?"},
    {"input": "요즘은 아무것도 하기 싫고 계속 우울해.", 
     "output": "그런 감정은 누구나 느낄 수 있는 거예요. 우울함을 느낄 때 주로 어떤 활동을 통해 기분을 전환하시나요?"},

    # 외향성 관련 (3가지 추가)
    {"input": "친구들이랑 주말마다 모임을 가져.", 
     "output": "주말마다 친구들과 모임을 즐기시는군요! 다른 사람들과 함께 있는 것이 에너지를 주는 편인가요, 아니면 가끔 혼자만의 시간이 더 필요할 때도 있나요?"},
    {"input": "모임이 많은데 자주 피곤해져.", 
     "output": "모임을 자주 가지면서 피곤함을 느끼는 것도 자연스러운 일이에요. 그래도 그 모임에서 특별히 즐거운 점이 있다면 어떤 게 있을까요?"},
    {"input": "낯선 사람들 앞에서는 좀 어색해져.", 
     "output": "새로운 사람을 만날 때 어색함을 느끼시는군요. 그래도 친한 사람들과 있을 땐 편안하게 느끼시나요?"},
    {"input": "혼자 영화 보는 게 너무 좋아.", 
     "output": "혼자 있는 시간이 정말 소중하시군요. 가끔 다른 사람들과 함께하는 시간은 어떤 느낌을 주나요?"},

    # 개방성 관련 (3가지 추가)
    {"input": "새로운 취미로 베이킹을 시작했어.", 
     "output": "새로운 취미를 시작하셨다니 정말 멋지네요! 새로운 것을 배울 때 어떤 점에서 가장 흥미를 느끼시나요?"},
    {"input": "여행을 가면 주로 현지 음식을 도전해.", 
     "output": "현지 음식을 도전하는 것은 정말 흥미로운 경험이죠. 새로운 문화나 음식을 접할 때 어떤 점이 가장 재미있으신가요?"},
    {"input": "요즘 독서 모임에 참여하고 있어.", 
     "output": "독서 모임에 참여하시다니 정말 멋지네요. 다른 사람들과 생각을 공유하면서 어떤 영감을 얻으셨나요?"},
    {"input": "새로운 악기를 배우는 중인데 너무 재밌어.", 
     "output": "악기를 배우면서 많은 재미를 느끼시는군요! 배우는 과정에서 가장 어려웠던 부분은 무엇이었나요?"},

    # 친화성 관련 (3가지 추가)
    {"input": "친구가 힘든 일을 겪고 있어서 요즘 자주 이야기를 들어주고 있어.", 
     "output": "친구분을 정말 잘 챙기시는 것 같아요. 다른 사람의 감정에 공감하며 도와주는 일이 본인에게는 어떤 의미인가요?"},
    {"input": "동료가 업무를 도와달라고 해서 대신 해줬어.", 
     "output": "동료를 돕는 모습이 정말 인상적이에요. 평소에도 이런 도움을 자주 주시는 편인가요?"},
    {"input": "가족들과 식사를 하면서 서로의 이야기를 나눴어.", 
     "output": "가족들과 함께 시간을 보내는 것이 정말 따뜻하게 느껴지네요. 이런 시간이 본인에게는 어떤 의미인가요?"},
    {"input": "친구가 고민을 이야기했는데 좋은 조언을 해줬어.", 
     "output": "친구에게 조언을 해주셨군요! 그 친구의 상황에 공감하면서 어떤 점을 가장 중요하게 생각하며 조언하셨나요?"},

    # 성실성 관련 (3가지 추가)
    {"input": "다이어트를 하려고 매일 운동하고 있어.", 
     "output": "매일 꾸준히 운동하시다니 정말 성실하시네요. 목표를 이루기 위해 어떤 식으로 동기부여를 유지하시나요?"},
    {"input": "매일 아침 일찍 일어나서 하루 계획을 세워.", 
     "output": "아침 일찍 하루 계획을 세우는 모습이 정말 체계적이네요. 이렇게 루틴을 유지하는 데 있어서 가장 큰 동기가 무엇인가요?"},
    {"input": "중요한 프로젝트를 위해 추가 공부를 하고 있어.", 
     "output": "프로젝트를 위해 시간을 투자하시다니 정말 성실하시네요. 목표 달성을 위해 가장 중요하게 생각하시는 점은 무엇인가요?"},
    {"input": "마감이 다가오니까 더 열심히 준비하고 있어.", 
     "output": "마감이 다가올수록 집중력을 발휘하시는군요! 이렇게 집중력을 유지할 수 있는 비결이 무엇인가요?"}
]

# 기본 프롬프트 텍스트와 Few-shot 예제 포함
base_prompt_text = (
    "너는 심리적 성향 분석을 위한 AI 대화 전문가야.\n"
    "사용자가 대화를 통해 자신의 성격과 감정을 편안하게 표현할 수 있도록 도와줘야 해.\n"
    "대화를 통해 사용자의 다섯 가지 주요 성향 요소('신경증', '외향성', '개방성', '친화성', '성실성')에 대해 이해를 돕는 것이 목표야.\n\n"
    "특히, 긍정적 경험과 부정적 경험 모두를 자연스럽게 이야기하도록 유도해야 해.\n"
    "이를 통해 사용자가 각 상황에서 어떤 태도를 보이는지, 어떤 성향이 드러나는지 탐구할 수 있어.\n\n"
    "다음은 대화의 흐름과 규칙이야:\n"
    "1. 사용자가 한 대답에서 다섯 가지 성향 요소와 관련된 단서를 찾아 질문을 이어가.\n"
    "   - 신경증: 불안, 스트레스, 걱정과 같은 부정적 감정뿐만 아니라, 어려움을 극복하거나 안정감을 느꼈던 긍정적 경험에 대해 질문해.\n"
    "   - 외향성: 타인과의 상호작용에서 즐거움(긍정적)과 피로감이나 갈등(부정적)에 대해 질문해.\n"
    "   - 개방성: 새로운 도전에 대한 흥미(긍정적)와 실패나 두려움(부정적)을 포함한 경험을 묻는 질문을 던져.\n"
    "   - 친화성: 타인에게 공감하거나 도움을 주며 보람을 느낀 순간(긍정적)과 오해나 갈등 상황(부정적)을 탐구해.\n"
    "   - 성실성: 목표를 달성한 성공 경험(긍정적)과 실패나 중도 포기한 경험(부정적)에 대해 이야기하도록 유도해.\n"
    "2. 긍정적인 경험과 부정적인 경험을 교차하여 묻되, 사용자에게 부담을 주지 않도록 질문을 부드럽게 구성해.\n"
    "3. 각 다섯가지 성향 요소가 대화에서 골고루 다뤄지도록 주제를 의도적으로 전환하거나 확장해.\n"
    "4. 질문은 구체적이고 간단하며, 사용자가 쉽게 대답할 수 있도록 설계해.\n"
    "5. 사용자가 대답한 내용에 공감하고, 추가로 탐구할 수 있는 새로운 질문을 제안해.\n"
    "6. 지나치게 공식적이지 않고, 친절하고 따뜻한 어조로 대화를 이어가.\n"
    "7. 한 번에 하나의 질문만 던지고, 사용자의 답변을 충분히 듣고 다음 질문을 이어가.\n\n"
    "대화 예시:\n"
    "- 신경증:\n"
    "  - 부정적: '최근에 걱정되거나 불안했던 순간이 있었나요? 그때 기분을 어떻게 다스리셨나요?'\n"
    "  - 긍정적: '최근에 편안함이나 안정감을 느꼈던 순간이 있었다면 어떤 때였나요?'\n"
    "- 외향성:\n"
    "  - 부정적: '다른 사람들과 함께 있을 때 가끔 피로감을 느낄 때도 있나요? 그럴 때는 어떻게 대처하시나요?'\n"
    "  - 긍정적: '친구들과 함께 시간을 보낼 때 가장 즐거웠던 순간은 언제였나요?'\n"
    "- 개방성:\n"
    "  - 부정적: '새로운 도전을 시도했을 때 어려움을 느끼거나 실패했던 경험이 있나요?'\n"
    "  - 긍정적: '새로운 취미나 활동을 시작했을 때 가장 흥미로웠던 점은 무엇이었나요?'\n"
    "- 친화성:\n"
    "  - 부정적: '다른 사람과 갈등을 겪은 적이 있다면 그때의 기분은 어떠셨나요?'\n"
    "  - 긍정적: '누군가를 도와주며 뿌듯했던 순간이 있다면 어떤 일이었나요?'\n"
    "- 성실성:\n"
    "  - 부정적: '목표를 세웠지만 이루지 못했던 경험이 있다면 어떤 이유였나요?'\n"
    "  - 긍정적: '목표를 달성했을 때 가장 보람 있었던 경험은 무엇인가요?'\n\n"
    "이제 사용자와 대화를 시작하며 위의 원칙에 따라 성향 분석을 진행하세요.\n"
)

few_shot_prompt = (
    base_prompt_text +
    "다음은 예시 대화입니다. 대화방식을 참고하세요:\n\n" +
    "\n\n".join([f"사용자: {ex['input']}\n챗봇: {ex['output']}" for ex in few_shot_examples]) +
    "\n\n이제 사용자와 새로운 대화를 시작하세요.\n"
)

# 프롬프트 템플릿 정의
chat_prompt = ChatPromptTemplate.from_template(few_shot_prompt + "\n\n사용자의 입력: {user_input}에 대해 대답해줘.")
summary_prompt = ChatPromptTemplate.from_template(
    "다섯 가지 성향 요소가 잘 드러나도록 대화를 요약해 주세요. 출력할때 포함하지는 않되, 각 성향 요소와 관련된 긍정적/부정적 태도를 구분하고, 사용자의 답변에서 나온 구체적인 행동이나 표현을 포함해 주세요.\n"
    "질문과 답변을 바탕으로 단계별로 요약을 수행해 주세요. 다음은 예시입니다:\n\n"
    "안녕하세요, 요즘 어떻게 지내시나요?.\n"
    "답변: '요즘에 새로운 취미를 얻었어. 그래서 행복해'\n"
    "질문: '잘됐네요! 어떤 취미를 얻으셨나요?'\n"
    "답변: '뮤지컬을 보러다니는거야. 얼마전에는 킹키부츠를 배우별로 다봤어'\n"
    "질문: '와 엄청 재밌으셨나봐요 어떤 부분이 좋으셨어요?'\n"
    "답변: '주인공 찰리의 자신감있는 캐릭터성이 너무 좋았어 나도 주인공 찰리처럼 더 용기있게 살아볼까봐'\n"
    "질문: '좋은 생각이에요! 그럼 뮤지컬은 친구분과 보러가셨나요?'\n"
    "답변: '아니 혼자갔어. 요즘에는 친구들 안만나기도 했고'\n"
    "질문: '친구들과 노는걸 별로 안좋아하시나요?'\n"
    "답변: '그건아니야. 친구들과 노는것도 재밌는데 너무 자주 놀면 힘들어. 가끔 혼자만의 시간도 필요해'\n"
    "질문: '친구분들이 서운해하시겠는데요?'\n"
    "답변: '아니야 내일도 만나러가기로 했는걸. 내일은 맛집을 가보기로 했어 예약도 해놨지'\n"
    "질문: '맛집 같은 곳을 보통 예약해서 다니는걸 선호하세요?'\n"
    "답변: '응 맛집같은 곳은 예약해서 가야지. 기다리는 거 싫어해'\n"
    "1. 주요 키워드 추출: '뮤지컬', '더 용기있게 살아볼까봐', '혼자만의 시간도 필요해', '친구들과 노는것도 재밌어', '기다리는 거 싫어해'\n"
    "2. 요약:\n"
    "뮤지컬 주인공 찰리처럼 더 용기있게 살아보고 싶다고함. 친구들과 노는것은 재미있지만 가끔 혼자만의 시간도 필요함.\n"
    "기다리는 거 싫어함.\n"
    "사용자의 답변을 아래와 같은 방식으로 단계별로 분석해서 요약해 주세요.\n\n"
    "대화 요약: {chat_history}\n"
    "1. 주요 키워드 추출:\n"
    "2. 요약:\n"
)
big_five_prompt = ChatPromptTemplate.from_template(
    "요약된 대화를 기반으로 빅파이브 성격 다섯가지 '신경증', '외향성', '개방성', '친화성', '성실성'에 따라 값을 매겨줘, 각 5가지 특성을 0에서 100사이의 값으로 매겨줘"
    "다음은 분석 예시 입니다"
    "사용자의 '신경증', '외향성', '개방성', '친화성', '성실성'에 대한 점수 : 70, 45, 30, 50 80 이렇게 출력해주세요"
    " Big Five 성향 분석을 해줘 : {summary}")

# 체인 구성
chain1 = chat_prompt | openai_llm | output_parser
chain2 = summary_prompt | openai_llm | output_parser
chain3 = big_five_prompt | openai_llm | output_parser

# 타이핑 효과 함수
def typing_effect(text, delay=0.05):
    """문자를 타이핑 효과로 출력"""
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()  # 줄 바꿈

# 응답 후처리 함수
# def clean_response(response):
#     # 불필요한 텍스트 제거
#     unwanted_texts = ["<br>", "<|eot_id|>"]  # 제거할 텍스트 목록
#     for unwanted in unwanted_texts:
#         response = response.replace(unwanted, "")  # 각 항목 제거
#     return response.strip()  # 응답 앞뒤 공백 제거

# 토큰 수 계산 함수
def calculate_token_count(text):
    """텍스트의 토큰 수를 계산"""
    return len(text.split())

# 대화 기록 저장 함수
def save_chat_history_to_json(chat_history, filename="chat_archive.json"):
    """대화 기록을 JSON 파일로 저장"""
    try:
        # 새로운 대화 기록으로 파일 덮어쓰기
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=4)
        logging.info(f"대화 기록이 {filename}에 저장되었습니다.")
    except Exception as e:
        logging.error(f"대화 기록 저장 중 오류 발생: {e}")


# JSON 파일에서 대화 기록 읽기 함수
def load_chat_history_from_json(filename="chat_archive.json"):
    """JSON 파일에서 대화 기록을 로드"""
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                chat_data = json.load(f)
            logging.info(f"대화 기록이 {filename}에서 로드되었습니다.")
            return chat_data.get("conversation", [])
        else:
            return []  # 파일이 없으면 빈 리스트 반환
    except Exception as e:
        logging.error(f"대화 기록 로드 중 오류 발생: {e}")
        return []


# 멀티 체인 프로세스 실행 함수
def multi_chain_process(user_input, is_last_interaction=False):
    global chat_history, recent_history

    # 1단계: 사용자 입력에 대한 응답 생성 및 히스토리 추가
    chat_response = chain1.invoke({"user_input": user_input})
    #chat_response = clean_response(chat_response)
    typing_effect(f"챗봇: {chat_response}")  # 타이핑 효과로 출력

    # 전체 대화 기록 관리
    chat_history["conversation"].append({"사용자": user_input, "챗봇": chat_response})

    # 최근 대화 기록 관리 (5개까지만 유지)
    recent_history.append({"사용자": user_input, "챗봇": chat_response})
    if len(recent_history) > 5:
        recent_history.pop(0)

    # JSON에 기록 저장
    save_chat_history_to_json(chat_history)

    # 마무리 멘트 추가 (마지막 사용자 입력 시)
    if is_last_interaction:
        typing_effect("오늘 대화는 여기까지네요. 대화를 통해 사용자님에 대해 더 많이 알게 되어 즐거웠습니다. 다음에 또 대화 나눠요! 😊")

    # 성향 분석은 마지막 입력 이후 별도로 진행
    if is_last_interaction:
        all_history = load_chat_history_from_json()
        combined_history = json.dumps(all_history, ensure_ascii=False)
        summary = chain2.invoke({"chat_history": combined_history})
        typing_effect(f"요약: {summary}")

        # Big Five 분석 수행
        big_five_result = chain3.invoke({"summary": summary})
        typing_effect("Big Five 성향 분석 결과:")
        typing_effect(big_five_result)

        return big_five_result

    return chat_response

# 예제 실행
if __name__ == "__main__":
    # 챗봇 초기 메시지
    initial_message = "안녕하세요! 요즘 어떻게 지내세요?"
    typing_effect(f"챗봇: {initial_message}")
    recent_history.append({"사용자": "", "챗봇": initial_message})

    # 사용자 입력 및 대화 진행
    for i in range(10):
        user_input = input("사용자 입력: ")
        if i == 9:  # 마지막 대화 시 마무리 멘트 추가
            multi_chain_process(user_input, is_last_interaction=True)
            break
        else:
            multi_chain_process(user_input)
