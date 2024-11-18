import os
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType

os.environ['OPENAI_API_KEY'] = "sk-svcacct-Q77qk6FjT99qZpKYNEmm72Dr7N28CSt9QO0WYtfqgO3XyQJtFildaVBnYaQhs-uj0T3BlbkFJ2h191cVCfPlwajbvNHvkZkIpgUFm7Mgd8x8OmyctJaAJ7KyMB9kCHDaYN1rXxgBAA"

# CSV 파일 경로
csv_file_path = "data/medicine_update.csv"

# CSV 데이터 로드
df = pd.read_csv(csv_file_path, encoding='utf-8')

# 필수 컬럼 확인
required_columns = ['제품명', '효능']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"CSV 파일에 필수 컬럼 {required_columns} 중 일부가 없습니다.")

# 에이전트 생성
agent = create_pandas_dataframe_agent(
    ChatOpenAI(temperature=0, model='gpt-4'),  # 모델 정의
    df,  # 데이터프레임
    verbose=True,  # 추론과정 출력
    agent_type=AgentType.OPENAI_FUNCTIONS,  # 최신 에이전트 타입
    allow_dangerous_code=True,  # 위험한 코드를 허용
)

# 질문
question = "열이 37도정도 되고 목이 아프고 콧물이 많이나요"
prompt = f"""
{question}
라는 질문을 보고 증상을 파악하여, csv 파일의 효능 부분을 참고해
가장 적절한 약품을 '제품명' 컬럼에서 서로 다르게 3가지 추천해주세요.
답변은 반드시 한국어로 해주세요.
제품은 반드시 csv 파일 안에 있는 제품으로 추천해주고, 제품명의 왜곡 없이 출력해주세요.

다음 형식으로 답변해주세요:
1. [제품명] - [주요 효능]
2. [제품명] - [주요 효능]
3. [제품명] - [주요 효능]
"""

# 에이전트 실행
response = agent.run(prompt)
print("추천 결과:")
print(response)

