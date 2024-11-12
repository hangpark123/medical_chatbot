import openai
import gradio as gr
from konlpy.tag import Kkma

# OpenAI API 키 설정
openai.api_key = ""
# 형태소 분석기 초기화
kkma = Kkma()


def morph(input_data):
    # 형태소 분석을 수행하고 결과를 반환
    return kkma.pos(input_data)


def generate_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # GPT-3.5 모델 사용
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            n=1,  # 응답 1개 생성
            temperature=0.7,
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"오류가 발생했습니다: {e}"


def extract_nouns_and_verbs(text):
    pos_tags = morph(text)
    nouns = [word for word, tag in pos_tags if tag.startswith('N')]
    verbs = [word for word, tag in pos_tags if tag.startswith('V')]
    return nouns, verbs


def chatbot(input_text):
    # 입력된 텍스트에서 명사와 동사 추출
    nouns, verbs = extract_nouns_and_verbs(input_text)

    if nouns and verbs:
        response_text = f"{nouns[0]}(이)가 {verbs[0]}시군요. 이에 대한 약을 추천하겠습니다."
    else:
        response_text = "형태소 분석에 실패했습니다."

    return {"GPT 응답": response_text}


# Gradio 인터페이스 정의
interface = gr.Interface(
    fn=chatbot,
    inputs="text",
    outputs=["json"],  # JSON 형식으로 형태소 분석 결과와 GPT 응답을 반환
    title="약 추천 챗봇입니다",
    description="부위가 어디신지, 함께 상태와 증상을 알려주세요."
)

# 인터페이스 실행
if __name__ == "__main__":
    interface.launch()
