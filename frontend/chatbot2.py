import gradio as gr

# 기본 증상 목록
symptoms = [
    "따끔해요", "피가 나요", "멍이 생겼어요", "저려요", "부어올라요", "욱신거려요", "가려워요",
    "열나요", "통증이 있어요", "염증이 생겼어요", "답답해요", "아프고 붓습니다", "건조해요",
    "쑤셔요", "찌릿해요", "무거워요", "뻐근해요", "붓고 있어요", "뜨거워요", "딱딱해졌어요",
    "물집이 생겼어요", "따가워요", "붉어졌어요", "시큰거려요", "아파요", "쓰려요", "얼얼해요", "감각이 둔해요"
]

exclude_symptoms = {
    "목": ["무거워요", "저려요", "쑤셔요", "딱딱해졌어요", "얼얼해요", "감각이 둔해요"],
    "손": ["열나요", "무거워요", "쑤셔요"],
    "팔": ["따끔해요", "열나요", "답답해요"],
    "다리": ["따끔해요", "무거워요", "답답해요", "따가워요"],
    "어깨": ["따끔해요", "따가워요", "열나요", "무거워요", "감각이 둔해요"],
    "허리": ["따끔해요", "저려요", "부어올라요", "열나요", "무거워요", "답답해요", "쓰려요"],
    "머리": ["저려요", "물집이 생겼어요", "따끔해요", "염증이 생겼어요", "건조해요", "쑤셔요", "쓰려요", "딱딱해졌어요", "답답해요", "얼얼해요", "감각이 둔해요"],
    "발": ["무거워요", "답답해요"],
    "손목": ["답답해요"],
    "발목": ["열나요", "저리네요", "답답해요"],
    "무릎": ["딱딱해졌어요", "무거워요", "답답해요"],
    "가슴": ["따끔해요", "쑤셔요", "저려요", "무거워요", "쓰려요", "얼얼해요", "감각이 둔해요"],
    "배": ["따끔해요", "피가 나요", "멍이 생겼어요", "답답해요", "저려요", "염증이 생겼어요", "무거워요", "쓰려요", "얼얼해요", "감각이 둔해요"]
}

# 특정 부위 전용 증상
exclusive_symptoms = {
    "목": ["잠겨요", "칼칼해요", "갈라져요", "이물감이 느껴져요", "부었어요"],
    "팔": ["힘이 없어요", "발진이 생겨요"],
    "머리": ["두통이 있어요", "어지러워요", "지끈거려요"]
}

# 상태 초기화 함수
def initialize_state():
    return {
        "step": "ask_start",
        "body_part": None,
        "symptoms": [],
        "chat_history": [],
        "placeholder": "부위를 입력해주세요."
    }

def conversation(input_text, state):
    # "처음" 입력 시 상태 초기화
    if input_text.lower() == "처음":
        state = initialize_state()
        body_parts = ["목", "손", "팔", "다리", "어깨", "허리", "머리", "발", "손목", "발목", "무릎", "가슴", "배"]
        response = f"안녕하세요. 어디가 아프신가요? 다음 부위 중 선택하여 적어주세요 : {', '.join(body_parts)} 등"
        state["step"] = "ask_body_part"
        state["chat_history"].append((input_text, response))
        return state["chat_history"], state, gr.update(placeholder="부위를 입력해주세요.", value="")

    if state["step"] == "ask_start":
        if input_text.lower() == "시작":
            body_parts = ["목", "손", "팔", "다리", "어깨", "허리", "머리", "발", "손목", "발목", "무릎", "가슴", "배"]
            response = f"안녕하세요. 어디가 아프신가요? 다음 부위 중 선택하여 적어주세요 : {', '.join(body_parts)} 등"
            state["step"] = "ask_body_part"
            state["chat_history"].append(("시작", response))
            return state["chat_history"], state, gr.update(placeholder="부위를 입력해주세요.", value="")

        response = "'시작'을 입력해주세요."
        state["chat_history"].append((input_text, response))
        return state["chat_history"], state, gr.update(placeholder="'시작'을 입력해주세요.", value="")

    elif state["step"] == "ask_body_part":
        state["body_part"] = input_text
        state["step"] = "ask_symptoms"

        filtered_symptoms = [
            symptom for symptom in symptoms
            if symptom not in exclude_symptoms.get(input_text, [])
        ]
        filtered_symptoms += exclusive_symptoms.get(input_text, [])

        response = f"{input_text} 부위가 선택되었습니다. 다음 증상 중 선택해주세요 : {', '.join(filtered_symptoms)} 등"
        state["chat_history"].append((input_text, response))
        return state["chat_history"], state, gr.update(placeholder="증상을 입력해주세요.", value="")

    elif state["step"] == "ask_symptoms":
        if input_text.lower() in ["끝", "완료"]:
            sentence = build_sentences(state["body_part"], state["symptoms"])
            state["step"] = "get_recommendation"
            state["placeholder"] = "약 추천 중"
            response = f"선택한 증상 : {sentence}. 이에 대한 약을 추천하겠습니다."
            state["chat_history"].append((input_text, response))
            return state["chat_history"], state, gr.update(placeholder="약 추천 중", value="")

        state["symptoms"].append(input_text)
        response = f"'{input_text}' 증상을 추가했습니다. 더 추가할 증상이 있나요? 완료하려면 '끝'이라고 입력하세요."
        state["chat_history"].append((input_text, response))
        return state["chat_history"], state, gr.update(placeholder="증상을 입력해주세요.", value="")

    elif state["step"] == "get_recommendation":
        recommendation = f"추천 약은 '{state['body_part']}의 증상에 맞는 약 3가지'입니다."
        response = f"추천 약 : {recommendation}"
        state["step"] = "completed"
        state["chat_history"].append((input_text, response))
        return state["chat_history"], state, gr.update(placeholder="약 추천 중", value="")

    response = "잘 이해하지 못했습니다. 다시 입력해주세요."
    state["chat_history"].append((input_text, response))
    return state["chat_history"], state, gr.update(placeholder="어느 부위가 아프신가요?", value="")

def build_sentences(body_part, selected_symptoms):
    if not selected_symptoms:
        return "증상을 선택해주세요."

    sentences = []
    symptoms_with_location_particle = [
        "염증이 생겼어요", "통증이 있어요", "열나요", "멍이 생겼어요", "열감이 있어요", "물집이 생겼어요",
        "이물감이 느껴져요", "두통이 있어요", "힘이 없어요", "발진이 생겨요"
    ]
    for symptom in selected_symptoms:
        particle = "에" if symptom in symptoms_with_location_particle else ("이" if (ord(body_part[-1]) - 44032) % 28 != 0 else "가")
        sentences.append(f"{body_part}{particle} {symptom}")

    return " 그리고 ".join(sentences)

def get_recommendation(sentence):
    return f"'{sentence}'에 맞는 약 3가지를 추천합니다."

# Gradio 인터페이스 구성
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=700)
    state = gr.State(initialize_state())
    user_input = gr.Textbox(label="사용자 입력", placeholder="'시작'을 입력해주세요.")

    user_input.submit(conversation, inputs=[user_input, state], outputs=[chatbot, state, user_input])

if __name__ == "__main__":
    demo.launch()