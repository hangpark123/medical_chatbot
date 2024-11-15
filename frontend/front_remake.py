import gradio as gr
from konlpy.tag import Kkma

kkma = Kkma()


def morph(input_data):
    if isinstance(input_data, list):
        input_data = ' '.join(input_data)
    preprocessed = kkma.pos(input_data)
    return preprocessed


def extract_nouns_and_verbs(text):
    pos_tags = morph(text)
    if pos_tags is None:
        return [], []
    nouns = [word for word, tag in pos_tags if tag.startswith('N')]
    verbs = [word for word, tag in pos_tags if tag.startswith('V')]
    return nouns, verbs


# 증상 키워드 매핑
symptom_normalization_map = {
    "아픈 것 같아요": "아파요",
    "아프다": "아파요",
    "쓰라리다": "쓰라려요",
    "따끔거리다": "따끔해요",
    # 필요에 따라 더 추가 가능
}

symptoms = {
    "공통": ["따끔해요", "피가 나요", "멍이 생겼어요", "저려요", "부어올라요", "욱신거려요", "가려워요",
           "열나요", "통증이 있어요", "염증이 생겼어요", "답답해요", "아프고 붓습니다", "건조해요",
           "쑤셔요", "찌릿해요", "무거워요", "뻐근해요", "붓고 있어요", "뜨거워요", "딱딱해졌어요",
           "물집이 생겼어요", "따가워요", "붉어졌어요", "시큰거려요", "아파요", "쓰려요", "얼얼해요", "감각이 둔해요"]
}

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

exclusive_symptoms = {
    "목": ["잠겨요", "칼칼해요", "갈라져요", "이물감이 느껴져요", "부었어요"],
    "팔": ["힘이 없어요", "발진이 생겨요"],
    "머리": ["두통이 있어요", "어지러워요", "지끈거려요"]
}

body_parts = ["목", "손", "팔", "다리", "어깨", "허리", "머리", "발", "손목", "발목", "무릎", "가슴", "배"]


def get_available_symptoms(body_part):
    if body_part not in body_parts:
        return []

    common_symptoms = set(symptoms["공통"])
    excluded = set(exclude_symptoms.get(body_part, []))
    exclusive = set(exclusive_symptoms.get(body_part, []))

    return list(common_symptoms - excluded) + list(exclusive)


def initialize_state():
    return {
        "step": "start",
        "body_part": None,
        "symptoms": [],
        "chat_history": [],
        "symptom_count": 0
    }

def normalize_symptom(text):
    # 형태소 분석으로 동사 또는 형용사를 추출
    _, verbs = extract_nouns_and_verbs(text)
    normalized_symptoms = []
    for verb in verbs:
        # 추출된 단어가 정규화 맵에 있는지 확인 후, 매핑된 증상으로 변환
        if verb in symptom_normalization_map:
            normalized_symptoms.append(symptom_normalization_map[verb])
        else:
            normalized_symptoms.append(verb)  # 매핑이 없으면 원래 형태로 사용
    return normalized_symptoms


def conversation(message, state):
    if state is None:
        state = initialize_state()

    chat_history = state["chat_history"]
    response = ""

    # 시작 단계
    if state["step"] == "start":
        if message.strip().lower() == "시작":
            response = f"안녕하세요. 어디가 아프신가요? 다음 부위 중 선택하여 적어주세요: 목, 손, 팔, 다리, 어깨, 허리, 머리, 발, 손목, 발목, 무릎, 가슴, 배"
            state["step"] = "body_part"
        else:
            response = "'시작'을 입력해주세요."

    # 부위 입력 단계
    elif state["step"] == "body_part":
        nouns, _ = extract_nouns_and_verbs(message)
        if nouns and nouns[0] in body_parts:
            state["body_part"] = nouns[0]
            available_symptoms = get_available_symptoms(nouns[0])
            response = f"'{nouns[0]}' 부위가 선택되었습니다. 다음 증상 중 선택해주세요:\n{', '.join(available_symptoms)}"
            state["step"] = "symptoms"
        else:
            response = f"올바른 부위를 선택해주세요."
#######
    # 증상 입력 단계
    elif state["step"] == "symptoms":
        # normalized_symptoms = normalize_symptom(input_text)

        if message.strip().lower() == "끝":
            if state["symptoms"]:
                symptoms_with_body_part = [f"{state['body_part']}가 {symptom}" for symptom in state["symptoms"]]
                response = f"입력하신 증상들: {', '.join(symptoms_with_body_part)}\n진단 결과를 분석중입니다..."
                chat_history.append((message, response))

                # 진단 결과를 자동으로 표시
                diseases = ["질병1", "질병2", "질병3"]
                medicines = ["약물1", "약물2", "약물3"]
                diagnosis_response = f"추천 진단 결과 (상위 3개):\n{', '.join(diseases)}\n\n추천 약물:\n" + "\n".join(
                    [f"* {medicine}" for medicine in medicines])
                chat_history.append(("시스템", diagnosis_response))

                # 재시작 안내 메시지 추가
                restart_message = "다시 하기를 원하시면 '시작'을 입력해주세요."
                chat_history.append(("시스템", restart_message))

                state["step"] = "complete"
                return restart_message, state, chat_history
            else:
                response = "최소 한 개의 증상을 입력해주세요."
        else:
            if state["symptom_count"] >= 3:
                response = "증상은 최대 3개까지만 입력 가능합니다."
            else:
                available_symptoms = get_available_symptoms(state["body_part"])
                if message.strip() in available_symptoms:
                    state["symptoms"].append(message.strip())
                    state["symptom_count"] += 1
                    response = f"'{message.strip()}' 증상이 추가되었습니다. 더 추가할 증상이 있으면 입력하시고, 없으면 '끝'을 입력해주세요."
                else:
                    normalized_symptoms = normalize_symptom(message.strip())
                    if normalized_symptoms:
                        state["symptoms"].extend(normalized_symptoms)
                        # .join(normalized_symptoms)
                        response = f"{message.strip()} 증상이 추가되었습니다. 더 추가할 증상이 있나요? 완료하려면 '끝'이라고 입력하세요."
                    elif message.strip().lower() == "끝":
                        state["step"] = "processing"
                        response = f"{state['body_part']} 부위에서 {', '.join(state['symptoms'])} 증상을 분석하여 질병명을 확인중입니다..."
                    else:
                        response = "올바른 증상을 입력하거나, '끝'을 입력해주세요."


    # 완료 상태에서의 추가 입력 처리
    elif state["step"] == "complete":
        if message.strip().lower() == "시작":
            state = initialize_state()
            response = f"안녕하세요. 어디가 아프신가요? 다음 부위 중 선택하여 적어주세요: 목, 손, 팔, 다리, 어깨, 허리, 머리, 발, 손목, 발목, 무릎, 가슴, 배"
            state["step"] = "body_part"
        else:
            response = "다시 하기를 원하시면 '시작'을 입력해주세요."

    chat_history.append((message, response))
    return response, state, chat_history


# Gradio 인터페이스 생성
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=("user.png", "assistant.png"),
        height=400
    )
    msg = gr.Textbox(
        placeholder="메시지를 입력하세요...",
        show_label=False,
        lines=1
    )
    state = gr.State(initialize_state())


    def respond(message, state):
        response, new_state, chat_history = conversation(message, state)
        return "", new_state, chat_history


    msg.submit(
        respond,
        [msg, state],
        [msg, state, chatbot]
    )

    demo.launch()