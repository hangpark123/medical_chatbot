import gradio as gr


# 기본 증상 목록
symptoms = [
    "따끔해요", "피가 나요", "멍이 생겼어요", "저려요", "부어올라요", "욱신거려요", "가려워요",
    "열나요", "통증이 있어요", "염증이 생겼어요", "답답해요", "아프고 붓습니다", "건조해요",
    "쑤셔요", "찌릿해요", "무거워요", "뻐근해요", "붓고 있어요", "뜨거워요", "딱딱해졌어요",
    "물집이 생겼어요", "따가워요", "붉어졌어요", "시큰거려요", "아파요", "쓰려요", "얼얼해요", "감각이 둔해요",
]


# 제외할 증상
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


# 특정 부위에만 나타나는 증상
exclusive_symptoms = {
    "목": ["잠겨요", "칼칼해요", "갈라져요", "이물감이 느껴져요", "부었어요"],
    "팔": ["힘이 없어요", "발진이 생겨요"],
    "머리": ["두통이 있어요", "어지러워요", "지끈거려요"],
}


def update_symptoms(body_part):
    filtered_symptoms = [symptom for symptom in symptoms if symptom not in exclude_symptoms.get(body_part, [])]

    filtered_symptoms += exclusive_symptoms.get(body_part, [])
    return gr.update(choices=filtered_symptoms, visible=True)


def sentence_builder(body_part, symptom):
    symptoms_with_location_particle = [
        "염증이 생겼어요", "통증이 있어요", "열나요", "멍이 생겼어요", "열감이 있어요", "물집이 생겼어요",
        "이물감이 느껴져요", "두통이 있어요", "힘이 없어요", "발진이 생겨요"
    ]

    if symptom in symptoms_with_location_particle:
        particle = "에"
    else:
        particle = "이" if (ord(body_part[-1]) - 44032) % 28 != 0 else "가"
    return f"{body_part}{particle} {symptom}"


# 인터페이스
with gr.Blocks() as demo:
    body_part = gr.Radio(
        ["목", "손", "팔", "다리", "어깨", "허리", "머리", "발", "손목", "발목", "무릎", "가슴", "배"],
        label="부위",
        info="어느 부위가 아프신가요?"
    )
    symptom = gr.Radio(
        choices=[],
        label="증상",
        info="증상이 어떤가요?",
        visible=False
    )

    body_part.change(fn=update_symptoms, inputs=body_part, outputs=symptom)

    output = gr.Textbox()
    button = gr.Button("문장 생성")
    button.click(sentence_builder, inputs=[body_part, symptom], outputs=output)


if __name__ == "__main__":
    demo.launch()
