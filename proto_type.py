import gradio as gr


def sentence_builder(body_part, symptom):
    # "에"가 자연스러운 증상 목록 설정
    symptoms_with_location_particle = [
        "염증이 생겼어요", "통증이 있어요", "열나요", "멍이 생겼어요", "열감이 있어요", "물집이 생겼어요"
    ]

    # 증상에 따라 조사 설정
    if symptom in symptoms_with_location_particle:
        particle = "에"
    else:
        # 한글의 마지막 글자를 유니코드 값으로 판별하여 받침 여부 확인
        if (ord(body_part[-1]) - 44032) % 28 != 0:
            particle = "이"
        else:
            particle = "가"
    print(f"{body_part}{particle} {symptom}")
    return f"{body_part}{particle} {symptom}"


demo = gr.Interface(
    sentence_builder,
    [
        gr.Radio(
            ["목", "손", "팔", "다리", "어깨", "허리", "머리", "발", "손목", "발목", "무릎", "가슴", "배"],
            label="부위",
            info="어느 부위가 아프신가요?"
        ),
        gr.Radio(
            ["따끔해요", "피가 나요", "멍이 생겼어요", "저려요", "부어올라요", "욱씬거려요", "가려워요",
             "열나요", "통증이 있어요", "염증이 생겼어요", "아프고 붓습니다", "건조해요", "붉게 변했어요",
             "쑤셔요", "찌릿해요", "무거워요", "뻐근해요", "붓고 있어요", "열감이 있어요", "딱딱해졌어요",
             "물집이 생겼어요", "따가워요", "붉어졌어요", "건조하고 가려워요", "시큰거려요", "저리네요", "답답해요"
             ],
            label="증상",
            info="증상이 어떤가요?"
        ),
    ],
    "text",
    examples=[
        ["목", "따끔해요"],
        ["손", "저려요"],
        ["발목", "부어올라요"],
        ["허리", "염증이 생겼어요"],
        ["다리", "쑤셔요"],
        ["머리", "무거워요"],
    ]
)

if __name__ == "__main__":
    demo.launch()
