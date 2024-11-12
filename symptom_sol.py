from konlpy.tag import Kkma
kkma = Kkma()

def morph(input_data):  # 형태소 분석
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

# 형태소 분석 적용
for symptom in symptoms["공통"]:
    print(morph(symptom))

# print('=='*30)

# 증상 집합 처리
# (exclude_symptoms["목"]) 부분을 수정. ["목"] 값에 user의 input text(부위:noun) 값을 받와아서 유효부위 처리
common_symptoms = set(symptoms["공통"])
excluded_symptoms = set(exclude_symptoms["목"])
exclusive_symptoms_set = set(exclusive_symptoms["목"])

result1 = list(common_symptoms - excluded_symptoms)
result2 = list(result1 + list(exclusive_symptoms_set))

# 업데이트 된 ["공통"]의 부위 값을 도출
print(result2)
