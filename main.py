import gradio as gr
from konlpy.tag import Kkma
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import Ollama
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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


def initialize_rag():
       
    # 영구 저장소 경로 설정
    persist_directory = "chroma_db"
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    # 이미 저장된 DB가 있는지 확인
    if not os.path.exists(persist_directory):
        # 처음 실행시에만 CSV 로드 및 임베딩 수행
        loader = CSVLoader(file_path="src/muchSimple.csv", encoding='UTF-8')
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        # 벡터 저장소 생성 및 저장
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vector_store.persist()
    else:
        # 이미 존재하는 DB 로드
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    
    retriever = vector_store.as_retriever()

    request_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
                Use ONLY the following pieces of retrieved context to answer the question.
                Do NOT use any other knowledge.
                If you don't know the answer based on the context, just say that you don't know.
                Response in Korean only.
                \nQuestion: {question} \nContext: {context} \nAnswer: """,
    input_variables=['context', 'question']
)
    
    chat_llm= ChatOpenAI(model_name="ft:gpt-3.5-turbo-1106:personal::ASgMZKtF",
                         temperature=0)

    def format_ret_docs(ret_docs):
        print("Retrieved Documents:", ret_docs) # 검색된 문서를 출력해서 확인함
        return "\n\n".join(doc.page_content for doc in ret_docs)
    
    rag_chain = (
        {'context': retriever | format_ret_docs, 'question': RunnablePassthrough()}
        | request_prompt
        | chat_llm
        | StrOutputParser()
    )
    
    return rag_chain

def get_available_symptoms(body_part):
    if body_part not in body_parts:
        return []

    common_symptoms = set(symptoms["공통"])
    excluded = set(exclude_symptoms.get(body_part, []))
    exclusive = set(exclusive_symptoms.get(body_part, []))

    return list(common_symptoms - excluded) + list(exclusive)


def initialize_state():
    state = {
        "step": "body_part",
        "body_part": None,
        "symptoms": [],
        "chat_history": [],
        "symptom_count": 0
    }

    # 초기 메시지 설정
    initial_message = "안녕하세요. 어디가 아프신가요?"

    # 초기 메시지를 대화 내역에 추가
    state["chat_history"].append((None, initial_message))

    return state


def normalize_symptom(text):
    _, verbs = extract_nouns_and_verbs(text)
    normalized_symptoms = []
    for verb in verbs:
        if verb in symptom_normalization_map:
            normalized_symptoms.append(symptom_normalization_map[verb])
        else:
            normalized_symptoms.append(verb)
    return normalized_symptoms


def get_recommendation(sentence):
    if not sentence or sentence == "증상을 말씀해주세요.":
        return "올바른 증상을 말씀해주세요."
    
    rag_chain = initialize_rag()
    
    question = f"사용자가 선택한 증상은 '{sentence}'입니다. 의심되는 증상을 한 단어로 표현하세요."\
                "그 다음 증상과 관련된 약을 finetuning 모델에서 학습된 데이터로 추천해주세요." \
                "만약 관련된 약이 없다면, 비슷한 효능이 있는 약을 제안해주세요."
    response = rag_chain.invoke(question)
    return response

def conversation(message, state):
    if state is None:
        state = initialize_state()

    chat_history = state["chat_history"]
    response = ""

    # "처음"을 입력하면 초기 메시지로 돌아가지만, 대화 내역은 유지
    if message.strip().lower() == "처음":
        state["step"] = "body_part"
        response = "안녕하세요. 어디가 아프신가요?"
        chat_history.append((None, response))
        return response, state, chat_history

    # 부위 입력 단계
    if state["step"] == "body_part":
        nouns, _ = extract_nouns_and_verbs(message)
        if nouns and nouns[0] in body_parts:
            state["body_part"] = nouns[0]
            available_symptoms = get_available_symptoms(nouns[0])
            response = f"'{nouns[0]}' 부위가 아프시군요. 증상에 대해 설명해주세요."
            state["step"] = "symptoms"
        else:
            response = "올바른 부위를 입력해주세요."

    # 증상 입력 단계
    elif state["step"] == "symptoms":
        if message.strip().lower() == "끝":
            if state["symptoms"]:
                symptoms_with_body_part = [f"{state['body_part']}가 {symptom}" for symptom in state["symptoms"]]
                sentence = build_sentences(state["body_part"], state["symptoms"])
                response = f"선택한 증상 : {sentence}. 이에 대한 약을 추천하겠습니다."

                # 증상에 따른 약 추천 호출
                recommendation = get_recommendation(sentence)
                response += "\n" + recommendation
                
                chat_history.append((message, response))
                print("RAG Recommendation:", recommendation)
                # 완전 종료
                restart_message = "다시 하기를 원하시면 '처음'을 입력해주세요."
                chat_history.append((None, restart_message))

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
                        response = f"{message.strip()} 증상이 추가되었습니다. 더 추가할 증상이 있나요? 완료하려면 '끝'이라고 입력하세요."
                    else:
                        response = "올바른 증상을 입력하거나, '끝'을 입력해주세요."

    elif state["step"] == "complete":
        print("RAG Recommendation:", recommendation)
        response = "다시 하기를 원하시면 '처음'을 입력해주세요."

    chat_history.append((message, response))
    return response, state, chat_history


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



# Gradio 인터페이스 생성
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=("user.png", "assistant.png"),            # 챗봇 이미지 바꾸기
        height=400 
    )
    msg = gr.Textbox(
        placeholder="메시지를 입력하세요...",
        show_label=False,
        lines=1
    )
    state = gr.State(initialize_state())

    demo.load(lambda: initialize_state()["chat_history"], outputs=[chatbot])

    def respond(message, state):
        response, new_state, chat_history = conversation(message, state)
        return "", new_state, chat_history

    msg.submit(
        respond,
        [msg, state],
        [msg, state, chatbot]
    )

    demo.launch()