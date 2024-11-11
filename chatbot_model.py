import gradio as gr
import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import Ollama

# 기본 증상 목록
symptoms = [
    "따끔해요", "피가 나요", "멍이 생겼어요", "저려요", "부어올라요", "욱신거려요", "가려워요",
    "열나요", "통증이 있어요", "염증이 생겼어요", "답답해요", "아프고 붓습니다", "건조해요",
    "쑤셔요", "찌릿해요", "무거워요", "뻐근해요", "붓고 있어요", "뜨거워요", "딱딱해졌어요",
    "물집이 생겼어요", "따가워요", "붉어졌어요", "시큰거려요", "아파요", "쓰려요", "얼얼해요", "감각이 둔해요"
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

# 특정 부위 전용 증상
exclusive_symptoms = {
    "목": ["잠겨요", "칼칼해요", "갈라져요", "이물감이 느껴져요", "부었어요"],
    "팔": ["힘이 없어요", "발진이 생겨요"],
    "머리": ["두통이 있어요", "어지러워요", "지끈거려요"]
}


def initialize_rag():
    # 영구 저장소 경로 설정
    persist_directory = "chroma_db"

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 이미 저장된 DB가 있는지 확인
    if not os.path.exists(persist_directory):
        # 처음 실행시에만 CSV 로드 및 임베딩 수행
        loader = CSVLoader(file_path="data/medicine.csv", encoding='UTF-8')
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)

        # 벡터 저장소 생성 및 저장 (배치로 나누어 추가)
        vector_store = Chroma(embedding=embeddings, persist_directory=persist_directory)

        batch_size = 5000  # Chroma의 최대 배치 크기인 5461보다 작은 값으로 설정
        for i in range(0, len(splits), batch_size):
            vector_store.add_texts(splits[i:i + batch_size])

        vector_store.persist()
    else:
        # 이미 존재하는 DB 로드
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

    retriever = vector_store.as_retriever()

    request_prompt = PromptTemplate(
        template="""
        \nQuestion: {question} \nContext: {context} \nAnswer: """,
        input_variables=['context', 'question']
    )

    chat_llm = Ollama(
        base_url="http://localhost:11434",
        model='kollama3'
    )

    def format_ret_docs(ret_docs):
        return "\n\n".join(doc.page_content for doc in ret_docs)

    rag_chain = (
            {'context': retriever | format_ret_docs, 'question': RunnablePassthrough()}
            | request_prompt
            | chat_llm
            | StrOutputParser()
    )

    return rag_chain


def update_symptoms(body_part):
    filtered_symptoms = [symptom for symptom in symptoms if symptom not in exclude_symptoms.get(body_part, [])]
    filtered_symptoms += exclusive_symptoms.get(body_part, [])
    return gr.update(choices=filtered_symptoms, visible=True)


def build_sentences(body_part, selected_symptoms):
    if not selected_symptoms:
        return "증상을 선택해주세요."

    sentences = []
    for symptom in selected_symptoms:
        symptoms_with_location_particle = [
            "염증이 생겼어요", "통증이 있어요", "열나요", "멍이 생겼어요",
            "물집이 생겼어요", "이물감이 느껴져요", "두통이 있어요", "힘이 없어요", "발진이 생겨요"
        ]

        particle = "에" if symptom in symptoms_with_location_particle else (
            "이" if (ord(body_part[-1]) - 44032) % 28 != 0 else "가")
        sentences.append(f"{body_part}{particle} {symptom}")

    return " 그리고 ".join(sentences)


def get_recommendation(sentence):
    if not sentence or sentence == "증상을 선택해주세요.":
        return "올바른 증상을 선택해주세요."
    rag_chain = initialize_rag()
    question = f"{sentence}. 이런 증상들에 맞는 약을 csv파일안에 효능항목을 보고 제품명을 3가지 추천해주세요."
    response = rag_chain.invoke(question)
    return response


# Gradio 인터페이스
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            body_part = gr.Radio(
                ["목", "손", "팔", "다리", "어깨", "허리", "머리", "발", "손목", "발목", "무릎", "가슴", "배"],
                label="부위",
                info="어느 부위가 아프신가요?"
            )
            symptom = gr.Checkboxgroup(  # Radio를 Checkboxgroup으로 변경
                choices=[],
                label="증상",
                info="증상을 여러개 선택할 수 있습니다.",
                visible=False
            )

    with gr.Row():
        sentence_output = gr.Textbox(label="선택한 증상")
        recommendation_output = gr.Textbox(label="약 추천")

    generate_btn = gr.Button("증상 문장 생성")
    recommend_btn = gr.Button("약 추천받기")

    body_part.change(fn=update_symptoms, inputs=body_part, outputs=symptom)
    generate_btn.click(build_sentences, inputs=[body_part, symptom], outputs=sentence_output)
    recommend_btn.click(get_recommendation, inputs=[sentence_output], outputs=recommendation_output)

if __name__ == "__main__":
    demo.launch()
