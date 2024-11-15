import gradio as gr
from konlpy.tag import Kkma
import os
from langchain.docstore.document import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
import pandas as pd


os.environ['OPENAI_API_KEY'] = "API_KEY"

def initialize_rag():
    # CSV 파일 경로
    csv_file_path = "src/simpleMedicine.csv"
    
    # embedding 저장 경로
    persist_directory = "chroma_db"
    
    # 단계 3: 임베딩 초기화 (항상 정의)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device':'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 이미 저장된 DB가 있는지 확인
    if not os.path.exists(persist_directory):
        # 단계 1: CSV 데이터를 Pandas로 로드
        df = pd.read_csv(csv_file_path, encoding='UTF-8')
            
        # CSV 데이터를 LangChain 문서 형태로 변환
        docs = [
            Document(page_content=row['효능'], metadata={"제품명": row['제품명']})
            for _, row in df.iterrows()
        ]
        
        # 단계 2: 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        splits = text_splitter.split_documents(docs)
        
        # 벡터 저장소 생성
        vector_store = Chroma.from_documents(
            splits,
            embeddings,
            persist_directory=persist_directory
        )
        vector_store.persist()

    else:
        # 이미 존재하는 DB 로드
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function= embeddings
        )
    
    # 단계 4: 검색기 설정
    retriever =  vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # 단계 5: 프롬프트 정의
    request_prompt = PromptTemplate(
        template=
        """
        아래의 증상과 맥락을 바탕으로 어떤 질병인지 예측하여 해당 질병의 증상을 파악해주세요.
        해당 증상을 바탕으로 rag 데이터 베이스의 csv 파일을 내의 효능 부분을 참고해 가장 적절한 약품의 '제품명' 컬럼에서 제품명을 3가지 추천해주세요.
        추천 시 약의 이름과 효능만 제공해주세요.

        Context: {context}

        증상: {question}

        다음 형식으로 답변해주세요:
        1. [제품명] - [주요 효능]
        2. [제품명] - [주요 효능]
        3. [제품명] - [주요 효능]
        """,
        input_variables=['context', 'question']
    )
    
    # 단계 6: LLM 생성
    chat_llm = ChatOpenAI(model_name="gpt-4o-2024-05-13", temperature=0)
    
    # 단계 7: 체인 생성
    def format_ret_docs(ret_docs):
        print("Retrieved Documents:", ret_docs) #testing용 코드
        return "\n\n".join(doc.page_content for doc in ret_docs)
    
    rag_chain = (
        {'context': retriever | format_ret_docs, 'question': RunnablePassthrough()}
        | request_prompt
        | chat_llm
        | StrOutputParser()
    )
    
    return rag_chain


def initialize_state():
    state = {
        "chat_history": []
    }
    # 초기 메시지 설정
    initial_message = "안녕하세요. 어디가 아프신가요?"

    # 초기 메시지를 대화 내역에 추가
    state["chat_history"].append((None, initial_message))

    return state


def get_recommendation(sentence):
    if not sentence or sentence == "증상을 말씀해주세요.":
        return "올바른 증상을 말씀해주세요."
    
    rag_chain = initialize_rag()
    
    question = f"사용자가 선택한 증상은 '{sentence}'입니다. 의심되는 증상을 한 단어로 표현하세요."\
                "그 다음 증상과 관련된 약을 RAG 데이터베이스에 올려져있는 데이터로 추천해주세요." \
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
        response = "시스템을 다시 시작하겠습니다.\n안녕하세요. 어디가 아프신가요?"
        chat_history.append((None, response))
        return response, state, chat_history

    # 문장을 그대로 RAG 모델에 전달
    sentence = message.strip()
    print("for debugging:", sentence) #testing용 코드
    response = f"입력된 증상: {sentence}. 이에 대한 약을 추천하겠습니다."

    # 증상에 따른 약 추천 호출
    recommendation = get_recommendation(sentence)
    response += "\n" + recommendation + "\n\n 처음으로 돌아가고 싶으시다면 '처음'을 입력해주세요."

    chat_history.append((message, response))
    return response, state, chat_history



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