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

os.environ["OPENAI_API_KEY"] = "sk-svcacct-Q77qk6FjT99qZpKYNEmm72Dr7N28CSt9QO0WYtfqgO3XyQJtFildaVBnYaQhs-uj0T3BlbkFJ2h191cVCfPlwajbvNHvkZkIpgUFm7Mgd8x8OmyctJaAJ7KyMB9kCHDaYN1rXxgBAA"


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
        response = "안녕하세요. 어디가 아프신가요?"
        chat_history.append((None, response))
        return response, state, chat_history

    # 문장을 그대로 RAG 모델에 전달
    sentence = message.strip()
    print("for debugging:", sentence) #testing용 코드
    response = f"입력된 증상: {sentence}. 이에 대한 약을 추천하겠습니다."

    # 증상에 따른 약 추천 호출
    recommendation = get_recommendation(sentence)
    response += "\n" + recommendation

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