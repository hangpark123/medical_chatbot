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
from langchain.docstore.document import Document

# 데이터 로딩 및 벡터 저장소 설정
persist_directory = "chroma_db"

# CSV 로더 설정 수정 (메타데이터 포함)
loader = CSVLoader(
    file_path="/home/minsu/coding/medicine_update.csv",
    encoding='UTF-8',
    csv_args={
        'delimiter': ',',
        'quotechar': '"'
    },
    metadata_columns=['제품명', '효능']
)
docs = loader.load()

# 임베딩 설정
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

def process_in_batches(documents, embeddings, persist_directory, batch_size=5000):
    if not os.path.exists(persist_directory):
        vector_store = None
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            if vector_store is None:
                vector_store = Chroma.from_documents(batch, embeddings, persist_directory=persist_directory)
                vector_store.persist()
            else:
                vector_store.add_documents(batch)
        return vector_store
    else:
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

vector_store = process_in_batches(splits, embeddings, persist_directory)

# retriever 설정
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 10
    }
)

# Ollama 모델 초기화
chat_llm = Ollama(
    base_url='http://127.0.0.1:11434',
    model='kollama4'
)

# 프롬프트 템플릿
request_prompt = PromptTemplate(
    template="""당신은 제공된 의약품 데이터베이스의 정보만을 사용하여 정확히 3개의 약품을 추천하는 시스템입니다.

    증상: {question}
    
    약품 데이터베이스:
    {context}
    
    [규칙]
    1. 정확히 3개의 약품만 추천할 것
    2. 데이터베이스에 있는 제품명과 효능만 사용할 것
    3. 제품명과 해당 제품의 실제 효능을 정확히 매칭할 것
    4. 증상과 관련이 있는 약품만 선택할 것
    
    다음 형식으로 답변해주세요:
    
    [분석된 증상]
    사용자의 증상에 대한 간단한 분석
    
    [데이터베이스 기반 추천 약품]
    1. [제품명]
    - 효능: [실제 효능 내용]
    
    2. [제품명]
    - 효능: [실제 효능 내용]
    
    3. [제품명]
    - 효능: [실제 효능 내용]

    [복용 시 주의사항]
    약품 복용 시 주의해야 할 점들
    """,
    input_variables=['context', 'question']
)

def format_ret_docs(ret_docs):
    formatted_docs = []
    for doc in ret_docs:
        try:
            formatted_docs.append(f"제품명: {doc.metadata.get('제품명', '')}\n효능: {doc.metadata.get('효능', '')}")
        except AttributeError:
            continue
    return "\n\n".join(formatted_docs)

# RAG 체인 설정
rag_chain = (
    {'context': retriever | format_ret_docs, 'question': RunnablePassthrough()}
    | request_prompt
    | chat_llm
    | StrOutputParser()
)

def initialize_state():
    return {
        "chat_history": [(None, "안녕하세요. 어떤 증상이 있으신지 자세히 말씀해 주세요.")]
    }

def get_recommendation(message):
    try:
        response = rag_chain.invoke(message)
        return response
    except Exception as e:
        return f"죄송합니다. 오류가 발생했습니다: {str(e)}"

def conversation(message, state):
    if state is None:
        state = initialize_state()
    
    chat_history = state["chat_history"]
    
    if message.strip().lower() == "처음":
        state = initialize_state()
        return "안녕하세요. 어떤 증상이 있으신지 자세히 말씀해 주세요.", state, state["chat_history"]
    
    response = get_recommendation(message)
    response += "\n\n다른 증상을 문의하시려면 '처음'을 입력해주세요."
    
    chat_history.append((message, response))
    return response, state, chat_history

# Gradio 인터페이스
with gr.Blocks(theme=gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
), css="""
    .container {
        max-width: 1024px;
        margin: auto;
        padding: 0 2rem;
        height: 100vh;  /* 전체 높이 사용 */
    }
    #chatbot { 
        height: calc(100vh - 140px) !important;  /* 채팅창 높이 증가 */
        overflow-y: auto;
        background-color: #ffffff;
        border-radius: 0;
        border: 1px solid #e5e7eb;
        box-shadow: none;
        margin: 0;
        padding: 1.5rem;
        font-size: 1.1rem;
    }
    #chatbot > div {
        padding: 0;
        max-width: 100%;
        min-height: calc(100vh - 200px);  /* 최소 높이 설정 */
    }
    .message {
        padding: 2rem !important;
        border-bottom: 1px solid #e5e7eb !important;
        line-height: 1.7 !important;
        margin: 0.5rem 0 !important;
    }
    .message:last-child {
        border-bottom: none !important;
    }
    .user-message {
        background-color: #ffffff;
    }
    .bot-message {
        background-color: #f7f7f8;
    }
    .input-row {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #ffffff;
        border-top: 1px solid #e5e7eb;
        padding: 1rem;
        z-index: 100;
    }
    .input-container {
        max-width: 1024px;
        margin: auto;
        display: flex;
        gap: 1rem;
        align-items: center;
    }
    .input-box { 
        border: 1px solid #e5e7eb !important;
        border-radius: 0.75rem !important;
        padding: 1rem 1.25rem !important;
        font-size: 1.1rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
        background-color: #ffffff !important;
        min-height: 50px !important;
    }
    .input-box:focus {
        outline: none !important;
        border-color: #2563eb !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1) !important;
    }
    .send-btn, .clear-btn {
        padding: 0.75rem 1.5rem !important;
        border-radius: 0.5rem !important;
        font-size: 1rem !important;
        height: 50px !important;
    }
    .header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        padding: 1rem;
        z-index: 100;
        height: 60px;  /* 헤더 높이 고정 */
    }
    .header-content {
        max-width: 1024px;
        margin: auto;
    }
    .content {
        margin-top: 60px;  /* 헤더 높이와 동일 */
        margin-bottom: 80px;  /* 입력창 높이 고려 */
        min-height: calc(100vh - 140px);  /* 전체 높이에서 헤더와 입력창 높이 제외 */
        display: flex;
        flex-direction: column;
    }
""") as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown(
            """
            <div class="header-content">
                <h1 style="font-size: 1.5rem; font-weight: 600; color: #111827; margin: 0;">
                    🏥 AI 헬스케어 어시스턴트
                </h1>
            </div>
            """,
            elem_classes="header"
        )
        
        with gr.Column(elem_classes="content"):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                bubble_full_width=False,
                avatar_images=("/home/minsu/coding/user.png", "/home/minsu/coding/bot.png"),
                height=None,
            )

        with gr.Column(elem_classes="input-row"):
            with gr.Row(elem_classes="input-container"):
                msg = gr.Textbox(
                    placeholder="증상을 입력하세요...",
                    show_label=False,
                    container=False,
                    elem_classes="input-box",
                    scale=20,
                )
                send_btn = gr.Button(
                    "전송",
                    variant="primary",
                    elem_classes="send-btn",
                    scale=2,
                )
                clear_btn = gr.Button(
                    "새로운 상담",
                    variant="secondary",
                    elem_classes="clear-btn",
                    scale=2,
                )

    state = gr.State(initialize_state())

    def clear_conversation():
        new_state = initialize_state()
        return new_state["chat_history"], new_state

    def handle_message(message, state):
        response, new_state, chat_history = conversation(message, state)
        return "", new_state, chat_history

    clear_btn.click(clear_conversation, [], [chatbot, state])
    send_btn.click(handle_message, [msg, state], [msg, state, chatbot])
    msg.submit(handle_message, [msg, state], [msg, state, chatbot])
    demo.load(lambda: initialize_state()["chat_history"], outputs=[chatbot])

if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=True,
        favicon_path="/home/minsu/coding/user.png"
    )
