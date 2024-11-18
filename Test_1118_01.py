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

# 데이터 로딩 및 벡터 저장소 설정
persist_directory = "chroma_db"
loader = CSVLoader(file_path="/home/minsu/coding/medicine_update.csv", encoding='UTF-8')
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
retriever = vector_store.as_retriever()

# Ollama 모델 초기화
chat_llm = Ollama(
    base_url='http://localhost:11434',
    model='kollama3'
)

# 프롬프트 템플릿
request_prompt = PromptTemplate(
    template="""사용자의 증상을 분석하고 적절한 약품을 추천해주세요.
    증상: {question}
    
    아래의 맥락을 참고하여 가장 적합한 약품 3가지를 추천해주세요:
    맥락: {context}
    
    다음 형식으로 답변해주세요:
    
    [분석된 증상]
    사용자의 증상에 대한 간단한 분석
    
    [추천 약품]
    1. [제품명] - [주요 효능]
    2. [제품명] - [주요 효능]
    3. [제품명] - [주요 효능]
    
    [주의사항]
    약품 복용 시 주의해야 할 점들
    """,
    input_variables=['context', 'question']
)

def format_ret_docs(ret_docs):
    return "\n\n".join(doc.page_content for doc in ret_docs)

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
    #chatbot { 
        background-color: #f8fafc;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .message { font-size: 1.1rem !important; }
    .input-box { 
        border-radius: 8px !important;
        border: 1px solid #e2e8f0 !important;
        padding: 12px !important;
        font-size: 1rem !important;
    }
""") as demo:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1 style="font-size: 2.8rem; font-weight: 700; color: #2d5ca9; margin-bottom: 1rem;">
                🏥 AI 헬스케어 어시스턴트
            </h1>
            <p style="font-size: 1.3rem; color: #4a5568; line-height: 1.6;">
                증상을 자세히 말씀해 주시면 적절한 약품을 추천해 드립니다.
            </p>
        </div>
        """
    )
    
    chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=("/home/minsu/coding/user.png", "/home/minsu/coding/bot.png"),
        height=600,
    )

    with gr.Row():
        with gr.Column(scale=9):
            msg = gr.Textbox(
                placeholder="증상을 자세히 입력해 주세요...",
                show_label=False,
                container=False,
                elem_classes="input-box",
                min_width=600,
            )
        with gr.Column(scale=1, min_width=100):
            send_btn = gr.Button(
                "전송",
                variant="primary"
            )

    with gr.Row():
        clear_btn = gr.Button(
            "새로운 상담 시작",
            variant="secondary",
            size="sm"
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
        server_name="0.0.0.0",
        show_error=True,
        favicon_path="/home/minsu/coding/user.png"
    )