import os

# from dotenv import load_dotenv
# print(load_dotenv()) # 환경변수 값 로딩
# # OpenAIEmbeddings() 유료 임베딩으로 필요
# os.environ['OPENAI_API_KEY']  = os.getenv('OPENAI_API_KEY')


from langchain_community.document_loaders.csv_loader import CSVLoader
loader = CSVLoader(file_path="data/medicine_update.csv", encoding='UTF-8')
persist_directory = "chroma_db"
docs = loader.load()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter  # text 분할

# HuggingFaceEmbeddings 모델 활용 임베딩 클래스 초기화
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
# 사전 학습된 sentence-transformers 지원 모델
# https://www.sbert.net/docs/pretrained_models.html 참조
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 단계 3: 분할 Documents 임베딩화 및 vectorstore 저장
from langchain_community.vectorstores.chroma import Chroma

def process_in_batches(documents, embeddings, persist_directory, batch_size=5000):
    if not(os.path.exists(persist_directory)):
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
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        return vector_store

vector_store = process_in_batches(splits, embeddings, persist_directory)

# 단계 3: 검색(retriever) 객체 생성
retriever = vector_store.as_retriever()

# 단계 4: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
from langchain.prompts.prompt import PromptTemplate

request_prompt = PromptTemplate(
    template="""아래의 증상과 맥락을 바탕으로 어떤 질병인지 예측하여 해당 질병의 증상을 파악해주세요.
    해당 증상을 바탕으로 효능 부분을 참고해 가장 적절한 약품의 '제품명' 컬럼에서 서로 다른 제품명을 3가지 추천해주세요.

Context: {context}

증상: {question}

다음 형식으로 답변해주세요:
1. [제품명] - [주요 효능]
2. [제품명] - [주요 효능]
3. [제품명] - [주요 효능]
""",
    input_variables=['context', 'question']
)
# # 단계 5: 언어모델 생성(Create LLM)
# # 모델(LLM) 을 생성합니다
from langchain_community.llms import Ollama

chat_llm = Ollama(
    base_url='http://localhost:11434',
    model = 'kollama3' # 명령 메시지에 따라 약 2~5분 정도 추론 소요
    #model = 'mistral', # 한국어보다 영어로 질문 입력
    #model = 'EEVE-Korean'  # 한국어 학습 잘된 모델
)

def format_ret_docs(ret_docs):
    # 질문 query를 바탕으로 retriever가 유사 문장을 검색한 문서 결과(page_content)를
    # 하나의 문단으로 합쳐주는 함수 구현
    return "\n\n".join(doc.page_content for doc in ret_docs)

# 단계 6: 구성요소를 결합하는 체인 생성 / LCEL방식
rag_chain = (
    {'context':retriever | format_ret_docs, 'question': RunnablePassthrough()}
    | request_prompt
    | chat_llm
    | StrOutputParser()
)

# 단계 7: 체인 실행(Run Chain)
question = "내가 지금 근육통이 너무 심해. 근육통을 완화하고 싶은데 csv 파일을 랜덤하게 섞고 효능을 참조해서 나에게 제품명을 3가지 추천해줘"
response = rag_chain.invoke(question)
print(response)