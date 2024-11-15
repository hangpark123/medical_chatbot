import os
import pandas as pd
from langchain.docstore.document import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

# OpenAI API Key 설정
os.environ['OPENAI_API_KEY'] = "API_KEY"

# CSV 파일 경로
csv_file_path = "data/medicine_update.csv"
persist_dir = "chroma_db"

# 단계 1: CSV 데이터를 Pandas로 로드
df = pd.read_csv(csv_file_path, encoding='utf-8')

# CSV 데이터를 LangChain 문서 형태로 변환
docs = [
    Document(page_content=row['효능'], metadata={"제품명": row['제품명']})
    for _, row in df.iterrows()
]

# 단계 2: 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
splits = text_splitter.split_documents(docs)

# 단계 3: 임베딩 초기화 및 벡터 저장소 생성
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)


def create_vector_store(documents, embeddings, persist_directory):
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        vector_store = Chroma.from_documents(documents, embeddings, persist_directory=persist_directory)
        vector_store.persist()
    else:
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vector_store


# 모든 문서를 한 번에 추가하여 Vector Store 생성
vector_store = create_vector_store(splits, embeddings, persist_dir)

# 단계 4: 검색기 설정
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# 단계 5: 프롬프트 정의
request_prompt = PromptTemplate(
    template="""아래의 증상과 맥락을 바탕으로 어떤 질병인지 예측하여 해당 질병의 증상을 파악해주세요.
    해당 증상을 바탕으로 csv 파일을 랜덤하게 셔플한 후 효능 부분을 참고해 가장 적절한 약품의 '제품명' 컬럼에서 제품명을 3가지 추천해주세요.

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
    return "\n\n".join(doc.page_content for doc in ret_docs)


rag_chain = (
        {'context': retriever | format_ret_docs, 'question': RunnablePassthrough()}
        | request_prompt
        | chat_llm
        | StrOutputParser()
)

# 단계 8: 실행 및 테스트
question = "목이 따끔해요 그리고 목이 칼칼해요"
response = rag_chain.invoke(question)

print("추천 결과:")
print(response)
