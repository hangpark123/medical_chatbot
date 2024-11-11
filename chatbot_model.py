from langchain_community.document_loaders.csv_loader import CSVLoader

# encoding 오류시 encoding='UTF-8' OR 'CP949' 추가
loader = CSVLoader(file_path="data/medicine.csv", encoding='UTF-8')
docs = loader.load()

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter  # text 분할

# HuggingFaceEmbeddings 모델 활용 임베딩 클래스 초기화
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
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
def process_in_batches(documents, embeddings, batch_size=5000):
    """문서를 배치로 나누어 처리하는 함수"""
    vector_store = None
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        if vector_store is None:
            vector_store = Chroma.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)
    return vector_store
# from_documents() : Document항목 list를 임베딩과 동시 vectorstore에 저장 메서드
# Document항목 list가 아닌 text항목 list 경우 from_texts()메서드 활용

vector_store = process_in_batches(splits, embeddings)

# 단계 4: 검색(retriever) 객체 생성
retriever = vector_store.as_retriever()

# 단계 5: 프롬프트 생성(Create Prompt)
from langchain.prompts.prompt import PromptTemplate


request_prompt = PromptTemplate(
template="""
\nQuestion: {question} \nContext: {context} \nAnswer:  """,
input_variables=['context', 'question']
)

# # 단계 6: 언어모델 생성(Create LLM)
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

# 단계 7: 구성요소를 결합하는 체인 생성 / LCEL방식
rag_chain = (
    {'context':retriever | format_ret_docs, 'question': RunnablePassthrough()}
    | request_prompt
    | chat_llm
    | StrOutputParser()
)

# 단계 8: 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
question = "내가 지금 근육통이 너무 심해. 근육통을 완화하고 싶은데 csv 파일을 랜덤하게 섞고 효능을 참조해서 나에게 제품명을 3가지 추천해줘"
response = rag_chain.invoke(question)
print(response)


