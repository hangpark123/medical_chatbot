import gradio as gr
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import Ollama
from pydantic import BaseModel, Field
from typing import List

class MedicineRecommendation(BaseModel):
    recommendations: List[dict] = Field(
        description="List of recommended medicines with their details"
    )

@dataclass
class SymptomData:
    """증상 관련 데이터를 관리하는 클래스"""
    base_symptoms: List[str]
    exclude_symptoms: Dict[str, List[str]]
    exclusive_symptoms: Dict[str, List[str]]
    location_particle_symptoms: List[str]

class MedicalRecommendationSystem:
    """의료 추천 시스템의 핵심 로직을 관리하는 클래스"""
    
    def __init__(self, csv_path: str, persist_directory: str = "chroma_db"):
        self.persist_directory = persist_directory
        self.csv_path = csv_path
        self.symptom_data = self._initialize_symptom_data()
        self.rag_chain = self._initialize_rag()

    def _initialize_symptom_data(self) -> SymptomData:
        """증상 관련 데이터 초기화"""
        return SymptomData(
            base_symptoms=[
                "따끔해요", "피가 나요", "멍이 생겼어요", "저려요", "부어올라요", 
                "욱신거려요", "가려워요", "열나요", "통증이 있어요", "염증이 생겼어요",
                "답답해요", "아프고 붓습니다", "건조해요", "쑤셔요", "찌릿해요",
                "무거워요", "뻐근해요", "붓고 있어요", "뜨거워요", "딱딱해졌어요",
                "물집이 생겼어요", "따가워요", "붉어졌어요", "시큰거려요", "아파요",
                "쓰려요", "얼얼해요", "감각이 둔해요"
            ],
            exclude_symptoms={
                "목": ["무거워요", "저려요", "쑤셔요", "딱딱해졌어요", "얼얼해요", "감각이 둔해요"],
                "손": ["열나요", "무거워요", "쑤셔요"],
                "팔": ["따끔해요", "열나요", "답답해요"],
                "다리": ["따끔해요", "무거워요", "답답해요", "따가워요"],
                "어깨": ["따끔해요", "따가워요", "열나요", "무거워요", "감각이 둔해요"],
                "허리": ["따끔해요", "저려요", "부어올라요", "열나요", "무거워요", "답답해요", "쓰려요"],
                "머리": ["저려요", "물집이 생겼어요", "따끔해요", "염증이 생겼어요", "건조해요", 
                      "쑤셔요", "쓰려요", "딱딱해졌어요", "답답해요", "얼얼해요", "감각이 둔해요"],
                "발": ["무거워요", "답답해요"],
                "손목": ["답답해요"],
                "발목": ["열나요", "저리네요", "답답해요"],
                "무릎": ["딱딱해졌어요", "무거워요", "답답해요"],
                "가슴": ["따끔해요", "쑤셔요", "저려요", "무거워요", "쓰려요", "얼얼해요", "감각이 둔해요"],
                "배": ["따끔해요", "피가 나요", "멍이 생겼어요", "답답해요", "저려요", 
                     "염증이 생겼어요", "무거워요", "쓰려요", "얼얼해요", "감각이 둔해요"]
            },
            exclusive_symptoms={
                "목": ["잠겨요", "칼칼해요", "갈라져요", "이물감이 느껴져요", "부었어요"],
                "팔": ["힘이 없어요", "발진이 생겨요"],
                "머리": ["두통이 있어요", "어지러워요", "지끈거려요"]
            },
            location_particle_symptoms=[
                "염증이 생겼어요", "통증이 있어요", "열나요", "멍이 생겼어요",
                "물집이 생겼어요", "이물감이 느껴져요", "두통이 있어요", 
                "힘이 없어요", "발진이 생겨요"
            ]
        )

    def _initialize_rag(self):
        """RAG 시스템 초기화"""
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        if not os.path.exists(self.persist_directory):
            loader = CSVLoader(file_path=self.csv_path, encoding='UTF-8')
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)
            
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=self.persist_directory
            )
            vector_store.persist()
        else:
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings
            )
        
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}  # 관련성 높은 상위 5개 문서 검색
        )
        
        template = """당신은 전문 약사입니다. 환자의 증상을 듣고 가장 적절한 약품을 추천해주세요.

환자의 증상: {question}

참고할 약품 정보: {context}

다음 형식으로 답변해주세요:

[추천 약품 1]
- 약품명: (약품 이름)
- 추천 이유: (이 약품을 추천하는 구체적인 이유)
- 주요 효능: (주요 효과)
- 주의사항: (복용 시 주의할 점)

[추천 약품 2]
- 약품명: (약품 이름)
- 추천 이유: (이 약품을 추천하는 구체적인 이유)
- 주요 효능: (주요 효과)
- 주의사항: (복용 시 주의할 점)

[추천 약품 3]
- 약품명: (약품 이름)
- 추천 이유: (이 약품을 추천하는 구체적인 이유)
- 주요 효능: (주요 효과)
- 주의사항: (복용 시 주의할 점)

각 약품은 환자의 증상과 직접적으로 관련이 있어야 하며, 부작용과 주의사항도 반드시 설명해주세요.
"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=['context', 'question']
        )
        
        llm = Ollama(
            base_url="http://localhost:11434",
            model='kollama3',
            temperature=0.3  # 일관된 출력을 위해 낮은 temperature 설정
        )
        
        return (
            {'context': retriever | (lambda x: "\n\n".join(doc.page_content for doc in x)), 
             'question': RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def get_filtered_symptoms(self, body_part: str) -> List[str]:
        """특정 부위에 해당하는 증상 목록 필터링"""
        filtered = [
            symptom for symptom in self.symptom_data.base_symptoms 
            if symptom not in self.symptom_data.exclude_symptoms.get(body_part, [])
        ]
        filtered.extend(self.symptom_data.exclusive_symptoms.get(body_part, []))
        return filtered

    def build_symptom_sentence(self, body_part: str, selected_symptoms: List[str]) -> str:
        """선택된 증상들을 자연스러운 문장으로 변환"""
        if not selected_symptoms:
            return "증상을 선택해주세요."
        
        sentences = []
        for symptom in selected_symptoms:
            particle = "에" if symptom in self.symptom_data.location_particle_symptoms else (
                "이" if (ord(body_part[-1]) - 44032) % 28 != 0 else "가"
            )
            sentences.append(f"{body_part}{particle} {symptom}")
        
        return " 그리고 ".join(sentences)

    def get_recommendation(self, symptoms_sentence: str) -> str:
        """증상 문장을 기반으로 약품 추천"""
        if not symptoms_sentence or symptoms_sentence == "증상을 선택해주세요.":
            return "올바른 증상을 선택해주세요."
        
        question = f"{symptoms_sentence}. 이러한 증상들에 대해 적절한 약품을 추천해주세요."
        return self.rag_chain.invoke(question)

class MedicalRecommendationUI:
    """의료 추천 시스템의 UI를 관리하는 클래스"""
    
    def __init__(self, recommendation_system: MedicalRecommendationSystem):
        self.system = recommendation_system
        self.interface = self._create_interface()

    def _create_interface(self) -> gr.Blocks:
        """Gradio 인터페이스 생성"""
        with gr.Blocks(title="의약품 추천 시스템", theme="soft") as interface:
            gr.Markdown("# 증상 기반 의약품 추천 시스템")
            
            with gr.Row():
                with gr.Column(scale=1):
                    body_part = gr.Radio(
                        choices=list(self.system.symptom_data.exclude_symptoms.keys()),
                        label="아픈 부위",
                        info="어느 부위가 불편하신가요?",
                        container=True
                    )
                    
                    symptoms = gr.Checkboxgroup(
                        choices=[],
                        label="증상",
                        info="해당하는 증상을 모두 선택해주세요.",
                        visible=False,
                        container=True
                    )
            
            with gr.Row():
                with gr.Column(scale=2):
                    sentence_output = gr.Textbox(
                        label="선택하신 증상",
                        info="선택하신 증상을 바탕으로 생성된 문장입니다.",
                        lines=2
                    )
                    recommendation_output = gr.Textbox(
                        label="추천 약품",
                        info="증상에 맞는 약품 추천 결과입니다.",
                        lines=10
                    )
            
            with gr.Row():
                generate_btn = gr.Button("증상 확인", variant="primary")
                recommend_btn = gr.Button("약품 추천받기", variant="secondary")
            
            # 이벤트 핸들러 연결
            body_part.change(
                fn=lambda x: gr.update(
                    choices=self.system.get_filtered_symptoms(x),
                    visible=True
                ),
                inputs=body_part,
                outputs=symptoms
            )
            
            generate_btn.click(
                fn=self.system.build_symptom_sentence,
                inputs=[body_part, symptoms],
                outputs=sentence_output
            )
            
            recommend_btn.click(
                fn=self.system.get_recommendation,
                inputs=[sentence_output],
                outputs=recommendation_output
            )
            
            return interface

    def launch(self, **kwargs):
        """UI 실행"""
        self.interface.launch(**kwargs)

def main():
    system = MedicalRecommendationSystem(
        csv_path="/home/minsu/coding/model_data.csv"
    )
    ui = MedicalRecommendationUI(system)
    ui.launch(share=True)

if __name__ == "__main__":
    main()
