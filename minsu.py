import gradio as gr
import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import Ollama

@dataclass
class SymptomData:
   """증상 관련 데이터를 관리하는 클래스"""
   base_symptoms: List[str]
   exclude_symptoms: Dict[str, List[str]]
   exclusive_symptoms: Dict[str, List[str]]
   location_particle_symptoms: List[str]

class MedicalRecommendationSystem:
   def __init__(self, csv_path: str, persist_directory: str = "chroma_db"):
       self.persist_directory = persist_directory
       self.csv_path = csv_path
       self.symptom_data = self._initialize_symptom_data()
       self.vector_store = self._initialize_vector_store()
       self.llm = self._initialize_llm()

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

   def _initialize_vector_store(self):
       """벡터 스토어 초기화"""
       embeddings = HuggingFaceEmbeddings(
           model_name="sentence-transformers/all-MiniLM-L6-v2",
           model_kwargs={'device': 'cuda'},
           encode_kwargs={
               'normalize_embeddings': True,
               'batch_size': 32
           }
       )
       
       if not os.path.exists(self.persist_directory):
           loader = CSVLoader(
               file_path=self.csv_path,
               encoding='UTF-8',
               csv_args={
                   'delimiter': ',',
                   'quotechar': '"'
               }
           )
           docs = loader.load()
           text_splitter = RecursiveCharacterTextSplitter(
               chunk_size=100,
               chunk_overlap=0,
               separators=["\n", ".", ";", ",", " ", ""]
           )
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
       
       return vector_store

   def _initialize_llm(self):
       """LLM 초기화"""
       return Ollama(
           model="kollama3",
           base_url="http://localhost:11434",
           temperature=0.3,
           num_ctx=2048,
           num_gpu=1,
           timeout=120
       )

   def _format_medicine_info(self, medicines: List[dict]) -> str:
       """약품 정보를 프롬프트에 적합한 형식으로 변환"""
       formatted_info = []
       for i, med in enumerate(medicines, 1):
           info = f"""[약품 {i}]
제품명: {med['name']}
주성분: {med['ingredients']}
분류: {med['type']}
효능효과: {med['effects']}
---"""
           formatted_info.append(info)
       return "\n".join(formatted_info)

   def _clean_response(self, response: str) -> str:
       """LLM 응답을 정리하여 필요한 정보만 추출"""
       try:
           lines = response.strip().split("\n")
           cleaned_lines = []
           valid_sections = ["[추천 약품 1]", "[추천 약품 2]", "[추천 약품 3]"]
           valid_fields = ["약품명:", "효능:", "복용법:", "주의사항:"]
           
           current_section = None
           skip_until_next_section = False
           
           for line in lines:
               line = line.strip()
               if not line:
                   continue
               
               # 새로운 섹션 시작
               if any(section in line for section in valid_sections):
                   current_section = line
                   cleaned_lines.append(line)
                   skip_until_next_section = False
                   continue
               
               # 유효한 필드인 경우만 포함
               if any(field in line for field in valid_fields) and not skip_until_next_section:
                   cleaned_lines.append(line)
           
           return "\n".join(cleaned_lines)
       except Exception as e:
           print(f"Error cleaning response: {str(e)}")
           return response

   def get_recommendation(self, symptoms_sentence: str) -> str:
       if not symptoms_sentence or symptoms_sentence == "증상을 선택해주세요.":
           return "올바른 증상을 선택해주세요."

       try:
           docs = self.vector_store.similarity_search(
               symptoms_sentence, 
               k=3
           )
           
           medicines = []
           for doc in docs:
               try:
                   content = doc.page_content.split(',')
                   if len(content) >= 30:
                       medicine = {
                           'name': content[1].strip().strip('"').replace("'", ""),
                           'ingredients': content[10].strip().strip('"').replace("'", ""),
                           'type': content[12].strip().strip('"').replace("'", ""),
                           'effects': content[-1].strip().strip('"').replace("'", "")
                       }
                       medicines.append(medicine)
               except Exception as e:
                   print(f"Error parsing medicine data: {str(e)}")
                   continue

           if not medicines:
               return "죄송합니다. 적절한 약품을 찾지 못했습니다."

           prompt = f"""의사로서 환자의 증상에 가장 적합한 약품을 추천해주세요.

환자의 증상: {symptoms_sentence}

사용 가능한 약품 정보:
{self._format_medicine_info(medicines)}

위 정보를 바탕으로 환자에게 가장 적합한 약품 3가지를 추천해주세요.
각 약품에 대해 다음 형식으로 작성해주세요:

[추천 약품 1]
약품명: (약품 이름)
효능: (주요 효능을 1-2줄로)
복용법: (일반적인 복용 방법)
주의사항: (주요 주의사항 1-2개)

[추천 약품 2]
약품명: (약품 이름)
효능: (주요 효능을 1-2줄로)
복용법: (일반적인 복용 방법)
주의사항: (주요 주의사항 1-2개)

[추천 약품 3]
약품명: (약품 이름)
효능: (주요 효능을 1-2줄로)
복용법: (일반적인 복용 방법)
주의사항: (주요 주의사항 1-2개)

응답에는 위의 형식만 포함하고, 그 외의 설명이나 부가 정보는 제외해주세요."""

           response = self.llm.predict(
               prompt,
               max_tokens=1000,
               temperature=0.3
           )

           cleaned_response = self._clean_response(response)
           
           if not cleaned_response or cleaned_response.isspace():
               return "죄송합니다. 추천을 생성하지 못했습니다. 다시 시도해주세요."
               
           return cleaned_response
           
       except Exception as e:
           print(f"Error in get_recommendation: {str(e)}")
           return "죄송합니다. 약품 추천 중 오류가 발생했습니다. 다시 시도해주세요."

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

class MedicalRecommendationUI:
   def __init__(self, recommendation_system: MedicalRecommendationSystem):
       self.system = recommendation_system
       self.interface = self._create_interface()

   def _create_interface(self) -> gr.Blocks:
       """Gradio 인터페이스 생성"""
       with gr.Blocks(title="의약품 추천 시스템", theme="soft") as interface:
           gr.Markdown("# 증상 기반 의약품 추천 시스템")
           
           # 왼쪽 컬럼 (증상 선택)
           with gr.Row():
               with gr.Column(scale=1):
                   body_part = gr.Radio(
                       choices=list(self.system.symptom_data.exclude_symptoms.keys()),
                       label="아픈 부위를 선택하세요",
                       info="어느 부위가 불편하신가요?",
                       container=True,
                       interactive=True
                   )
                   
                   symptoms = gr.Checkboxgroup(
                       choices=[],
                       label="증상을 선택하세요",
                       info="해당하는 증상을 모두 선택해주세요.",
                       visible=False,
                       container=True,
                       interactive=True
                   )

                   with gr.Row():
                       generate_btn = gr.Button("증상 확인", variant="primary")
                       recommend_btn = gr.Button("약품 추천받기", variant="secondary")

               # 오른쪽 컬럼 (결과 표시)
               with gr.Column(scale=2):
                   sentence_output = gr.Textbox(
                       label="선택하신 증상",
                       info="선택하신 증상을 바탕으로 생성된 문장입니다.",
                       lines=2,
                       interactive=False
                   )
                   recommendation_output = gr.Textbox(
                       label="추천 약품",
                       info="증상에 맞는 약품 추천 결과입니다.",
                       lines=15,
                       interactive=False
                   )
           
           # 이벤트 핸들러
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
       try:
           self.interface.launch(
               server_name="0.0.0.0",  # 모든 IP에서 접근 가능5
               share=True,             # 공유 링크 생성
               debug=True              # 디버그 모드 활성화
           )
       except Exception as e:
           print(f"Error launching interface: {str(e)}")

def main():
   try:
       system = MedicalRecommendationSystem(
           csv_path="/home/minsu/coding/model_data.csv"
       )
       ui = MedicalRecommendationUI(system)
       ui.launch()
   except Exception as e:
       print(f"Error in main: {str(e)}")

if __name__ == "__main__":
   main()
