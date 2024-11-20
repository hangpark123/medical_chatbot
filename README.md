
# 🏥 AI-Driven Medicine Recommendation System

AI 기반 의약품 추천 시스템은 사용자의 증상을 입력받아 관련 데이터를 분석하고 적절한 의약품을 추천해주는 헬스케어 어시스턴트입니다. 이 프로젝트는 Gradio 기반 인터페이스와 LangChain, HuggingFace, Chroma를 활용하여 사용자 친화적인 약품 추천 서비스를 제공합니다.

---

## 📌 **프로젝트 주요 기능**
- **증상 기반 약품 추천**: 사용자가 입력한 증상과 관련된 약품을 데이터베이스에서 검색 및 추천.
- **사용자 맞춤형 추천**: 기저질환 및 알레르기 정보를 기반으로 안전한 약품만 추천.
- **추천 약품의 정보 제공**: 약품의 효능, 주의사항, 부작용 등 상세 정보 제공.
- **데이터 기반 검색**: LangChain과 Chroma를 활용한 검색 및 추천.
- **직관적인 사용자 인터페이스**: Gradio 기반 UI를 통해 누구나 쉽게 접근 가능.

---

## 🚀 **기술 스택**
- **Backend**: Python, LangChain, HuggingFace
- **Frontend**: Gradio
- **Database**: CSV 기반 의약품 데이터 (Chroma Vector Store)
- **Model**: Ollama (LLM 기반 모델)
- **Deployment**: 로컬 서버 또는 클라우드 호스팅

---

## 📂 **프로젝트 구조**
```
.
├── src/
│   ├── medicine_update.csv      # 의약품 데이터베이스
│   ├── user.png                 # 사용자 아바타 이미지
│   ├── bot.png                  # 봇 아바타 이미지
├── chroma_db/                   # Chroma 벡터 데이터 저장소
├── main.py                      # 메인 Python 코드
├── requirements.txt             # Python 의존성 파일
└── README.md                    # 프로젝트 설명 파일
```

---

## 💻 **설치 및 실행 방법**

### 1. **프로젝트 클론**
```bash
git clone https://github.com/<your-repo-name>/medicine-recommendation.git
cd medicine-recommendation
```

### 2. **필요한 라이브러리 설치**
```bash
pip install -r requirements.txt
```

### 3. **Chroma 데이터베이스 초기화**
`medicine_update.csv` 파일을 사용하여 Chroma 데이터베이스를 생성합니다. 코드는 실행 시 자동으로 데이터베이스를 생성합니다.

### 4. **서버 실행**
```bash
python main.py
```

### 5. **접속**
로컬 서버에 접속:
```
http://localhost:7860
```

---

## 🛠 **주요 코드 설명**

### **1. 데이터 로딩**
`medicine_update.csv` 파일을 로드하고, LangChain의 CSVLoader와 Chroma Vector Store를 사용하여 데이터 검색을 준비합니다.

### **2. 증상 기반 검색**
사용자가 입력한 증상에 대해 Chroma Vector Store에서 관련 문서를 검색하고, 검색된 내용을 기반으로 LangChain Prompt Template에 데이터를 전달하여 결과를 생성합니다.

### **3. 추천 결과**
LLM(Ollama)을 통해 추천 결과를 생성하며, 약품의 효능, 주의사항, 부작용 등을 함께 제공합니다.

---

## 🌟 **특징**
- **데이터 기반 추천**: 사용자 입력과 데이터베이스를 매칭하여 신뢰할 수 있는 약품 추천.
- **개인화된 필터링**: 기저질환 및 알레르기 정보를 반영한 약품 추천.
- **유연한 확장성**: 새로운 약품 데이터 추가 및 다양한 증상 지원 가능.

---

## 📝 **앞으로 추가할 기능**
- 다국어 지원 (예: 영어, 일본어, 중국어)
- 부작용 및 약물 상호작용 정보 제공
- 지역 약국 정보와의 연동
- 심각한 증상에 대한 응급 경고 시스템

---

## 🤝 **기여 방법**
1. 이 저장소를 포크합니다.
2. 새로운 기능이나 버그를 수정한 후 PR(Pull Request)을 생성합니다.
3. 프로젝트 발전에 기여해주셔서 감사합니다! 😊

---

## 📜 **라이선스**
이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 📧 **문의**
- **팀명**: AI Healthcare Project Team
- **이메일**: support@aihealthcare.com
