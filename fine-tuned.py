from openai import OpenAI
import os
# import Confing

# 참조 사이트
# https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
os.environ['OPENAI_API_KEY'] = "API KEY"

client = OpenAI()

question = input("Question : ")
completion = client.chat.completions.create(
  model="ft:gpt-4o-2024-08-06:personal::ASJVS2m1",
  #model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "너는 참고용으로 의약품을 추천하는 챗봇이야. 정확한 진단은 병원을 방문해 주세요 라고 항상 안내해야 해."},
    {"role": "user", "content": f"{question}"}
  ]
)
print(completion.choices[0].message.content)