import requests
import re

url = "http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
payload = {
    "serviceKey": "KyPdMMO24VFJ9wrLqUUC7fmfchd2sZLJKUlzfPe2YaUtJ0gnRZ9Mtqp76sfqpcGfm65lNp2PZ4LbUVcPQrerBQ==",
    "pageNo": 1,
    "numOfRows": 3, #결과 수
    "efcyQesitm": "오십견" #사용자 증상
}

response = requests.get(url, params=payload)

entityList = ["entpName", "itemName", "efcyQesitm", "useMethodQesitm", "atpnWarnQesitm",
              "atpnQesitm", "intrcQesitm", "seQesitm", "depositMethodQesitm"] #api 내부 항목명 list

items = re.findall(r'<item>(.+?)</item>',response.text, re.DOTALL) #<item>박스 내용 regex

#--- data fetch version 1 (전체 내용 fetch) ---
# for item in items:
#     for entity in entityList:
#         medicine = re.findall(fr'<{entity}>(.+?)</{entity}>', item, re.DOTALL)
#         print(medicine[0])

#--- data fetch version 2 (eneity별 fetch) ---
for item in items:
    for entity in entityList:
        
        entpName = re.findall(r'<entpName>(.+?)</entpName>', item, re.DOTALL)
        itemName = re.findall(r'<itemName>(.+?)</itemName>', item, re.DOTALL)
        efcyQesitm = re.findall(r'<efcyQesitm>(.+?)</efcyQesitm>', item, re.DOTALL)
        useMethodQesitm = re.findall(r'<useMethodQesitm>(.+?)</useMethodQesitm>', item, re.DOTALL)
        atpnWarnQesitm = re.findall(r'<atpnWarnQesitm>(.+?)</atpnWarnQesitm>', item, re.DOTALL)
        atpnQesitm = re.findall(r'<atpnQesitm>(.+?)</atpnQesitm>', item, re.DOTALL)
        intrcQesitm = re.findall(r'<intrcQesitm>(.+?)</intrcQesitm>', item, re.DOTALL)
        seQesitm = re.findall(r'<seQesitm>(.+?)</seQesitm>', item, re.DOTALL)
        depositMethodQesitm = re.findall(r'<depositMethodQesitm>(.+?)</depositMethodQesitm>', item, re.DOTALL)
        
    print("업체명:", entpName[0])
    print("제품명:", itemName[0])
    print("효능:", efcyQesitm[0])
    print("사용법:", useMethodQesitm[0])
    # print("주의사항경고:", atpnWarnQesitm[0]) #어떤 경우는 해당 항목이 기재되어 있지 않아 index out of range가 되는 경우가 있음
    print("주의사항:", atpnQesitm[0])
    print("상호작용:", intrcQesitm[0]) #위와 마찬가지로 가끔 outOfRange 에러 발생
    print("부작용:", seQesitm[0])
    print("보관법:", depositMethodQesitm[0])
    print("-"*50)


 