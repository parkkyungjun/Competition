from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings
# from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain.vectorstores import Chroma

import os
from read_pdf import *
import pandas as pd
# PyMuPDFLoader 을 이용해 PDF 파일 로드

df = pd.read_csv('/home/pkj/rag/data/train.csv')
preds, gts = [], []

model = ChatOllama(model="llama3.1:70b", temperature=0)


# Prompt 템플릿 생성
template = '''모든 대답은 한국어로 대답해줘. 한 문장으로 혹은 두 문장으로 대답해주고 주어진 자료를 요약하는 식으로 대답해줘":
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])


b_src = ''
ps = ['1-1 2024 주요 재정통계 1권', '2024 나라살림 예산개요', '2024년도 성과계획서(총괄편)', '월간 나라재정 2023년 12월호', '재정통계해설']
for i in range(len(df)):
    src = df['Source'][i]
    qs = df['Question'][i]
    gt = df['Answer'][i]
    if src in ps:
        continue
    if src != b_src:
        b_src = src
        file_path = f'/home/pkj/rag/data/train_source/{src}.pdf'
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        docs = text_splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings(
            model_name='BAAI/bge-m3',
            model_kwargs={'device':'cuda'},
            encode_kwargs={'normalize_embeddings':True},
        )

        vectorstore = Chroma.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
        rag_chain = (
            {'context': retriever | format_docs, 'question': RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

    answer = rag_chain.invoke(qs)


    def is_korean(string):
        for char in string:
            if '\uac00' <= char <= '\ud7a3':
                return True
        return False

    def is_english(string):
        for char in string:
            if 'a' <= char <= 'z' or 'A' <= char <= 'Z':
                return True
        return False
    import re

    def normalize_whitespace(text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    if is_korean(answer):
        answer = normalize_whitespace(answer)
        preds.append(answer)
        gts.append(gt)
        print("Query:", qs)
        print("Answer:", answer)
    
result = calculate_average_f1_score(gts, preds)
print(result)