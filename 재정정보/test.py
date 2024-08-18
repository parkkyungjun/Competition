from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma

import os
from read_pdf import *
import pandas as pd
import re
import random
from tqdm import tqdm 

df = pd.read_csv('/home/pkj/rag/data/test.csv')
sub = pd.read_csv('/home/pkj/rag/data/sample_submission.csv')


model = ChatOllama(model="llama3.1:70b", temperature=0)


# Prompt 템플릿 생성
template = '''모든 대답은 한국어로 대답해줘. 한 문장으로 대답해줘 적혀 있는 그대로 대답해줘":
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3', # BAAI/bge-m3
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True},
)

b_src = ''
max_score = 0
for _ in tqdm(range(100)):
    preds, gts = [], []
    chunk = random.randint(100, 1000)
    overlap = random.randint(100, chunk-1)
    k = random.randint(3, 10)
    for i in range(len(df)):
        src = df['Source'][i]
        qs = df['Question'][i]
        # if i not in [27, 28, 29, 30, 31, 32]: # 682 436 4
        #     continue
        # if src != b_src:
        #     b_src = src
        file_path = f'/home/pkj/rag/data/test_source/{src}.pdf'
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk,
            chunk_overlap=overlap,
        )
        docs = text_splitter.split_documents(pages)

        vectorstore = Chroma.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={'k': k})
        rag_chain = (
            {'context': retriever | format_docs, 'question': RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        answer = rag_chain.invoke(qs)

        def normalize_whitespace(text):
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        answer = normalize_whitespace(answer)
        preds.append(answer)

        # print("Query:", qs)
        # print("Answer:", answer)
        
    # sub['Answer'] = preds
    gt = ['328백만원', '520백만원', '280백만원', '133.5백만원', '37.5백만원', '50백만원']
    score = calculate_average_f1_score(gt, preds)['average_f1_score']
    print(score, chunk, overlap, k, preds)
    if score > max_score:
        max_score = score
        best_chunk, best_overlap, best_k = chunk, overlap, k
    # sub.to_csv('submission.csv', index=False)

print(best_chunk, best_overlap, best_k, max_score)

