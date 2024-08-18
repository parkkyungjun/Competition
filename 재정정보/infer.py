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

df = pd.read_csv('/home/pkj/rag/data/test.csv')
sub = pd.read_csv('/home/pkj/rag/data/sample_submission.csv')
preds, gts = [], []

model = ChatOllama(model="llama3.1:70b", temperature=0)


# Prompt 템플릿 생성
template = '''모든 대답은 한국어로 대답해줘. 한 문장으로 대답해줘 적혀 있는 그대로 대답해줘":
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])


b_src = ''
embeddings = HuggingFaceEmbeddings(
    model_name='BAAI/bge-m3', # BAAI/bge-m3
    model_kwargs={'device':'cuda'},
    encode_kwargs={'normalize_embeddings':True},
)
for i in range(len(df)):
    src = df['Source'][i]
    qs = df['Question'][i]

    if src != b_src:
        b_src = src
        file_path = f'/home/pkj/rag/data/test_source/{src}.pdf'
        loader = PyMuPDFLoader(file_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=250,
        )
        docs = text_splitter.split_documents(pages)

        vectorstore = Chroma.from_documents(docs, embeddings)
        
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
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

    print("Query:", qs)
    print("Answer:", answer)
    
sub['Answer'] = preds
sub.to_csv('submission.csv', index=False)