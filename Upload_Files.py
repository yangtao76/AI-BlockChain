import streamlit as st
import openai
import os
import pandas as pd
import io 
from io import BytesIO
from streamlit_chat import message
from pypdf import PdfReader
import chardet
from pathlib import Path
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import ElasticVectorSearch, pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain





# Load env vars
load_dotenv()

# Set up ChatGPT
llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo") 
prompt = PromptTemplate(template='{text}\n', input_variables=['text'])
chain = LLMChain(llm=llm, prompt=prompt)

# 创建一个 uploads 文件夹用于存储上传的文件
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# 获取 uploads 文件夹中所有的 PDF 文件
pdf_files = [f for f in os.listdir("uploads") if f.endswith(".pdf")]

# 创建一个函数来上传和保存文件
def save_uploaded_file(uploaded_file):
    # 检查文件是否已经存在
    if os.path.exists(os.path.join("uploads", uploaded_file.name)):
        return st.warning("文件已经存在：{}".format(uploaded_file.name))
    with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return st.success("文件已保存：{}".format(uploaded_file.name))


# 创建一个函数来提取 PDF 文件中的文本
def extract_text_from_pdf(pdf_file):
    # 读取 PDF 文件
    with open(os.path.join("uploads", pdf_file), "rb") as f:
        pdf_data = BytesIO(f.read())
    # 提取 PDF 文件中的文本
    reader = PdfReader(pdf_data)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    # 将文本拆分成较小的块
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    return texts



def main():
    # 创建菜单栏 
    st.set_page_config(page_title="Multipage App", layout="centered", initial_sidebar_state="auto")

    # st.title("Main Page")
    st.sidebar.success("Select a page above", icon=None)

    st.title("提取 PDF 文件中的文本")
    # 添加文件上传组件
    uploaded_file = st.file_uploader("请选择 PDF 文件进行上传", type="pdf")
    # query_text = st.text_input('有什么问题吗?')
    # 如果用户上传了文件，则保存它并显示成功消息
    if uploaded_file is not None:
        save_uploaded_file(uploaded_file)
        st.success("文件保存成功，请刷新页面.....")

#        st.experimental_rerun()
    # 显示所有上传的 PDF 文件
    if len(pdf_files) > 0:
        selected_pdf_file = st.selectbox("请选择一个 PDF 文件", pdf_files)
        if st.button("提取文本"):
            # 提取 PDF 文件中的文本
            raw_text = extract_text_from_pdf(selected_pdf_file)

            st.session_state['aa'] = raw_text

    else:
        st.warning("没有上传任何 PDF 文件")

if __name__ == "__main__":
    main()
