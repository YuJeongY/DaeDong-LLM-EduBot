
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool
import pandas as pd 
from langchain.docstore.document import Document

# .env 파일 로드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

# .env 파일 로드
@st.cache_resource
def load_excel_data(excel_path):
    index_path = 'faiss_index'
    if os.path.exists(index_path):
        # 기존 FAISS 인덱스 로드
        vector = FAISS.load_local(index_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        # Excel 데이터 로드 및 처리
        df = pd.read_excel(excel_path)
        df = df.fillna("")  # NaN 값 채우기
        documents = [
            Document(
                page_content=row['교육내용'],
                metadata={
                    "수강사이트": row['수강사이트'],
                    "과정명": row['과정명'],
                    "URL": row['URL'],
                    "교육비": row['교육비'],
                    "수강방법": row['수강방법'],
                    "교육시간": row['교육시간']
                }
            )
            for _, row in df.iterrows()
            if row['교육내용'].strip()
        ]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        split_docs = text_splitter.split_documents(documents)
        vector = FAISS.from_documents(split_docs, OpenAIEmbeddings())
        # FAISS 인덱스 저장
        vector.save_local(index_path)
    retriever = vector.as_retriever()
    tool = Tool(
        name="edu_courses_search",
        func=retriever.get_relevant_documents,
        description="Search for relevant edu courses in excel file",
    )
    return tool

# 에이전트와 대화하는 함수
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    response = result['output']  # 명시적으로 출력 키를 처리
    return response

# 대화 내용 출력하는 함수
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])

# Streamlit 메인 코드
def main():
    # 페이지 설정
    st.set_page_config(page_title="에듀봇 대동 비전이", layout="wide", page_icon="✏️")

    st.image('에듀봇 대동이.png', width=600)
    st.markdown('---')
    st.title("안녕하세요! 임직원 역량강화 도우미 '에듀봇 대동 비전이' 입니다.")  # 시작 타이틀

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    

    excel_path = './data/courses.csv' 
    edu_courses_search = load_excel_data(excel_path)
    tools = [edu_courses_search]

    # LLM 설정
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system",
            """
            

                Hello! 😊 I’m *Daedong Vision-i*, your dedicated assistant chatbot for employee capability enhancement!  
                My mission is to help employees quickly and conveniently find the training courses they need.
                "Make sure to use the `edu_courses_search` tool for searching information from the Excel document. "

                Here’s the detailed information using `edu_courses_search` I’ll provide:  
                - I can also provide a **full list of available courses** if required!  
                1️⃣ **Course Name**: The exact name of the course so you can easily identify it.  
                2️⃣ **Training Purpose**: The goals and objectives of the training to understand how it benefits you or your team.  
                3️⃣ **Course Content**: A detailed outline of what will be covered in the training.  
                4️⃣ **Training Site**:  
                - (*Important!*) I will only use the **third column of the courses.xlsx Excel file** to provide accurate site information. I will never create or modify this data on my own.  
                5️⃣ **Training Cost**: A breakdown of the costs associated with the course.  
                6️⃣ **Attendance Method**: Whether the course is online, offline, or a hybrid model.  
                7️⃣ **Training Duration**: The length of the course so you can plan your schedule effectively.  

                I’ll share this information in **table format** or **bullet points**, making it easy to read and understand. 😊  
                To make things even more engaging, I’ll use friendly emojis to guide you along the way.  

                If you have any questions or need more details, feel free to ask anytime! 🚀
            """
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    

    # 에이전트 생성
    agent = create_tool_calling_agent(llm, tools, prompt)

    # AgentExecutor 정의
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 사용자 입력 처리
    user_input = st.chat_input('질문이 무엇인가요?')

    if user_input:
        response = chat_with_agent(user_input, agent_executor)

        # 메시지를 세션에 추가
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": response})

    # 대화 내용 출력
    print_messages()

if __name__ == "__main__":
    main()
