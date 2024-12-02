
import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
import pandas as pd

# .env 파일 로드
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# API 키를 환경변수로 관리하기 위한 설정 파일
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]


# 폴더 경로 설정
folder_path = "./data"  # 분석할 파일이 저장된 폴더 경로
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


# 엑셀 문서 로드 함수
def load_excel_with_metadata(file_path):
    documents = []
    xls = pd.ExcelFile(file_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        loader = DataFrameLoader(df, page_content_column=df.columns[0])
        sheet_docs = loader.load_and_split(text_splitter)
        for doc in sheet_docs:
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["sheet_name"] = sheet_name
            doc.metadata["cell_range"] = f"A1:{df.columns[-1]}{len(df)}"  # 추가 셀 범위 정보
        documents.extend(sheet_docs)
    return documents

def load_documents_from_folder(folder_path):
    documents = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
            documents.extend(load_excel_with_metadata(file_path))
    return documents


# 에이전트와 대화하는 함수
def chat_with_agent(user_input, agent_executor):
    result = agent_executor({"input": user_input})
    response = result['output']  # 명시적으로 출력 키를 처리
    return response

# 세션 기록 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state.session_history:
        st.session_state.session_history[session_ids] = ChatMessageHistory()
    return st.session_state.session_history[session_ids]

# 대화 내용 출력하는 함수
def print_messages():
    for msg in st.session_state["messages"]:
        st.chat_message(msg['role']).write(msg['content'])


# 모든 문서 로드
all_docs = load_documents_from_folder(folder_path)


# FAISS 인덱스 설정 및 생성
vector = FAISS.from_documents(all_docs, OpenAIEmbeddings())
retriever = vector.as_retriever()

# 도구 정의
excel_tool = create_retriever_tool(
    retriever,
    name="excel_search",
    description="Use this tool to search information from the excel document"
)

# Streamlit 메인 코드
def main():
    # 페이지 설정
    st.set_page_config(page_title="에듀봇 대동 비전이", layout="wide", page_icon="✏️")

    st.image('에듀봇 대동이.png', width=600)
    st.markdown('---')
    st.title("안녕하세요! 임직원 역량강화 도우미 '에듀봇 대동 비전이' 입니다")  # 시작 타이틀

    # 세션 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "session_history" not in st.session_state:
        st.session_state["session_history"] = {}


# return retriever_tool
    tools = [excel_tool]

    # LLM 설정
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # Prompt 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            (
            "system",
            """
            

                Hello! 😊 I’m *Daedong Vision-i*, your dedicated assistant chatbot for employee capability enhancement!  
                My mission is to help employees quickly and conveniently find the training courses they need.
                "Make sure to use the `excel_search` tool for searching information from the Excel document. "

                Here’s the detailed information using `excel_search` I’ll provide:  
                - I can also provide a **full list of available courses** if required!  
                1️⃣ **Course Name**: The exact name of the course so you can easily identify it.  
                2️⃣ **Training Purpose**: The goals and objectives of the training to understand how it benefits you or your team.  
                3️⃣ **Course Content**: A detailed outline of what will be covered in the training.  
                4️⃣ **Training Site**:  
                - (*Important!*) I will only use the **third column of the "세개사이트합본" Excel file** to provide accurate site information. I will never create or modify this data on my own.  
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

    # 에이전트 생성 (initialize_agent 대신 create_tool_calling_agent 사용)
    agent = create_tool_calling_agent(llm, tools, prompt)

    # AgentExecutor 정의
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 사용자 입력 처리
    user_input = st.chat_input('질문이 무엇인가요?')

    if user_input:
        session_id = "default_session"
        session_history = get_session_history(session_id)

        if session_history.messages:
            previous_messages = [{"role": msg['role'], "content": msg['content']} for msg in session_history.messages]
            response = chat_with_agent(user_input + "\n\nPrevious Messages: " + str(previous_messages), agent_executor)
        else:
            response = chat_with_agent(user_input, agent_executor)

        # 메시지를 세션에 추가
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # 세션 기록에 메시지를 추가
        session_history.add_message({"role": "user", "content": user_input})
        session_history.add_message({"role": "assistant", "content": response})

    # 대화 내용 출력
    print_messages()


if __name__ == "__main__":
    main()
