import json
import os
import urllib

import streamlit as st
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import AzureChatOpenAI
from langchain.agents import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langchain_community.agent_toolkits import create_sql_agent

# Load environment variables
load_dotenv()

def load_config():
    with open("db_config.json", "r") as f:
        return json.load(f)

# Database initialization functions
# Modified database initialization section
def init_mysql(config):
    user = urllib.parse.quote_plus(config['user'])
    password = urllib.parse.quote_plus(config['password'])
    host = config['host']
    port = config['port']
    database = config['database']

    db_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

# Initialize MySQL agent
def init_mysql_agent(config):
    mysql_llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
    )
    # Create SQL agent using the LLM and MySQL database
    db = init_mysql(config)
    mysql_agent = create_sql_agent(
        llm=mysql_llm, 
        db=db,
        agent_type="openai-tools", 
        verbose=False, 
        handle_parsing_errors=True
    )
    return mysql_agent

def init_postgresql(config):
    db_uri = f"postgresql://{config['postgresql']['user']}:{config['postgresql']['password']}@{config['postgresql']['host']}:{config['postgresql']['port']}/{config['postgresql']['database']}"
    return SQLDatabase.from_uri(db_uri)

def init_postgresql_agent(config):
    pg_llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
    )
    db = init_postgresql(config)
    postgresql_agent = create_sql_agent(
        llm=pg_llm,
        db=db,
        agent_type="openai-tools",
        verbose=False,
        handle_parsing_errors=True
    )
    return postgresql_agent

def init_sqlite(config):
    if not os.path.exists(config['path']):
        raise FileNotFoundError(f"SQLite file {config['path']} does not exist.")
    db_uri = f"sqlite:///{config['path']}"
    return SQLDatabase.from_uri(db_uri)

def init_sqlite_agent(config):
    sqlite_llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
    )
    db = init_sqlite(config)
    sqlite_agent = create_sql_agent(
        llm=sqlite_llm,
        db=db,
        agent_type="openai-tools",
        verbose=False,
        handle_parsing_errors=True
    )
    return sqlite_agent



# Tool creation for each database
# Define the tool for MySQL query
@tool
def query_mysql(query: str):
    """
    Query the MySQL database. Input should be a SQL query.
    Use ONLY the tables and schema from the MySQL database.
    """
    try:
        if "mysql_agent" in st.session_state:
            # Call the agent to execute the query
            result = st.session_state.mysql_agent.invoke({"input": query})
            return result
        return "MySQL agent not initialized."
    except Exception as e:
        return f"Error executing MySQL query: {str(e)}"


@tool
def query_postgresql(query: str):
    """
    Query the PostgreSQL database. Input should be a SQL query.
    Use ONLY the tables and schema from the PostgreSQL database.
    """
    try:
        if "postgresql_db" in st.session_state:
            result = st.session_state.postgresql_agent.invoke({"input": query})
            return result
        return "PostgreSQL database not connected"
    except Exception as e:
        return f"Error executing PostgreSQL query: {str(e)}"

@tool
def query_sqlite(query: str):
    """
    Query the SQLite database. Input should be a SQL query.
    Use ONLY the tables and schema from the SQLite database.
    """
    try:
        if "sqlite_db" in st.session_state:
            result = st.session_state.sqlite_agent.invoke({"input": query})
            return result
        return "SQLite database not connected"
    except Exception as e:
        return f"Error executing SQLite query: {str(e)}"

# Setup LLM
def get_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0,
    )

# State management
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Tool execution node
class ToolNode:
    def __init__(self, tools: list):
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: State):
        messages = state["messages"]
        message = messages[-1]
        outputs = []

        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# Graph setup
def setup_graph():
    tools = [query_mysql, query_postgresql, query_sqlite]
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(State)

    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    tool_node = ToolNode(tools=tools)

    graph.add_node("chatbot", chatbot)
    graph.add_node("tools", tool_node)

    def should_continue(state: State) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    graph.add_conditional_edges("chatbot", should_continue, ["tools", END])
    graph.add_edge("tools", "chatbot")
    graph.add_edge(START, "chatbot")

    return graph.compile()

# Initialize session state for connection status
if 'connection_status' not in st.session_state:
    st.session_state.connection_status = {
        'mysql': False,
        'postgresql': False,
        'sqlite': False,
        'chat_initialized': False
    }

# Streamlit UI
st.title("Multi-Database Chat Interface")

# Sidebar for database connections
with st.sidebar:
    st.subheader("Database Connections")

    try:
        config = load_config()

        # MySQL connection
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Connect MySQL"):
                try:
                    # Initialize both the database and the agent
                    st.session_state.mysql_db = init_mysql(config["mysql"])
                    st.session_state.mysql_agent = init_mysql_agent(config["mysql"])
                    st.session_state.connection_status['mysql'] = True
                    st.success("Connected to MySQL!")
                except Exception as e:
                    st.error(f"MySQL connection failed: {str(e)}")

        with col2:
            st.write("Status:")
            st.write("🟢" if st.session_state.connection_status['mysql'] else "🔴")

        # PostgreSQL connection
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Connect PostgreSQL"):
                try:
                    st.session_state.postgresql_db = init_postgresql(config)
                    st.session_state.postgresql_agent = init_postgresql_agent(config)
                    st.session_state.connection_status['postgresql'] = True
                    st.success("Connected to PostgreSQL!")
                except Exception as e:
                    st.error(f"PostgreSQL connection failed: {str(e)}")
        with col2:
            st.write("Status:")
            st.write("🟢" if st.session_state.connection_status['postgresql'] else "🔴")

        # SQLite connection
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("Connect SQLite"):
                try:
                    st.session_state.sqlite_db = init_sqlite(config["sqlite"])
                    st.session_state.sqlite_agent = init_sqlite_agent(config["sqlite"])
                    st.session_state.connection_status['sqlite'] = True
                    st.success("Connected to SQLite!")
                except Exception as e:
                    st.error(f"SQLite connection failed: {str(e)}")
        with col2:
            st.write("Status:")
            st.write("🟢" if st.session_state.connection_status['sqlite'] else "🔴")

        # Initialize chat system
        st.divider()
        if st.button("Initialize Chat System"):
            try:
                st.session_state.graph = setup_graph()
                st.session_state.connection_status['chat_initialized'] = True
                st.success("Chat system initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize chat system: {str(e)}")

    except Exception as e:
        st.error(f"Configuration error: {str(e)}")

# Main chat interface
if st.session_state.connection_status.get('chat_initialized', False):
    if "chat_history" not in st.session_state:
        connected_dbs = [db for db, status in st.session_state.connection_status.items() 
                        if status and db != 'chat_initialized']
        st.session_state.chat_history = [
            AIMessage(content=f"Hello! I can help you query multiple databases. "
                            f"Currently connected databases: {', '.join(connected_dbs)}")
        ]

    # Display chat history
    for message in st.session_state.chat_history:
        role = "assistant" if isinstance(message, (AIMessage, ToolMessage)) else "user"
        with st.chat_message(role):
            if isinstance(message, ToolMessage):
                try:
                    content = json.loads(message.content)
                    st.write(content)
                except json.JSONDecodeError:
                    st.write(message.content)
            else:
                st.write(message.content)

    # Chat input
    if any(st.session_state.connection_status[db] for db in ['mysql', 'postgresql', 'sqlite']):
        user_query = st.chat_input("Ask a question about the databases...")
        if user_query and user_query.strip():
            user_message = HumanMessage(content=user_query)
            st.session_state.chat_history.append(user_message)

            with st.chat_message("user"):
                st.write(user_query)

            try:
                result = st.session_state.graph.invoke({
                    "messages": [user_message]
                })

                for message in result["messages"]:
                    st.session_state.chat_history.append(message)
                    with st.chat_message("assistant"):
                        if isinstance(message, ToolMessage):
                            try:
                                content = json.loads(message.content)
                                st.write(content)
                            except json.JSONDecodeError:
                                st.write(message.content)
                        else:
                            st.write(message.content)

            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)
                st.session_state.chat_history.append(AIMessage(content=error_message))
    else:
        st.warning("Please connect to at least one database before starting the chat.")
else:
    st.info("Please connect to databases and initialize the chat system using the sidebar options.")
