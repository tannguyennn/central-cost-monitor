from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
import streamlit as st
from semantic_kernel.functions.kernel_function_decorator import kernel_function
import os
import urllib

# Load environment variables
load_dotenv()

# Validate Azure OpenAI configuration
if not all([os.getenv('AZURE_OPENAI_ENDPOINT'),
           os.getenv('AZURE_OPENAI_API_KEY'),
           os.getenv('AZURE_OPENAI_API_VERSION')]):
    raise ValueError("Missing Azure OpenAI configuration. Please check your .env file.")

# Initialize Streamlit page configuration
st.set_page_config(page_title="Chat with Multiple Databases", page_icon=":speech_balloon:")

# Database configurations (3 databases)
DATABASES = {
    "MySQL": "mysql+pymysql://root:{password}@35.198.228.62:3306/cost_central_monitor".format(
        password=urllib.parse.quote_plus('UP?_]sBRY42@=)=;')
    ),
    "PostgreSQL": "postgresql://postgres:N~s~sh?]r{DY6.8D@35.247.174.112:5432/cost_central_monitor?sslmode=require"
}

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. I can query multiple databases for you."),
    ]
def db():
    if "db_instances" not in st.session_state:
        st.session_state.db_instances = {name: SQLDatabase.from_uri(uri) for name, uri in DATABASES.items()}

db()

def get_llm():
    return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0,
        )

class DataPlugin:
    def __init__(self):
        self.llm = get_llm()

    @kernel_function(name="select_databases", description="Choose relevant databases for the query")
    def select_databases(self, user_query, db_schemas):
        """
        This function uses the Kernel to select relevant databases for the user query.
        The result will be a list of database names.
        """
        
        schemas_context = "\n".join([f"{name}: {schema}" for name, schema in db_schemas.items()])
        
        # Cập nhật prompt để yêu cầu danh sách cơ sở dữ liệu phù hợp
        prompt = f"""
        Here are the schemas of the available databases:
        {schemas_context}

        User query: {user_query}

        Please return a comma-separated list of database names that are most relevant for the question. 
        Do not include any explanations or extra text, just the names of the databases.
        """
        
        # Truyền chuỗi vào invoke() và lấy kết quả
        result = self.llm.invoke(prompt)
        
        # Trả về danh sách cơ sở dữ liệu sau khi tách chuỗi
        return [db.strip() for db in result.content.strip().split(',')]

    def get_sql_chain(self, db):
        template = """
        Based on the schema, write a SQL query to answer the user's question.
        <SCHEMA>{schema}</SCHEMA>

        Question: {question}

        Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
        """
        prompt = ChatPromptTemplate.from_template(template)
        llm = get_llm()

        def get_schema(_):
            return db.get_table_info()

        return (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | llm
            | StrOutputParser()
        )

    def process_results_with_ai(self, raw_results, user_query):
        if not raw_results.strip():
            return "No data available"
        
        prompt = f"Here are the results from the query: {user_query}. Format the results nicely:\n{raw_results}"

        result = self.llm.invoke(prompt)  
        return result.content.strip()

    def get_sql_response(self, db, user_query):
        sql_query = self.get_sql_chain(db).invoke({"question": user_query})
        raw_results = db.run(sql_query)
        return raw_results

    @kernel_function(name="query_multiple_databases")
    def query_multiple_databases(self, user_query, db_instances):
        # Step 1: Get schemas from all databases
        db_schemas = {name: db.get_table_info() for name, db in db_instances.items()}

        # Step 2: Use LLM to select the most relevant databases
        selected_db_names = self.select_databases(user_query, db_schemas)
        
        # Step 3: Generate SQL and fetch results for each selected database
        results = []
        for db_name in selected_db_names:
            db_name = db_name.strip()
            selected_db = db_instances.get(db_name)
            if selected_db:
                raw_result = self.get_sql_response(selected_db, user_query)
                processed_results = self.process_results_with_ai(raw_result, user_query)
                results.append((db_name, processed_results))
        
        return results
db_plugin = DataPlugin()

# Streamlit UI
st.title("Chat with Multiple Databases")

if st.button("Connect to Databases"):
    with st.spinner("Initializing database connections..."):
        try:
            st.session_state.db_instances = {name: SQLDatabase.from_uri(uri) for name, uri in DATABASES.items()}
            st.success("Connected to all databases!")
        except Exception as e:
            st.error(f"Failed to initialize database connections: {str(e)}")

# Chat interface
if st.session_state.db_instances:
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_query = st.chat_input("Ask a database question...")
    if user_query:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            try:
                db()
                results = db_plugin.query_multiple_databases(user_query, st.session_state.db_instances)
                response = ""
                for db_name, processed_results in results:
                    response += f"**Database: {db_name}**\n{processed_results}\n\n"
                st.markdown(response)
                st.session_state.chat_history.append(AIMessage(content=response))
            except Exception as e:
                error_message = f"Error: {str(e)}"
                st.error(error_message)
                st.session_state.chat_history.append(AIMessage(content=error_message))
else:
    st.info("Please initialize databases first.")