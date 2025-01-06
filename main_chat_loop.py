from user_intent import classify_text
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from sqlalchemy import create_engine
import pandas as pd

# our file agents.py
from agents import MySQLQueryWriter, PythonCodeWriterGraph, PythonCodeWriterCluster  # My agents

# Database connection parameters
db_config = {
    'user': 'cctrxn',
    'password': 'cctrxnpwd',
    'host': 'localhost',
    'database': 'cctransreview'
}

# Create the SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")
db = SQLDatabase(engine=engine)

# Initialize agents
model = ChatMistralAI(model="mistral-large-latest")
mysql_agent = MySQLQueryWriter(db, model)
python_agent_graph = PythonCodeWriterGraph(model)
python_agent_cluster = PythonCodeWriterCluster(model)

# ===== function to understand user's intent
def understand_user_intent(user_input):
    # set possible intent labels
    intent_labels = ["draw graph", "customer portrait", "cluster", "select data from database"]
    user_intent = classify_text(user_input, intent_labels)
    return user_intent

# ===== function to select agent for particular intent
def select_agent(intent):
    # Select the appropriate agent based on the intent
    if intent == 'select data from database':
        return mysql_agent
    elif intent == 'draw graph':
        return python_agent_graph
    elif intent == 'cluster':
        return python_agent_cluster
    elif intent == 'customer portrait':
        return python_agent_cluster  # Assuming the same agent for clustering
    else:
        raise ValueError("Unknown intent")

# ===== function to execute agent on the basis of intent
def execute_agent_task(agent, intent, user_input, df=None):
    # Execute the agent's task based on the intent
    if intent == 'select data from database':
        state = {
            "question": user_input,
            "query": "",
            "result": "",
            "answer": ""
        }
        state = agent.write_query(state)
        state = agent.execute_query(state)
        return state
    elif intent == 'draw graph':
        state = {
            "question": user_input,
            "code": "",
            "result": "",
            "answer": ""
        }
        state = agent.write_code(state, df)
        return state['code']['query']
    elif intent == 'cluster' or intent == 'customer portrait':
        state = {
            "question": user_input,
            "code": "",
            "result": "",
            "answer": ""
        }
        state = agent.write_code(state, df)
        return state['code']['query']
    else:
        raise ValueError("Unknown intent")

# ===== Function to pre-fetch data. Data defined by LLM on the basis of user intent
def pre_fetch_data(intent, user_input):
    # Generate SQL query dynamically based on the intent and user input
#   reserved call to llm
#    sql_query =  execute_agent_task(mysql_agent, intent, user_input)
    sql_query = '''
 SELECT trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender, street, city,
    state, zip, lat, longitude, city_pop, job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud,
    merch_zipcode
 FROM
 ccoperations
 '''
    print("Debug: pre-fetching data to dataframe... ")

    # Fetch data from the database
    df = pd.read_sql(sql_query, engine)
    return df

# ===== Main chat loop =====
def main_chat_loop(welcome_message="Welcome to Chatbot!"):
    """Main function to run the chatbot."""
    print(welcome_message)

    while True:
        user_input = input("Please enter your query (type 'exit' to quit): ")

        if user_input.lower() == 'exit':
            print("Exiting the chatbot. Goodbye!")
            break
        else:
            print(f"Getting extra information from local data for query: '{user_input}'...")

            # call function to define user's intent
            intent = understand_user_intent(user_input)
            print(f"User needs to: {intent['top_label']}")
            print(intent)

            agent = select_agent(intent['top_label'])
            print(f"Selected agent: {agent}")

            # Pre-fetch data if the intent requires it
            if intent['top_label'] in ['draw graph', 'cluster']:
                df = pre_fetch_data(intent['top_label'], user_input)
                print(f"Data fetched to df:{df.head()}")
            else:
                df = None

            

            results = execute_agent_task(agent, intent['top_label'], user_input, df)
            print(f"Debug: agent results: {results}")

            # Execute the generated Python code if the intent is 'draw graph'
            if intent['top_label'] == 'draw graph':
                try:
                    exec(results)
                    print("Graph generated successfully.")
                except Exception as e:
                    print(f"Error executing the generated code: {e}")

            # Execute the generated Python code if the intent is 'draw graph'
            if intent['top_label'] == 'cluster':
                try:
                    exec(results)
                    print("Graph generated successfully.")
                except Exception as e:
                    print(f"Error executing the generated code: {e}")


if __name__ == "__main__":
    main_chat_loop()
