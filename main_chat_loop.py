from user_intent import classify_text
from user_intent import clarify_intent
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from sqlalchemy import create_engine
import pandas as pd

# our file agents.py
from agents import MySQLQueryWriter, PythonCodeWriterGraph, PythonCodeWriterCluster  # My agents

# Global debug mode variable
debug_mode = True

# Database connection parameters
db_config = {
    'user': 'svbo',
    'password': 'svbopwd',
    'host': 'localhost',
    'database': 'botransactions'
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
    # call LLM to clarify intent
    if debug_mode:
        print(f"Debug: Understanding user intent for input: {user_input}")
    user_input_clarified = clarify_intent(user_input)
    # structured output from LLM - dictionary
    intent = user_input_clarified['query']

    # glue user input with clarified intent and pass to classification
    intent = f"{user_input} {intent}"

    # set possible intent labels
    intent_labels = ["draw graph", "customer portrait or profile", "cluster analysis", "select data from database", "unknown"]
    user_intent = classify_text(intent, intent_labels)
    if debug_mode:
        print(f"Debug: Understood user intent: {user_intent}")
    return user_intent

# ===== function to select agent for particular intent
def select_agent(intent):
    # Select the appropriate agent based on the intent
    if debug_mode:
        print(f"Debug: Selecting agent for intent: {intent}")
    if intent == 'select data from database':
        return mysql_agent
    elif intent == 'draw graph':
        return python_agent_graph
    elif intent == 'cluster analysis':
        return python_agent_cluster
    elif intent == 'customer portrait':
        return python_agent_cluster  # Assuming the same agent for clustering
    else:
        return None

# ===== function to execute agent on the basis of intent
def execute_agent_task(agent, intent, user_input, df=None):
    # Execute the agent's task based on the intent
    if debug_mode:
        print(f"Debug: Executing agent task for intent: {intent} with user input: {user_input}")
    if intent == 'select data from database':
        state = {
            "question": user_input,
            "query": "",
            "result": "",
            "answer": ""
        }
        state = agent.write_query(state)
        state = agent.execute_query(state)
        state = agent.generate_answer(state)
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
    elif intent == 'cluster analysis' or intent == 'customer portrait':
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
    SELECT oper_date, issuer_card_id, mcc, merchant_city, merchant_country, oper_amount_amount_value, oper_amount_currency, oper_type
    FROM operations
    WHERE is_reversal = 0
    AND oper_type IS NOT NULL
    AND mcc <> ''
    AND merchant_country REGEXP '^[0-9]+$'
    AND merchant_country <> '0'
    AND oper_amount_amount_value > 1.00
    AND oper_amount_amount_value < 1000000000.00
    '''
    if debug_mode:
        print(f"Debug: Pre-fetching data for intent: {intent} with user input: {user_input}")
        print("Debug: pre-fetching data to dataframe... ")

    # Fetch data from the database
    df = pd.read_sql(sql_query, engine)
    if debug_mode:
        print(f"Debug: Pre-fetched data: {df.head()}")
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
            if debug_mode:
                print(f"Getting extra information from local data for query: '{user_input}'...")

            # call function to define user's intent
            intent = understand_user_intent(user_input)
            if debug_mode:
                print(f"User needs to: {intent['top_label']}")
                print(intent)

            if intent['top_label'] == 'unknown':
                print("Sorry, I do not understand your question. Please try again.")
                continue

            agent = select_agent(intent['top_label'])
            if debug_mode:
                print(f"Selected agent: {agent}")

            # Pre-fetch data if the intent requires it
            if intent['top_label'] in ['draw graph', 'cluster analysis']:
                df = pre_fetch_data(intent['top_label'], user_input)
                if debug_mode:
                    print(f"Data fetched to df:{df.head()}")
            else:
                df = None

            results = execute_agent_task(agent, intent['top_label'], user_input, df)
            if debug_mode:
                print(f"Debug: agent results: {results}")

            # Execute the generated Python code if the intent is 'draw graph'
            if intent['top_label'] == 'draw graph':
                if debug_mode:
                    print(f"Debug: Executing generated code for graph drawing: {results}")
                try:
                    exec(results)
                    print("Graph generated successfully.")
                except Exception as e:
                    print(f"Error executing the generated code: {e}")

            # Print the answer if the intent is 'select data from database'
            if intent['top_label'] == 'select data from database':
                print(f"Answer: {results['answer']}")

if __name__ == "__main__":
    main_chat_loop()
