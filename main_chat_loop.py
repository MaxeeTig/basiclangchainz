from user_intent import classify_text
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from sqlalchemy import create_engine

# our file agents.py
from agents import MySQLQueryWriter, PythonCodeWriter  # Hypothetical agents

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
python_agent = PythonCodeWriter(model)

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
        return python_agent
    elif intent == 'cluster' or intent == 'customer portrait':
        return python_agent  # Assuming the same agent for clustering
    else:
        raise ValueError("Unknown intent")

# ===== function to execute agent on the basis of intent
def execute_agent_task(agent, intent, user_input):
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
        return agent.write_code(state)
    elif intent == 'cluster' or intent == 'customer portrait':
        return agent.info(user_input)
    else:
        raise ValueError("Unknown intent")

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

            results = execute_agent_task(agent, intent['top_label'], user_input)
            print(f"Debug: agent results: {results}")

if __name__ == "__main__":
    main_chat_loop()
