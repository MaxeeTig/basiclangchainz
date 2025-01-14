from langchain_community.utilities import SQLDatabase
from langchain_mistralai import ChatMistralAI
from sqlalchemy import create_engine
import agent_test
from agent_test import intents_agents, TestAgent
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
model = ChatMistralAI(model="mistral-large-latest")

def get_agent_type():
    try:
        agent_type = input("\nInput the type of agent to create testing scenarios: ")
        if agent_type.isdigit() and 0 <= int(agent_type) < len(intents_agents):
            return int(agent_type)
        else:
            raise ValueError
    except ValueError:
        raise

def get_amnt_scenarios():
    try:
        amnt = input("Input amount of testing scenarios (default = 3): ")
        if amnt.isdigit() and 0 < int(amnt):
            return int(amnt)
        if amnt =="":
            return None
        else:
            raise ValueError
    except ValueError:
        raise

def main_loop():
    test_agent = TestAgent(db, model)
    print(test_agent.info())
    print(agent_test.existing_queries())
    try:
        agent_type = get_agent_type()
        amnt_scenarios = get_amnt_scenarios()
        if amnt_scenarios is None:
            test_agent(agent_type)  # Только один аргумент
        else:
            test_agent(agent_type, amnt_scenarios)  # Два аргумента
    except ValueError:
        print("\nFailed to get a valid agent or amount of scenarios type, please try again.")

if __name__ == "__main__":
    main_loop()