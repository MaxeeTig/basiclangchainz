import json
from datetime import datetime
from typing import List, TypedDict
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

intents_agents = {
    0: {"intent": "select data from database",
        "i_description": "extract from database date that suits user input",
        "agent": "mysql_agent",
        "a_description": "generates executable SQL code based on user input"},
    1: {"intent": "draw graph",
        "i_description": "create code that will draw a graph for data described in user input",
        "agent": "python_agent_graph",
        "a_description": "selects required columns from dataframe and generates Python code using Matplotlib to create graphs that visualizes data from a Pandas DataFrame"},
    2: {"intent": "cluster analysis",
        "i_description": "create code that will perform clustering based on user input",
        "agent": "python_agent_cluster",
        "a_description": "generate Python code that performs clustering analysis on dataframe using libraries like Scikit-learn"},
    3: {"intent": "customer portrait",
        "i_description": "based on user input creates customer portraits",
        "agent": "python_agent_cluster",
        "a_description": "generate Python code that performs clustering analysis on dataframe using libraries like Scikit-learn"},
}

def format_intents_agents():
    result = []
    for key, value in intents_agents.items():
        result.append(f"Index: {key}")
        result.append(f"  Intent: {value['intent']}")
        result.append(f"  Agent: {value['agent']}")
    return "\n".join(result)

def correspondence_intents_agents():
    return ", ".join(f"{value['intent']} - {value['agent']}" for value in intents_agents.values())


def existing_queries():
    return f"here is the list of existing queries and used agents\n{format_intents_agents()}"

class QueryObject(TypedDict):
    """Structure for a single query object."""
    input: str
    expected_intent: str
    expected_agent: str
    expected_output: str

class QueryOutput(TypedDict):
    """Structure for an array of query objects."""
    queries: List[QueryObject]

class TestAgent:
    def __init__(self, db: SQLDatabase, model: ChatMistralAI):
        self.db = db
        self.model = model

    def info(self):
        return f"{self.__class__.__name__}: This agent creates testing scenarios for chatbot."

    def __call__(self, agent_type: int, amnt_scenarios: int = 3):
        base_prompt = '''
You create test scenarios for llm agents.
given a type of agent and intent create {amnt_scenarios} test scenarios for that agent and suitable for intent in a json format.

###INSTRUCTIONS###
1. **Parameters**: Use following provided parameters to generate testing scenarios:
    - `intent`: the intent for which testing scenarios should be generated
    - `agent_type`: the type of llm agent that process the provided `intent` for which testing scenarios should be generated
    - `amnt_scenarios`: amount of test scenarios to create
    
2. **Scenario**: testing scenario must be in such format:
    - `input`: create a user's question.
    - `expected_intent`: intent that suits the best and must be chosen by tested classification model.
    - `expected_agent`: agent from that can process `expected_intent` the best.
    - `expected_output`: your suggestions of the right answer of the chosen agent on user input.

3. **Output**: generate array of json objects with {amnt_scenarios} scenarios

###PARAMETERS###
`intent`: {testing_intent}
`agent_type`: {testing_agent}
`amnt_scenarios`: {amnt_scenarios}

###DESCRIPTON###
{testing_agent} is llm agent that {testing_agent_descr}
{testing_intent} means that agent must {testing_intent_descr}
List of `expected_intent` - `expected_agent` that shows which intents could be done by which agents:
{correspondence_intents_agents}
You must always follow this list when creating scenario

Only use the following tables:
{table_info}

###EXAMPLES###
1. Given the parameters:
- `intent`: "select data from database"
- `agent_type`: "mysql_agent"
- `amnt_scenarios`: 2

Generated scenarios in your answer must look like:
```json
[
    {{
        "input": "Show me the transactions for the last month",
        "expected_intent": "select data from database",
        "expected_agent": "mysql_agent",
        "expected_output": "SQL query that gets transactions from corresponding table and results of it"
    }},
    {{
        "input": "show Transactions which amount is more than 10000",
        "expected_intent": "select data from database",
        "expected_agent": "mysql_agent",
        "expected_output": "SQL query that gets data from table of transactions and field amount is > 10000 and results of it"
    }},
]
```

2. Given the parameters:
- `intent`: "draw graph"
- `agent_type`: "python_agent_graph"
- `amnt_scenarios`: 4

Generated scenarios in your answer must look like:
```json
[
    {{
        "input": "Visualize the transaction amounts over time",
        "expected_intent": "draw graph",
        "expected_agent": "python_agent_graph",
        "expected_output": "Python code for drawing a graph of transaction amounts over time"
    }},
    {{
        "input": "Generate a bar chart for transactions exceeding 10,000",
        "expected_intent": "draw graph",
        "expected_agent": "python_agent_graph",
        "expected_output": "Python code that filters transactions > 10,000 and generates a bar chart"
    }},
    {{
        "input": "Plot a line graph for last month's transactions",
        "expected_intent": "draw graph",
        "expected_agent": "python_agent_graph",
        "expected_output": "Python code that filters last month's transactions and plots a line graph"
    }},
    {{
        "input": "Create a scatter plot for high-value transactions",
        "expected_intent": "draw graph",
        "expected_agent": "python_agent_graph",
        "expected_output": "Python code that identifies high-value transactions and creates a scatter plot"
    }},
]
```

3. Given the parameters:
- `intent`: "cluster analysis"
- `agent_type`: "python_agent_cluster"
- `amnt_scenarios`: 1

Generated scenarios in your answer must look like:
```json
[
    {{
        "input": "Cluster the customers based on spending",
        "expected_intent": "cluster analysis",
        "expected_agent": "python_agent_cluster",
        "expected_output": "Python code for clustering transactions"
    }},
]
```
###TASK###
Generate json array of scenarios based on provided parameters 
'''
        query_prompt_template = ChatPromptTemplate([
            ("system", base_prompt),
            ("user", ""),
        ])
        prompt = query_prompt_template.invoke({
            "amnt_scenarios": amnt_scenarios,
            "testing_agent":intents_agents[agent_type]["agent"],
            "testing_agent_descr":intents_agents[agent_type]["a_description"],
            "testing_intent": intents_agents[agent_type]["intent"],
            "testing_intent_descr":intents_agents[agent_type]["i_description"],
            "correspondence_intents_agents": correspondence_intents_agents(),
            "table_info": self.db.get_table_info(),
        })

        structured_llm = self.model.with_structured_output(QueryOutput)
        response = structured_llm.invoke(prompt)

        file_path = 'scenarios/scenarios_'+intents_agents[agent_type]["agent"]+'_'+datetime.now().strftime('%H-%M-%S_%d-%m-%Y')+'.json'

        # Запись данных в файл .json
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=4)
