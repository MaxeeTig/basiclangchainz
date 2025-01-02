from typing_extensions import TypedDict, Annotated
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_mistralai import ChatMistralAI

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class MySQLQueryWriter:
    def __init__(self, db: SQLDatabase, model: ChatMistralAI):
        self.db = db
        self.model = model

    def info(self, user_input):
        return f"MySQLQueryWriter: This agent is responsible for writing MySQL queries. User Input: {user_input}"

    def validate_query(self, state: State):
        """Validate SQL query. State object as input"""
        query = state['query']['query']
        print(f"Debug: validating SQL: {query}")
        try:
            result = self.db.run(query)
            return True, None
        except Exception as e:
            return False, str(e)

    def write_query(self, state: State, max_attempts=3):
        """Generate SQL query to fetch information. State object is input"""
        prompt_template = '''
Given an input question, create a syntactically correct {dialect} query to run to help find the answer.
Unless the user specifies in his question a specific number of examples they wish to obtain, always limit your query
to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Only use the following tables:
{table_info}
Question: {input}
'''

        query_prompt_template = ChatPromptTemplate([
                    ("system", prompt_template),
                    ("user", ""),
                    ])

        prompt = query_prompt_template.invoke ({
                "dialect": "MySQL",
                "top_k": 10,
                "table_info": self.db.get_table_info(),
                "input": state["question"],
            })

        structured_llm = self.model.with_structured_output(QueryOutput)

        for attempt in range(max_attempts):
            state['query'] = structured_llm.invoke(prompt)
            print(f"Debug: generated SQL: {state['query']}")
            is_valid, error_message = self.validate_query(state)
            if is_valid:
                return state
            else:
                prompt = (f"The generated query is invalid. Error: {error_message}. "
                          f"Attempt {attempt + 1} of {max_attempts}. "
                          f"Previous invalid query: {state['query']}. "
                          f"Please correct the query.")

        state['query'] = "Failed to generate a valid query after multiple attempts."
        return state

    def execute_query(self, state: State):
        """Execute SQL query. State object as input"""
        execute_query_tool = QuerySQLDataBaseTool(db=self.db)
        state['result'] = execute_query_tool.invoke(state["query"])
        return state

class PythonCodeWriter:
    def info(self, user_input):
        return f"PythonCodeWriter: This agent is responsible for writing Python code. User Input: {user_input}"

    def write_code(self, state: State):
        """Generate Python code based on the input question. State object as input"""
        system_prompt = '''You are a Python coding assistant that only outputs Python code without any explanations or comments.
Given an instruction and the suggested Python code, return the correct Python code.
'''
        code_prompt_template = ChatPromptTemplate([
            ("system", system_prompt),
            ("user", ""),
        ])

        prompt = code_prompt_template.invoke({
            "input": state["question"],
        })

        # Assuming the model is passed to the PythonCodeWriter class similar to MySQLQueryWriter
        structured_llm = self.model.with_structured_output(CodeOutput)
        state['code'] = structured_llm.invoke(prompt)
        return state
