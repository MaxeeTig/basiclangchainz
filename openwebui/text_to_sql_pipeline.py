"""
title: DB Pipeline
author: Maxim Tigulev
date: 16 Jan 2025
version: 1.1
license: MIT
description: A pipeline for using text-to-SQL for retrieving relevant information from a database.
"""

from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from sqlalchemy import create_engine
import ast
from typing_extensions import TypedDict, Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

# Global debug mode variable
debug_mode = True

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

class Pipeline:
    class Valves(BaseModel):
# These fields will be available in UI
        DB_USER: str
        DB_PASSWORD: str
        DB_HOST: str
        DB_DATABASE: str

    # Update valves/ environment variables based on your selected database
    def __init__(self):
        self.name = "Data Finder Companion"
        self.engine = None
        self.nlsql_response = ""

        # Initialize
        self.valves = self.Valves(
            **{
                "DB_USER": "svbo",
                "DB_PASSWORD": "svbopwd",
                "DB_HOST": "localhost",
                "DB_DATABASE": "botransactions"
            }
        )

    def init_db_connection(self):
        # Update your DB connection string based on selected DB engine - current connection string is for MySQL
        self.engine = create_engine(f"mysql+pymysql://{self.valves.DB_USER}:{self.valves.DB_PASSWORD}@{self.valves.DB_HOST}/{self.valves.DB_DATABASE}")
        return self.engine

    async def on_startup(self):
        # This function is called when the server is started.
        self.init_db_connection()

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def validate_query(self, query_output):
        """Validate SQL query."""
        query = query_output['query']
        if debug_mode:
            print(f"Debug: validating SQL: {query}")
        try:
            self.db.run(query)
            return True, None
        except Exception as e:
            return False, str(e)

    def write_query(self, user_message, max_attempts=3):
        """Generate SQL query to fetch information."""
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
                "input": user_message,
            })

        structured_llm = self.model.with_structured_output(QueryOutput)

        if debug_mode:
            print(f"Debug: Generating SQL query for question: {user_message}")
        for attempt in range(max_attempts):
            query_output = structured_llm.invoke(prompt)
            if debug_mode:
                print(f"Debug: Generated SQL query: {query_output}")
            is_valid, error_message = self.validate_query(query_output)
            if is_valid:
                return query_output
            else:
                prompt = (f"The generated query is invalid. Error: {error_message}. "
                          f"Attempt {attempt + 1} of {max_attempts}. "
                          f"Previous invalid query: {query_output}. "
                          f"Please correct the query.")

        return {"query": "Failed to generate a valid query after multiple attempts."}

    def execute_query(self, query_output):
        """Execute SQL query."""
        query = query_output['query']
        execute_query_tool = QuerySQLDataBaseTool(db=self.db)
        if debug_mode:
            print(f"Debug: Executing SQL query: {query}")
        result = execute_query_tool.invoke(query)
        if debug_mode:
            print(f"Debug: SQL query result: {result}")
        return result

    def generate_answer(self, user_message, query_output, result):
        """Answer question using retrieved information as context."""
        query = query_output['query']
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f'Question: {user_message}\n'
            f'SQL Query: {query}\n'
            f'SQL Result: {result}'
        )
        if debug_mode:
            print(f"Debug: Generating answer for question: {user_message}")
        response = self.model.invoke(prompt)
        if debug_mode:
            print(f"Debug: Generated answer: {response.content}")
        return response.content

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # Debug logging is required to see what SQL query is generated by the LlamaIndex library; enable on Pipelines server if needed

        # Create database reader for MySQL
        self.db = SQLDatabase(self.engine)

        # Set up LLM connection; uses mistral-large-latest model
        self.model = ChatMistralAI(model="mistral-large-latest")

        query_output = self.write_query(user_message)
        result = self.execute_query(query_output)
        answer = self.generate_answer(user_message, query_output, result)

        return answer
