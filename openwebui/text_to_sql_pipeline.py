"""
title: Llama Index DB Pipeline
author: 0xThresh
date: 2024-08-11
version: 1.1
license: MIT
description: A pipeline for using text-to-SQL for retrieving relevant information from a database using the Llama Index library.
requirements: llama_index, sqlalchemy, psycopg2-binary
"""

from typing import List, Union, Generator, Iterator
import os
from pydantic import BaseModel
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from sqlalchemy import create_engine
import ast

class Pipeline:
    class Valves(BaseModel):
        DB_HOST: str
        DB_PORT: str
        DB_USER: str
        DB_PASSWORD: str
        DB_DATABASE: str
        DB_TABLE: str
        MISTRAL_HOST: str
        TEXT_TO_SQL_MODEL: str

    # Update valves/ environment variables based on your selected database
    def __init__(self):
        self.name = "Database RAG Pipeline"
        self.engine = None
        self.nlsql_response = ""

        # Initialize
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],                                                           # Connect to all pipelines
                "DB_HOST": os.getenv("DB_HOST", "http://localhost"),                     # Database hostname
                "DB_PORT": os.getenv("DB_PORT", 5432),                                        # Database port
                "DB_USER": os.getenv("DB_USER", "postgres"),                                  # User to connect to the database with
                "DB_PASSWORD": os.getenv("DB_PASSWORD", "password"),                          # Password to connect to the database with
                "DB_DATABASE": os.getenv("DB_DATABASE", "postgres"),                          # Database to select on the DB instance
                "DB_TABLE": os.getenv("DB_TABLE", "table_name"),                            # Table(s) to run queries against
                "MISTRAL_HOST": os.getenv("MISTRAL_HOST", "http://host.docker.internal:11434"), # Make sure to update with the URL of your Mistral host, such as http://localhost:11434 or remote server address
                "TEXT_TO_SQL_MODEL": os.getenv("TEXT_TO_SQL_MODEL", "mistral:latest")            # Model to use for text-to-SQL generation
            }
        )

    def init_db_connection(self):
        # Update your DB connection string based on selected DB engine - current connection string is for Postgres
        self.engine = create_engine(f"postgresql+psycopg2://{self.valves.DB_USER}:{self.valves.DB_PASSWORD}@{self.valves.DB_HOST}:{self.valves.DB_PORT}/{self.valves.DB_DATABASE}")
        return self.engine

    async def on_startup(self):
        # This function is called when the server is started.
        self.init_db_connection()

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def validate_query(self, query):
        """Validate SQL query."""
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
            query = structured_llm.invoke(prompt)
            if debug_mode:
                print(f"Debug: Generated SQL query: {query}")
            is_valid, error_message = self.validate_query(query)
            if is_valid:
                return query
            else:
                prompt = (f"The generated query is invalid. Error: {error_message}. "
                          f"Attempt {attempt + 1} of {max_attempts}. "
                          f"Previous invalid query: {query}. "
                          f"Please correct the query.")

        return "Failed to generate a valid query after multiple attempts."

    def execute_query(self, query):
        """Execute SQL query."""
        execute_query_tool = QuerySQLDataBaseTool(db=self.db)
        if debug_mode:
            print(f"Debug: Executing SQL query: {query}")
        result = execute_query_tool.invoke(query)
        if debug_mode:
            print(f"Debug: SQL query result: {result}")
        return result

    def generate_answer(self, user_message, query, result):
        """Answer question using retrieved information as context."""
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f'Question: {user_message}\n'
            f'SQL Query: {query}\n'
            f'SQL Result: {result}'
        )
        if debug_mode:
            print(f"Debug: Generating answer for question: {user_message} with SQL result: {result}")
        response = self.model.invoke(prompt)
        if debug_mode:
            print(f"Debug: Generated answer: {response.content}")
        return response.content

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # Debug logging is required to see what SQL query is generated by the LlamaIndex library; enable on Pipelines server if needed

        # Create database reader for Postgres
        self.db = SQLDatabase(self.engine, include_tables=[self.valves.DB_TABLE])

        # Set up LLM connection; uses phi3 model with 128k context limit since some queries have returned 20k+ tokens
        self.model = ChatMistralAI(model=self.valves.TEXT_TO_SQL_MODEL, base_url=self.valves.MISTRAL_HOST, request_timeout=180.0, context_window=30000)

        query = self.write_query(user_message)
        result = self.execute_query(query)
        answer = self.generate_answer(user_message, query, result)

        return answer
