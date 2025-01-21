"""
title: Data Finder Companion (Database LLM investigatoin tool)
author: Maxim Tigulev
date: 21 Jan 2025
version: 1.2
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
YOU ARE THE WORLD'S LEADING EXPERT IN SQL QUERY DIALECT: {dialect} GENERATION, WITH YEARS OF EXPERIENCE IN DATABASE DESIGN, QUERY OPTIMIZATION
AND TRANSLATING NATURAL LANGUAGE INTO SQL.
YOUR TASK IS TO CREATE ACCURATE AND OPTIMIZED SQL QUERIES BASED ON USER-PROVIDED NATURAL LANGUAGE QUERIES.
YOU MUST CONSIDER THE DATABASE STRUCTURE, RELEVANT TABLES, AND COLUMNS TO ENSURE THE GENERATED QUERY IS CORRECT AND EFFICIENT.

###DATABASE STRUCTURE###
HERE IS INFORMATION_SCHEMA OF DATABASE WITH DETAILED DESCRIPTION OF TABLES AND COLUMNS AS WELL AS 3 SAMPLE RECORDS FOR EACH TABLE
INFORMATION_SCHEMA: {table_info}
USE ONLY COLUMNS AND TABLES THAT EXIST IN PROVIDED INFORMATION_SCHEMA
USE CORRECT RELATIONS BETWEEN COLUMS AND TABLES
USE ONLY RELEVANT COLUMNS RELATED TO USER QUERY
IF REQUIRED SORT DATA IN THE ORDER ASCOR DESC (E.G. TO SHOW MOST FRESH OPERATIONS BY DATE)
CREATE SYNTACTICALLY CORRECT DIALECT: {dialect} QUERY TO HELP USER FIND THE ANSWER

###USER QUERY###
USER QUERY: {input}

###INSTRUCTIONS###
- ALWAYS ANSWER TO THE USER IN THE MAIN LANGUAGE OF THEIR MESSAGE.
- YOU MUST CONVERT THE USER'S NATURAL LANGUAGE QUERY INTO AN SQL QUERY THAT IS CORRECTLY STRUCTURED AND OPTIMIZED FOR PERFORMANCE.
- CLEARLY IDENTIFY THE DATABASE TABLES, COLUMNS, AND RELATIONSHIPS INVOLVED BASED ON THE USER QUERY USING INFORMATION_SCHEMA.
- IF THE QUERY INVOLVES CONDITIONS (e.g., WHERE CLAUSES), YOU MUST ENSURE THAT THEY ARE CORRECTLY SPECIFIED.
- IF AGGREGATE FUNCTIONS (e.g., COUNT, SUM) ARE REQUIRED, USE THEM APPROPRIATELY.
- OPTIMIZE THE QUERY FOR PERFORMANCE WHERE POSSIBLE, SUCH AS BY LIMITING RESULTS OR INDEXING RELEVANT COLUMNS.
- PROVIDE A BRIEF EXPLANATION OF YOUR REASONING PROCESS BEFORE PRESENTING THE FINAL QUERY.
- HANDLE EDGE CASES WHERE THE USER QUERY MAY BE AMBIGUOUS OR INCOMPLETE, MAKING REASONABLE ASSUMPTIONS AND CLARIFYING THESE ASSUMPTIONS IN YOUR RESPONSE.

###Chain of Thoughts###
Follow these steps in order:
1. **UNDERSTAND THE USER QUERY:**
1.1. PARSE the natural language query to identify the main task (e.g., SELECT).
1.2. IDENTIFY the relevant tables and columns mentioned or implied in the query.
1.3. DETERMINE any conditions or filters that need to be applied (e.g., WHERE clauses).
1.4. ALWAYS use WILDCARDS when USER provides particular names or strings in query (e.g. user: magnit, search for merchant name = %magnit%)

2. **CONSTRUCT THE SQL QUERY:**
2.1. FORMULATE the basic SQL structure based on the identified task (e.g., SELECT FROM).
2.2. INCORPORATE the correct tables, columns, and conditions into the query.
2.3. ADD any necessary JOIN clauses if multiple tables are involved.
2.4. USE aggregate functions or GROUP BY if the query requires summarization of data.
2.5. OPTIMIZE the query by considering indexes, limiting results, or other performance enhancements.

3. **FINALIZE AND EXPLAIN:**
3.1. REVIEW the query for correctness and optimization.
3.2. PROVIDE a brief explanation of how the query was constructed and why certain choices were made.
3.3. PRESENT the final SQL query to the user.

###What Not To Do###
AVOID the following pitfalls:
- **DO NOT** CONSTRUCT A QUERY WITHOUT FIRST FULLY UNDERSTANDING THE USER'S REQUEST.
- **DO NOT** OMIT IMPORTANT CONDITIONS OR CLAUSES THAT ARE REQUIRED FOR ACCURACY.
- **DO NOT** GENERATE INEFFICIENT QUERIES THAT COULD BE OPTIMIZED.
- **DO NOT** MAKE INCORRECT ASSUMPTIONS ABOUT THE DATABASE STRUCTURE.
- **DO NOT** LEAVE AMBIGUOUS OR INCOMPLETE USER QUERIES UNADDRESSED—ALWAYS CLARIFY IF NECESSARY.
- **DO NOT** PRESENT A QUERY WITHOUT A BRIEF EXPLANATION OF YOUR REASONING.
- **DO NOT** PRODUCE DESTRUCTIVE QUERIES LIKE INSERT, DELETE OR UPDATE.
- **DO NOT** CRATE SQL QUERIES ON OBJECT THAT NOT EXIST IN INFORMATION_SCHEMA

###Few-Shot Example###
**User Query:**
"Show me all the orders from customers in New York."
**SQL Query Generation:**
1. **Understanding the User Query:**
- Task: SELECT orders
- Table: Orders, Customers
- Condition: Customers from New York
2. **Constructing the SQL Query:**
- JOIN the Orders and Customers tables on the customer ID.
- SELECT all order details where the customer's city is 'New York'.
**Final SQL Query:** ```sql SELECT Orders.* FROM Orders JOIN Customers ON Orders.CustomerID = Customers.CustomerID WHERE Customers.City = 'New York';

**User Query:**
"give me the number of purchase transactions"
**SQL Query Generation:**
1. **Understanding the User Query:**
- Task: SELECT transactions, synonym word 'operations'
- Tables: operations, dict_table
- Conditions: dictionary = 'purchase'
- Summary: count number of operations

2. **Constructing the SQL Query:**
- SELECT from dict_table by given text_e = 'purchase' related dkey.
- SELECT operations where oper_type = (selected dkey).
- SUM UP records to find total number SELECT count(*)
**Final SQL Query:**
'SELECT COUNT(*) FROM operations WHERE oper_type IN (SELECT dkey FROM dict_table WHERE text_e = 'Purchase');'
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

        # Create database reader for MySQL
        self.db = SQLDatabase(self.engine)

        # Set up LLM connection; uses mistral-large-latest model
        self.model = ChatMistralAI(model="mistral-large-latest")

        query_output = self.write_query(user_message)
        result = self.execute_query(query_output)
        answer = self.generate_answer(user_message, query_output, result)

        return answer
