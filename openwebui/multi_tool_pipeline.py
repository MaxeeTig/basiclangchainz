"""
title: ClusterTools Code Assistant (Database LLM graph cluster-plotting tool)
author: Mikael Kondakhchan
date: 23 Jan 2025
version: 0.5
license: MIT
requirements: langchain==0.3.1,  langchain-mistralai==0.2.4, langchain_openai==0.2.1, langchain_text_splitters==0.3.2, pydantic==2.8.2, pymysql==1.1.1
description: A pipeline for using LLM to build graphs on the basis of database information.
"""

import base64
from langchain.memory import ConversationBufferWindowMemory
from typing_extensions import TypedDict, Annotated
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai import ChatMistralAI
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import io
import os
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool, tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from typing import List, Sequence
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

# Initialize the Mistral LLM
model = ChatMistralAI(model="mistral-large-latest")

#create variables to use across the tools
df = pd.DataFrame()
html_string =''

#                   ~~~~~ SQL query tool ~~~~~~

class QueryInput(BaseModel):
    """user's question to generate SQL query and extract data from db."""
    user_input: str = Field(..., description="User's question for getting information from database, creating a sql query")

class QueryOutput(BaseModel):
    """Generated SQL query and graph type."""
    query: str = Field(..., description="Syntactically valid SQL query.")

def validate_query(query_output):
    """Validate SQL query."""
    query = query_output.query
    if debug_mode:
        print(f"Debug: validating SQL: {query}")
    try:
        db.run(query)
        return True, None
    except Exception as e:
        return False, str(e)

@tool("pre_fetch_data", args_schema=QueryInput, return_direct=True)
def pre_fetch_data(user_input, max_attempts=3):
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
USE CORRECT RELATIONS BETWEEN COLUMNS AND TABLES
USE ONLY RELEVANT COLUMNS RELATED TO USER QUERY
IF REQUIRED SORT DATA IN THE ORDER ASCOR DESC (E.G. TO SHOW MOST FRESH OPERATIONS BY DATE)
CREATE SYNTACTICALLY CORRECT DIALECT: {dialect} QUERY TO HELP USER FIND THE ANSWER
THIS QUERY WILL BE USED FOR DIFFERENT GRAPHS DRAWING
SELECTED DATA WILL BE PLACED TO PANDAS DATAFRAME

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
1.5. IDENTIFY TYPE OF GRAPH WHICH USER WANTS TO DRAW AND PUT IT IN THE STRUCTURED OUTPUT IN 'graph_type' dictionary item
1.6. IF USER HAVE NOT PROVIDED TYPE OF GRAPH SUGGEST OPTION THE BEST SUITE USER NEEDS (e.g: Bar Chart,  Line Chart, Pie Chart, Scatter Plot,
Histogram, Heatmap)

2. **CONSTRUCT THE SQL QUERY:**
2.1. FORMULATE the basic SQL structure based on the identified task (e.g., SELECT FROM).
2.2. INCORPORATE the correct tables, columns, and conditions into the query.
2.3. ADD any necessary JOIN clauses if multiple tables are involved.
2.4. USE aggregate functions or GROUP BY if the query requires summarization of data.
2.5. OPTIMIZE the query by considering indexes, limiting results, or other performance enhancements.
2.6. ATTENTION to columns order in query, for example for Pie Chart column with attribute (e.g customer_person_gender) goes first, count (e.g customer_count) goes second

3. **FINALIZE AND EXPLAIN:**
3.1. REVIEW the query for correctness and optimization.
3.2. PROVIDE a brief explanation of how the query was constructed and why certain choices were made.
3.3. PRESENT the final SQL query to the user.
3.4. IDENTIFY type of chart on user query or that THE BEST SUITE USER NEEDS
3.5. INCLUDE type of chart in response as separate IN 'graph_type' dictionary item


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
"Draw line chart of operations counts by month."
**SQL Query Generation:**
1. **Understanding the User Query:**
- Task: SELECT operations
- Table: operations
- Condition: all operations
- Group: GROUP operation_id by month
- Summary: count number of operations
2. **Constructing the SQL Query:**
- SELECT operation_id from operations
- GROUP BY month with date conversion
- SORT ASCENDING
**Final SQL Query:** "SELECT DATE_FORMAT(oper_date, '%Y-%m') AS month, COUNT(*) AS operation_count FROM operations GROUP BY month ORDER BY month ASC;"
3. **Understand graph type**
- IF user mentioned graph type in query identify graph type
- place graph type in the answer in structured output to 'graph_type' dictionary item
- IF user have not specified graph type in query suggest suitable graph type
**Final Answer:** 'graph_type': 'Line Chart'

**User Query:**
"show me pie chart diagram of distribution of my customers by gender"
**SQL Query Generation:**
1. **Understanding the User Query:**
- Task: SELECT customers
- Tables: customers
- Conditions: all data
- GROUP by gender
- Summary: count number of customers of each gender and 'none'
2. **Constructing the SQL Query:**
- SELECT customer_person_gender.
- GROUP by customer_person_gender
- SUM UP records to find total number SELECT count(*)
**Final SQL Query:**
"SELECT customer_person_gender, COUNT(*) AS customer_count FROM customers GROUP BY customer_person_gender;"
3. **Understand graph type**
- IF user mentioned graph type in query identify graph type
- place graph type in the answer in structured output to 'graph_type' dictionary item
- IF user have not specified graph type in query suggest suitable graph type
**Final Answer:** 'graph_type': 'Pie Chart'
'''

    query_prompt_template = ChatPromptTemplate([
                ("system", prompt_template),
                ("user", ""),
                ])

    prompt = query_prompt_template.invoke ({
            "dialect": "MySQL",
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": user_input,
        })
    global html_string
    html_string = ''
    structured_llm = model.with_structured_output(QueryOutput)

    if debug_mode:
        print(f"Debug: Generating SQL query for question: {user_input}")
    for attempt in range(max_attempts):
        query_output = structured_llm.invoke(prompt)
        if debug_mode:
            print(f"Debug: Generated SQL query: {query_output}")
        is_valid, error_message = validate_query(query_output)
        if is_valid:
            query = query_output.query
            query = query.replace('%Y', '%%Y')
            query = query.replace('%m', '%%m')
            if debug_mode:
                print(f"Debug: Executing SQL query: {query}")
            global df
            df = pd.read_sql(query, engine)
            if debug_mode:
                print(f"Debug: SQL query result: {df.head()}")
            return query
        else:
            prompt = (f"The generated query is invalid. Error: {error_message}. "
                      f"Attempt {attempt + 1} of {max_attempts}. "
                      f"Previous invalid query: {query_output}. "
                      f"Please correct the query.")

    return {"query": "Failed to generate a valid query after multiple attempts."}


#                   ~~~~~ python code for graphs tool ~~~~~~

class PythonGraphInput(BaseModel):
    """user's question and graph type to generate python code for making a matplotlib graph."""
    user_input: str = Field(..., description="User's question for creating a graph.")
    graph_type: str = Field(..., description="Type of graph to generate.(e.g., 'barchart', 'scatter', 'box plot', 'line chart, 'histogram', 'pie chart').")

class PythonGraphOutput(TypedDict):
    """Generated python code."""
    query: Annotated[str, ..., "Syntactically valid Python code."]

@tool("python_graph_code", args_schema=PythonGraphInput, return_direct=True)
def python_graph_code(user_input,graph_type, max_attempts=3):
    system_prompt = """
    You are a Python code writing agent specialized in creating plots and graphs using library matplotlib.
    Your task is to generate Python code that that creates a graph using data from Pandas dataframe based on user's request and provided graph type.
    This is user request {input}. and this is the type of graph you need to create {graph_type}
    The data is pre-fetched into a DataFrame named 'df'. You should use matplotlib to create the graph of needed type and save it into png.
    Analyze provided sample data, drop from 'df' useless features, select appropriate features to draw a graph.
    Do not create sample data dataframe, use for creating a graph initial 'df' to create from it cleaned dataframe like:
    df_cleaned = df.drop(columns=['column1','column2']) where column1, column2, columnN are useless features.
    Always do this in the end of your code:
    ``` 
    filename = 'graph.png'
    plt.savefig(filename)
    plt.close()
    globals()['filename'] = filename
    ```

    ### Instructions:
    1. **DataFrame**: Assume the data is already loaded into a Pandas DataFrame named 'df' and NEVER create a new df.
    2. **Matplotlib**: Use Matplotlib to create the graph.
    3. **Parameters**: Use the following parameters from the operations table to customize the graph:
       - `features`: The columns to use as features for creating a graph.
       - `graph_type`: The type of graph to create (e.g., 'barchart', 'scatter', 'box plot', 'line chart, 'histogram', 'pie chart').
    4. **Output**: Generate the Python code as a string and return it.
    5. **Code**: save created plot in `filename` variable

    ### DataFrame Columns:
    {columns}

    ### Example:
    Given the parameters:
    - `features`: ['feature1', 'feature2']
    - `graph_type`: 'scatter'

    The generated code should look like this:
    ```python
    import matplotlib.pyplot as plt

    # Assuming df is already defined and contains the data
    features = df[['feature1', 'feature2']]

    plt.figure(figsize=(10, 6))
    plt.scatter(features['feature1'], features['feature2'], cmap='viridis')
    plt.title('Customer Segmentation')
    plt.xlabel('Feature 1') 
    plt.ylabel('Feature 2')
    filename = 'graph.png'
    plt.savefig(filename)
    plt.close()
    globals()['filename'] = filename
    ```

    ### Task:
    Generate the Python code based on the provided parameters.
    """
    code_prompt_template = ChatPromptTemplate([
        ("system", system_prompt),
        ("user", ""),
    ])

    prompt = code_prompt_template.invoke({
        "columns": df.columns.tolist(),
        "input": user_input,
        "graph_type": graph_type,
    })

    structured_llm = model.with_structured_output(CodeOutput)
    for attempt in range(max_attempts):
        code_response = structured_llm.invoke(prompt)
        code = code_response['query']
        print(code)
        try:
            # Execute the generated Python code
            exec(code, globals(), locals())

            # Retrieve the 'filename' object from the local variables
            filename = globals().get('filename', None)
            print(filename)
            if filename is None:
                return "no graph has been generated"
            else:
                with open(filename, 'rb') as f:
                    image_data = f.read()

                # Convert the image data to a base64 string
                image_base64 = base64.b64encode(image_data).decode('utf-8')

                # Remove the local file
                os.remove(filename)
                global html_string
                html_string = f"\n\n```html\n<div><img src='data:image/png;base64,{image_base64}' alt='Graph'></div>\n```"

                # Return the base64 string embedded in HTML
                return f"Generated a cluster graph based on the user's query: {user_input}."

        except Exception as e:
            print(f"Exception occurred {e}")
            prompt = (f"The generated query is invalid. Error: {e}. "
                      f"Attempt {attempt + 1} of {max_attempts}. "
                      f"Previous invalid query: {code}. "
                      f"Please correct the query.")
    return "Failed to generate a valid code for cluster graph after multiple attempts."
#                   ~~~~~ Python code for clustering tool ~~~~~~


#                   ~~~~~ some analysis tool? ~~~~~~

#code generation

class CodeInput(BaseModel):
    """user's question to generate python code for making a matplotlib graph."""
    user_input: str = Field(..., description="User's question for creating a graph.")

class CodeOutput(TypedDict):
    """Generated python code."""
    query: Annotated[str, ..., "Syntactically valid Python code."]

@tool("graph_code", args_schema=CodeInput, return_direct=True)
def graph_code(user_input, max_attempts=3):
    """
    A tool for generating python code for clustering and drawing a plot
    """
    system_prompt = """
    You are a Python code writing agent specialized in creating clustering analysis using libraries like Scikit-learn.
    Your task is to generate Python code that performs clustering analysis on data from a Pandas DataFrame.
    This is user request {input}.
    The data is pre-fetched into a DataFrame named 'df'. You should use Scikit-learn to create the clustering analysis.
    Analyze provided sample data, drop from 'df' useless features, select appropriate features to build clusters.
    Do not create sample data dataframe, use for cluster analysis initial 'df' to create from it cleaned dataframe like:
    df_cleaned = df.drop(columns=['column1','column2']) where column1, column2, columnN are useless features.
    Always do this in the end of your code:
    ``` 
    filename = 'graph_date.png'
    plt.savefig(filename)
    plt.close()
    globals()['filename'] = filename
    ```

    ### Instructions:
    1. **DataFrame**: Assume the data is already loaded into a Pandas DataFrame named 'df'.
    2. **Scikit-learn**: Use Scikit-learn to create the clustering analysis.
    3. **Parameters**: Use the following parameters from the operations table to customize the clustering:
       - `features`: The columns to use as features for clustering.
       - `n_clusters`: The number of clusters to create.
       - `algorithm`: The clustering algorithm to use (e.g., 'kmeans', 'dbscan', 'hierarchical').
       - `title`: The title of the clustering analysis.
    4. **Output**: Generate the Python code as a string and return it.
    5. **Code**: save created plot in `filename` variable

    ### DataFrame Columns:
    {columns}

    ### Example:
    Given the parameters:
    - `features`: ['feature1', 'feature2']
    - `n_clusters`: 3
    - `algorithm`: 'kmeans'
    - `title`: 'Customer Segmentation'

    The generated code should look like this:
    ```python
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    # Assuming df is already defined and contains the data
    features = df[['feature1', 'feature2']]
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(features)
    labels = kmeans.labels_

    plt.figure(figsize=(10, 6))
    plt.scatter(features['feature1'], features['feature2'], c=labels, cmap='viridis')
    plt.title('Customer Segmentation')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    filename = 'graph_date.png'
    plt.savefig(filename)
    plt.close()
    globals()['filename'] = filename
    ```

    ### Task:
    Generate the Python code based on the provided parameters.
    """

    code_prompt_template = ChatPromptTemplate([
        ("system", system_prompt),
        ("user", ""),
    ])

    prompt = code_prompt_template.invoke({
        "columns": df.columns.tolist(),
        "input": user_input,
    })

    structured_llm = model.with_structured_output(CodeOutput)
    for attempt in range(max_attempts):
        code_response = structured_llm.invoke(prompt)
        code = code_response['query']
        print(code)
        try:
            # Execute the generated Python code
            exec(code, globals(), locals())

            # Retrieve the 'filename' object from the local variables
            filename = globals().get('filename', None)
            print(filename)
            if filename is None:
                return "no graph has been generated"
            else:
                with open(filename, 'rb') as f:
                    image_data = f.read()

                # Convert the image data to a base64 string
                image_base64 = base64.b64encode(image_data).decode('utf-8')

                # Remove the local file
                os.remove(filename)
                html_string = f"\n\n```html\n<div><img src='data:image/png;base64,{image_base64}' alt='Graph'></div>\n```"

                # Return the base64 string embedded in HTML
                return f"Generated a cluster graph based on the user's query: {user_input}."

        except Exception as e:
            print(f"Exception occurred {e}")
            prompt = (f"The generated query is invalid. Error: {e}. "
                      f"Attempt {attempt + 1} of {max_attempts}. "
                      f"Previous invalid query: {code}. "
                      f"Please correct the query.")
    return "Failed to generate a valid code for cluster graph after multiple attempts."




class Pipeline:
    def __init__(self):
        self.name = "MultiTool analysis assistant"
        self.model = model  # Ensure the model is accessible within the class
        self.db = db
        self.df = None
        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)

    async def on_startup(self):
        if debug_mode:
            print(f"Debug: on_startup:{__name__}")

    async def on_shutdown(self):
        if debug_mode:
            print(f"Debug: on_shutdown:{__name__}")


    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict):
        try:
            if debug_mode:
                print(f"Debug: pipe:{__name__}")

            if body.get("title", False):
                if debug_mode:
                    print("Debug: Title Generation")
                return "Cluster Code Assistant"

            tools: Sequence[BaseTool] = [pre_fetch_data, python_graph_code]

            prompt = ChatPromptTemplate.from_messages([
                ("system", """
            You are a helpful assistant that can communicate with person in chat and use tools for prefetching data from database or drawing a graph.
            - Use the tools provided to perform specific tasks.
            - If the user's query is unclear, ask for clarification.
            - Always be polite and informative.
                """),
                MessagesPlaceholder("chat_history"),  # Placeholder for chat history
                ("user", "{input}"),  # Placeholder for user input
                MessagesPlaceholder("agent_scratchpad"),  # Placeholder for intermediate steps
            ])
            global html_string
            html_string = ''
            agent = create_tool_calling_agent(model, tools, prompt)
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=self.memory)
            response = agent_executor.invoke({"input": user_message, "chat_history": messages})
            if not df.empty:
                print("df passed to pipe successfully")
                print(df.head())
            else:
                print("empty df")
            print("response from last question")
            print(response)
            return response['output']+ html_string
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise