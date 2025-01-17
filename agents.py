from typing_extensions import TypedDict, Annotated
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_mistralai import ChatMistralAI
import ast

# Global debug mode variable
debug_mode = True

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

class CodeOutput(TypedDict):
    """Generated python code."""
    query: Annotated[str, ..., "Syntactically valid Python code."]

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
        if debug_mode:
            print(f"Debug: validating SQL: {query}")
        try:
            result = self.db.run(query)
            return True, None
        except Exception as e:
            return False, str(e)

    def write_query(self, state: State, max_attempts=3):
        """Generate SQL query to fetch information. State object is input"""

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
                "input": state["question"],
            })

        structured_llm = self.model.with_structured_output(QueryOutput)

        if debug_mode:
            print(f"Debug: Generating SQL query for question: {state['question']}")
        for attempt in range(max_attempts):
            state['query'] = structured_llm.invoke(prompt)
            if debug_mode:
                print(f"Debug: Generated SQL query: {state['query']}")
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
        if debug_mode:
            print(f"Debug: Executing SQL query: {state['query']}")
        state['result'] = execute_query_tool.invoke(state["query"])
        if debug_mode:
            print(f"Debug: SQL query result: {state['result']}")
        return state

    def generate_answer(self, state: State):
        """Answer question using retrieved information as context. State object as input"""
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f'Question: {state["question"]}\n'
            f'SQL Query: {state["query"]}\n'
            f'SQL Result: {state["result"]}'
        )
        if debug_mode:
            print(f"Debug: Generating answer for question: {state['question']} with SQL result: {state['result']}")
        response = self.model.invoke(prompt)
        if debug_mode:
            print(f"Debug: Generated answer: {response.content}")
        state['answer'] = response.content
        return state

class PythonCodeWriterGraph:
    def __init__(self, model: ChatMistralAI):
        self.model = model

    def info(self, user_input):
        return f"PythonCodeWriterGraph: This agent is responsible for writing Python code. User Input: {user_input}"

    def validate_code(self, state: State):
        """Validate Python code using static analysis. State object as input"""
        code = state['code']['query']
        if debug_mode:
            print(f"Debug: validating code: {code}")
        try:
            # Parse the code to check for syntax errors
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def write_code(self, state: State, df, max_attempts=3):
        """Generate Python code based on the input question. State object as input"""

        # Get the columns of the DataFrame
        columns = df.columns.tolist()

        system_prompt = """
You are a Python code writing agent specialized in creating graphs using Matplotlib.
Your task is to select required columns from dataframe and generate Python code that visualizes data from a Pandas DataFrame.
This is user request {input}.
Read user request carefully and define: graph type need to draw, parameters for required graph type.
Assume the data is pre-fetched into a DataFrame named 'df'.
Do not create sample dataframe.
You should use Matplotlib to create the graphs.

### Instructions:
1. **DataFrame**: Assume the data is already loaded into a Pandas DataFrame named 'df'. Select required columns from DataFrame. No sample dataframe required.
2. **Matplotlib**: Use Matplotlib to create the graphs.
3. **Parameters**: Use the following parameters from the operations table to customize the graph:
   - `x_column`: The column name to use for the x-axis.
   - `y_column`: The column name to use for the y-axis.
   - `graph_type`: The type of graph to create (e.g., 'line', 'bar', 'scatter', 'histogram').
   - `title`: The title of the graph.
   - `xlabel`: The label for the x-axis.
   - `ylabel`: The label for the y-axis.
4. **Output**: Generate the Python code as a string and return it.

### DataFrame Columns:
{columns}

### Task:
Generate the Python code based on the provided parameters.
"""

        code_prompt_template = ChatPromptTemplate([
            ("system", system_prompt),
            ("user", ""),
        ])

        prompt = code_prompt_template.invoke({
            "columns": columns,
            "input": state["question"],
        })

        structured_llm = self.model.with_structured_output(CodeOutput)

        if debug_mode:
            print(f"Debug: Generating Python code for question: {state['question']}")
        for attempt in range(max_attempts):
            state['code'] = structured_llm.invoke(prompt)
            if debug_mode:
                print(f"Debug: Generated Python code: {state['code']}")
            is_valid, error_message = self.validate_code(state)
            if is_valid:
                return state
            else:
                prompt = (f"The generated code is invalid. Error: {error_message}. "
                          f"Attempt {attempt + 1} of {max_attempts}. "
                          f"Previous invalid code: {state['code']}. "
                          f"Please correct the code.")

        state['code'] = "Failed to generate a valid code after multiple attempts."
        return state

class PythonCodeWriterCluster:
    def __init__(self, model: ChatMistralAI):
        self.model = model

    def info(self, user_input):
        return f"PythonCodeWriterCluster: This agent is responsible for writing Python code. User Input: {user_input}"

    def validate_code(self, state: State):
        """Validate Python code using static analysis. State object as input"""
        code = state['code']['query']
        if debug_mode:
            print(f"Debug: validating code: {code}")
        try:
            # Parse the code to check for syntax errors
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def write_code(self, state: State, df, max_attempts=3):
        """Generate Python code based on the input question. State object as input"""

        # Get the columns of the DataFrame
        columns = df.columns.tolist()
        # Get the first few rows of the DataFrame as sample data
        sample_data = df.head().to_dict(orient='split')

        system_prompt = """
You are a Python code writing agent specialized in creating clustering analysis using libraries like Scikit-learn.
Your task is to generate Python code that performs clustering analysis on data from a Pandas DataFrame.
This is user request {input}.
The data is pre-fetched into a DataFrame named 'df'. You should use Scikit-learn to create the clustering analysis.
Analyze provided sample data, drop from 'df' useless features, select appropriate features to build clusters.
Do not create sample data dataframe, use for cluster analysis initial 'df' to create from it cleaned dataframe like:
df_cleaned = df.drop(columns=['column1','column2']) where column1, column2, columnN are useless features.

### Instructions:
1. **DataFrame**: Assume the data is already loaded into a Pandas DataFrame named 'df'.
2. **Scikit-learn**: Use Scikit-learn to create the clustering analysis.
3. **Parameters**: Use the following parameters from the operations table to customize the clustering:
   - `features`: The columns to use as features for clustering.
   - `n_clusters`: The number of clusters to create.
   - `algorithm`: The clustering algorithm to use (e.g., 'kmeans', 'dbscan', 'hierarchical').
   - `title`: The title of the clustering analysis.
4. **Output**: Generate the Python code as a string and return it.

### DataFrame Columns:
{columns}

### Sample Data:
{sample_data}

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
plt.show()
```
### Task:
Generate the Python code based on the provided parameters.
"""

        code_prompt_template = ChatPromptTemplate([
            ("system", system_prompt),
            ("user", ""),
        ])

        prompt = code_prompt_template.invoke({
            "columns": columns,
            "sample_data": sample_data,
            "input": state["question"],
        })

        structured_llm = self.model.with_structured_output(CodeOutput)

        if debug_mode:
            print(f"Debug: Generating Python code for question: {state['question']}")
        for attempt in range(max_attempts):
            state['code'] = structured_llm.invoke(prompt)
            if debug_mode:
                print(f"Debug: Generated Python code: {state['code']}")
            is_valid, error_message = self.validate_code(state)
            if is_valid:
                return state
            else:
                prompt = (f"The generated code is invalid. Error: {error_message}. "
                          f"Attempt {attempt + 1} of {max_attempts}. "
                          f"Previous invalid code: {state['code']}. "
                          f"Please correct the code.")

        state['code'] = "Failed to generate a valid code after multiple attempts."
        return state
