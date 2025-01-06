from typing_extensions import TypedDict, Annotated
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_mistralai import ChatMistralAI

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

class PythonCodeWriterGraph:
    def __init__(self, model: ChatMistralAI):
        self.model = model

    def info(self, user_input):
        return f"PythonCodeWriterGraph: This agent is responsible for writing Python code. User Input: {user_input}"

    def write_code(self, state: State, df):
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
        state['code'] = structured_llm.invoke(prompt)
        return state

class PythonCodeWriterCluster:
    def __init__(self, model: ChatMistralAI):
        self.model = model

    def info(self, user_input):
        return f"PythonCodeWriterCluster: This agent is responsible for writing Python code. User Input: {user_input}"

    def write_code(self, state: State, df):
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
        state['code'] = structured_llm.invoke(prompt)
        return state
