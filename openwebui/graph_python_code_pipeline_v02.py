from typing_extensions import TypedDict, Annotated
from typing import List, Union, Generator, Iterator
import subprocess
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from transformers import pipeline
import base64 

# Global debug mode variable
debug_mode = True

class QueryOutput(TypedDict):
    """Generated SQL query and graph type."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]
    graph_type: Annotated[str, ..., "Graph type that the user wants to create."]

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

class Pipeline:
    def __init__(self):
        self.name = "InfoGraph Code Assistant"
        self.model = model  # Ensure the model is accessible within the class
        self.db = db

    async def on_startup(self):
        if debug_mode:
            print(f"Debug: on_startup:{__name__}")

    async def on_shutdown(self):
        if debug_mode:
            print(f"Debug: on_shutdown:{__name__}")

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
        # Replace %Y with %%Y to escape it for pymysql
        query = query.replace('%Y', '%%Y')
        query = query.replace('%m', '%%m')
        if debug_mode:
            print(f"Debug: Executing SQL query: {query}")
        df = pd.read_sql(query, engine)
        if debug_mode:
            print(f"Debug: SQL query result: {df.head()}")
        return df

    def map_query_to_data(self, label):
        if 'Line Chart' in label:
            return self.generate_line_chart
        elif 'Bar Chart' in label:
            return self.generate_bar_chart
        elif 'Pie Chart' in label:
            return self.generate_pie_chart
        # Add more mappings as needed
        else:
            return None

    def generate_line_chart(self, df, x_col, y_col):
        """
        Generate a line chart from a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The column name for the x-axis.
        y_col (str): The column name for the y-axis.
        """
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x=x_col, y=y_col)
        plt.title(f'Line Chart of {y_col} by {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.savefig('graph.png')
        plt.close()

    def generate_bar_chart(self, df, x_col, y_col):
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x=x_col, y=y_col)
        plt.title(f'Bar Chart of {y_col} by {x_col}')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.savefig('graph.png')
        plt.close()

    def generate_pie_chart(self, df, col):
        plt.figure(figsize=(10, 6))
        df[col].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'Pie Chart of {col}')
        plt.ylabel('')
        plt.savefig('graph.png')
        plt.close()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:

        # Write SQL query
        query_output = self.write_query(user_message)

        # Validate SQL query
        is_valid, error_message = self.validate_query(query_output)
        if not is_valid:
            return f"Invalid SQL query: {error_message}"

        # Execute SQL query
        df = self.execute_query(query_output)

        # Map query to data
        graph_function = self.map_query_to_data(query_output['graph_type'])

        if graph_function is None:
            return "Unsupported graph type"

        # Determine column names dynamically
        x_col = df.columns[0] if len(df.columns) > 0 else None
        y_col = df.columns[1] if len(df.columns) > 1 else None

        if x_col is None or y_col is None:
            return "DataFrame does not have enough columns to generate the graph."

        # Execute selected graph drawing function
        graph_function(df, x_col=x_col, y_col=y_col)

        # Read the image file into a bytes buffer
        with open('graph.png', 'rb') as f:
            image_data = f.read()

        # Convert the image data to a base64 string
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Remove the local file
        #os.remove('graph.png')

        return f"```html\n<div><img src='data:image/png;base64,{image_base64}' alt='Graph'></div>\n```"