from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import subprocess
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_mistralai import ChatMistralAI
import matplotlib.pyplot as plt
import io

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

class Pipeline:
    def __init__(self):
        self.name = "InfoGraph Code Assistant"

    async def on_startup(self):
        if debug_mode:
            print(f"Debug: on_startup:{__name__}")

    async def on_shutdown(self):
        if debug_mode:
            print(f"Debug: on_shutdown:{__name__}")

    def pre_fetch_data(self, intent, user_input):
        sql_query = '''
        SELECT oper_date, issuer_card_id, mcc, merchant_city, merchant_country, oper_amount_amount_value, oper_amount_currency, oper_type
        FROM operations
        WHERE is_reversal = 0
        AND oper_type IS NOT NULL
        AND mcc <> ''
        AND merchant_country REGEXP '^[0-9]+$'
        AND merchant_country <> '0'
        AND oper_amount_amount_value > 1.00
        AND oper_amount_amount_value < 1000000000.00
        '''
        if debug_mode:
            print(f"Debug: Pre-fetching data for intent: {intent} with user input: {user_input}")
            print("Debug: pre-fetching data to dataframe... ")

        # Fetch data from the database
        df = pd.read_sql(sql_query, engine)
        if debug_mode:
            print(f"Debug: Pre-fetched data: {df.head()}")
        return df

    def execute_python_code(self, code, df):
        try:
            exec(code, globals(), locals())
            return locals()['result'], 0
        except Exception as e:
            return str(e), 1

    def generate_graph(self, code, df):
        try:
            exec(code, globals(), locals())
            fig = locals().get('fig', None)
            if fig:
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                return buf.getvalue(), 0
            else:
                return None, 1
        except Exception as e:
            return str(e), 1

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        if debug_mode:
            print(f"Debug: pipe:{__name__}")
            print(messages)
            print(user_message)

        if body.get("title", False):
            if debug_mode:
                print("Debug: Title Generation")
            return "InfoGraph Code Assistant"
        else:
            # Pre-fetch data
            df = self.pre_fetch_data('draw graph', user_message)
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
            "input": user_message,
            })

            structured_llm = self.model.with_structured_output(CodeOutput)

            code_response = structured_llm.invoke(prompt)
            code = code_response['query']

            # Execute the generated Python code
            if debug_mode:
                print(f"Debug: Executing generated code: {code}")
            graph_image, return_code = self.generate_graph(code, df)
            if return_code == 0:
                return graph_image
            else:
                return graph_image
