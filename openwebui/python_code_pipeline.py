from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import subprocess
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_mistralai import ChatMistralAI
import matplotlib.pyplot as plt
import io

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
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

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
        print(f"Debug: Pre-fetching data for intent: {intent} with user input: {user_input}")
        print("Debug: pre-fetching data to dataframe... ")

        # Fetch data from the database
        df = pd.read_sql(sql_query, engine)
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
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)

        if body.get("title", False):
            print("Title Generation")
            return "InfoGraph Code Assistant"
        else:
            # Pre-fetch data
            df = self.pre_fetch_data('draw graph', user_message)

            # Generate Python code for graph drawing
            code_generation_prompt = f"Generate Python code to draw a graph using Matplotlib for the following data:\n{df.head()}"
            code_response = model.generate_code(code_generation_prompt)
            code = code_response['query']

            # Execute the generated Python code
            graph_image, return_code = self.generate_graph(code, df)
            if return_code == 0:
                return graph_image
            else:
                return graph_image
