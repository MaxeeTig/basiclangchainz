"""
title: ClusterGraph Code Assistant (Database LLM graph cluster-plotting tool)
author: Mikael Kondakhchan
date: 23 Jan 2025
version: 0.5
license: MIT
requirements: langchain==0.3.1,  langchain-mistralai==0.2.4, langchain_openai==0.2.1, langchain_text_splitters==0.3.2, pydantic==2.8.2, pymysql==1.1.1
description: A pipeline for using LLM to build graphs on the basis of database information.
"""


import base64

from typing_extensions import TypedDict, Annotated
from typing import List, Union, Generator, Iterator
import subprocess
import pandas as pd
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt
import io
import os

# Global debug mode variable
debug_mode = True

class CodeOutput(TypedDict):
    """Generated python code."""
    query: Annotated[str, ..., "Syntactically valid Python code."]

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
        self.name = "Cluster Code Assistant"
        self.model = model  # Ensure the model is accessible within the class

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

    def generate_cluster_graph_code(self, df, columns, user_message, max_attempts=3):
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
            "columns": columns,
            "input": user_message,
        })

        structured_llm = self.model.with_structured_output(CodeOutput)
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
                return filename, 0
            except Exception as e:
                print(f"Exception occurred {e}")
                prompt = (f"The generated query is invalid. Error: {e}. "
                          f"Attempt {attempt + 1} of {max_attempts}. "
                          f"Previous invalid query: {code}. "
                          f"Please correct the query.")
        return "Failed to generate a valid code for cluster graph after multiple attempts.", 1

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        if debug_mode:
            print(f"Debug: pipe:{__name__}")

        if body.get("title", False):
            if debug_mode:
                print("Debug: Title Generation")
            return "Cluster Code Assistant"
        else:
            # Pre-fetch data
            df = self.pre_fetch_data('cluster analysis', user_message)
            columns = df.columns.tolist()


        answer, return_code = self.generate_cluster_graph_code(df, columns, user_message)
        if return_code == 0:
            if answer is None:
                return "no graph has been generated"
            else:
                with open(answer, 'rb') as f:
                    image_data = f.read()

                # Convert the image data to a base64 string
                image_base64 = base64.b64encode(image_data).decode('utf-8')

                # Remove the local file
                os.remove(answer)

                # Return the base64 string embedded in HTML
                return f"```html\n<div><img src='data:image/png;base64,{image_base64}' alt='Graph'></div>\n```"
        else:
            return answer