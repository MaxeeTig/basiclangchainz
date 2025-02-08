# database_info.py - getting information about tables in Oracle database
from langchain_community.utilities import SQLDatabase
import cx_Oracle
import sys
import os

# Initialize Oracle Client
def initialize_oracle_client():
    try:
        if sys.platform.startswith("win32"):
            lib_dir = r"C:\oracle\instantclient_19_6"
            cx_Oracle.init_oracle_client(lib_dir=lib_dir)
    except Exception as err:
        print("Whoops!")
        print(err)
        sys.exit(1)

# ===== Function to connect to Oracle database and get information about it
def get_database_info(db_config, tables_list = None):
    """
    db_config - dictionary with connection parameters
    tables_list - allows specify particular tables to return
    """
    # Create connection string
    dsn = cx_Oracle.makedsn(db_config['host'], db_config['port'], service_name=db_config['service_name'])
    db = SQLDatabase.from_uri(f"oracle+cx_oracle://{db_config['user']}:{db_config['password']}@{dsn}")

    # Get all tables names list
    table_names = db.get_usable_table_names()

    # Get tables information and 3 records from them
    table_info = db.get_table_info(tables_list)

    return table_names, table_info
    #return table_names

def main():
# Initialize Oracle Client
    initialize_oracle_client()
    # Database connection parameters
    db_config = {
        'user': 'MAIN',
        'password': 'MAIN1',
        'host': 'AVERS.BT.BPC.IN',
        'port': 1521,  # Default Oracle port
        'service_name': 'SV'
    }

    # Option: Define particular tables to get info about
    tables = ['vis_card','vis_currency_rate']

    # Get the database information
    table_info, db_info = get_database_info(db_config, tables)
    #table_info = get_database_info(db_config)

    # Print the database info on the screen
    print(f"Here is info about tables: {table_info}")
    print(f"Here are details: {db_info}")

if __name__ == "__main__":
    main()