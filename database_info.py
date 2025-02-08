# oracle_database_info.py - getting information about tables in Oracle database
from langchain_community.utilities import SQLDatabase
import cx_Oracle

# ===== Function to connect to Oracle database and get information about it
def get_database_info(db_config):
    """
    db_config - dictionary with connection parameters
    """
    # Create connection string
    dsn = cx_Oracle.makedsn(db_config['host'], db_config['port'], service_name=db_config['service_name'])
    db = SQLDatabase.from_uri(f"oracle+cx_oracle://{db_config['user']}:{db_config['password']}@{dsn}")

    # Get all tables names list
    table_names = db.get_usable_table_names()

    # Get tables information and 3 records from them
    table_info = db.get_table_info()

    return table_names, table_info

def main():
    # Database connection parameters
    db_config = {
        'user': 'your_username',
        'password': 'your_password',
        'host': 'your_host',
        'port': 1521,  # Default Oracle port
        'service_name': 'your_service_name'
    }

    # Get the database information
    table_info, db_info = get_database_info(db_config)

    # Print the database info on the screen
    print(f"Here is info about tables: {table_info}")
    print(f"Here are details: {db_info}")

if __name__ == "__main__":
    main()
