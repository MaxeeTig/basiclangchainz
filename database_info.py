from langchain_community.utilities import SQLDatabase

# ===== Function to connect to database and get information about it
def get_database_info(db_uri, tables_list = None):
    """
    db_uri - connection string
    tables_list - allows specify particular tables to return
    """
    # Connect to the database
    db = SQLDatabase.from_uri(db_uri)
    # Get all tables names list 
    table_names = db.get_usable_table_names()
    # Get tables information and 3 records from them
    table_info = db.get_table_info(tables_list)

    return table_names, table_info 

def main():
# Database connection parameters
    db_config = {
    'user': 'svbo',
    'password': 'svbopwd',
    'host': 'localhost',
    'database': 'botransactions'
    }

# Create connection string 
    db_uri = (f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")

# Option: Define particular tables to get info about
    tables = ['operations','mcc']
    table_info, db_info = get_database_info(db_uri, tables)

# Get the database information
#    table_info, db_info = get_database_info(db_uri)

# Print the database info on the screen
    print(f"Here is info about tables: {table_info}")
    print(f"Here ae details: {db_info}")


if __name__ == "__main__":
    main()
