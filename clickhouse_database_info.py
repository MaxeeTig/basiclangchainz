# database_info.py - getting information about tables in ClickHouse database
from clickhouse_driver import Client

# ===== Function to connect to ClickHouse database and get information about it
def get_database_info(db_config):
    """
    db_config - dictionary with connection parameters
    """
    # Connect to the ClickHouse database
    client = Client(host=db_config['host'], port=db_config['port'], user=db_config['user'],
password=db_config['password'])

    # Get all tables names list
    table_names_query = """
    SELECT name
    FROM system.tables
    WHERE database = '{}'
    """.format(db_config['database'])
    table_names = [row[0] for row in client.execute(table_names_query)]

    # Get tables information and 3 records from them
    table_info = {}
    for table_name in table_names:
        table_info_query = """
        SELECT *
        FROM {}
        LIMIT 3
        """.format(table_name)
        table_info[table_name] = client.execute(table_info_query)

    return table_names, table_info

def main():
    # Database connection parameters
    db_config = {
        'user': 'default',
        'password': '',
        'host': 'localhost',
        'port': 9000,
        'database': 'default'
    }

    # Get the database information
    table_info, db_info = get_database_info(db_config)

    # Print the database info on the screen
    print(f"Here is info about tables: {table_info}")
    print(f"Here are details: {db_info}")

if __name__ == "__main__":
    main()
