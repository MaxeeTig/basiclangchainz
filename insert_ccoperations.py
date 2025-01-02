import sys
import csv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pymysql

# Database connection parameters
db_config = {
    'user': 'cctrxn',
    'password': 'cctrxnpwd',
    'host': 'localhost',
    'database': 'cctransreview'
}

# Create the SQLAlchemy engine
engine = create_engine(f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")

def insert_data_from_csv(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile, delimiter=',')
        rows = list(csvreader)
        total_records = len(rows)
        with engine.connect() as connection:
            with connection.begin() as transaction:  # Start a transaction
                for index, row in enumerate(rows, start=1):
                    query = text("""
                        INSERT INTO ccoperations (
                            trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender, street, city, state, zip, lat, longitude, city_pop, job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud, merch_zipcode
                        ) VALUES (
                            :trans_date_trans_time, :cc_num, :merchant, :category, :amt, :first, :last, :gender, :street, :city, :state, :zip, :lat, :longitude, :city_pop, :job, :dob, :trans_num, :unix_time, :merch_lat, :merch_long, :is_fraud, :merch_zipcode
                        )
                    """)
                    params = {
                        'trans_date_trans_time': row['trans_date_trans_time'],
                        'cc_num': row['cc_num'],
                        'merchant': row['merchant'],
                        'category': row['category'],
                        'amt': row['amt'],
                        'first': row['first'],
                        'last': row['last'],
                        'gender': row['gender'],
                        'street': row['street'],
                        'city': row['city'],
                        'state': row['state'],
                        'zip': row['zip'],
                        'lat': row['lat'],
                        'longitude': row['long'],
                        'city_pop': row['city_pop'],
                        'job': row['job'],
                        'dob': row['dob'],
                        'trans_num': row['trans_num'],
                        'unix_time': row['unix_time'],
                        'merch_lat': row['merch_lat'],
                        'merch_long': row['merch_long'],
                        'is_fraud': row['is_fraud'],
                        'merch_zipcode': row['merch_zipcode']
                    }

                    try:
                        connection.execute(query, params)
                        print(f"Insert record {index} of {total_records}")
                    except SQLAlchemyError as e:
                        print(f"Error inserting data: {e}")
                        transaction.rollback()  # Rollback in case of error
                        raise
                transaction.commit()  # Commit the transaction

if __name__ == '__main__':
    #if len(sys.argv) != 2:
    #    print("Usage: python insert_ccoperations.py <csv_file_path>")
    #    sys.exit(1)

    #csv_file_path = sys.argv[1]
    csv_file_path = "1.csv"
    insert_data_from_csv(csv_file_path)
    print("Data inserted successfully.")
