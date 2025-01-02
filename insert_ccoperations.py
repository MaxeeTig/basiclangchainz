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
        with engine.connect() as connection:
            for row in csvreader:
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

                # Debug print for city_pop value
                print(f"city_pop value from CSV: {row['city_pop']}")
                print(f"city_pop value in params: {params['city_pop']}")

                # Replace placeholders with actual values for debug printing
                debug_query = str(query).replace(':trans_date_trans_time', f"'{params['trans_date_trans_time']}'")
                debug_query = debug_query.replace(':cc_num', f"{params['cc_num']}")
                debug_query = debug_query.replace(':merchant', f"'{params['merchant']}'")
                debug_query = debug_query.replace(':category', f"'{params['category']}'")
                debug_query = debug_query.replace(':amt', f"{params['amt']}")
                debug_query = debug_query.replace(':first', f"'{params['first']}'")
                debug_query = debug_query.replace(':last', f"'{params['last']}'")
                debug_query = debug_query.replace(':gender', f"'{params['gender']}'")
                debug_query = debug_query.replace(':street', f"'{params['street']}'")
                debug_query = debug_query.replace(':city', f"'{params['city']}'")
                debug_query = debug_query.replace(':state', f"'{params['state']}'")
                debug_query = debug_query.replace(':zip', f"'{params['zip']}'")
                debug_query = debug_query.replace(':lat', f"{params['lat']}")
                debug_query = debug_query.replace(':longitude', f"{params['longitude']}")
                debug_query = debug_query.replace(':city_pop', f"{params['city_pop']}")
                debug_query = debug_query.replace(':job', f"'{params['job']}'")
                debug_query = debug_query.replace(':dob', f"'{params['dob']}'")
                debug_query = debug_query.replace(':trans_num', f"'{params['trans_num']}'")
                debug_query = debug_query.replace(':unix_time', f"{params['unix_time']}")
                debug_query = debug_query.replace(':merch_lat', f"{params['merch_lat']}")
                debug_query = debug_query.replace(':merch_long', f"{params['merch_long']}")
                debug_query = debug_query.replace(':is_fraud', f"{params['is_fraud']}")
                debug_query = debug_query.replace(':merch_zipcode', f"'{params['merch_zipcode']}'")

                print(f"Prepared Query: {debug_query}")
                print(f"Params: {params}")

                try:
                    connection.execute(query, params)
                except SQLAlchemyError as e:
                    print(f"Error inserting data: {e}")

if __name__ == '__main__':
    #if len(sys.argv) != 2:
    #    print("Usage: python insert_ccoperations.py <csv_file_path>")
    #    sys.exit(1)

    #csv_file_path = sys.argv[1]
    csv_file_path = "1.csv"
    insert_data_from_csv(csv_file_path)
    print("Data inserted successfully.")
