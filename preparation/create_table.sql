USE cctransreview;

CREATE TABLE ccoperations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    trans_date_trans_time DATETIME,
    cc_num BIGINT,
    merchant VARCHAR(255),
    category VARCHAR(255),
    amt DECIMAL(10, 2),
    first VARCHAR(255),
    last VARCHAR(255),
    gender CHAR(1),
    street VARCHAR(255),
    city VARCHAR(255),
    state CHAR(2),
    zip VARCHAR(10),
    lat DECIMAL(9, 6),
    longitude DECIMAL(9, 6),
    city_pop INT,
    job VARCHAR(255),
    dob DATE,
    trans_num VARCHAR(255),
    unix_time BIGINT,
    merch_lat DECIMAL(9, 6),
    merch_long DECIMAL(9, 6),
    is_fraud TINYINT(1),
    merch_zipcode VARCHAR(10)
);
