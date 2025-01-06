CREATE database cctransreview;
CREATE USER 'cctrxn'@'localhost' IDENTIFIED BY 'cctrxnpwd';
GRANT ALL PRIVILEGES ON *.* TO 'cctrxn'@'localhost' WITH GRANT OPTION;
