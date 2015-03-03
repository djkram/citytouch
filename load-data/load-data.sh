#!/bin/bash

# Python dependencies
pip install psycopg2 rarfile nltk django greenlet
python -m nltk.downloader all

# Start Postgres
service postgresql restart 

# Load data
cd /data/scripts
echo "Loading data to postgres..."
python DB_import.py -f completeMWC.rar -d citytouch -t spanish_twitter_mwc -u citytouch
echo "Creating TS..."
python TS_generate.py -d citytouch -t spanish_twitter_mwc -u citytouch

mv database_table.json /var/www/html/data/NEW-database_table.json
