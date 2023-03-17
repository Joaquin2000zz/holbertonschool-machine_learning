#!/usr/bin/env bash
read -p "file: " file # 3-insert_value.sql 
read -p "database: " db # db_0
read -p "host: " host # localhost
cat $file | mysql -h $host -u root -p $db
cat 2-main.sh | sh
