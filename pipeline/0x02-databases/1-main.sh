#!/usr/bin/env bash
read -p "file: " file # 1-first_table.sql 
read -p "database: " db # db_0
read -p "host: " host # localhost
cat $file | mysql -h $host -u root -p $db
echo 'SHOW TABLES;' | mysql -h localhost -u root -p $db
