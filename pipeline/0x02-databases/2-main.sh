#!/usr/bin/env bash
read -p "file: " file # 2-list_values.sql 
read -p "batabase: " db # db_0
read -p "host: " host # localhost
cat $file | mysql -h $host -u root -p $db