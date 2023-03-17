#!/usr/bin/env bash
read -p "file: " file # 4-best_score.sql
read -p "database: " db # db_0
read -p "host: " host # localhost
read -p "table: " table # second_table
echo "CREATE DATABASE IF NOT EXISTS $db;" | mysql -h $host -u root -p
cat $table.sql | mysql -h $host -u root -p $db
cat $file | mysql -h $host -u root -p $db
