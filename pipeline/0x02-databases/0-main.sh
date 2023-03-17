#!/usr/bin/env bash
read -p "file: " file # 0-create_database_if_missing.sql
read -p "host: " host # localhost
cat $file | mysql -h $host -u root -p
echo "SHOW DATABASES;" | mysql -h $host -u root -p
