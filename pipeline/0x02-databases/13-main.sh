#!/usr/bin/env bash
read -p "file: " file # 13-uniq_users.sql
read -p "database: " db # holberton
read -p "host: " host # localhost
echo "CREATE DATABASE IF NOT EXISTS $db;" | mysql -h $host -u root -p
cat $file | mysql -h $host -u root -p $db
echo 'INSERT INTO users (email, name) VALUES ("bob@dylan.com", "Bob");' | mysql -h $host -u root -p $db
echo 'INSERT INTO users (email, name) VALUES ("bob@dylan.com", "Bob");' | mysql -h $host -u root -p $db