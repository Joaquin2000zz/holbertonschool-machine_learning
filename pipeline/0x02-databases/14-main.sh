#!/usr/bin/env bash
read -p "file: " file # 14-country_users.sql
read -p "database: " db # holberton
read -p "host: " host # localhost
echo "DROP DATABASE IF EXISTS $db;" | mysql -h $host -u root -p
echo "CREATE DATABASE IF NOT EXISTS $db;" | mysql -h $host -u root -p
echo "DROP TABLE IF EXISTS users;" | mysql -h $host -u root -p $db
cat $file | mysql -h $host -u root -p $db
echo 'INSERT INTO users (email, name, country) VALUES ("bob@dylan.com", "Bob", "US");' | mysql -h $host -u root -p $db
echo 'INSERT INTO users (email, name) VALUES ("billy@idol.com", "billy");' | mysql -h $host -u root -p $db
echo 'INSERT INTO users (email, name) VALUES ("billy@idol.com", "billy");' | mysql -h $host -u root -p $db
echo 'SELECT * FROM users;' | mysql -h $host -u root -p $db
