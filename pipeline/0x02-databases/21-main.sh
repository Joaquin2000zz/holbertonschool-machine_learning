#!/usr/bin/env bash
read -p "file: " file # 21-div.sql
read -p "database: " db # holberton
read -p "host: " host # localhost
read -p "table: " table # numbers
echo "DROP DATABASE IF EXISTS $db;" | mysql -h $host -u root -p
echo "CREATE DATABASE IF NOT EXISTS $db;" | mysql -h $host -u root -p
echo "CREATE TABLE IF NOT EXISTS $table(id INT NOT NULL);" | mysql -h $host -u root -p $db
if [ "echo 'SELECT COUNT(*) FROM '.$table.';' | mysql -h $host -u root -p $db" == 0 ];
then
    cat $table.sql | mysql -h $host -u root -p $db
fi
cat $file | mysql -h $host -u root -p $db
cat 20-main.sql | mysql $host -u root -p $db
echo "SELECT (a / b) FROM numbers;" | mysql -h $host -r root -p $db
echo "--------------------------------------------"
echo "SELECT SafeDiv(a, b) FROM numbers;" | mysql -h $host -r root -p $db
