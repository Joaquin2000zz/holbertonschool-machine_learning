#!/usr/bin/env bash
read -p "file: " file # 8-genre_id_by_show.sql
read -p "database: " db # hbtn_0d_tvshows
read -p "host: " host # localhost
read -p "table: " table # tv_shows
echo "CREATE DATABASE IF NOT EXISTS $db;" | mysql -h $host -u root -p
echo "CREATE TABLE IF NOT EXISTS $table(id INT NOT NULL);" | mysql -h $host -u root -p $db
if [ "echo 'SELECT COUNT(*) FROM '.$table.';' | mysql -h $host -u root -p $db" == 0 ];
then
    cat $table.sql | mysql -h $host -u root -p $db
fi
cat $file | mysql -h $host -u root -p $db