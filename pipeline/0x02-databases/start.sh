#!/usr/bin/env bash
sudo service mysql start
echo "SHOW DATABASES;" | mysql -h localhost -u root -p
