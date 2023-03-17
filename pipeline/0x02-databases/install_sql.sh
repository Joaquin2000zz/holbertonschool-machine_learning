wget https://dev.mysql.com/get/mysql-apt-config_0.8.12-1_all.deb
sudo apt update
sudo apt install -f mysql-client mysql-server
mysql --version