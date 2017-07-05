#!/bin/bash
# sudo docker run --name mysql -d -v /images/mysql/:/var/lib/mysql -v /images/mysqldata/conf/:/etc/mysql/conf.d -p 3306:3306  -e MYSQL_ROOT_PASSWORD=123456  daocloud.io/mysql
# sudo docker run -it -d -v /home/suntopo/Recog/:/Recog -v /images/:/images -p 20001:20001 --name pycv0.3 4217bf3ea7d4 bash -c "cd /Recog/recog/ && python MainApplication.py"
sed -i '' 's/"debug": True/"debug": False/g' Setting.py
git status
git commit -a
git push
sed -i '' 's/"debug": False/"debug": True/g' Setting.py
ssh -t suntopo@120.77.80.118 "bash -c 'cd Recog/recog/ && git pull && sudo docker restart pycv0.3'"