#!/bin/bash
# sudo docker run -it -d -v /home/suntopo/Recog/:/Recog -v /images/:/images -p 20001:20001 --name pycv0.3 4217bf3ea7d4 bash -c "cd /Recog/recog/ && python MainApplication.py"
ssh -t suntopo@120.77.80.118 "bash -c 'cd Recog/recog/ && git pull && sudo docker restart pycv0.3'"