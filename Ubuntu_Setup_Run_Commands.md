# Setup & Installation Commands:

## 1.Update System Packages
```
sudo apt update 
sudo apt upgrade -y
sudo apt install -y software-properties-common curl git build-essential
sudo apt update
```

## 2.Steps to Change TimeZone from Server UTC to LocalTime Zone/Important for logging
## List available timezones to confirm the correct name
```
timedatectl list-timezones | grep 'Asia/Kolkata'
```
### Set the system timezone to Asia/Kolkata (IST)
```
sudo timedatectl set-timezone Asia/Kolkata
```
### Verify the change
```
timedatectl
```

## 3.Verify Python Installation
```
python3 --version
```


## Note:Tested on Python 3.10.12 and ubuntu 22.04

## 4.Install/Upgrade Pip
```
sudo apt install -y python3-pip
```

## 5.Create Application Directory
```
mkdir ~/symptoms_disease
cd ~/symptoms_disease
```

## 6.Create Python Virtual Environement
```
sudo apt install python3.10-venv
python3 -m venv pyvenv
```

## 7.Activate Virtual Environement
```
source pyvenv/bin/activate
```

## 8.Install required packages
```
pip install --no-cache-dir flask pandas nltk scikit-learn waitress google requests
```
```
pip install gunicorn
```

## 9.Check if Virtal Environement is created Successfully
```
cd ~/symptoms_disease
ls
```
### Output:
```
pyvenv                   only pyvenv folder is visible in output
```

## 10.Under symptoms_disease create Dataset Folder
```
mkdir -p Dataset
ls
```
### Output:
```
Dataset pyvenv                   Dataset and venv folder is visible in output
```
## 10.Move to Dataset folder and download Dataset file
```
cd ~/symptoms_disease/Dataset
wget https://raw.githubusercontent.com/dhavalrohit/Disease_Suggestion_API/refs/heads/testing_V2/Dataset/diseasesymp_updated.csv
```

```
ls
```
### Output:
```
diseasesymp_updated.csv                      is visbible in output
```

## 11.In main folder download other files
```
cd ~/symptoms_disease
wget https://raw.githubusercontent.com/dhavalrohit/Disease_Suggestion_API/refs/heads/testing_V2/app.py
wget https://raw.githubusercontent.com/dhavalrohit/Disease_Suggestion_API/refs/heads/testing_V2/README.md
```

### check
```
ls
```
### Output:
```
Dataset  README.md    app.py  venv       should be visible in output
```
### Verify downloaded files:
```
(venv) ubuntu@ubuntuserver:~/symptoms_disease$ ls
```
### Output:
```
Dataset  README.md  Treatment.py  app.py  venv
```
### Check all  Files:
```
(venv) ubuntu@ubuntuserver:~/symptoms_disease$ ls -l
```
### Output:
```
total 24
drwxrwxr-x 2 ubuntu ubuntu 4096 Nov 10 05:15 Dataset
-rw-rw-r-- 1 ubuntu ubuntu 1236 Nov 10 05:29 README.md
-rw-rw-r-- 1 ubuntu ubuntu 5378 Nov 10 05:29 app.py
drwxrwxr-x 6 ubuntu ubuntu 4096 Nov 10 05:07 venv
#similar output you should see
```
## 12.Running the API
```
cd ~/symptoms_disease
```
```
source pyvenv/bin/activate
```
```
gunicorn --bind 0.0.0.0:6000 app:app --log-level info --capture-output --enable-stdio-inheritance
```
## Command to Check if Gunicorn Process is active(API is running)
```
sudo lsof -i:6000
```
## Command to Stop the API(Gunicorn proccess) if Actively Running in Background
```
sudo pkill gunicorn
```

## Commands to Check Log File
```
cd ~/symptoms_disease
cat api.log
```

## Facing API Endpoint Connection Issues try this Commands:
## Run Command:sudo iptables -I INPUT 1 -p tcp --dport <your_port_number> -j ACCEPT
```
sudo iptables -I INPUT 1 -p tcp --dport 6000 -j ACCEPT
sudo iptables-save | sudo tee /etc/iptables/rules.v4
```

# To Check Receive endpoint(List Last 10 rows in CSV File)
```
cd ~/symptoms_disease
tail -n 10 Dataset/diseasesymp_updated.csv
```
```
tail -n 10 diseasesymp_updated.csv
```


## Steps to Host Your API(Using tmux)
## Note-Used Only for testing or developement 
### login to your instance
```
ssh -i your_key.pem ubuntu@<your_public_ip>
```
### install tmux
```
sudo apt update && sudo apt install tmux -y
```

### start tmux session #tmux new -s symptoms_disease_app
```
tmux new -s symptoms_disease_app
```
```
cd ~/symptoms_disease
```
```
source pyvenv/bin/activate   
```
```
gunicorn --bind 0.0.0.0:6000 app:app --log-level info --capture-output --enable-stdio-inheritance
```
### Detach the session (keep it running in background)
### Ctrl + B, then D

### Reattach anytime to view logs
```
tmux attach -t symptoms_disease_app
```
### Stop the app
Reattach (tmux attach -t symptoms_disease_app)
Press Ctrl + C to stop
Type exit to close tmux

### Check tmux status is main ssh
```
tmux ls
```
