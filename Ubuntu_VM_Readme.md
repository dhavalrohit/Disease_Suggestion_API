# Setup & Installation Commands:

## 1.Update System Packages
```
sudo apt update 
sudo apt upgrade -y
sudo apt install -y software-properties-common curl git build-essential
sudo apt update
```


## 2.Verify Python Installation
```
python3 --version
```


## Note:Tested on Python 3.10.12 and ubuntu 22.04

## 3.Install/Upgrade Pip
```
sudo apt install -y python3-pip
```

## 4.Create Application Directory
```
mkdir ~/symptoms_disease
cd ~/symptoms_disease
```

## 5.Create Python Virtual Environement
```
sudo apt install python3.10-venv
python3 -m venv pyvenv
```

## 6.Activate Virtual Environement
```
source pyvenv/bin/activate
```

## 7.Install required packages
```
pip install --no-cache-dir flask pandas nltk scikit-learn waitress google requests
```

## 8.Check if Virtal Environement is created Successfully
```
cd ~/symptoms_disease
ls
```
### Output:
```
pyvenv                   only pyvenv folder is visible in output
```

## 9.Under symptoms_disease create Dataset Folder
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
wget https://raw.githubusercontent.com/dhavalrohit/Disease_Suggestion_API/refs/heads/main/Dataset/diseasesymp_updated.csv
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
wget https://raw.githubusercontent.com/dhavalrohit/Disease_Suggestion_API/main/app.py
wget https://raw.githubusercontent.com/dhavalrohit/Disease_Suggestion_API/main/Treatment.py
wget https://raw.githubusercontent.com/dhavalrohit/Disease_Suggestion_API/main/README.md
```

### check
```
ls
```
### Output:
```
Dataset  README.md  Treatment.py  app.py  venv       should be visible in output
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
-rw-rw-r-- 1 ubuntu ubuntu 1992 Nov 10 05:29 Treatment.py
-rw-rw-r-- 1 ubuntu ubuntu 5378 Nov 10 05:29 app.py
drwxrwxr-x 6 ubuntu ubuntu 4096 Nov 10 05:07 venv
#similar output you should see
```




# Running the API
```
cd ~/symptoms_disease
```
```
source pyvenv/bin/activate
```
```
python3 app.py
```


# Facing API Endpoint Connection Issues try this Commands:
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
