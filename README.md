# 👁️ Eye Lab: Gaze Tracker API

Eye Lab is an open-source tool to create eye-tracking usability tests. It started as a final undergraduate work for the Computer Engineering student [Karine Pistili](https://www.linkedin.com/in/karine-pistili/) who made the prototype. The idea is to evolve it into a more complete and useful tool with the community's help.

The current version of the software allows users to create their usability sessions of a website, recording the webcam, screen, and mouse movements and use this information to find out where the user has been looking into the screen by using heatmaps.

## 👩‍💻 Setting up project locally

The project consists of two parts, this repository contains the backend of the application, and the frontend can be found [here](https://github.com/uramakilab/web-eye-tracker-front). Install it as well to have the full application running.

### Prerequisites

- [Python 3x](https://www.python.org/downloads/)

## Setting Up a Virtual Environment
### Virtual Environment Setup

It is highly recommended to run this project inside a Python virtual environment to avoid dependency conflicts. 

**For Windows (Command Prompt / PowerShell):**
```bash
# 1. Create the virtual environment
python -m venv venv

# 2. Activate it
.\venv\Scripts\activate

#### **Linux & macOS**
##### **Step 1: Create a virtual environment**
```sh
python3 -m venv venv
```


##### **Step 2: Activate the virtual environment**
```sh
source venv/bin/activate
```

##### **Step 3: Install dependencies**
```sh
pip install -r requirements.txt
```

##### **Step 4: Run Flask**
```sh
flask run
```

---

#### **Windows**
##### **Step 1: Create a virtual environment**
```sh
python -m venv venv
```

##### **Step 2: Activate the virtual environment**
```sh
venv\Scripts\activate
```

##### **Step 3: Install dependencies**
```sh
pip install -r requirements.txt
```

##### **Step 4: Run Flask**
```sh
flask run
```

---

### **2. Using Conda (Works on Linux, macOS, and Windows)**

#### **Step 1: Create a Conda virtual environment**
```sh
conda create --name flask_env python=3.10
```

#### **Step 2: Activate the environment**
```sh
conda activate flask_env
```

#### **Step 3: Install dependencies**
```sh
pip install -r requirements.txt
```

#### **Step 4: Run Flask**
```sh
flask run
```


### **Additional Notes**
- If you face issues running `flask run`, try:

  ```sh
  python -m flask run
  ```
- If Flask is not installed, install it manually:

  ```sh
  pip install flask
  ```
- On **Windows**, if `venv\Scripts\activate` doesn't work, **run PowerShell as Administrator** and enable scripts:

  ```sh
  Set-ExecutionPolicy Unrestricted -Scope Process
  ```

## 🧑‍🤝‍🧑 Contributing

Anyone is free to contribute to this project. Just do a pull request with your code and if it is all good we will accept it. You can also help us look for bugs if you find anything that creates an issue.
To see the full list of contributions, check out the ahead commits of the "develop" branch concerning the "main" branch. Full logs of the project development can be found in the [Daily Work Progress](https://docs.google.com/document/d/1RjCnGjYYgPKvFUrN8hSjPX29aayWr6eEopeCN3QZwEQ/edit?usp=sharing) file. Hoping to see your name in the list of contributors soon! 🚀


## 📃 License

This software is under the [MIT License](https://opensource.org/licenses/MIT).

Copyright 2021 Uramaki Lab
