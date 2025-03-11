# 👁️ Eye Lab: Gaze Tracker API

Eye Lab is an open source tool to create eye tracking usability tests. It started as a final undergraduation work for Computer Engineering of student [Karine Pistili](https://www.linkedin.com/in/karine-pistili/) that created the first prototype. The idea is to evolve it to a more complete and useful tool with the help of the community.

The current version of the software allows users to create their usability sessions of an website, recording the webcam, screen and mouse movements and use this information to find out where the user has been looking into the screen by using heatmaps.

## 👩‍💻 Setting up project locally

The project consists of two parts, this repository contains the backend of the application and the frontend can be found [here](https://github.com/uramakilab/web-eye-tracker-front). Install it as well.

### Prerequisites

* [Python 3x](https://www.python.org/downloads/)

### 1. Create virtual environment

Before installing all dependencies and starting your Flask Server, it is better to create a python virtual environment. You can use the [venv package](https://docs.python.org/3/library/venv.html)

```
python -m venv /path/to/new/virtual/environment
```

Then activate your env. On windows for example you can activate with the script:

```
name-of-event/Scripts/activate
```

### 2. Install dependencies

Install all dependencies listed on the requirements.txt with:

```
pip install -r requirements.txt
```

### 3. Run the Flask API

```
flask run
```

---

## 💻 Setup for macOS / Linux

The following steps provide instructions for setting up the **Eye Lab API** on **macOS** and **Linux**.

### For macOS / Linux

1. **Install Python 3 and pip**

   - On **macOS**:
     macOS usually comes with **Python 3** pre-installed. You can check by running:

     ```bash
     python3 --version
     ```

     If it's not installed, you can install **Python 3** via [Homebrew](https://brew.sh/). First, install **Homebrew** (if you don't have it) with:

     ```bash
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```

     Then install **Python 3**:

     ```bash
     brew install python
     ```

   - On **Linux**:
     Use the following commands to install **Python 3** and **pip** on Ubuntu or other Debian-based systems:

     ```bash
     sudo apt update
     sudo apt install python3 python3-pip
     ```

     Alternatively, if you want to manage different Python versions, you can use [pyenv](https://github.com/pyenv/pyenv).

2. **Create a virtual environment**

   It's best to create a virtual environment to isolate your dependencies.

   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**

   Once the virtual environment is created, activate it.

   - On **macOS/Linux**:

     ```bash
     source venv/bin/activate
     ```

4. **Install the dependencies**

   Once the virtual environment is activated, install the required dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Flask API**

   To run the Flask API, use the following command:

   ```bash
   flask run
   ```

   By default, this will start the Flask server at `http://127.0.0.1:5000/`.

---

## 🧑‍🤝‍🧑 Contributing

Anyone is free to contribute to this project. Just do a pull request with your code and if it is all good we will accept it. You can also help us look for bugs, if you find anything create an issue.

## 📃 License

This software is under the [MIT License](https://opensource.org/licenses/MIT). 

Copyright 2021 Uramaki Lab
