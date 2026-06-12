# 👁️ Eye Lab: Gaze Tracker API

Eye Lab is an open-source tool to create eye-tracking usability tests. It started as a final undergraduate work for the Computer Engineering student [Karine Pistili](https://www.linkedin.com/in/karine-pistili/) who made the prototype. The idea is to evolve it into a more complete and useful tool with the community's help.

The current version of the software allows users to create their usability sessions of a website, recording the webcam, screen, and mouse movements and use this information to find out where the user has been looking into the screen by using heatmaps.

## 👩‍💻 Setting up project locally

The project consists of two parts, this repository contains the backend of the application, and the frontend can be found [here](https://github.com/uramakilab/web-eye-tracker-front). Install it as well to have the full application running.

### Prerequisites

- [Python 3x](https://www.python.org/downloads/)

## Setting Up a Virtual Environment


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

## **Deployment**

- **Files:** use the environment files to configure deployments: `.env.development` and `.env.production` (keep secrets out of the repo).
**Script:** deploy with `./deploy_cloud_run.sh` (supports `-e dev` or `-e prod`, CLI flags override env files).

Quick examples:

```bash
# Deploy using development defaults in .env
./deploy_cloud_run.sh -e dev

# Deploy to production using .env.production (or override values with flags)
./deploy_cloud_run.sh -e prod -p my-prod-project -s my-service-name -r europe-west6 -i my-image -a
```

Configuration variables (place in `.env` for local/dev or `.env.production` for prod):

- `PROJECT_ID` — GCP project id
- `REGION` — Cloud Run region (example: `europe-west6`)
- `SERVICE_NAME` — Cloud Run service name
- `IMAGE_NAME` — container image name (pushed to `gcr.io/PROJECT/IMAGE:timestamp`)
- `ALLOW_UNAUTH` — `true` or `false` (whether to allow unauthenticated access)

Files added by this project:

- `deploy_cloud_run.sh` — helper script that builds with Cloud Build and deploys to Cloud Run.
- `.env.development` — development defaults (committed).
- `.env.production` — production template (do not commit secrets; see `.gitignore`).

Run locally with `.env.development` or `.env.production`:

```bash
# Run using development env file
./run_local.sh -e dev --port 3000

# Run using production env file (if you have it configured locally)
./run_local.sh -e prod --port 3000
```

This script sources the selected env file and runs the app via `python wsgi.py`.
Security note: never commit production credentials or service account keys. Keep `.env.production` and any secret files out of source control.
