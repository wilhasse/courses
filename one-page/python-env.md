# Install

Python
https://python.org/downloads/

Add Environment Variables

```
PATH
C:\Users\YourUsername\AppData\Local\Programs\Python\Python3xx
C:\Users\YourUsername\AppData\Local\Programs\Python\Python3xx\Scripts
```

Pip

```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

# New Env

# Create and navigate to your project folder

```bash
mkdir project-name
cd project-name

# Create virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows cmd:
venv\Scripts\activate

# On Windows PowerShell:
.\venv\Scripts\Activate
```

Install pip packages

Example:

```bash
pip install pywin32 svn minio mysql-connector-python pyyaml
```

Freeze

```bash
pip freeze > requirements.txt
```

# New Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
