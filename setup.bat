@echo off
:: Create a virtual environment
python -m venv venv

:: Activate the virtual environment
call venv\Scripts\activate

:: Install the required packages
pip install -r requirements.txt

:: Start Jupyter Notebook
jupyter notebook

