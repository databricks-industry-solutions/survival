# Databricks notebook source
# MAGIC %md The purpose of this notebook is to download and set up the data we will use for the solution accelerator. Before running this notebook, make sure you have entered your own credentials for Kaggle and accepted the rules of this contest [dataset](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/rules).

# COMMAND ----------

# MAGIC %pip install kaggle py7zr

# COMMAND ----------

# MAGIC %md 
# MAGIC Set Kaggle credential configuration values in the block below: You can set up a [secret scope](https://docs.databricks.com/security/secrets/secret-scopes.html) to manage credentials used in notebooks. For the block below, we have set up the `solution-accelerator-cicd` secret scope and saved our Kaggle username and key in it. See the `./RUNME` notebook for instructions on how to set up the this secret scope. 
# MAGIC
# MAGIC Don't forget to accept the [terms of the competition](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/data).

# COMMAND ----------

import os
# os.environ['kaggle_username'] = 'YOUR KAGGLE USERNAME HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ['kaggle_username'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_username")

# os.environ['kaggle_key'] = 'YOUR KAGGLE KEY HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ['kaggle_key'] = dbutils.secrets.get("solution-accelerator-cicd", "kaggle_key")

# COMMAND ----------

# MAGIC %md Download and unzip data:

# COMMAND ----------

# MAGIC %sh 
# MAGIC rm -rf /dbfs/tmp/kkbox-survival/
# MAGIC mkdir -p /dbfs/tmp/kkbox-survival/
# MAGIC cd /dbfs/tmp/kkbox-survival/
# MAGIC export KAGGLE_USERNAME=$kaggle_username
# MAGIC export KAGGLE_KEY=$kaggle_key
# MAGIC kaggle competitions download -c kkbox-churn-prediction-challenge
# MAGIC unzip kkbox-churn-prediction-challenge.zip

# COMMAND ----------

# MAGIC %md Extract the downloaded data to the folder used throughout the accelerator:

# COMMAND ----------

# MAGIC %sh 
# MAGIC cd /dbfs/tmp/kkbox-survival/
# MAGIC py7zr x members_v3.csv.7z members/
# MAGIC py7zr x transactions.csv.7z transactions/
# MAGIC py7zr x transactions_v2.csv.7z transactions/
# MAGIC py7zr x user_logs.csv.7z user_logs/
# MAGIC py7zr x user_logs_v2.csv.7z user_logs/
