# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/survival. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/survivorship-and-churn.

# COMMAND ----------

# MAGIC %md Our goal over the next few notebooks is to examine how two core Survival Analysis techniques can be applied to better understand patterns around customer attrition in a subscription model.  In this notebook, we will prepare a publicly available dataset for analysis.  This data will then serve as the basis for some exploratory analysis and modeling intended to assist us in understanding and predicting customer churn.

# COMMAND ----------

# MAGIC %md ##Step 1: Access the Raw Data Files
# MAGIC 
# MAGIC The dataset we will be working with is from the [KKBox Churn Prediction Challenge](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data) hosted on Kaggle in 2018. Access to the data will require you to authenticate to Kaggle and agree to the organization's terms and conditions.  Once you've done this, you can download the data, decompress it and upload it to your preferred cloud storage layer.  For the purposes of this blog, we assume this has already been done and each file has been uploaded to an appropriate subfolder under a [mount point](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) named */mnt/kkbox-survival/*. 
# MAGIC 
# MAGIC While there are numerous files available with this dataset, we will be making use of the transaction log and member files only. The information in some of the other files could be very interesting for this analysis. However, only a fraction of the subscriptions referenced in the transaction log have data in these other files.  Instead of limiting our analysis to a small number of customers for whom we can create a more complete picture of service utilization, we'll focus our analysis to the more foundational information in the transaction log and member files.
# MAGIC 
# MAGIC Under our mount point, *i.e.* */mnt/kkbox-survival/*, it is expected that the transactions.csv and transactions_v2.csv files are uploaded to a subfolder named *transactions* and members_v3.csv is uploaded to a subfolder named *members*. 
# MAGIC 
# MAGIC We have automated this data extraction step and set up the data in a temporary folder */tmp/kkbox-survival/*, which will be used throughout this accelerator. 

# COMMAND ----------

# MAGIC %run "./config/Data Extract"

# COMMAND ----------

# MAGIC %md With these files loaded to the proper locations, we can now read the files into [Delta Lake-backed tables](https://databricks.com/product/delta-lake-on-databricks) to make them more easily consumable in subsequent steps:

# COMMAND ----------

# DBTITLE 1,Load Needed Libraries
import shutil
from pyspark.sql.types import *

# COMMAND ----------

# DBTITLE 1,Drop Old Database Objects
# delete the old database and tables if needed
_ = spark.sql('DROP DATABASE IF EXISTS kkbox CASCADE')

# drop any old delta lake files that might have been created
dbutils.fs.rm('/tmp/kkbox-survival/silver', recurse=True)

# COMMAND ----------

# DBTITLE 1,Create Database to Hold Data
# create database to house SQL tables
_ = spark.sql('CREATE DATABASE kkbox')

# COMMAND ----------

# DBTITLE 1,Prep Transactions Dataset
# transaction dataset schema
transaction_schema = StructType([
  StructField('msno', StringType()),
  StructField('payment_method_id', IntegerType()),
  StructField('payment_plan_days', IntegerType()),
  StructField('plan_list_price', IntegerType()),
  StructField('actual_amount_paid', IntegerType()),
  StructField('is_auto_renew', IntegerType()),
  StructField('transaction_date', DateType()),
  StructField('membership_expire_date', DateType()),
  StructField('is_cancel', IntegerType())  
  ])

# read data from csv
transactions = (
  spark
    .read
    .csv(
      '/tmp/kkbox-survival/transactions',
      schema=transaction_schema,
      header=True,
      dateFormat='yyyyMMdd'
      )
    )

# persist in delta lake format
( transactions
    .write
    .format('delta')
    .partitionBy('transaction_date')
    .mode('overwrite')
    .save('/tmp/kkbox-survival/silver/transactions')
  )

# create table object to make delta lake queryable
spark.sql('''
  CREATE TABLE kkbox.transactions
  USING DELTA 
  LOCATION '/tmp/kkbox-survival/silver/transactions'
  ''')

# COMMAND ----------

# DBTITLE 1,Prep Members Dataset
# members dataset schema
member_schema = StructType([
  StructField('msno', StringType()),
  StructField('city', IntegerType()),
  StructField('bd', IntegerType()),
  StructField('gender', StringType()),
  StructField('registered_via', IntegerType()),
  StructField('registration_init_time', DateType())
  ])

# read data from csv
members = (
  spark
    .read
    .csv(
      '/tmp/kkbox-survival/members/members_v3.csv',
      schema=member_schema,
      header=True,
      dateFormat='yyyyMMdd'
      )
    )

# persist in delta lake format
(
  members
    .write
    .format('delta')
    .mode('overwrite')
    .save('/tmp/kkbox-survival/silver/members')
  )

# create table object to make delta lake queryable
spark.sql('''
  CREATE TABLE kkbox.members 
  USING DELTA 
  LOCATION '/tmp/kkbox-survival/silver/members'
  ''')

# COMMAND ----------

# MAGIC %md ##Step 2: Construct Membership Event Dataset
# MAGIC 
# MAGIC With the data now loaded into a more accessible format, let's begin constructing subscription lifespan information.  To do this, we need to examine the membership-relevant information contained in the transaction logs. As is typical of many applications, the transaction log data records quite a bit more information than just subscription updates.  We'll need to weed out entries that are not relevant to our analysis and begin stitching together a picture of changes to subscriptions that indicate when a customer joined, how the subscription changed over time, and ultimately when a customer abandoned the subscription, should that event have transpired.
# MAGIC 
# MAGIC Given some limitations of the dataset and some quirks of the KKBox business model addressed in the Kaggle challenge documentation, this process is quite involved.  To assist us in developing the logic required, let's narrow our focus to one customer, *msno = WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8=*, examining the transaction log information associated with this person:

# COMMAND ----------

# DBTITLE 1,Transaction History
# MAGIC %sql  -- all transactions for customer msno WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8=
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM kkbox.transactions a
# MAGIC WHERE a.msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC ORDER BY a.transaction_date, a.membership_expire_date

# COMMAND ----------

# MAGIC %md The transaction log is quite busy, often with multiple entries recorded on given transaction date (*transaction_date*) as can be observed for this customer on several dates including 
# MAGIC 2016-09-25. Based on documentation and the analysis of others who have used this dataset, we might consider records with a value of zero or less for payment plan days (*payment_plan_days*) as not relevant for our analysis.  This still leaves a lot of entries in the log for a given transaction date. 
# MAGIC 
# MAGIC Looking a little more closely at the transaction entries on a given date, it appears many are changing the subscription's expiration date (*membership_expire_date*).  There is no way to determine which is the *final* entry for a given date as the time part of the transaction date field is truncated. Given this limitation, we'll make the assumption that the expiration date furthest into the future on a given transaction date is the expiration date on the account at that point in time.  While this may not be a perfectly valid assumption, it appears that subscription change events trigger follow up transactions on later transaction dates and the expiration date associated with these seems to be more stable.  So while we may not have the information exactly right on a given transaction date, we can still get an accurate understanding of the expiration date as we look over the range of entries associated with the account:

# COMMAND ----------

# DBTITLE 1,Condensed Transaction History
# MAGIC %sql  -- drop payment_plan_days of zero or less
# MAGIC       --   and select largest exp date on a trans date  
# MAGIC 
# MAGIC SELECT 
# MAGIC   a.msno, 
# MAGIC   a.transaction_date as trans_at, 
# MAGIC   MAX(a.membership_expire_date) as expires_at 
# MAGIC FROM kkbox.transactions a
# MAGIC WHERE  
# MAGIC   a.payment_plan_days > 0 AND 
# MAGIC   a.msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC GROUP BY
# MAGIC   a.msno, a.transaction_date
# MAGIC ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md Examining the more condensed transaction history for this customer, we can see another quirk in the data occurring on December 24, 2015 (*trans_at = 2015-12-24*). On this date, the expiration date on the account is reset to a date that occurs prior to subscription initiation.  The next transaction date, the account is reset to a future date.
# MAGIC 
# MAGIC The backdated record is likely due to some kind of subscription management activity such as a change in auto-renewal status or the like.  Regardless of the reason, it's clear the backdated value should not be considered for churn identification purposes.  Instead, we might simply add logic so that if a backdated expiration date appears in the log, it is reset to the transaction date with which it is associated:

# COMMAND ----------

# DBTITLE 1,Corrected, Condensed Transaction History
# MAGIC %sql  -- correct expiration dates that have been backdated
# MAGIC 
# MAGIC SELECT
# MAGIC   x.msno,
# MAGIC   x.trans_at,
# MAGIC   CASE   -- if expiration date is prior to transaction date, then expiration date = transaction date
# MAGIC     WHEN x.expires_at < x.trans_at THEN trans_at
# MAGIC     ELSE expires_at 
# MAGIC     END as expires_at
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     a.msno, 
# MAGIC     a.transaction_date as trans_at, 
# MAGIC     MAX(a.membership_expire_date) as expires_at 
# MAGIC   FROM kkbox.transactions a
# MAGIC   WHERE  
# MAGIC     a.payment_plan_days > 0 
# MAGIC   GROUP BY
# MAGIC     a.msno, a.transaction_date
# MAGIC   ) x
# MAGIC WHERE x.msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md Finally, we must note that the overall transaction dataset terminates on March 31, 2017.  In that regard, we are observing the transaction history backward from an *effective current date* of April 1, 2017.
# MAGIC 
# MAGIC In the Kaggle challenge documentation associated with this dataset, KKBox explains that customers should not be considered as having churned until 30-days have passed since a subscription's expiration with no changes to the account that push the expiration date forward.  With this in mind, we need to place a dummy entry into the transaction dataset for April 1, 2017 that will prevent us from considering accounts that have not yet completed the 30-day expiration window as churned.  To fully understand the effect of this entry, you'll need to take a look at how we identify churn in the next section of this notebook:

# COMMAND ----------

# DBTITLE 1,Add Dummy Transactions (for Apr 1, 2017)
# MAGIC %sql  -- add dummy transaction entries for 2017-04-01 to prevent misclassification of churn in subsequent steps 
# MAGIC 
# MAGIC SELECT
# MAGIC   x.msno,
# MAGIC   x.trans_at,
# MAGIC   CASE 
# MAGIC     WHEN x.expires_at < x.trans_at THEN trans_at
# MAGIC     ELSE expires_at 
# MAGIC     END as expires_at
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     a.msno, 
# MAGIC     a.transaction_date as trans_at, 
# MAGIC     MAX(a.membership_expire_date) as expires_at 
# MAGIC   FROM kkbox.transactions a
# MAGIC   WHERE  
# MAGIC     a.payment_plan_days > 0
# MAGIC   GROUP BY
# MAGIC     a.msno, a.transaction_date
# MAGIC   UNION ALL
# MAGIC   SELECT DISTINCT  -- dummy entries to protect churn calculations
# MAGIC     msno,
# MAGIC     TO_DATE('2017-04-01') as trans_at,
# MAGIC     TO_DATE('2017-04-01') as expires_at
# MAGIC   FROM kkbox.transactions
# MAGIC   ) x
# MAGIC WHERE x.msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md ##Step 3: Identify Churn Events
# MAGIC 
# MAGIC We now have a relatively clean history of membership-relevant events for this customer. This will allow us to compare each transaction to its immediately preceding transaction to see if it resulted in a new expiration date being set.  Such events may represent subscription renewals or extensions:

# COMMAND ----------

# DBTITLE 1,Identify Membership-Meaningful Events 
# MAGIC %sql  -- compare membership-relevant transactions to prior transactions to identify expiration changes
# MAGIC 
# MAGIC WITH trans AS (  -- membership event dataset (from previous section)
# MAGIC   SELECT  -- -----------------------------------------------
# MAGIC     x.msno,
# MAGIC     x.trans_at,
# MAGIC     CASE 
# MAGIC       WHEN x.expires_at < x.trans_at THEN trans_at
# MAGIC       ELSE expires_at 
# MAGIC       END as expires_at
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       a.msno, 
# MAGIC       a.transaction_date as trans_at, 
# MAGIC       MAX(a.membership_expire_date) as expires_at 
# MAGIC     FROM kkbox.transactions a
# MAGIC     WHERE  
# MAGIC       a.payment_plan_days > 0
# MAGIC     GROUP BY
# MAGIC       a.msno, a.transaction_date
# MAGIC     UNION ALL
# MAGIC     SELECT DISTINCT 
# MAGIC       msno,
# MAGIC       TO_DATE('2017-04-01') as trans_at,
# MAGIC       TO_DATE('2017-04-01') as expires_at
# MAGIC     FROM kkbox.transactions
# MAGIC     ) x
# MAGIC   )      -- -----------------------------------------------
# MAGIC   SELECT
# MAGIC     msno,
# MAGIC     LAG(expires_at) OVER(PARTITION BY msno ORDER BY trans_at) as previous_expires_at,
# MAGIC     trans_at,
# MAGIC     expires_at,
# MAGIC     CASE   -- identify meaningful events that change expiration date
# MAGIC       WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) IS NULL THEN 1  -- new customer registration
# MAGIC       WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) != expires_at THEN 1 -- change in expiration date
# MAGIC       ELSE 0
# MAGIC       END as meaningful_event
# MAGIC   FROM trans
# MAGIC   WHERE msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC   ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md If we narrow our focus to just the meaningful membership events, we can now examine the days between the previous expiration date and the current transaction date (where the expiration date is pushed forward).  If more than 30 days has passed since expiration without a meaningful transaction, *i.e.* one that moves the expiration date forward, we would consider that churn has taken place.  
# MAGIC 
# MAGIC Of course, this calculation depends on a transaction having been recorded after the expiration date.  With many accounts, customers appear to have walked away for 6-months to a year to then renew their account.  In these scenarios, a churn event would be recorded with the subsequent activation forming a new potential chain of meaningful membership events.  In situations where no subsequent transactions occur, our dummy records inserted for April 1, 2017 will ensure we can correctly identify a churn event as having taken place:

# COMMAND ----------

# DBTITLE 1,Add Churned Flag
# MAGIC %sql  -- identify churn events as those there the next transaction is 30+ days from prior expiration date
# MAGIC 
# MAGIC WITH trans AS (  
# MAGIC   SELECT  
# MAGIC     x.msno,
# MAGIC     x.trans_at,
# MAGIC     CASE 
# MAGIC       WHEN x.expires_at < x.trans_at THEN trans_at
# MAGIC       ELSE expires_at 
# MAGIC       END as expires_at
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       a.msno, 
# MAGIC       a.transaction_date as trans_at, 
# MAGIC       MAX(a.membership_expire_date) as expires_at 
# MAGIC     FROM kkbox.transactions a
# MAGIC     WHERE  
# MAGIC       a.payment_plan_days > 0
# MAGIC     GROUP BY
# MAGIC       a.msno, a.transaction_date
# MAGIC     UNION ALL
# MAGIC     SELECT DISTINCT  
# MAGIC       msno,
# MAGIC       TO_DATE('2017-04-01') as trans_at,
# MAGIC       TO_DATE('2017-04-01') as expires_at
# MAGIC     FROM kkbox.transactions
# MAGIC     ) x
# MAGIC   )      
# MAGIC   
# MAGIC   SELECT
# MAGIC     msno,
# MAGIC     previous_expires_at,
# MAGIC     trans_at,
# MAGIC     expires_at,
# MAGIC     CASE  -- identify churn events
# MAGIC       WHEN DATEDIFF(trans_at, previous_expires_at) > 30 THEN 1
# MAGIC       ELSE 0
# MAGIC       END as churn_event
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       msno,
# MAGIC       LAG(expires_at) OVER(PARTITION BY msno ORDER BY trans_at) as previous_expires_at,
# MAGIC       trans_at,
# MAGIC       expires_at,
# MAGIC       CASE 
# MAGIC         WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) IS NULL THEN 1  
# MAGIC         WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) != expires_at THEN 1 
# MAGIC         ELSE 0
# MAGIC         END as meaningful_event
# MAGIC     FROM trans
# MAGIC     )
# MAGIC   WHERE 
# MAGIC     meaningful_event=1
# MAGIC     AND msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC   ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md Notice in the results that the downstream transaction, whether a resubscription or simply a dummy transaction, is the one flagged as being associated with the churn.  Pulling the flags back one transaction date associates the flag with the event where we'd intuitively understand churn to have occurred:

# COMMAND ----------

# DBTITLE 1,Position Churned Flag Appropriately
# MAGIC %sql -- move churn flag to prior transaction record
# MAGIC 
# MAGIC WITH trans AS ( 
# MAGIC   SELECT  
# MAGIC     x.msno,
# MAGIC     x.trans_at,
# MAGIC     CASE 
# MAGIC       WHEN x.expires_at < x.trans_at THEN trans_at
# MAGIC       ELSE expires_at 
# MAGIC       END as expires_at
# MAGIC   FROM (
# MAGIC     SELECT 
# MAGIC       a.msno, 
# MAGIC       a.transaction_date as trans_at, 
# MAGIC       MAX(a.membership_expire_date) as expires_at 
# MAGIC     FROM kkbox.transactions a
# MAGIC     WHERE  
# MAGIC       a.payment_plan_days > 0
# MAGIC     GROUP BY
# MAGIC       a.msno, a.transaction_date
# MAGIC     UNION ALL
# MAGIC     SELECT DISTINCT  
# MAGIC       msno,
# MAGIC       TO_DATE('2017-04-01') as trans_at,
# MAGIC       TO_DATE('2017-04-01') as expires_at
# MAGIC     FROM kkbox.transactions
# MAGIC     ) x
# MAGIC   )      
# MAGIC   SELECT
# MAGIC     msno,
# MAGIC     previous_expires_at,
# MAGIC     trans_at,
# MAGIC     expires_at,
# MAGIC     LEAD(churn_event, 1) OVER (PARTITION BY msno ORDER BY trans_at) as churned -- adjust churn flag assignment
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       msno,
# MAGIC       previous_expires_at,
# MAGIC       trans_at,
# MAGIC       expires_at,
# MAGIC       CASE
# MAGIC         WHEN DATEDIFF(trans_at, previous_expires_at) > 30 THEN 1
# MAGIC         ELSE 0
# MAGIC         END as churn_event
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         msno,
# MAGIC         LAG(expires_at) OVER(PARTITION BY msno ORDER BY trans_at) as previous_expires_at,
# MAGIC         trans_at,
# MAGIC         expires_at,
# MAGIC         CASE 
# MAGIC           WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) IS NULL THEN 1  
# MAGIC           WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) != expires_at THEN 1
# MAGIC           ELSE 0
# MAGIC           END as meaningful_event
# MAGIC       FROM trans
# MAGIC       )
# MAGIC     WHERE meaningful_event=1
# MAGIC     )
# MAGIC   WHERE msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC   ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md This now leaves our dummy transaction records as having a NULL flag value.  By removing these, we now have an accurate subscription history log for our customer:

# COMMAND ----------

# DBTITLE 1,Remove Dummy Transactions
# MAGIC %sql  -- remove the dummy transactions added earlier
# MAGIC 
# MAGIC WITH trans AS (  
# MAGIC   SELECT  
# MAGIC     x.msno,
# MAGIC     x.trans_at,
# MAGIC     CASE 
# MAGIC       WHEN x.expires_at < x.trans_at THEN trans_at
# MAGIC       ELSE expires_at 
# MAGIC       END as expires_at
# MAGIC   FROM (
# MAGIC     SELECT 
# MAGIC       a.msno, 
# MAGIC       a.transaction_date as trans_at, 
# MAGIC       MAX(a.membership_expire_date) as expires_at 
# MAGIC     FROM kkbox.transactions a
# MAGIC     WHERE  
# MAGIC       a.payment_plan_days > 0
# MAGIC     GROUP BY
# MAGIC       a.msno, a.transaction_date
# MAGIC     UNION ALL
# MAGIC     SELECT DISTINCT  
# MAGIC       msno,
# MAGIC       TO_DATE('2017-04-01') as trans_at,
# MAGIC       TO_DATE('2017-04-01') as expires_at
# MAGIC     FROM kkbox.transactions
# MAGIC     ) x
# MAGIC   )      
# MAGIC SELECT *
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     msno,
# MAGIC     previous_expires_at,
# MAGIC     trans_at,
# MAGIC     expires_at,
# MAGIC     LEAD(churn_event, 1) OVER (PARTITION BY msno ORDER BY trans_at) as churned 
# MAGIC   FROM (
# MAGIC     SELECT
# MAGIC       msno,
# MAGIC       previous_expires_at,
# MAGIC       trans_at,
# MAGIC       expires_at,
# MAGIC       CASE
# MAGIC         WHEN DATEDIFF(trans_at, previous_expires_at) > 30 THEN 1
# MAGIC         ELSE 0
# MAGIC         END as churn_event
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         msno,
# MAGIC         LAG(expires_at) OVER(PARTITION BY msno ORDER BY trans_at) as previous_expires_at,
# MAGIC         trans_at,
# MAGIC         expires_at,
# MAGIC         CASE 
# MAGIC           WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) IS NULL THEN 1 
# MAGIC           WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) != expires_at THEN 1 
# MAGIC           ELSE 0
# MAGIC           END as meaningful_event
# MAGIC       FROM trans
# MAGIC       )
# MAGIC     WHERE meaningful_event=1
# MAGIC     )
# MAGIC   )
# MAGIC WHERE churned IS NOT NULL  -- remove dummy transaction entries
# MAGIC       AND msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md Let's remove the customer-specific constraint, clean up our final SELECT-list and convert our query to a view to make its use across all customer subscriptions a little easier going forward:

# COMMAND ----------

# DBTITLE 1,Convert to View for Reuse
# MAGIC %sql  -- create view from query (with customer-filter removed)
# MAGIC 
# MAGIC DROP VIEW IF EXISTS kkbox.membership_history;
# MAGIC 
# MAGIC CREATE VIEW kkbox.membership_history
# MAGIC AS
# MAGIC   WITH trans AS (  
# MAGIC     SELECT  
# MAGIC       x.msno,
# MAGIC       x.trans_at,
# MAGIC       CASE 
# MAGIC         WHEN x.expires_at < x.trans_at THEN trans_at
# MAGIC         ELSE expires_at 
# MAGIC         END as expires_at
# MAGIC     FROM (
# MAGIC       SELECT 
# MAGIC         a.msno, 
# MAGIC         a.transaction_date as trans_at, 
# MAGIC         MAX(a.membership_expire_date) as expires_at 
# MAGIC       FROM kkbox.transactions a
# MAGIC       WHERE  
# MAGIC         a.payment_plan_days > 0
# MAGIC       GROUP BY
# MAGIC         a.msno, a.transaction_date
# MAGIC       UNION ALL
# MAGIC       SELECT DISTINCT  
# MAGIC         msno,
# MAGIC         TO_DATE('2017-04-01') as trans_at,
# MAGIC         TO_DATE('2017-04-01') as expires_at
# MAGIC       FROM kkbox.transactions
# MAGIC       ) x
# MAGIC     )      
# MAGIC SELECT 
# MAGIC   msno,
# MAGIC   trans_at,
# MAGIC   expires_at,
# MAGIC   churned
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC       msno,
# MAGIC       previous_expires_at,
# MAGIC       trans_at,
# MAGIC       expires_at,
# MAGIC       LEAD(churn_event, 1) OVER (PARTITION BY msno ORDER BY trans_at) as churned 
# MAGIC     FROM (
# MAGIC       SELECT
# MAGIC         msno,
# MAGIC         previous_expires_at,
# MAGIC         trans_at,
# MAGIC         expires_at,
# MAGIC         CASE
# MAGIC           WHEN DATEDIFF(trans_at, previous_expires_at) > 30 THEN 1
# MAGIC           ELSE 0
# MAGIC           END as churn_event
# MAGIC       FROM (
# MAGIC         SELECT
# MAGIC           msno,
# MAGIC           LAG(expires_at) OVER(PARTITION BY msno ORDER BY trans_at) as previous_expires_at,
# MAGIC           trans_at,
# MAGIC           expires_at,
# MAGIC           CASE 
# MAGIC             WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) IS NULL THEN 1
# MAGIC             WHEN (LAG(expires_at, 1) OVER(PARTITION BY msno ORDER BY trans_at)) != expires_at THEN 1
# MAGIC             ELSE 0
# MAGIC             END as meaningful_event
# MAGIC         FROM trans
# MAGIC         )
# MAGIC       WHERE meaningful_event=1
# MAGIC       )
# MAGIC     )
# MAGIC WHERE churned IS NOT NULL 
# MAGIC       --AND msno='WAMleIuG124oDSSdhcZIECwLcHHHbk4or0y6gxkK7t8='
# MAGIC ORDER BY msno, trans_at, expires_at

# COMMAND ----------

# MAGIC %md And now we can explore all customer's history of subscription activations, expirations and renewals:

# COMMAND ----------

# DBTITLE 1,Membership Change History for All Members
# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM kkbox.membership_history

# COMMAND ----------

# MAGIC %md ##Step 4: Derive Subscription Info
# MAGIC 
# MAGIC Now that we have our membership history assembled and flagged for churn events, we can assemble a subscription dataset.  This dataset should have one entry per customer for each subscription.  A subscription will be identified as having ended on the earliest churn event that is on or after a given transaction date.  It will be identified as having started on the earliest transaction associated with that date. A single customer (as identified by the *msno* value) may have multiple subscriptions, though only one would be active at a given time:

# COMMAND ----------

# DBTITLE 1,Subscription History View
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS kkbox.subscriptions;
# MAGIC 
# MAGIC CREATE TABLE kkbox.subscriptions
# MAGIC USING delta
# MAGIC AS
# MAGIC   SELECT
# MAGIC     ROW_NUMBER() OVER(ORDER BY p.starts_at) as subscription_id,
# MAGIC     p.msno,
# MAGIC     p.starts_at,
# MAGIC     p.ends_at,
# MAGIC     1 + DATEDIFF(COALESCE(p.ends_at,'2017-04-01'), p.starts_at) as duration_days,
# MAGIC     CASE WHEN ends_at IS NULL THEN 0 ELSE 1 END as churned,
# MAGIC     COUNT(*) OVER(PARTITION BY p.msno ORDER BY p.starts_at) - 1 as prior_subscriptions
# MAGIC   FROM (  -- first transaction prior to or on churn date
# MAGIC     SELECT 
# MAGIC       x.msno,
# MAGIC       MIN(x.trans_at) as starts_at,
# MAGIC       x.churns_at as ends_at
# MAGIC     FROM (  -- membership history with associated churn dates (for each transaction)
# MAGIC       SELECT  
# MAGIC         a.msno,
# MAGIC         a.trans_at,
# MAGIC         a.expires_at,
# MAGIC         a.churned,
# MAGIC         MIN(b.expires_at) as churns_at
# MAGIC       FROM kkbox.membership_history a  -- all events 
# MAGIC       LEFT OUTER JOIN kkbox.membership_history b  -- churn events only
# MAGIC         ON a.msno = b.msno AND  -- match customer
# MAGIC            1 = b.churned AND    -- the right-hand (b) dataset is limited to churn events
# MAGIC            a.trans_at <= b.expires_at -- transactions prior to or on date of churn
# MAGIC       GROUP BY 
# MAGIC         a.msno,
# MAGIC         a.trans_at,
# MAGIC         a.expires_at,
# MAGIC         a.churned
# MAGIC         ) x
# MAGIC     GROUP BY
# MAGIC       x.msno,
# MAGIC       x.churns_at
# MAGIC     ) p
# MAGIC   ORDER BY p.msno, starts_at, ends_at

# COMMAND ----------

# DBTITLE 1,Subscription History
# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM kkbox.subscriptions
# MAGIC ORDER BY subscription_id
