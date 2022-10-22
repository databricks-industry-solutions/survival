# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/survival. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/survivorship-and-churn.

# COMMAND ----------

# MAGIC %md With our subscription data prepared, we can now begin examining patterns of customer attrition observed with the KKBox music service.  In this notebook, we'll get oriented to general patterns of customer dropout in preparation for the more detailed work taking place in the next notebook.

# COMMAND ----------

# MAGIC %md ##Step 1: Prepare the Environment
# MAGIC 
# MAGIC The techniques we'll use in this and subsequent notebooks come from the domain of [Survival Analysis](https://en.wikipedia.org/wiki/Survival_analysis#:~:text=Survival%20analysis%20is%20a%20branch,and%20failure%20in%20mechanical%20systems.). While there are several notebooks available in Python that support these techniques, we'll leverage [lifelines](https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html), the most popular of the survival analysis libraries currently available.  To do this, we'll first need to install and load the library to our cluster:
# MAGIC 
# MAGIC **NOTE** The next cell assumes you are running this notebook on a Databricks cluster that does not make use of the ML runtime.  If using an ML runtime, please follow these [alternative steps](https://docs.databricks.com/libraries.html#workspace-library) to load the lifelines library to your environment. 

# COMMAND ----------

# DBTITLE 1,Install Needed Libraries
# MAGIC %pip install lifelines

# COMMAND ----------

# DBTITLE 1,Load Needed Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import pairwise_logrank_test

# COMMAND ----------

# MAGIC %md ##Step 2: Examine Population-Level Survivorship
# MAGIC 
# MAGIC Using the full subscription dataset, we can take a look at how members dropout of the subscription service over time.  To do this, we will derive a [Kaplan-Meier curve](https://towardsdatascience.com/kaplan-meier-curves-c5768e349479#:~:text=The%20Kaplan%2DMeier%20estimator%20is,at%20a%20certain%20time%20interval.) which will identify the probability of survival at a point in time (in terms of days since subscription inception) using some simple statistical calculations derived from the observed data:

# COMMAND ----------

# DBTITLE 1,Retrieve Subscription Data
# retrieve subscription data to pandas DF
subscriptions_pd = spark.table('kkbox.subscriptions').toPandas()
subscriptions_pd.head()

# COMMAND ----------

# DBTITLE 1,Derive Survival Curve for Population
kmf = KaplanMeierFitter(alpha=0.05) # calculate a 95% confidence interval
kmf.fit(subscriptions_pd['duration_days'], subscriptions_pd['churned'])

# COMMAND ----------

# MAGIC %md The output of the previous step tells us we fit the KM model using nearly 3.1 million subscription records of which about 1.3 million were still active as of April 1, 2017.  (The term *right-censored* tells us that the event of interest, *i.e.* churn, has not occurred within our observation window.)  Using this model, we can now calculate the median survival time for any given subscription:

# COMMAND ----------

# DBTITLE 1,Calculate Median Survival Time
median_ = kmf.median_survival_time_
median_

# COMMAND ----------

# MAGIC %md Per documentation associated with the dataset, KKBox members typically subscribe to the service on a 30-day cycle. The median survival time of 184-days would indicate that most customers sign-up for an initial 30-day term and renew subscriptions month over month for 6-months on average before dropping out.
# MAGIC 
# MAGIC Passing the model various values that correspond with 30-day initial registration, a 1-year commitment, and a 2-year renewal, we can calculate the probability a customer continues with the service, *i.e.* survives past that point in time:

# COMMAND ----------

# DBTITLE 1,Portion of Population Surviving at Point in Time
kmf.predict([30, 365, 730])

# COMMAND ----------

# MAGIC %md Graphing this out, we can see how customer drop-out varies as subscriptions age:

# COMMAND ----------

# DBTITLE 1,The Survival Rate over Time
# plot attributes
plt.figure(figsize=(12,8))
plt.title('All Subscriptions', fontsize='xx-large')

# y-axis
plt.ylabel('Survival Rate', fontsize='x-large')
plt.ylim((0.0, 1.0))
plt.yticks(np.arange(0.0,1.0,0.05))
plt.axhline(0.5, color='red', alpha=0.75, linestyle=':') # median line in red

# x-axis
plt.xlabel('Timeline (Days)', fontsize='x-large')
plt.xticks(range(0,800,30))
plt.axvline(30, color='gray', alpha=0.5, linestyle=':')  # 30-day gray dashed line
plt.axvline(180, color='gray', alpha=0.5, linestyle=':')  # 30-day gray dashed line
plt.axvline(365, color='gray', alpha=0.5, linestyle=':') # 1-year gray dashed line
plt.axvline(365*2, color='gray', alpha=0.5, linestyle=':') # 2-year gray dashed line

kmf.plot_survival_function()

# COMMAND ----------

# MAGIC %md ##Step 3: Examine How Survivorship Varies
# MAGIC 
# MAGIC The overall pattern of survivorship tells a pretty compelling story about customer attrition at KKBox, but it can be interesting to examine how this pattern varies by subscription attributes. By focusing on attributes of the subscription known at the time of its creation, we may be able to identify variables that tell us something about the long-term retention probability on an account at a time when we may be offering the greatest incentives to acquire a customer. With this in mind, we'll examine the registration channel associated with the subscription along with the initial payment method and payment plan days selected at initial registration:

# COMMAND ----------

# DBTITLE 1,Subscriptions with Initial Attributes
# MAGIC %sql  -- identify registration channel, initial payment method and initial payment plan days by subscription
# MAGIC DROP VIEW IF EXISTS subscription_attributes;
# MAGIC 
# MAGIC CREATE TEMP VIEW subscription_attributes 
# MAGIC AS
# MAGIC   WITH transaction_attributes AS ( -- get payment method and plan days for each subscriber's transaction date
# MAGIC       SELECT
# MAGIC         a.msno,
# MAGIC         a.trans_at,
# MAGIC         FIRST(b.payment_method_id) as payment_method_id,
# MAGIC         FIRST(b.payment_plan_days) as payment_plan_days
# MAGIC       FROM (  -- base transaction dates
# MAGIC         SELECT 
# MAGIC           msno,
# MAGIC           transaction_date as trans_at,
# MAGIC           MAX(membership_expire_date) as expires_at
# MAGIC         FROM kkbox.transactions
# MAGIC         GROUP BY msno, transaction_date
# MAGIC         ) a
# MAGIC       INNER JOIN kkbox.transactions b
# MAGIC         ON  a.msno=b.msno AND
# MAGIC             a.trans_at=b.transaction_date AND 
# MAGIC             a.expires_at=b.membership_expire_date
# MAGIC       WHERE b.payment_plan_days > 0
# MAGIC       GROUP BY
# MAGIC         a.msno,
# MAGIC         a.trans_at
# MAGIC       )
# MAGIC   SELECT
# MAGIC     m.*,
# MAGIC     n.payment_method_id as init_payment_method_id,
# MAGIC     n.payment_plan_days as init_payment_plan_days,
# MAGIC     COALESCE(CAST(o.registered_via as String), 'Unknown') as registered_via
# MAGIC   FROM kkbox.subscriptions m
# MAGIC   INNER JOIN transaction_attributes n
# MAGIC     ON  m.msno=n.msno AND      -- match on customer
# MAGIC         m.starts_at=n.trans_at -- and transaction date at start of transaction
# MAGIC   LEFT OUTER JOIN kkbox.members o  -- membership info (assume stable across subscription)
# MAGIC     ON m.msno=o.msno
# MAGIC   ORDER BY m.subscription_id

# COMMAND ----------

# DBTITLE 1,Subscriptions with Initial Attributes
# capture output to Spark DataFrame
subscriptions = spark.table('subscription_attributes')

# capture output to pandas DataFrame
subscriptions_pd = subscriptions.toPandas()
subscriptions_pd.head()

# COMMAND ----------

# MAGIC %md With attributes now associated with our subscriptions, let's examine the registration channels by which customers subscribe to the service:

# COMMAND ----------

# DBTITLE 1,Members by Registration Channel
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   registered_via,
# MAGIC   COUNT(DISTINCT msno) as members
# MAGIC FROM subscription_attributes
# MAGIC GROUP BY registered_via
# MAGIC ORDER BY members DESC

# COMMAND ----------

# MAGIC %md We don't know anything about the numbered channels but it's clear that channels 7 & 9 are the most popular by far.  Several other channels are pretty popular while there are a few which have a nominal number of subscribers associated with them. 
# MAGIC 
# MAGIC To keep our analysis simple, let's eliminate channels 13, 10 & 16 which combined are associated with less than 0.3% of our unique subscribers.  Doing this, we can now revisit our survival chart, presenting separate curves for each of the remaining channels:

# COMMAND ----------

# DBTITLE 1,Survival Rate by Registration Channel
# eliminate nominal channels
channels_pd = subscriptions_pd[~subscriptions_pd['registered_via'].isin(['10','13','16'])]

# configure the plot
plt.figure(figsize=(12,8))
ax = plt.subplot(111)
plt.title('By Registration Channel', fontsize='xx-large')

# configure the x-axis
plt.xlabel('Timeline (Days)', fontsize='x-large')
plt.xticks(range(0,800,30))

# configure the y-axis
plt.ylabel('Survival Rate', fontsize='x-large')
plt.ylim((0.0, 1.0))
plt.yticks(np.arange(0.0,1.0,0.05))

# graph each curve on the plot
for name, grouped_pd in channels_pd.groupby('registered_via'):
    kmf = KaplanMeierFitter(alpha=0.05)
    kmf.fit(
      grouped_pd['duration_days'], 
      grouped_pd['churned'], 
      label='Channel {0}'.format(name)
      )
    kmf.plot(ax=ax)

# COMMAND ----------

# MAGIC %md Before attempting to interpret these different curves, it's important we evaluate whether they are statistically different from one another.  Comparing each curve to the others, we can calculate the probability these curves do not differ from one another using the [log-rank test](https://en.wikipedia.org/wiki/Logrank_test):
# MAGIC 
# MAGIC **NOTE** By adding an argument for t_0 to the call below, you can calculate the same metrics for each curve at a specific point in time, instead of across all times as shown here.

# COMMAND ----------

# DBTITLE 1,Probability Curves Are the Same (Overall)
log_rank = pairwise_logrank_test(channels_pd['duration_days'], channels_pd['registered_via'], channels_pd['churned'])
log_rank.summary

# COMMAND ----------

# DBTITLE 1,Probability Curves Are the Same (at Day 184)
log_rank = pairwise_logrank_test(channels_pd['duration_days'], channels_pd['registered_via'], channels_pd['churned'], t_0=184)
log_rank.summary

# COMMAND ----------

# MAGIC %md Overall and specifically at day 184, the median survival date as identified above, most of these curves is significantly different from one another (as indicated by nearly all p-values being < 0.05). This tells us that the different representations in the chart above are meaningful.  But how?  Without additional information regarding the numbered channels, it's hard to tell a compelling story as to why some customers see higher attrition than others.  Still, KKBox may want to explore why some channels seem to have higher retention rates and examine differences in cost associated with each channel in order to maximize the effectiveness of their customer acquisition efforts.
# MAGIC 
# MAGIC Now, let's do this same analysis for the payment method used when the subscription was created:

# COMMAND ----------

# DBTITLE 1,Members by Initial Payment Method
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   init_payment_method_id,
# MAGIC   COUNT(DISTINCT msno) as customers
# MAGIC FROM subscription_attributes
# MAGIC GROUP BY init_payment_method_id
# MAGIC ORDER BY customers DESC

# COMMAND ----------

# MAGIC %md The number of payment methods in the dataset is quite large.  Unlike with registration channels where there was a clear delineation between the popular and the unpopular channels, the decline in subscription counts with payment methods is more gradual.  With this in mind, we'll set an arbitrary cutoff of 10,000 members associated with a payment method for it to be included in our analysis:

# COMMAND ----------

# DBTITLE 1,Get Subscription Data for Those Associated with Popular Initial Payment Methods
payment_methods_pd = spark.sql('''
  SELECT
    duration_days,
    churned,
    init_payment_method_id
  FROM subscription_attributes
  WHERE init_payment_method_id IN (
    SELECT
      init_payment_method_id
    FROM subscription_attributes
    GROUP BY init_payment_method_id
    HAVING COUNT(DISTINCT msno)>10000
    )''').toPandas()

# COMMAND ----------

# DBTITLE 1,Survival Rate by Initial Payment Method
# configure the plot
plt.figure(figsize=(12,8))
ax = plt.subplot(111)
plt.title('By Initial Payment Method', fontsize='xx-large')   

# configure the y-axis
plt.ylabel('Survival Rate', fontsize='x-large')
plt.ylim((0.0, 1.0))
plt.yticks(np.arange(0.0,1.0,0.05))

# configure the x-axis
plt.xlabel('Timeline (Days)', fontsize='x-large')
plt.xticks(range(0,800,30))

# calculate the surival rates 
for name, grouped_pd in payment_methods_pd.groupby('init_payment_method_id'):
    kmf = KaplanMeierFitter(alpha=0.05)
    kmf.fit(
      grouped_pd['duration_days'], 
      grouped_pd['churned'], 
      label='Method {0}'.format(name)
      )
    _ = kmf.plot(ax=ax)
    _.legend(loc='upper right')

# COMMAND ----------

# MAGIC %md Even when focusing just on the more popular methods, the chart above is quite a bit busier than the one before. This is a chart for which we should carefully consider the statistical differences between any two payment methods before drawing too many hard conclusions.  But in the interest of space, we'll interpret this chart now, addressing the pairwise statistical comparisons in the cell below.
# MAGIC 
# MAGIC Without knowledge with which to speculate why, it's very interesting that some methods have very different drop-off rates.  For example, compare method 34 at the top of the chart and method 35 at the bottom.  Do these different payment methods indicate differences in ability to pay for a service over the long-term, *e.g.* credit cards associated with higher earners or those with higher credit scores *vs.* cards associated with lower earners or those with lower scores?  Alternatively, could some of these initial payment methods be tied to vouchers where the customer somehow is foregoing payment or receiving the service at a discount during an initial 30-day window.  When the customer is then asked to pay the regular price, the customer may be dropping out of the service as they weren't terribly invested in the service from the outset.  (It's important to remember we only know the initial payment method used and not subsequent payment methods employed, though that data is presumably in our transaction log dataset.) Again, without more business knowledge, we can only speculate, but given the large magnitude differences show here, this would be an aspect of customer acquisition worth exploring in more detail.
# MAGIC 
# MAGIC Here is the pairwise comparison, limited to those curves that don't quite differ enough from one another to be considered statistically different:

# COMMAND ----------

# DBTITLE 1,Probability Curves Are the Same
log_rank = pairwise_logrank_test(payment_methods_pd['duration_days'], payment_methods_pd['init_payment_method_id'], payment_methods_pd['churned'])
summary = log_rank.summary
summary[summary['p']>=0.05]

# COMMAND ----------

# MAGIC %md From the Log Rank test results, it would appear that payment methods 22 and 32 are statistically undifferentiated (as are 20 and 40). 
# MAGIC 
# MAGIC Next, let's examine the days configured for the payment plan at the initiation of a subscription.  This is an odd attribute of the subscriptions as it could be viewed as either continuous or discrete. Let's treat it as descrete here to see how we might handle it in later analysis.
# MAGIC 
# MAGIC At this point, I assume the pattern for this analysis is familiar:

# COMMAND ----------

# DBTITLE 1,Members by Initial Payment Plan Days
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   init_payment_plan_days,
# MAGIC   COUNT(DISTINCT msno) as customers
# MAGIC FROM subscription_attributes
# MAGIC GROUP BY init_payment_plan_days
# MAGIC ORDER BY customers DESC

# COMMAND ----------

# DBTITLE 1,Plot Popular Initial Payment Play Days
payment_plan_days_pd = spark.sql('''
  SELECT
    duration_days,
    churned,
    init_payment_plan_days
  FROM subscription_attributes
  WHERE init_payment_plan_days IN (
    SELECT
      init_payment_plan_days
    FROM subscription_attributes
    GROUP BY init_payment_plan_days
    HAVING COUNT(DISTINCT msno)>10000
    )''').toPandas()
   
plt.figure(figsize=(12,8))
ax = plt.subplot(111)
plt.title('By Initial Payment Plan Days', fontsize='xx-large')

plt.ylabel('Survival Rate', fontsize='x-large')
plt.ylim((0.0, 1.0))
plt.yticks(np.arange(0.0,1.0,0.05))

plt.xlabel('Timeline (Days)', fontsize='x-large')
plt.xticks(range(0,800,30))

# calculate the surival rates 
for name, grouped_pd in payment_plan_days_pd.groupby('init_payment_plan_days'):
    kmf = KaplanMeierFitter(alpha=0.05)
    kmf.fit(
      grouped_pd['duration_days'], 
      grouped_pd['churned'], 
      label='Days {0}'.format(name)
      )
    _ = kmf.plot(ax=ax)
    _.legend(loc='upper right')

# COMMAND ----------

# MAGIC %md It appears as if most customers make it to their first renewal and then there is a sizeable drop. Not every plan sees the same rate of drop-out at that time with the 7-day and 10-days plans having massive drop-out at the first renewal compared to the 30-day plan which KKBox indicates is the traditional plan. Also interesting is that the 100-day plan sees a steep drop in customers at it's first renewal point (with it's survival rate following below that of the 30-day subscribers) while the 90-day subscribers follow a very different trajectory until later in the subscription lifecycle.
# MAGIC 
# MAGIC While the confidence intervals on these curves are more visible than some of the earlier ones, statistically, all the curves are significant:

# COMMAND ----------

# DBTITLE 1,Probability Curves Are the Same
log_rank = pairwise_logrank_test(payment_plan_days_pd['duration_days'], payment_plan_days_pd['init_payment_plan_days'], payment_plan_days_pd['churned'])
summary = log_rank.summary
summary[summary['p']>=0.05]

# COMMAND ----------

# MAGIC %md Lastly, let's consider whether subscribers who previously churned might follow different paths from those who have only had a single subscription:

# COMMAND ----------

# DBTITLE 1,Plot Prior Subscription Count
plt.figure(figsize=(12,8))
ax = plt.subplot(111)
plt.title('By Prior Subscription Count', fontsize='xx-large')

plt.ylabel('Survival Rate', fontsize='x-large')
plt.ylim((0.0, 1.0))
plt.yticks(np.arange(0.0,1.0,0.05))

plt.xlabel('Timeline (Days)', fontsize='x-large')
plt.xticks(range(0,800,30))
 
for name, grouped_pd in subscriptions_pd.groupby('prior_subscriptions'):
    kmf = KaplanMeierFitter(alpha=0.05)
    kmf.fit(
      grouped_pd['duration_days'], 
      grouped_pd['churned'], 
      label='Prior Subscriptions {0}'.format(name)
      )
    kmf.plot(ax=ax)

# COMMAND ----------

# DBTITLE 1,Probability Curves Are the Same
log_rank = pairwise_logrank_test(subscriptions_pd['duration_days'], subscriptions_pd['prior_subscriptions'], subscriptions_pd['churned'])
summary = log_rank.summary
summary[summary['p']>=0.05]

# COMMAND ----------

# MAGIC %md  As the number of prior subscription increases, the number of subscribers falling into each category declines leading to larger and larger confidence intervals. There does appear to be a general pattern of subscribers with a few prior subscriptions being retained at higher rates but only up to a point.  And then it appears the number of priors doesn't really offer any protection from drop-out.  The lack of statistical significance would encourage us not to put too much emphasis on this narrative but still it might be interesting for KKBox to examine.
