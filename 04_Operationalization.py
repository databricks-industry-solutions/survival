# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/survival. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/survivorship-and-churn.

# COMMAND ----------

# MAGIC %md For most organizations, the development, evaluation and interpretation of the Cox PH model is the end goal. The information it provides is incredibly useful in exploring when and why customers are dropping out of a subscription service, helping organizations build better customer acquisition and retention strategies.  Still, they are capable of predicting retention probabilities for specific customers that can be useful in applications such as the calculation of customer lifetime value.  With that in mind, we'll examine how we might leveraging our Cox PH models for survival estimation.
# MAGIC 
# MAGIC While many Data Science libraries available in Python make use of a SciKit-Learn API, the [lifelines](https://lifelines.readthedocs.io/en/latest/index.html#) library we are using does not.  While an experimental sklearn_adapter interface is available with the library, it does not yet accept the numpy arrays that are typically produced by the various sklearn transformations, making the bundling of transformation logic with a model difficult.   
# MAGIC 
# MAGIC With that in mind, we will explore an alternative means of deploying the model, one that recognizes that the baseline hazard rates and survival functions can be easily extracted from our trained model and combined with similarly extracted coefficients to arrive at the same predictions. As the sklearn adapter interface evolves on the lifelines library, this may be a pattern we revisit.

# COMMAND ----------

# MAGIC %md ##Step 1: Setup the Environment
# MAGIC 
# MAGIC Here, we will reload the dataset and trained model persisted to storage in the last notebook:

# COMMAND ----------

# DBTITLE 1,Install Needed Libraries
# MAGIC %pip install lifelines

# COMMAND ----------

# DBTITLE 1,Load Libraries
import pandas as pd 
import numpy as np

from mlflow import sklearn

import shutil

# COMMAND ----------

# DBTITLE 1,Reload Persisted Dataframes from Last Notebook
dataset_path = '/dbfs/tmp/kkbox-survival/tmp/datasets/'

# read datasets from temp storage
subscriptions_pd = pd.read_pickle(dataset_path + 'subscriptions_pd')
survival_pd = pd.read_pickle(dataset_path + 'survival_pd')

# COMMAND ----------

# DBTITLE 1,Reload Persisted Models from Last Notebook
# path to saved models
model_path = '/dbfs/tmp/kkbox-survival/tmp/models/'

# load models from storage
cph = sklearn.load_model(model_path + 'cph')

# COMMAND ----------

# MAGIC %md ##Step 2: Extract the Data Required for Baseline Hazard Calculations
# MAGIC 
# MAGIC To understand how we might deploy our models, let's extract the baseline cumulative hazard data from it:

# COMMAND ----------

# DBTITLE 1,Extract the Baseline Hazard Data
baseline_chazard = cph.baseline_cumulative_hazard_.copy()
baseline_chazard.head(10)

# COMMAND ----------

# MAGIC %md Notice how the hazard, *i.e.* rate of drop-out, increases with each passing day (recorded in the DataFrame's index).  Notice too that each payment plan days option has its own column of hazard data. To predict the rate of customer dropout for subscribers aligned with our reference features, *i.e.* a channel 7 and payment method 41, up to a given point in time, we simply retrieve the value associated with that day for the payment plan on which the customer is on.
# MAGIC 
# MAGIC When a customer does not elect reference features, we simply lookup the coefficients for the differing channels and  payment methods:

# COMMAND ----------

# DBTITLE 1,Extract the Coefficients
coefficients = cph.params_
coefficients

# COMMAND ----------

# MAGIC %md It's important to note that these values are the coefficients presented in the model summaries reviewed in the last notebook.  To convert them to the factors discussed in that notebook, we must transform them using the exponential function, *i.e.* *exp()*. If we are considering multiple factors, we can simply add them to one another before applying the exponential function in order to arrive at a combined factor to be applied to the baseline.  This is described in the Cox PH hazard rate formula:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/kkbox_hazardrate.png' width=500>
# MAGIC 
# MAGIC Notice the (*x_i* - *x_i_baseline*) term that's applied to the *b_i* term as part of the partial hazard calculation. In that part of the function, we'd take the value of our feature and subtract from it the baseline value. Because all our features are one-hot encoded as 1 for the non-reference feature and 0 for the baseline reference feature, this part of the calculation resolves to a 1x multiplier, allowing us to simply sum our coefficients, transform and apply them. 
# MAGIC 
# MAGIC With this in mind, let's try to calculate the cumulative hazard rate for an arbitrarily selected subscriber registering via channel 7, paying with method 28 and subscribing to an initial payment plan of 30 days:

# COMMAND ----------

# DBTITLE 1,Retrieve Coefficients
# function to look up coefficient value
def get_coefficient(key, coefficients):
  if key in coefficients.keys():
    return coefficients[key]
  else:
    return 0

#feature values
channel = 7
method = 28

# retrieve coefficients
channel_coefficient = get_coefficient('channel_{0}'.format(channel), coefficients) # is part of the baseline so should not have it's own coeff
method_coefficient = get_coefficient('method_{0}'.format(method), coefficients) 

# display results
print('Channel Coefficient: {0:.4f}'.format(channel_coefficient))
print('Method Coefficient: {0:.4f}'.format(method_coefficient))

# COMMAND ----------

# MAGIC %md Notice that channel 7 is not found in the coefficients dataset because it's part of the baseline (which uses channel 7 and method 41 as its reference features).  By returning 0 for any coefficient not found in the coefficients dataset, we are in effect stating the baseline captures that feature's contribution to subscriber hazard.
# MAGIC 
# MAGIC Let's now apply our coefficients to the baseline hazard (using the formula above) to predict the cumulative hazard for this customer on day 184. It's important to remember that we have separate baselines for different initial payment plan days so that we'll need to be sure to select the appropriate baseline hazard column: 

# COMMAND ----------

# DBTITLE 1,Apply Coefficients to Baseline to Predict Hazard
#feature values
channel = 7
method = 28
plan = 30
t = 184

# retrieve coefficients
channel_coefficient = get_coefficient('channel_{0}'.format(channel), coefficients)
method_coefficient = get_coefficient('method_{0}'.format(method), coefficients) 

# calculate hazard
baseline_at_t = baseline_chazard.loc[t, plan]
partial_hazard = np.exp(channel_coefficient + method_coefficient)
hazard_rate = baseline_at_t * partial_hazard

# display results
print('Cumulative Hazard Rate @ Day {1}:\t{0:.4f}'.format(hazard_rate, t))
print('   Baseline Hazard @ Day {1}:\t\t{0:.4f}'.format(baseline_at_t, t))
print('   Partial Hazard:\t\t\t{0:.4f}'.format(partial_hazard))

# COMMAND ----------

# MAGIC %md Just to verify we are doing this correctly, let's compare our calculated result against the one returned by the lifelines model for this same subscription at the same point in time.  We'll grab a subscriber from our input dataset that matches these criteria so that we don't have to juggle the encoding logic:

# COMMAND ----------

# DBTITLE 1,Use Model to Predict Hazard
# retrieve encoded value for our selected customer
X = survival_pd[
  (subscriptions_pd['registered_via']==channel) &
  (subscriptions_pd['init_payment_method_id']==method) &
  (subscriptions_pd['init_payment_plan_days']==plan)
  ].head(1)

# predict cumulative hazard
cph.predict_cumulative_hazard(X, times=[t])

# COMMAND ----------

# MAGIC %md **The model produces a different result from the one we calculated!!!**  To skip over a bunch of investigative work that was done to determine why this is, I'll simply note that there appears to be a **hidden partial hazard** in the model that we need to apply to all calculations.
# MAGIC 
# MAGIC To see this *hidden* value, consider that the baseline represents the time-dependent hazard rate for a reference customer, *i.e.* one that registered via channel 7 and paid via method 41.  The partial hazard for this subscriber should be 0, but look at what the model returns:

# COMMAND ----------

# DBTITLE 1,Get the Partial Hazard for the Baseline (Reference) Member
#feature values
baseline_channel = 7
baseline_method = 41
plan = 30  # arbitrarily selected plan (doesn't change the results)

# retrieve a baseline subscription
X = survival_pd[
  (subscriptions_pd['registered_via']==baseline_channel) &
  (subscriptions_pd['init_payment_method_id']==baseline_method) &
  (subscriptions_pd['init_payment_plan_days']==plan)
  ].head(1)

# get the partial hazard (factor) for the baseline member
hidden_partial_hazard = cph.predict_partial_hazard(X).iloc[0]
print(hidden_partial_hazard)

# COMMAND ----------

# MAGIC %md I don't have an explanation for why the model has a partial hazard for the reference subscription, but by applying it to the baseline, we can correct the issue, allowing us to use the formula above without additional modification:
# MAGIC 
# MAGIC **NOTE** The hidden partial hazard is a rate, not a coefficient; it has already been transformed into a factor which can be applied directly to the baseline hazard.

# COMMAND ----------

# DBTITLE 1,Adjust the Baseline Hazard for the Hidden Partial Hazard
# if baseline factor is not present, set it to 1 so that math works in next step
if np.isnan(hidden_partial_hazard): hidden_partial_hazard=1

# adjust the baseline by the baseline factor
baseline_chazard_adj= baseline_chazard * hidden_partial_hazard
baseline_chazard_adj.head(10)

# COMMAND ----------

# MAGIC %md With our baseline hazard adjusted, let's now attempt to calculate the cumulative hazard for our arbitrarily selected sample subscriber and compare our results to those returned by the model:

# COMMAND ----------

# DBTITLE 1,Calculate Hazard
#feature values
channel = 7
method = 28
plan = 30
t = 184

# calculate hazard
baseline_at_t = baseline_chazard_adj.loc[t, plan]
partial_hazard = np.exp(channel_coefficient + method_coefficient)
hazard_rate = baseline_at_t * partial_hazard

# display results
print('Cumulative Hazard Rate @ Day {1}:\t{0:.4f}'.format(hazard_rate, t))
print('   Baseline Hazard @ Day {1}:\t\t{0:.4f}'.format(baseline_at_t, t))
print('   Partial Hazard:\t\t\t{0:.4f}'.format(partial_hazard))

# COMMAND ----------

# DBTITLE 1,Use Model to Predict Hazard
#feature values 
channel = 7
method = 28
plan = 30
t = 184

# baseline member
X = survival_pd[
  (subscriptions_pd['registered_via']==channel) &
  (subscriptions_pd['init_payment_method_id']==method) &
  (subscriptions_pd['init_payment_plan_days']==plan)
  ].head(1)

# model prediction
cph.predict_cumulative_hazard(X, times=[t])

# COMMAND ----------

# MAGIC %md ###Step 3: Extract the Data Required for Survival Ratio Calculations
# MAGIC 
# MAGIC The baseline hazard formula is the most widely documented formula surrounding Cox PH models.  Still, baseline hazard is not an intuitive value for most analysts.  Instead, most analysts tend to think about survival over time in terms of the ratio of a given population that survives to a given point in time.  This value is known as the survival ratio:
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/kkbox_survivalratio.png' width=500>
# MAGIC 
# MAGIC What you should notice about this formula is that the partial hazard term is identical to the one in the hazard rate formula.  It's applied a bit differently against a baseline survival ratio which calculates the probability a reference subscriber makes it to a given point in time. As with the baseline hazard rate, the survival ratio can be extracted and must be adjusted for the hidden partial hazard:

# COMMAND ----------

# DBTITLE 1,Extract Baseline Survival Data
# retrieve baseline survival ratio 
baseline_survival = cph.baseline_survival_.copy()
baseline_survival.head(10)

# COMMAND ----------

# DBTITLE 1,Retrieve Hidden Partial Hazard (Same as Above)
#feature values
baseline_channel = 7
baseline_method = 41
plan = 30  # arbitrarily selected plan (doesn't change the results)

# retrieve a baseline subscription
X = survival_pd[
  (subscriptions_pd['registered_via']==baseline_channel) &
  (subscriptions_pd['init_payment_method_id']==baseline_method) &
  (subscriptions_pd['init_payment_plan_days']==plan)
  ].head(1)

# get the partial hazard (factor) for the baseline member
hidden_partial_hazard = cph.predict_partial_hazard(X).iloc[0]

# if baseline factor is not present, set it to 1 so that math works in next step
if np.isnan(hidden_partial_hazard): hidden_partial_hazard=1

# COMMAND ----------

# DBTITLE 1,Adjust Baseline Survival by Hidden Partial Hazard
# adjust the baseline by the baseline factor
baseline_survival_adj= np.power(baseline_survival, hidden_partial_hazard)
baseline_survival_adj.head(10)

# COMMAND ----------

# DBTITLE 1,Calculate Survival Ratio for Subscriber
#feature values
channel = 7
method = 28
plan = 30
t = 184

# retrieve feature coefficients
channel_coefficient = get_coefficient('channel_{0}'.format(channel), coefficients) 
method_coefficient = get_coefficient('method_{0}'.format(method), coefficients) 

# calculate hazard ratio
baseline_at_t = baseline_survival_adj.loc[t, plan]
partial_hazard = np.exp(channel_coefficient + method_coefficient)
survival_rate = np.power(baseline_at_t, partial_hazard)

print('Calculated:\t{0:.4f}'.format(survival_rate))

# COMMAND ----------

# DBTITLE 1,Retrieve Survival Rate for Subscriber
#feature values
channel = 7
method = 28
plan = 30
t = 184

X = survival_pd[
  (subscriptions_pd['registered_via']==channel) &
  (subscriptions_pd['init_payment_method_id']==method) &
  (subscriptions_pd['init_payment_plan_days']==plan)
  ].head(1)

# model prediction
cph.predict_survival_function(X, times=[t])

# COMMAND ----------

# MAGIC %md Again, pre-applying the partial hazard to our baseline survival ratio allows us to generate a survival ratio prediction that matches what the model returns.  Now we can focus on persisting these data in a manner that will allow us to predict survival without the model object. 

# COMMAND ----------

# MAGIC %md ###Step 4: Persist Model Data
# MAGIC 
# MAGIC All the data we need to calculate a survival ratio or hazard rate now sits in a set of pandas DataFrames.  While the Databricks platform provides JDBC integration with external relational databases for this, we'll persist our data to Delta Lake tables to make the demonstration more contained.  Given this, we'll modify the DataFrames slightly to make their conversion a bit easier:

# COMMAND ----------

# DBTITLE 1,Restructure Baseline Hazard Data
# copy the index to a dataframe column
baseline_chazard_adj['t'] = baseline_chazard_adj.index

# unpivot the dataframe to eliminate separate fields for different payment plans
baseline_chazard_unpivot = pd.melt(baseline_chazard_adj, id_vars=['t'], var_name='plan', value_name='baseline')

# display results
baseline_chazard_unpivot.head(10)

# COMMAND ----------

# DBTITLE 1,Restructure Baseline Survival Data
# copy the index to a dataframe column
baseline_survival_adj['t'] = baseline_survival_adj.index

# unpivot the dataframe to eliminate separate fields for different payment plans
baseline_survival_unpivot = pd.melt(baseline_survival_adj, id_vars=['t'], var_name='plan', value_name='baseline')

# display results
baseline_survival_unpivot.head(10)

# COMMAND ----------

# DBTITLE 1,Restructure Coefficients Data
# convert coefficients series into dataframe
coefficients = pd.DataFrame(coefficients)

# move index to a dataframe column
coefficients['feature'] = coefficients.index

# display results
coefficients.head(10)

# COMMAND ----------

# MAGIC %md Now we can persist the data:

# COMMAND ----------

# DBTITLE 1,Write the Baseline Hazard Data to Storage (& Present as SQL Table)
spark.sql('drop table if exists kkbox.cumulative_hazard')
shutil.rmtree('/dbfs/tmp/kkbox-survival/gold/cumulative_hazard', ignore_errors=True)

( # write baseline data to storage
  spark.createDataFrame(baseline_chazard_unpivot)
    .distinct()
    .write
    .format('delta')
    .mode('overwrite')
    .save('/tmp/kkbox-survival/gold/cumulative_hazard')
  )

# define table for baseline data
spark.sql('create table kkbox.cumulative_hazard using delta location \'/tmp/kkbox-survival/gold/cumulative_hazard\'')

# retrieve stored data
display(spark.sql('SELECT * FROM kkbox.cumulative_hazard ORDER BY t'))

# COMMAND ----------

# DBTITLE 1,Write the Baseline Survival Data to Storage (& Present as SQL Table)
spark.sql('drop table if exists kkbox.survival_ratio')
shutil.rmtree('/dbfs/tmp/kkbox-survival/gold/survival_ratio', ignore_errors=True)

( # write baseline data to storage
  spark.createDataFrame(baseline_survival_unpivot)
    .distinct()
    .write
    .format('delta')
    .mode('overwrite')
    .save('/tmp/kkbox-survival/gold/survival_ratio')
  )

# define table for baseline data
spark.sql('create table kkbox.survival_ratio using delta location \'/tmp/kkbox-survival/gold/survival_ratio\'')

# retrieve stored data
display(spark.sql('SELECT * FROM kkbox.survival_ratio ORDER BY t'))

# COMMAND ----------

# DBTITLE 1,Write the Coefficient Data to Storage (& Present as SQL Table)
shutil.rmtree('/dbfs/tmp/kkbox-survival/gold/coefficients', ignore_errors=True)

( # write coefficients data to storage
  spark.createDataFrame(coefficients)
    .distinct()
    .write
    .format('delta')
    .mode('overwrite')
    .save('/tmp/kkbox-survival/gold/coefficients')
  )

# define table for coefficient data
spark.sql('drop table if exists kkbox.coefficients')
spark.sql('create table kkbox.coefficients using delta location \'/tmp/kkbox-survival/gold/coefficients\'')

# retrieve stored data
display(spark.table('kkbox.coefficients'))

# COMMAND ----------

# MAGIC %md ##Step 5: Calculate Hazard Rates & Survival Ratios
# MAGIC 
# MAGIC With our data organized as a accessible  asset, we can now easily calculate hazard at any given point in time.  Here, we'll do this using a SQL statement, calculating the cumulative hazard leveraging both baseline and feature-aligned coefficients over the entire range for which we have subscriber data:

# COMMAND ----------

# DBTITLE 1,Cumulative Hazard
# MAGIC %sql
# MAGIC 
# MAGIC SELECT
# MAGIC   b.t,
# MAGIC   b.baseline,
# MAGIC   EXP(c.coefs) as partial_hazard,
# MAGIC   b.baseline * EXP(c.coefs) as hazard
# MAGIC FROM kkbox.cumulative_hazard b
# MAGIC CROSS JOIN (  
# MAGIC   SELECT
# MAGIC       COALESCE(SUM(coef),0) as coefs
# MAGIC   FROM kkbox.coefficients
# MAGIC   WHERE feature IN ('channel_7', 'method_28')
# MAGIC   ) c
# MAGIC WHERE b.plan=30

# COMMAND ----------

# MAGIC %md To see how we might implement this same logic a bit differently, we might move the SQL statement into a string that will allow us to more easily parameterize it:

# COMMAND ----------

channel = 7
method = 28
plan = 30

sql_statement = '''
  SELECT
    b.t,
    b.baseline,
    EXP(c.coefs) as partial_hazard,
    b.baseline * EXP(c.coefs) as hazard
  FROM kkbox.cumulative_hazard b
  CROSS JOIN (
    SELECT
        COALESCE(SUM(coef),0) as coefs
    FROM kkbox.coefficients
    WHERE feature IN ('channel_{0}', 'method_{1}')
    ) c
  WHERE b.plan={2}'''.format(channel, method, plan)

display(spark.sql(sql_statement))

# COMMAND ----------

# MAGIC %md And here we calculate survival ratios over time for this same subscription:

# COMMAND ----------

channel = 7
method = 28
plan = 30

sql_statement = '''
  SELECT
    b.t,
    b.baseline,
    EXP(c.coefs) as partial_hazard,
    POW(b.baseline, EXP(c.coefs)) as survival_ratio
  FROM kkbox.survival_ratio b
  CROSS JOIN (
    SELECT
        COALESCE(SUM(coef),0) as coefs
    FROM kkbox.coefficients
    WHERE feature IN ('channel_{0}', 'method_{1}')
    ) c
  WHERE b.plan={2}'''.format(channel, method, plan)

display(spark.sql(sql_statement))

# COMMAND ----------

# MAGIC %md So, is this fully operationalized?  Depending on how you intend to use the model, there are many ways these queries could be packaged to enable a full deployment.  As mentioned above, as the lifelines API evolves, we'll revisit this deployment strategy to move it into something that's more mainstream.  Still, it is our hope that for now, organizations needing to analyze subscription drop-out would be able to see how this approach could be applied.
