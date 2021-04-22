# Databricks notebook source
# DBTITLE 1,Group Project - Yelp Dataset
""" 
BHURJI Prineet Kaur
M G Bindhu
BRUNO Théau
"""

# COMMAND ----------

# DBTITLE 1,Importing the Data
import pyspark.sql
from pyspark.sql.functions import *

# COMMAND ----------

# Data path

filePath_covid = "/FileStore/tables/GP_Data/parsed_covid_sample.json"

filePath_users = "/FileStore/tables/GP_Data/parsed_user_sample.json"  

filePath_reviews = "/FileStore/tables/GP_Data/parsed_review_sample.json" 

filePath_business = "/FileStore/tables/GP_Data/parsed_business_sample.json" 

filePath_checkin = "/FileStore/tables/GP_Data/parsed_checkin_sample.json" 

filePath_tip = "/FileStore/tables/GP_Data/parsed_tip_sample.json" 


# COMMAND ----------

# import covid data

covid=spark\
           .read\
           .format("JSON")\
           .option("header","true")\
           .option("inferSchema","true")\
           .load(filePath_covid)

covid.createOrReplaceTempView("covid")

# drop rows with all nas
covid = covid.na.drop("all")

# COMMAND ----------

# import users data

users=spark\
           .read\
           .format("JSON")\
           .option("header","true")\
           .option("inferSchema","true")\
           .load(filePath_users)

users.createOrReplaceTempView("users")

# drop rows with all nas
users = users.na.drop("all")

# COMMAND ----------

# import reviews data

reviews=spark\
           .read\
           .format("JSON")\
           .option("header","true")\
           .option("inferSchema","true")\
           .load(filePath_reviews)

reviews.createOrReplaceTempView("reviews")

# drop rows with all nas
reviews = reviews.na.drop("all")

# COMMAND ----------

# import business data

business=spark\
           .read\
           .format("JSON")\
           .option("inferSchema","true")\
           .load(filePath_business)

business.createOrReplaceTempView("business")


# COMMAND ----------

# import checkin data

checkin=spark\
           .read\
           .format("JSON")\
           .option("header","true")\
           .option("inferSchema","true")\
           .load(filePath_checkin)

checkin.createOrReplaceTempView("checkin")

# drop rows with all nas
checkin = checkin.na.drop("all")

# COMMAND ----------

# import tip data

tip=spark\
           .read\
           .format("JSON")\
           .option("header","true")\
           .option("inferSchema","true")\
           .load(filePath_tip)

tip.createOrReplaceTempView("tip")

# drop rows with all nas
tip = tip.na.drop("all")

# COMMAND ----------

# DBTITLE 1,Exploring the Datasets 


# COMMAND ----------

# Checking the shape of Covid Dataset
print((covid.count(), len(covid.columns)))

# COMMAND ----------

# Checking the shape of Users Dataset
print((users.count(), len(users.columns)))

# COMMAND ----------

# Checking the shape of Users Dataset
print((reviews.count(), len(reviews.columns)))

# COMMAND ----------

# Checking the shape of Users Dataset
print((business.count(), len(business.columns)))

# COMMAND ----------

# Checking the shape of Users Dataset
print((checkin.count(), len(checkin.columns)))

# COMMAND ----------

# Checking the shape of Users Dataset
print((tip.count(), len(tip.columns)))

# COMMAND ----------

# DBTITLE 1, Data Processing


# COMMAND ----------

# DBTITLE 1,a) Covid Data processing
# covid data
covid.show(3, truncate=False)

# COMMAND ----------

# Check Duplicates
covid.distinct().select(count("business_id"), countDistinct("business_id")).show()


# COMMAND ----------

# Drop duplicates
covid = covid.dropDuplicates(["business_id"])

# COMMAND ----------

# check for nas in columns
covid.select([count(when(col(c).isNull(), c)).alias(c) for c in covid.columns]).show()

# COMMAND ----------

# Last verification
covid.distinct().select(count("business_id"), countDistinct("business_id")).show()

# COMMAND ----------

covid.columns

# COMMAND ----------

# One-hot encoding TRUE as 1 and FALSE as 0

covid = covid.withColumn('Call To Action enabled', when(col('Call To Action enabled')=="TRUE",1).otherwise(0))
covid = covid.withColumn('Covid Banner', when(col('Covid Banner')=="TRUE",1).otherwise(0))
covid = covid.withColumn('Grubhub enabled', when(col('Grubhub enabled')=="TRUE",1).otherwise(0))
covid = covid.withColumn('Request a Quote Enabled', when(col('Request a Quote Enabled')=="TRUE",1).otherwise(0))
covid = covid.withColumn('Temporary Closed Until', when(col('Temporary Closed Until')=="TRUE",1).otherwise(0))
covid = covid.withColumn('Virtual Services Offered', when(col('Virtual Services Offered')=="TRUE",1).otherwise(0))
covid = covid.withColumn('delivery or takeout', when(col('delivery or takeout')=="TRUE",1).otherwise(0))
covid = covid.withColumn('highlights', when(col('highlights')=="TRUE",1).otherwise(0))


covid.show()

# COMMAND ----------

# Rename the target columns to label for the machine learning 
covid = covid.withColumnRenamed("delivery or takeout","label")


# COMMAND ----------

# Creation of the target table to merge 
# extract last 2 columns dataset

business_target = covid.select('business_id','label')
business_target.show()

# COMMAND ----------

# Creation of the covid table features - subset 
# select the other variables as features

covid_features = covid.select('business_id',
                                   'Call To Action enabled',
                                   'Covid Banner',
                                   'Grubhub enabled',
                                   'Request a Quote Enabled',
                                   'Temporary Closed Until',
                                   'Virtual Services Offered',
                                   'highlights')

# COMMAND ----------

# DBTITLE 1,b) Business Data processing
# business data
business.show(3, truncate=False)

# COMMAND ----------

# Rename the columns 

# Autre methode
# business = business.toDF(*(c.replace('.', '_') for c in yelp_business.columns))

business = business.withColumnRenamed("hours.Friday",'hours_Friday')
business = business.withColumnRenamed('hours.Monday', 'hours_Monday')
business = business.withColumnRenamed('hours.Saturday','hours_Saturday')
business = business.withColumnRenamed('hours.Sunday', 'hours_Sunday')
business = business.withColumnRenamed('hours.Thursday', 'hours_Thursday')
business = business.withColumnRenamed('hours.Tuesday','hours_Tuesday')
business = business.withColumnRenamed('hours.Wednesday','hours_Wednesday')
business = business.withColumnRenamed("attributes.AcceptsInsurance","AcceptsInsurance")
business = business.withColumnRenamed("attributes.AgesAllowed","AgesAllowed")
business = business.withColumnRenamed("attributes.Alcohol","Alcohol")
business = business.withColumnRenamed("attributes.Ambience","Ambience")
business = business.withColumnRenamed("attributes.BYOB","BYOB")
business = business.withColumnRenamed("attributes.BYOBCorkage","BYOBCorkage")
business = business.withColumnRenamed("attributes.BestNights","BestNights")
business = business.withColumnRenamed("attributes.BikeParking","BikeParking")
business = business.withColumnRenamed("attributes.BusinessAcceptsBitcoin","Bitcoin")
business = business.withColumnRenamed("attributes.BusinessAcceptsCreditCards","BusinessAcceptsCreditCards")
business = business.withColumnRenamed("attributes.BusinessParking","Parking")
business = business.withColumnRenamed("attributes.ByAppointmentOnly","AppOnly")
business = business.withColumnRenamed("attributes.Caters","Caters")
business = business.withColumnRenamed("attributes.CoatCheck","Coat_check")
business = business.withColumnRenamed("attributes.Corkage",'Cortage')
business = business.withColumnRenamed("attributes.DietaryRestrictions","Restriction")
business = business.withColumnRenamed("attributes.DogsAllowed","DogsAllowed")
business = business.withColumnRenamed("attributes.DriveThru","Drivethru")
business = business.withColumnRenamed("attributes.GoodForDancing","GoodForDancing")
business = business.withColumnRenamed("attributes.GoodForKids","GoodForKids")
business = business.withColumnRenamed("attributes.GoodForMeal","GoodForMeal")
business = business.withColumnRenamed("attributes.HairSpecializesIn","HairSpecializeIn")
business = business.withColumnRenamed("attributes.HappyHour","HappyHour")
business = business.withColumnRenamed("attributes.HasTV","Has_TV")
business = business.withColumnRenamed("attributes.Music","Music")
business = business.withColumnRenamed("attributes.NoiseLevel","NoiseLevel")
business = business.withColumnRenamed("attributes.Open24Hours","24/7")
business = business.withColumnRenamed("attributes.OutdoorSeating","OutdoorSeating")
business = business.withColumnRenamed("attributes.RestaurantsAttire","Attire")
business = business.withColumnRenamed("attributes.RestaurantsCounterService","CounterService")
business = business.withColumnRenamed("attributes.RestaurantsDelivery","Delivery")
business = business.withColumnRenamed("attributes.RestaurantsGoodForGroups","GoodForGroups")
business = business.withColumnRenamed("attributes.RestaurantsPriceRange2","PriceRange")
business = business.withColumnRenamed("attributes.RestaurantsReservations","Reservation")
business = business.withColumnRenamed("attributes.RestaurantsTableService","TableService")
business = business.withColumnRenamed("attributes.RestaurantsTakeOut","TakeOut")
business = business.withColumnRenamed("attributes.Smoking","Smoking")
business = business.withColumnRenamed("attributes.WheelchairAccessible","Wheelchair")
business = business.withColumnRenamed("attributes.WiFi","Wifi")

# COMMAND ----------

# Analyse the column Delivery 
#business.where("Delivery is null").count()
#business.where("Delivery is False").count()
#business.where("Delivery is True").count()

business.groupby("Delivery").count().show()

# COMMAND ----------

# Analyse the column TakeOut 
#business.where("TakeOut is null").count()
#business.where("TakeOut is False").count()
#business.where("TakeOut is True").count()

business.groupby("TakeOut").count().show()

# COMMAND ----------

from pyspark.sql.window import Window 
import pyspark.sql.functions as F 
from pyspark.sql.functions import row_number

# Fill NaN with False on the columns TakeOut and Delivery 
business = business.na.fill('False', subset= ['TakeOut', 'Delivery'])

# Filter the business_id which never did Delivery and TakeOut 
business = business.filter(business.Delivery != True).filter(business.TakeOut != True)
business = business.drop("Delivery", "TakeOut")
business.count()

# COMMAND ----------

drop_cols = ['address',
             'AgesAllowed',
             'BYOB',
             'BYOBCorkage', 
             'Bitcoin',
             'Coat_check',
             'DogsAllowed',  
             'Has_TV',
             'Music',
             'Delivery',
             'TakeOut',
             'PriceRange',  
             'Smoking',
             'Wheelchair',
             'business_id',
             'name',
             'postal_code',
             'hours_Friday',
             'hours_Monday',
             'hours_Saturday',
             'hours_Sunday',
             'hours_Thursday',
             'hours_Tuesday',
             'hours_Wednesday',
             'Wifi',
             'HairSpecializeIn',
             'Cortage',
             'BestNights',
             'Attire',
             'NoiseLevel',
             'Parking',
             'Ambience', 
             'city',
            'GoodForDancing',
             'GoodForKids',
             'GoodForMeal', 'GoodForGroups']



keep_cols = ['AcceptsInsurance',
             'Alcohol',
             'BikeParking',
             'BusinessAcceptsCreditCards',
             'AppOnly',
             'Caters',
             'Restriction',
             'Drivethru',
             'HappyHour',
             '24/7',
             'OutdoorSeating',
             'CounterService',
             'Reservation',
             'TableService',
             'state']

# COMMAND ----------

# Column categories splitting/ processing and count the number of categories 

business = business\
.withColumn("categories", split(col("categories"),","))\
.withColumn("num_of_categories", size(col("categories")))

business = business.drop("categories")

# COMMAND ----------

# Count if duplicates
business.distinct().select(count("business_id"), countDistinct("business_id")).show()

# COMMAND ----------

# Array of string type columns
string = [item[0] for item in business.dtypes if item[1].startswith('string')]

# Fill NaN with missing 
business = business.na.fill('Missing', subset= string)

# COMMAND ----------

# Check subset col size
len(business[keep_cols].columns)

# COMMAND ----------

# Checking the string columns in the keep_cols 
col_kept = [item[0] for item in business[keep_cols].dtypes if item[1].startswith('string')]

len(col_kept) 

# COMMAND ----------

# One-hot encoding using a function 
import itertools
import pyspark.sql.functions as F

def spark_get_dummies(df):
    
    categories = []
    for i, values in enumerate(df.columns):
        categories.append(df.select(values).distinct().rdd.flatMap(lambda x: x).collect())
        
    expressions = []
    for i, values in enumerate(df.columns):
        expressions.append([F.when(F.col(values) == i, 1).otherwise(0).alias(str(values) + "_" + str(i)) for i in categories[i]])
    
    expressions_flat = list(itertools.chain.from_iterable(expressions))
    
    df_final = df.select(*expressions_flat)
    
    return df_final

# COMMAND ----------

# Apply get_dummies / One Hot encoding 
business_dummies = spark_get_dummies(business[col_kept])
business_dummies.count()

# COMMAND ----------

# Subset of the num of categories 
business_categories = business.select("num_of_categories")
business_categories.count()

# COMMAND ----------

# Create a subset for the merging 
business_id = business.select("business_id")
business_id.count()

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql.window import Window


# No common to merge so we create an adding row_index to join on 
business_dummies = business_dummies.withColumn('row_index', row_number().over(Window.orderBy(monotonically_increasing_id())))
business_categories = business_categories.withColumn('row_index', row_number().over(Window.orderBy(monotonically_increasing_id())))
business_id = business_id.withColumn('row_index', row_number().over(Window.orderBy(monotonically_increasing_id())))

# Merging 
business_features = business_id.join(business_categories, on=["row_index"])
business_features = business_features.join(business_dummies, on=["row_index"]).drop("row_index")
business_features.show()

# COMMAND ----------

## AUTRE TECHNIQUE POUR MERGE 

#from pyspark.sql.functions import col
#def addColumnIndex(df): 
  # Create new column names
#  oldColumns = df.schema.names
#  newColumns = oldColumns + ["columnindex"]

  # Add Column index
#  df_indexed = df.rdd.zipWithIndex().map(lambda (row, columnindex): \
#                                        row + (columnindex,)).toDF()

  #Rename all the columns
#  new_df = reduce(lambda data, idx: data.withColumnRenamed(oldColumns[idx], 
#                  newColumns[idx]), xrange(len(oldColumns)), df_indexed)   
#  return new_df

# Add index now...
#df1WithIndex = addColumnIndex(df1)
#df2WithIndex = addColumnIndex(df2)

#Now time to join ...
#newone = df1WithIndex.join(df2WithIndex, col("columnindex"),
#                           'inner').drop("columnindex")

# COMMAND ----------

# Final shape
len(business_features.columns)

# COMMAND ----------

# DBTITLE 1,c) Tip - Data Processing
tip.show(3)

# COMMAND ----------

# Check NAN
tip.select([count(when(col(c).isNull(), c)).alias(c) for c in tip.columns]).show()

# COMMAND ----------

# Check duplicate
tip.select(count("business_id"), countDistinct("business_id")).show()


# COMMAND ----------

# Convert the datetime type to a numerical value 
# Time from the tip to now 
date_diff = spark.sql("SELECT business_id, DATEDIFF( NOW(), date) from tip")
date_diff = date_diff.withColumnRenamed("datediff(CAST(now() AS DATE), CAST(date AS DATE))","date_diff")
date_diff.show()


# COMMAND ----------

# Feature engineering 
# For each business create min max count and tip average time of tip 

expr = [min(col("date_diff")),max(col("date_diff")), count(col("date_diff")), mean(col("date_diff"))]
date_diff = date_diff.groupBy("business_id").agg(*expr)
date_diff = date_diff.withColumnRenamed("min(date_diff)","min_date_diff")
date_diff = date_diff.withColumnRenamed("max(date_diff)","max_date_diff")
date_diff = date_diff.withColumnRenamed("count(date_diff)","count_tip")
date_diff = date_diff.withColumnRenamed("avg(date_diff)","avg_date_diff")
date_diff.show()

# COMMAND ----------

# Second way to do the feature engineering with sql - more simple 

date_diff.createOrReplaceTempView("date_diff")

tip_final = spark.sql("SELECT  business_id, min_date_diff, max_date_diff, count_tip, avg_date_diff, max_date_diff - min_date_diff as diff_date_diff FROM date_diff")

tip_final.show()

# COMMAND ----------

# DBTITLE 1,d) Review - Data Processing
reviews.show(3)

# COMMAND ----------

# Check NAN
reviews.select([count(when(col(c).isNull(), c)).alias(c) for c in reviews.columns]).show()

# COMMAND ----------

# Convert Date to good format
reviews = reviews.withColumn("date",to_timestamp(col("date"),"yyyy-MM-dd HH:mm:ss"))

# COMMAND ----------

#Drop columns that are not needed
reviews = reviews.drop("review_id", "user_id", "text")
reviews.columns

# COMMAND ----------

# Review aggregation per business id sum(cool), sum(funny), sum(useful), avg(stars)

reviews = reviews.groupBy("business_id").agg(sum(col("cool")).alias("sum_cool"), sum(col("funny")).alias("sum_funny"), sum(col("useful")).alias("sum_useful"), avg(col("stars")).alias("avg_stars"), max(col("date")).alias("most_recent_date"))
reviews.show(5)


# COMMAND ----------

# Create recency columns using most_recent_date col with spark 
reviews = reviews.withColumn("now", to_timestamp(lit("15-04-2021"), "dd-MM-yyyy"))\
.select(col("business_id"), col("sum_cool"), col("sum_funny"), col("sum_useful"), col("avg_stars"), datediff(col("now"), col("most_recent_date")).alias("review_recency"))


# COMMAND ----------

# rounding the average stars

reviews = reviews.withColumn("avg_stars", round("avg_stars",3))
reviews.show(5)

# COMMAND ----------

# DBTITLE 1,e) Checkin Data processing
# checkin data

checkin=spark\
           .read\
           .format("JSON")\
           .option("header","true")\
           .option("inferSchema","true")\
           .load(filePath_checkin)

checkin.createOrReplaceTempView("checkin")

# drop rows with all nas

checkin = checkin.na.drop("all")

checkin.show()

# COMMAND ----------

checkin.show(3)

# COMMAND ----------

# Access checkin with sql
checkin.createOrReplaceTempView("checkin")

# COMMAND ----------

# Clean date to extract the year 
checkin = checkin.select('business_id', regexp_replace(col("date"), " ", ""))
checkin = checkin.withColumnRenamed("regexp_replace(date,  , , 1)",'date')
checkin.show()

# COMMAND ----------

# Create year and month col 
checkin = checkin.withColumn('year', concat(checkin.date.substr(0, 4)))
checkin = checkin.withColumn('month', concat(checkin.date.substr(6, 2)))

checkin.show()

# COMMAND ----------

# Convert the col year and month in integer 
from pyspark.sql.types import IntegerType
checkin = checkin.withColumn("year", checkin["year"].cast(IntegerType()))
checkin = checkin.withColumn("month", checkin["month"].cast(IntegerType()))

checkin.show()

# COMMAND ----------

checkin_2 = checkin

checkin_2.createOrReplaceTempView("checkin_2")

checkin_2.show()

# COMMAND ----------

# STR_TO_DATE(date,'%Y-%m-%d %H:%i:%s') 
# Feature engineering using SQL - create min max count 
checkin_features = spark.sql("SELECT business_id, count(business_id) as count_checkin, min(year) as min_year, max(year) as max_year, max(year) - min(year) as checkin_diff FROM checkin_2 GROUP BY business_id")
checkin_features.show()

# COMMAND ----------

# DBTITLE 1,Merging the Datasets
#Did merging for checkin_features, reviews_cleansed, tip_final, covid_features, business_target

yelp = checkin_features.join(reviews,['business_id'],"inner")
yelp = yelp.join(tip_final,['business_id'],"inner")
yelp = yelp.join(covid_features,['business_id'],"inner")
yelp = yelp.join(business_features,['business_id'],"inner")

yelp = yelp.join(business_target,['business_id'],"inner")
yelp.show()

# COMMAND ----------

# DBTITLE 1,Building the Model
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

train, test = yelp.randomSplit([0.7, 0.3], seed=974)

# COMMAND ----------

# Security to not have to rerun the merging
yelp974 = yelp

# COMMAND ----------

# Shape of the Database 
len(yelp974.columns[1:-1])

# COMMAND ----------

# Vector Assembler of the features 

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


assembler = VectorAssembler(
    inputCols= yelp974.columns[1:-1],
    outputCol="features")

output = assembler.transform(yelp974)
db = output.select("business_id", "features", "label")
db.show(truncate=False)
db.head(2)

# COMMAND ----------

# DBTITLE 1,Feature Selection using ChiqSelector 
# From 83 reducing to 50 features

from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors


selector = ChiSqSelector(numTopFeatures=50, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="label")

result = selector.fit(db).transform(db)

print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
result.show()

# COMMAND ----------

# Database formated for the Machine Learning
db_selected = result.select("business_id", "selectedFeatures", "label")
db_selected = db_selected.withColumnRenamed("selectedFeatures","features")
db_selected.show()

# COMMAND ----------

# DBTITLE 1,Decision Tree
#Modelization DECISION TREE 

from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml import Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(db_selected)
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(db_selected)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = db_selected.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

treeModel = model.stages[2]
# summary only
print(treeModel)

# COMMAND ----------

#AUC
from pyspark.mllib.evaluation import BinaryClassificationMetrics

out = model.transform(testData)\
  .select("prediction", "indexedLabel")\
  .rdd.map(lambda x: (float(x[0]), float(x[1])))

metrics = BinaryClassificationMetrics(out)

print(metrics.areaUnderPR)
print(metrics.areaUnderROC)

# COMMAND ----------

# DBTITLE 1,Random Forest
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml import Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(db_selected)
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(db_selected)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = db_selected.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
dt = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

treeModel = model.stages[2]
# summary only
print(treeModel)

# COMMAND ----------

#AUC
from pyspark.mllib.evaluation import BinaryClassificationMetrics

out = model.transform(testData)\
  .select("prediction", "indexedLabel")\
  .rdd.map(lambda x: (float(x[0]), float(x[1])))

metrics = BinaryClassificationMetrics(out)

print(metrics.areaUnderPR)
print(metrics.areaUnderROC)

# COMMAND ----------

# DBTITLE 1,Gradient Boosting
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml import Pipeline
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(db_selected)
# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(db_selected)

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = db_selected.randomSplit([0.7, 0.3])

# Train a DecisionTree model.
dt = GBTClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", maxIter=10)

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g " % (1.0 - accuracy))

treeModel = model.stages[2]
# summary only
print(treeModel)

# COMMAND ----------

#AUC
from pyspark.mllib.evaluation import BinaryClassificationMetrics

out = model.transform(testData)\
  .select("prediction", "indexedLabel")\
  .rdd.map(lambda x: (float(x[0]), float(x[1])))

metrics = BinaryClassificationMetrics(out)

print(metrics.areaUnderPR)
print(metrics.areaUnderROC)

# COMMAND ----------

# DBTITLE 1,Business Insights
# Number of businnes per label
db_selected.createOrReplaceTempView("db_selected")
#spark.sql("SELECT COUNT(label) FROM db_selected WHERE label is True").show()
#db_selected.filter(db_selected.label == True).count()

db_selected.groupby("label").count()

# COMMAND ----------

#Feature importance of our best model Gradiant Boosting 
from pyspark.ml.classification import GBTClassifier

dt = GBTClassifier()
dtModel = dt.fit(db)
print(dtModel.featureImportances)

# COMMAND ----------

# TABLE OF FEATURE IMPORTANCE in %

from  pyspark.mllib.linalg import SparseVector  # code works the same
#from pyspark.ml.linalg import SparseVector     # code works the same

import pandas as pd

a = dtModel.featureImportances  # note the index starts at 0
df = pd.DataFrame(a.toArray()).reset_index()
df = df.rename(columns={0:"Feature_importance", "index":"Feature_number"})
df['Column_name'] = yelp974.columns[1:-1]
df = df.sort_values(by=['Feature_importance'], ascending=False)
df.Feature_importance = (df.Feature_importance*100).round(2)
print(df.head(15))
#     0

# COMMAND ----------

# Feature importance of the 5 first features 
df.Feature_importance.head(5).sum()

# COMMAND ----------

# Feature importance of the 5 first features 
df.Feature_importance.head(10).sum()

# COMMAND ----------

# Business per state 
groupbystate = business_target.join(business["business_id","state","review_count"], on=["business_id"])
business_per_state = groupbystate.groupby("state").count()
business_per_state = business_per_state.withColumnRenamed("count","all")
business_per_state.show()

# COMMAND ----------

# New delivery/Takeout since the covid 
new_delivery_business_per_state = groupbystate.where("label == True").groupby("state").count()
new_delivery_business_per_state = new_delivery_business_per_state.withColumnRenamed("count","new")
new_delivery_business_per_state.show()

# COMMAND ----------

# Ranking delivery per state
percentage_new_per_state = new_delivery_business_per_state.join(business_per_state, on=["state"])
percentage_new_per_state.show()

# COMMAND ----------

# Percentage of new delivery/TakeOut comparing to the total number of business per state
percentage_new_per_state.createOrReplaceTempView("percentage_new_per_state")
percentage = spark.sql("SELECT *, (new/all)*100 as percentage FROM percentage_new_per_state")
percentage.show()

# COMMAND ----------

# Ranking with the best improvement 
percentage.sort(desc("percentage")).show()

# COMMAND ----------

# Number of review per state
review_per_state = groupbystate.groupby("state").sum("review_count").sort(desc("sum(review_count)"))
review_per_state.show()

# COMMAND ----------

# DBTITLE 1,Cross Validation 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

rfParams = ParamGridBuilder()\
  .addGrid(dt.featureSubsetStrategy, ['auto', 'sqrt'])\
  .build()

rfCv = CrossValidator()\
  .setEstimator(pipeline)\
  .setEstimatorParamMaps(rfParams)\
  .setEvaluator(BinaryClassificationEvaluator())\
  .setNumFolds(10)

rfrModel = rfCv.fit(trainingData)

# COMMAND ----------

#Best feature subset
rfrModel.bestModel.stages[-1]._java_obj.parent().getFeatureSubsetStrategy()

# COMMAND ----------

#Get feature importances
ExtractFeatureImp(rfrModel.bestModel.stages[-1].featureImportances, rtrain, "features").head(10)

# COMMAND ----------

#AUC
from pyspark.mllib.evaluation import BinaryClassificationMetrics

out = model.transform(testData)\
  .select("prediction", "indexedLabel")\
  .rdd.map(lambda x: (float(x[0]), float(x[1])))

metrics = BinaryClassificationMetrics(out)

print(metrics.areaUnderPR)
print(metrics.areaUnderROC)
