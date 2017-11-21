# Mag. Hava Shabtai Data Scientist Portfolio 
[LinkedIn profile](https://www.linkedin.com/in/hava-shabtai/)

# Understanding and Predicting Property Maintenance Fines in Detroit City

This project is based on a data challenge from the Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)). It's the fourth assignment from the course 'Applied Machine Learning in Python' of University of Michigan, taken through coursera.

The Michigan Data Science Team ([MDST](http://midas.umich.edu/mdst/)) and the Michigan Student Symposium for Interdisciplinary Statistical Sciences ([MSSISS](https://sites.lsa.umich.edu/mssiss/)) have partnered with the City of Detroit to help solve one of the most pressing problems facing Detroit - blight. [Blight violations](http://www.detroitmi.gov/How-Do-I/Report/Blight-Complaint-FAQs) are issued by the city to individuals who allow their properties to remain in a deteriorated condition. Every year, the city of Detroit issues millions of dollars in fines to residents and every year, many of these fines remain unpaid. Enforcing unpaid blight fines is a costly and tedious process, so the city wants to know: how can we increase blight ticket compliance?

The first step in answering this question is understanding when and why a resident might fail to comply with a blight ticket. This is where predictive modeling comes in. The project's goal is to predict whether a given blight ticket will be paid on time.

All data for this project has been provided to coursera through the [Detroit Open Data Portal](https://data.detroitmi.gov/). Only the data included in the coursera directory will be used for training the model for this project. We will use the data of [Detroit Blight-Violations Records](https://data.detroitmi.gov/Property-Parcels/Blight-Violations/ti6p-wcg4) to get the score of the classifier in the AUC_score.
___

The course provides you with two data files for use in training and validating your models: train.csv and test.csv. Each row in these two files corresponds to a single blight ticket, and includes information about when, why, and to whom each ticket was issued. **The target variable is compliance, which is True if the ticket was paid early, on time, or within one month of the hearing data, False if the ticket was paid after the hearing date or not at all, and Null if the violator was found not responsible.** Compliance, as well as a handful of other variables that will not be available at test-time, are only included in train.csv.

Note: All tickets where the violators were found not responsible are not considered during evaluation. They are included in the training set as an additional source of data for visualization, and to enable unsupervised and semi-supervised approaches. However, they are not included in the test set.

<br>

**File descriptions** 

    train.csv - the training set (all tickets issued 2004-2011)
    test.csv - the test set (all tickets issued 2012-2016)
    addresses.csv & latlons.csv - mapping from ticket id to addresses, and from addresses to lat/lon coordinates. 
     Note: misspelled addresses may be incorrectly geolocated.

<br>

**Data fields**

train.csv & test.csv

    ticket_id - unique identifier for tickets
    agency_name - Agency that issued the ticket
    inspector_name - Name of inspector that issued the ticket
    violator_name - Name of the person/organization that the ticket was issued to
    violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
    mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
    ticket_issued_date - Date and time the ticket was issued
    hearing_date - Date and time the violator's hearing was scheduled
    violation_code, violation_description - Type of violation
    disposition - Judgment and judgement type
    fine_amount - Violation fine amount, excluding fees
    admin_fee - $20 fee assigned to responsible judgments
state_fee - $10 fee assigned to responsible judgments
    late_fee - 10% fee assigned to responsible judgments
    discount_amount - discount applied, if any
    clean_up_cost - DPW clean-up or graffiti removal cost
    judgment_amount - Sum of all fines and fees
    grafitti_status - Flag for graffiti violations
    
train.csv only

    payment_amount - Amount paid, if any
    payment_date - Date payment was made, if it was received
    payment_status - Current payment status as of Feb 1 2017
    balance_due - Fines and fees still owed
    collection_status - Flag for payments in collections
    compliance [target variable for prediction] 
     Null = Not responsible
     0 = Responsible, non-compliant
     1 = Responsible, compliant
    compliance_detail - More information on why each ticket was marked compliant or non-compliant


___

## Evaluation

The predictions will be given as the probability that the corresponding blight ticket will be paid on time.

The evaluation metric for this project is the Area Under the ROC Curve (AUC). 

The performance of the classifier will be tested based on the AUC score computed for the classifier. The goal of this project is to get AUC_score > 0.7.
___

For this project, a function is created that trains a model to predict blight ticket compliance in Detroit using `train.csv`. Using this model, it returns a series of length 61001 with the data being the probability that each corresponding ticket from `test.csv` will be paid, and the index being the ticket_id.

Example:

    ticket_id
       284932    0.531842
       285362    0.401958
       285361    0.105928
       285338    0.018572
                 ...
       376499    0.208567
       376500    0.818759
       369851    0.018528
       Name: compliance, dtype: float32

## Structure of the function

The structure of the function will be as follow:

PART I: Build the classifier and run it on the provided training data

1. Import all the needed packages.
2. read the needed files into a dataframe, then organize and fix the dataframes from small problems, i.e. mixed types value an etc. Afterward build the feature space and the label space.
3. Appraise the problem beforehand to decide whether it is an imbalanced classification problem or not.
4. Decide which classifier to use, organize the feature space and the label space appropriately and engage the classifier.
5. Plot the ROC for the y_test that was splitted from the training data train.csv file.

PART II: Run the classifier on the provided test file and calculate the 'AUC_score' 

1. Read the test file, organize the test file and merge it with the Blight_Violations file in order to get the label feature.
2. Engage the classifier on the feature space of the test.csv file.
3. Plot the ROC for the y_test that was gathered from the test.csv file.
4. returns a series of length 61001 with the data being the probability that each corresponding ticket from test.csv will be paid, and the index being the ticket_id.

## PART I: Build the classifier and run it on the provided training data

```python
# 1. Import all the needed packeges.

# needed
%matplotlib notebook
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# from the sklearn.ensemble module, we import the GradientBoostingClassifier class for the clasifier in use
from sklearn.ensemble import GradientBoostingClassifier
# can't pass str to the model fit() method hence import preprocessing from sklearn
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from adspy_shared_utilities import plot_class_regions_for_classifier_subplot

# in order to check the performance of the classifier import roc_curve, auc
from sklearn.metrics import roc_curve, auc
```

<br>
Next, I am going to handle the train.csv file. Some of the columns are redundant for the feature space and the label space, since they don't exist in the test.csv file, so they will be deleted.  Also, as mentioned, some of the columns have mixed type, which makes the reading of file longer and problematic, so I will use 'dtype' in the 'pd.read_csv' function.
<br>
