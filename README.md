# MSM predictor

**Cleaning Data:**

  "Unnamed: 0" column was useless therefore it was dropped.
  
  "Episode" and "Name of show" were totally the same, therefore "Name of show" was dropped.
  
  Keeping "Start_time", "End_time" and "Length" is unnecessary so "Length" was dropped.
  
  "Start_time" consists of date and time, and date part is same with "Date" columns so date part was removed and start time was converted to integer.
  
  Using "Date" column, "Year", "Month" and "Day" columns were created and "Date" column was dropped.
  
  "Year" and "Month" were converted to string to be treated as categorical features.
  
  Each "Length" unit was 15 minutes so it was divided by 4 to be in the same scale as "Start_time"(each unit presenting 1 hour 
 
  "Name of episode" was missing for 1/3 of the rows and also Top 20 frequent values of these column weren't carrying that much information. so this column was not used in the model.
  
  "Yes" and "No" were converted to 1 and 0 respectively.
  
  
**Filling NA values:**

There were 43 "Start_time" and "End_time" missing values in data.csv and also 22 missing values in test.csv.
For both of two files Mean of column was used to fill the NAs.

>TODO: Look for the same show in the other dates and use its start_time to fill NAs.

There were 83344 missing temperature values in data.csv file. Filling missing data was done using

>1- Temperature of that time if found in other rows

>2- Mean temperature of that day

**Encode Episode Column:**

There were 6687 unique Episode values. They were too much to be encoded using one-hot method, therefore their 3000(vocab_size) top frequent
values were given numerical IDs and the rest were given a specific id.
>These numerical IDs are later used in an embedding layer.

**Make dummies for categorical features:**

Categorical features such as "Season", were encoded using one-hot method and to reduce the computation "drop_first" attribute was set on True.

There were some missed categorical values in test.csv such as "12" in Month or "2018" in year, therefore their corrspanding column in one-hot encoded test dataframe was missing.
For each missing column in one-hot encoded test dataframe, a column with zero values was added to dataFrame.

**Split data:**

500000 rows of data were selected randomly as train_data and the rest 160000 were used as valid_data

>For some reasons "Episode" column was kept seperately from the main data.

>TODO: test.csv has only records of 2019, split data in a way to place most of 2019 records in valid_data for a better and more realistic validation

 
**Normalize Data:**
using mean and std of columns of train_x, valid_x and test_x were normalized. The reason mean and std of train_x is used and not the mean and std of all data 
is that I didn't want to share any information about valid_x with the model.

## Models

**A Naive model: Using Mean**

Let's predict mean for any situtation :)

> MAE of using Mean is 0.599 (for normalized Market Share_total with mean of 0 and std of 1)

**A Simple model: Using Linear Regression**

> MAE of using Mean is 0.364 (for normalized Market Share_total with mean of 0 and std of 1)

**Neural Network**

The model consists of three inner models, model1 takes "Episode" as input and after applying an embedding, flattenes the output. 
 model2 takes the rest of data as input and there is a Dense layer inside it.
 
 output of these two models is concatenated and used as input for the rest of the model.
 
> Two Dropout layers were added to handle the overfitting.
 
> MAE of using NN is 0.203 (for normalized Market Share_total with mean of 0 and std of 1)
> R2 score of NN is 0.878 (for normalized Market Share_total with mean of 0 and std of 1)

