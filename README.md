# NLP
# HOTEL REVIEW ANALYSIS - CUSTOMER SENTIMENT ORIENTATION STUDY 🙂 😐 ☹️
![](https://images.pexels.com/photos/60217/pexels-photo-60217.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500)

OBJECTIVE 💡
**The objective of this analysis is to understand the sentiment orientation of the customer relative to their hote stay. The secondary aim of this analysis is to identify topics and themes to create customer experience improvement related strategies based on the data insights.**

**Getting the data:**
To get the data within Kaggle you just need to run the below code block which will get the data ingested within the environment. In case you are using Google Colab you just need to get the data path from the lefthand side panel or you can also mount the google drive to access the data from your google drive. 

Google Colab Data Path: ('/content/filename.csv') should do the magic.

Importing pandas is key to perform dataframe related activities.

```python
from typing import Iterator

import pandas as pd
```
PyCaret is a low code Ml flow library that provides hossts of packages to solve various data related problems starting from simple regression to NL related acticities. In this notebook I will utilizing PyCAret's NLP library to perform the Hotel Review Sentiment Analysis, creating a Latent Dirichlet Allocation Model and assigning that model to new set of data for sentiment prediction. 

Options to install PyCaret:
* pip intall pycaret (basic)
* pip install pycaret [full] (entire package with all the dependencies)
* pip install pycaret-nightly (updated and full version)

PyCaret uses interactive plotting ability. In order to render interactive plots in Google Colab, run the below line of code in your colab notebook.

```python
from typing import Iterator
from pycaret.utils import enable_colab 
enable_colab()

data = pd.read_csv('/content/tripadvisor_hotel_reviews.csv')

data.head()

```



