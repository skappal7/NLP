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
![](https://github.com/skappal7/NLP/blob/main/Image/1%20Table.PNG?auto=compress&cs=tinysrgb&dpr=1&w=500)

```python
from typing import Iterator
data.shape
```
(20491, 2)

 Once the setup is succesfully executed it prints the information grid with the following information: 🛠️

**session_id :** A pseduo-random number distributed as a seed in all functions for later reproducibility. If no session_id is passed, a random number is automatically generated that is distributed to all functions. In this experiment session_id is set as 999 for later reproducibility.

**Number of Documents :** Number of documents (or samples in dataset if dataframe is passed).

**Vocab Size :** Size of vocabulary in the corpus after applying all text pre-processing such as removal of stopwords, bigram/trigram extraction, lemmatization etc.
Notice that all text pre-processing steps are performed automatically when you execute setup().
```python
from typing import Iterator
from pycaret.nlp import *
nlp_sent = setup(data = data, target = 'Review', session_id = 999,log_experiment = True, experiment_name = 'HotRev1')
```
# Let's Perform Topic Modeling 🎯

**What is Topic Model?** 
In machine learning and natural language processing, a topic model is a type of statistical model for discovering the abstract "topics" that occur in a collection of documents. Topic modeling is a frequently used text-mining tool for discovery of hidden semantic structures in a text body. Intuitively, given that a document is about a particular topic, one would expect particular words to appear in the document more or less frequently. In a hotel review dataset checkin, checkout, night stay etc. words will appear mostly relative to various customer experience intensities.

```python
from typing import Iterator
lda = create_model('lda')
print(lda2)
```
LdaModel(num_terms=32301, num_topics=6, decay=0.5, chunksize=100)

```python
from typing import Iterator
lda_results = assign_model(lda2)
lda_results.head()
```
![](https://github.com/skappal7/NLP/blob/main/Image/1%20Table.PNG?auto=compress&cs=tinysrgb&dpr=1&w=500)


