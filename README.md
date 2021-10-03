# HOTEL REVIEW ANALYSIS - CUSTOMER SENTIMENT ORIENTATION STUDY 🙂 😐 ☹️
![](https://images.pexels.com/photos/60217/pexels-photo-60217.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500)

OBJECTIVE 💡

**The objective of this analysis is to understand the sentiment orientation of the customer relative to their hotel stay. The secondary aim of this analysis is to identify topics and themes to create customer experience improvement related strategies based on the data insights.**

**Getting the data:**
To get the data within Kaggle you just need to run the below code block which will get the data ingested within the environment. In case you are using Google Colab you just need to get the data path from the lefthand side panel or you can also mount the google drive to access the data from your google drive. 

Google Colab Data Path: ('/content/filename.csv') should do the magic.

Importing pandas is key to perform dataframe related activities.

```python
from typing import Iterator

import pandas as pd
```
PyCaret is a low code Ml flow library that provides hossts of packages to solve various data related problems starting from simple regression to NLP related activities. In this notebook I will utilizing PyCAret's NLP library to perform the Hotel Review Sentiment Analysis, creating a Latent Dirichlet Allocation Model and assigning that model to new set of data for sentiment prediction. 

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
![](https://github.com/skappal7/NLP/blob/main/Image/1%20Table%202.PNG?auto=compress&cs=tinysrgb&dpr=1&w=500)

```python
from typing import Iterator
plot_model()
```
![](https://github.com/skappal7/NLP/blob/main/Image/1%20top%20100%20Words.png?auto=compress&cs=tinysrgb&dpr=1&w=500)

**Top 100 Biagrams**
```python
from typing import Iterator
plot_model(plot = 'bigram')
```
![](https://github.com/skappal7/NLP/blob/main/Image/2%20top%20100%20Bigrams.PNG?auto=compress&cs=tinysrgb&dpr=1&w=500)

**Frequency Distribution of Topic 5**
```python
from typing import Iterator
plot_model(lda2, plot = 'frequency', topic_num = 'Topic 5')
```
![](https://github.com/skappal7/NLP/blob/main/Image/3%20top%20100%20Post%20Remmoving%20Stop%20Words.PNG?auto=compress&cs=tinysrgb&dpr=1&w=500)

**Topic Distribution** 

```python
from typing import Iterator
plot_model(lda2, plot = 'topic_distribution')
```
![](https://github.com/skappal7/NLP/blob/main/Image/4%20Topic%20Distribution.PNG?auto=compress&cs=tinysrgb&dpr=1&w=500)

**Model Evaluation** 
```python
from typing import Iterator
evaluate_model(lda2)
```
![](https://github.com/skappal7/NLP/blob/main/Image/5%20Model%20Evaluation.PNG?auto=compress&cs=tinysrgb&dpr=1&w=500)

# Intrinsic Model Evaluation Method Using Coherence Value

**What is Intrinsic Evaluation Method?**

Intrinsic evaluation methods assess how well the word embeddings inherently capture the semantic or syntactic relationships between the words. Where Semantics refers to the meaning of words, whereas syntax refers to the grammar. You could also evaluate the embeddings on syntactic analogies, such as plurals, tenses and comparatives.
N
Hence, using the tune_model() we will create a topic coherence score by iterating on a pre-defined grid with different number of topics and create a model for each parameter.Topic coherence is then evaluated for different models and are visually presented in a graph that has the  Coherence Score on y-axis as a function of # Topics on x-axis. You can view the results below:

**Note: This part of the process took the longest around 4+ hours to create the semantic and syntactic relationships in between the topics.**

```python
from typing import Iterator
tuned_unsupervised = tune_model(model = 'lda', multi_core = True)
```
![](https://github.com/skappal7/NLP/blob/main/Image/6%20Topic%20Coherence.PNG?auto=compress&cs=tinysrgb&dpr=1&w=500)

```python
from typing import Iterator
evaluate_model(tuned_unsupervised )
```
![](https://github.com/skappal7/NLP/blob/main/Image/7%20Evaluate%20Model%20tuned%20LDA.PNG?auto=compress&cs=tinysrgb&dpr=1&w=500)

```python
from typing import Iterator
print(tuned_unsupervised)
```
LdaModel(num_terms=32301, num_topics=200, decay=0.5, chunksize=100)
```python
from typing import Iterator
plot_model(tuned_unsupervised, plot = 'topic_distribution')
```
![](https://github.com/skappal7/NLP/blob/main/Image/8%20Topic%20Distribution%20tuned%20LDA.PNG?auto=compress&cs=tinysrgb&dpr=1&w=500)

```python
from typing import Iterator
plot_model(tuned_unsupervised, plot = 'frequency', topic_num = 'Topic 70')
```
![](https://github.com/skappal7/NLP/blob/main/Image/9%20Topic%2070%20Top%20100%20Post%20Stop%20Words%20Removal.PNG?auto=compress&cs=tinysrgb&dpr=1&w=500)

```python
from typing import Iterator
save_model(tuned_unsupervised,'Final Tuned LDA Model 07072021')
```
Model Succesfully Saved
(<gensim.models.ldamulticore.LdaMulticore at 0x7f95e70d8f10>,
 'Final Tuned LDA Model 07072021.pkl')
 
```python
from typing import Iterator
saved_lda = load_model('Final Tuned LDA Model 07072021')
```
Model Sucessfully Loaded

```python
from typing import Iterator
print(saved_lda)
```
LdaModel(num_terms=32301, num_topics=200, decay=0.5, chunksize=100)


That's about it, this is how simple it is to create an end to end NLP model using PyCaret 😀😀😀, Please star this repo if you liked the content

 

