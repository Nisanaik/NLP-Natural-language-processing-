#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd


# In[2]:


get_ipython().system('pip install gensim')


# In[3]:


import gensim


# In[4]:


import os
import nltk


# In[5]:


nltk.download('punkt')


# In[6]:


from nltk import sent_tokenize
from gensim.utils import simple_preprocess

story = []
for filename in os.listdir('data'):
    
    f = open(os.path.join('data',filename))
    corpus = f.read()
    raw_sent = sent_tokenize(corpus)
    for sent in raw_sent:
        story.append(simple_preprocess(sent))


# In[7]:


story


# In[49]:


from nltk.corpus import stopwords
nltk_stop_words = set(stopwords.words('english'))

text = "This is an example sentence with some stop words."
words = text.split()
filtered_words = [word for word in words if word.lower() not in nltk_stop_words]
filtered_text = ' '.join(filtered_words)

print(filtered_text)


# In[51]:


pip install spacy


# In[54]:


import nltk
from nltk.corpus import stopwords
 
nltk.download('stopwords')
print(stopwords.words('english'))


# In[55]:


len(story)


# In[56]:


model = gensim.models.Word2Vec(
    window=10,
    min_count=2,
    #workers=4,
)


# In[57]:


model.build_vocab(story)


# In[58]:


model.train(story, total_examples=model.corpus_count, epochs=model.epochs)


# In[59]:


model.wv.most_similar('daenerys')


# In[60]:


model.wv.doesnt_match(['jon','rikon','robb','arya','sansa','bran'])


# In[61]:


model.wv.doesnt_match(['cersel','jaime','bronn','tyrion'])


# In[62]:


model.wv['jon']


# In[63]:


model.wv['king'].shape


# In[64]:


model.wv.similarity('arya','sansa')


# In[65]:


model.wv.similarity('tyrion','sansa')


# In[66]:


model.wv.get_normed_vectors()


# In[67]:


y= model.wv.index_to_key


# In[68]:


y


# In[69]:


from sklearn.decomposition import PCA


# In[70]:


pca= PCA(n_components=3)


# In[71]:


X = pca.fit_transform(model.wv.get_normed_vectors())


# In[72]:


X.shape


# In[73]:


import plotly.express as px
fig = px.scatter_3d(X[:100],x=0,y=1,z=2, color=y[:100])
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




