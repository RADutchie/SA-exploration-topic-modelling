import pandas as pd
import numpy as np

from gensim import models, corpora
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
from collections import defaultdict

def get_relevant_words(vis,lam=0.3,topn=20):
    """Function to extract the relevent words from each modelled topic

    Parameters
    ---------
        vis : PreparedData created using `pyLDAvis.gensim.prepare()`
        lam : int; lambda value for term relevence `(term w | topic t) = λ * p(w | t) + (1 - λ) * p(w | t)/p(w); see Sievert & Shirley (2014)`
        topn : The number number of relevent terms to return
    
    Returns
    ---------
        relevent_words : pandas.DataFrame containing the Topic, topic_id, n Relevant_words as list and Frequency of tokens in each topic
    """ 
    
    a = vis.topic_info
    a['relevance'] = a['logprob']*lam+(1-lam)*a['loglift'] # this calculates the relevance value as in plLDAvis
    a = a.loc[:,['Category','Term','relevance']].groupby(['Category'])\
    .apply(lambda x: x.sort_values(by='relevance',ascending=False).head(topn))
    a = a.loc[:,'Term'].reset_index().loc[:,['Category','Term']]
    a = a[a['Category']!='Default']
    topics = a.Category.tolist()
    terms = a.Term.tolist()
    d = defaultdict(list)
    for top, term in zip(topics,terms):
        d[top].append(term)
    relevent_words = pd.DataFrame([d]).T.reset_index()
    relevent_words.columns = ['Topic','Relevant_words']

    token_percent = vis.topic_coordinates.sort_values(by='topics').loc[:,['topics','Freq']].reset_index().rename(columns={'topic':'topic_id'})
    return relevent_words.join(token_percent,how='left').drop('topics',axis=1)



def main_topic_doc(ldamodel, corpus=corpus):
    """Function to extract the dominant topic, its percent contribution and 10 top keywords for each document in a corpus

    Parameters
    ---------
        ldamodel : `gensim.models.ldamodel.LdaModel` An already trained Gensim LdaModel.
        corpus : array-like list of bag of word docs in tuple form or scipy CSC matrix. The corpus in bag of word form, the same docs used to train the model.
    
    Returns
    ---------
        document_topics : pandas.DataFrame containing the Dominant_Topic, Percent_Contrib and Topic_keywords for each document in the corpus
    """ 
    
    doc_topics = pd.DataFrame()

    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)

        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = "' ".join([word for word, prop in wp])
                doc_topics = doc_topics.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    doc_topics.columns = ['Dominant_Topic', 'Percent_Contrib', 'Topic_keywords']
    return doc_topics


if __name__ == "__main__":
    get_relevant_words()
    main_topic_doc()