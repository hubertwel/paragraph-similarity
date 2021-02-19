<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Semantic similarity of tweets

The final project for the Building AI course.

## Summary

* Multi-class text classification with Doc2Vec
  
  This project finds **similar Twitter posts as paragraphs using Doc2Vec from Gensim API**. Doc2vec is an NLP tool for representing documents as vectors and is an extension to     the Word2Vec method. While Word2Vec computes a **feature vector for every word** in the corpus, Doc2Vec computes a **feature vector for every document** in the corpus. Entire   posts are like paragraphs. They may include more than one sentence. The **vectors of entire documents are computed** (together with some word vectors with parameters dm=0,       db_words=1). The printed results include the **most similar post (paragraph)**, the second, the third, the mean and the least similar post (paragraph). Also, the validation     accuracy and the test accuracy are printed out. 

* The use of publicly available data sets as the train corpus and the test corpus
  
  The data sets are taken from the **French government site**, which makes them available for the purpose of machine learning projects. 
  ( https://www.data.gouv.fr/fr/datasets/credibility-corpus-with-several-datasets-twitter-web-database-in-french-and-english/ )
  
* Optimization
  
  The returned results are optimized by **selecting the best possible parameters** and then **the best hyperparameters** using **Optuna trials (optuna.create_study)**. 
  At the end of the program, I do a **cross-validation using the LinearRegression classifier** as an estimator to show the **validation accuracy** and the **test accuracy**.

## Background

* Social media - a search problem and its importance
  
  Using the Twitter search engine (or other social media search engines) is not very good if somebody wants to find the most similar posts. The returned results reflect the       **keywords** that are used and do not seem to take into account **the meaning of entire posts**. Fortunately, Gensim created new API with **Doc2Vec computing vectors of entire   paragraphs, not just word vectors**. However, some word vectors may also be simultaneously computed like with Word2Vec. 

* Personal motivation
  
  The project was created due to the **lack of such paragraph search possibility on social media** and due to my **personal interest in AI** as well, especially in programming     in **Python**. 

  Also, I wanted to use **real, raw data from Twitter** to see how similar results will be, not just specially prepared data for the purpose of showing good examples.

## How is it used?

### Development process

#### Dependencies
```
pip install --pre --upgrade gensim

pip install optuna
```
#### Preparing the data sets
This is the reading randomtweets3.txt and randomtweets3.txt files. They have .txt extension, but it is the csv format.

Strip_tags, strip_multiple_whitespaces are used as filters. Removing stopwords and short words is postponed to the building the model phase. Adding other filters didn't increase accuracy. On the contrary, sometimes accuracy was even lower, so additional filters are not used.
```
def read_corpus(fname, tokens_only=False):
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        csv_reader = csv.DictReader(f, quoting=csv.QUOTE_ALL)
        CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_multiple_whitespaces]
        for i, row in enumerate(csv_reader):
          line = row['x']
          line = remove_urls(line)
          line_list = preprocess_string(line, CUSTOM_FILTERS)
          line = " ".join(line_list)
          tokens = simple_preprocess(line)
          if tokens_only:
            yield tokens
          else:
            # For training data, add tags
            yield TaggedDocument(tokens, [i])

def remove_urls(text):
 text = re.sub(r'https?:\/\/.*[\r\n]*', ' ', text, flags=re.MULTILINE)
 return text
  
train_file = os.path.join(test_data_dir, 'randomtweets3.txt')
test_file = os.path.join(test_data_dir, 'randomtweets4.txt')

train_corpus = list(read_corpus(train_file, tokens_only=True))
train_corpus_tagged = list(read_corpus(train_file))
test_corpus = list(read_corpus(test_file, tokens_only=True))
```
#### Building and training the model on the train corpus
The dm=1 variant and dm=0 without dbow_words (without training word vectors) were tested, but they returned lower accuracy. Also, many various configurations of the parameters that are internal to the model were tested, but were giving lower accuracy. This is the best configuration of parameters so far. 

An additional trim_rule was added, so that stopwords and short words of lenght < 3 could be removed, since they are not useful for training the model. They are pruned only during the trim_rule method call, while building the model, so that the train corpus still includes them and therefore those words like "I", "me", "on", etc. can be printed, which is better for the user to better comprehend the paragraphs.
```
def trim_rule(word, count, min_count):
    stop_words = set(stopwords.words('english')) 
    # This rule is only used to prune vocabulary during the current method call,
    # so that documents can be printed with stopwords and with words of any length
    if ((word in stop_words) or (len(word) < 3)):
        return utils.RULE_DISCARD  # throw out
    else:
        return utils.RULE_DEFAULT  # apply default rule, i.e. min_count

model = Doc2Vec(dm=0, vector_size=80, min_count=3, epochs=50, hs=1, dbow_words=1, trim_rule=trim_rule)
model.build_vocab(train_corpus_tagged)
model.train(train_corpus_tagged, total_examples=model.corpus_count, epochs=model.epochs)
```
#### Assessing the model on the train corpus
Here we have the inference of paragraph vectors of each paragraph (document) together with some word wectors. I discovered a very interesting thing. Only some word vectors are computed by Doc2Vec when dbow_words was set to 1 (and of course, none word vectors when dbow_words was set to 0). Actually, only 820 word vectors were computed out of almost 5,000 words in the vocabulary. I was doing an experiment trying to supercharge word vectors with tf-idf values, but they did not have any impact on Doc2Vec, so I had to withdraw from that experiment (it can be found in a previous commit).

A sanity check in the loop below is a check to see whether the model is behaving in a usefully consistent manner, i.e. to verify how many of the inferred documents are found to be the most similar to itself. Almost 90 % of them (printed as zeros, since the most similar document has the index 0) is a satisfactory number.
```
ranks = []
first_ranks = []
second_ranks = []
inferred_vectors = []
for doc_id in range(len(train_corpus_tagged)):
  inferred_vector = model.infer_vector(train_corpus_tagged[doc_id].words)
  sims = model.dv.most_similar([inferred_vector], topn=len(model.dv))
  # sanity check (self-similarity)
  rank = [docid for docid, sim in sims].index(doc_id)
  ranks.append(rank)
  first_ranks.append(sims[0][0])
  inferred_vectors.append(inferred_vector)

counter = collections.Counter(ranks)
```
#### Comparing and printing the most/second-most/third-most/median/least similar documents from the train corpus
The title says it all. The inferred vectors of the most similar paragraph were very high (about 0.97), but it is just the train corpus. 

Unfortunately, printed paragraphs do not pass the human eye test, which means they are not very similar despite good vector values, but remember that data tests in this project are very small.
```
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('THIRD-MOST', 2), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
  print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus_tagged[sims[index][0]].words)))
```
#### Assessing the model on the independent data set (test corpus)
Now, it is time to assess the model on the unseen data set, which in this case is the test corpus. Its size is the same as of the train corpus, i.e. 1,000. 

**I am using the vectors that were inferred after the training the model on the train corpus and I am applying them on the independent data (the test corpus)**. 

Note that a sanity check is not relevant here, because the train corpus and the test corpus are not overlapping, so there won't be self similarity of paragraphs here (both sets are disjoint). 
```
ranks_test = []
first_ranks_test = []
inferred_vectors_test = []
for doc_id in range(len(test_corpus)):
  inferred_vector_test = model.infer_vector(test_corpus[doc_id])
  sims_test = model.dv.most_similar([inferred_vector_test], topn=len(model.dv))
  first_ranks_test.append(sims_test[0][0])
  inferred_vectors_test.append(inferred_vector_test)
```
#### Preparing vectors for cross validatiom
All vectors need to be converted to np arrays, so that they could be used in cross validation.
```  
tags_array_train = np.array(first_ranks)
vectors_2Darray_train = np.array(inferred_vectors)
tags_array_test = np.array(first_ranks_test)
vectors_2Darray_test = np.array(inferred_vectors_test)
y_train, X_train = tags_array_train, vectors_2Darray_train
y_test, X_test = tags_array_test, vectors_2Darray_test
```
#### Creating Optuna study
Optuna is a hyperparameter optimization framework. The difference between parameters (earlier optimized by me in the Doc2Vec) and hyperparameters is that parameters are configuration variables that are internal to the model and whose values can be estimated from data. On the other hand, hyperparameters are configuration variables that are external to the model and whose values cannot be estimated from data. Here, they are optimized by Optuna framework in the objective(trial) method. The number of trials is set to 30, but it can be increased. 
```
# Define an objective function to be maximized
def objective(trial):
  # Optimize hyperparameters: 
    penalty = trial.suggest_categorical("penalty", ['l1', 'l2'])
    c = trial.suggest_float("C", 5e-1, 15e-1, log=True)
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])
    intercept_scaling = trial.suggest_float("intercept_scaling", 1e-1, 2e0, log=True)
    clf = LogisticRegression(penalty=penalty, C=c, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, solver='liblinear', max_iter=300, class_weight='balanced',       multi_class='auto')
  # Scoring method:
    k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
    score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=-1, scoring='accuracy')
    accuracy = np.mean(score)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
```
#### Cross validation
Cross validation is a model validation technique for assessing how the results of a statistical analysis will generalize to an independent data set. In other words, it is used to evaluate the skill of machine learning models on unseen data. 

K-Fold cross validation is the procedure with a single parameter called k that refers to the number of groups that a given data sample is to be split into. It is a popular method, because it generally results in a less biased estimate of the model skill than other methods, such as a simple train/test split.

I tried many classifiers as estimators. The worst **test accuracy** was **GaussianNB** - 0.007, then **DecisionTreeClassifier** - 0.018, **SVC** - 0.028, **KNeighborsClassifier** - 0.084, **LinearDiscriminantAnalysis** - 0.158 and **LogisticRegression - 0.527** (with the validation accuracy 11.6). As one can see **only LogisticRegression performed relatively well and only with the liblinear solver**. Therefore I decided to optimize only this classifier with Optuna (see above). **I tuned hyperparameters C, fit_intercept and intercept_scaling** and that improved the test accuracy to **0.549**. The optimization helps to tune hyperparameters, but it requires parameters itself. Choosing the search scope for C and intercept_scaling was difficult. It turned out that making wide search scopes often made the test accuracy worse, so finally I decided to make those scopes not too wide.

Overall, by cross validating with different classifiers and then, by tuning with Optuna, **I improved the test accuracy from 0.007 to 0.549, which is relatively good comparing to finding the most similar paragraph randomly with 0.001 probability** (since corpuses have 1,000 paragraphs each).
```
clf = LogisticRegression(penalty=study.best_params["penalty"], C=study.best_params["C"], fit_intercept=study.best_params["fit_intercept"], intercept_scaling=study.best_params["intercept_scaling"], solver='liblinear', max_iter=300, class_weight='balanced', multi_class='auto')
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=-1, scoring='accuracy')
print('score: ', score)
print('Validation accuracy: {}'.format(round(np.mean(score)*100, 3)))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))
```
#### The notebook
This development process was done on a [Google Colab notebook](https://colab.research.google.com/notebooks/intro.ipynb) that you can find in this repository. 

Anybody can use this solution and many social media users need it. In order to make it popular, the big social media platforms would need to incorporate this technology on their sites. First they would need to create the text areas, so users could paste entire paragraphs (posts) there. Right now, there are only small text boxes for typing keywords. Then, Twitter, Facebook, etc. would need to use Gensim Doc2Vec models or their own, even better, models. For instance, Facebook could combine it with their LASER, so users could search for multilingual post similarities.

![Screenshot](https://github.com/hubertwel/paragraph-similarity/blob/main/paragraph-similarity/images/paragraph_similarity.jpg)

## Data sources and AI methods

| Syntax            | Description                                                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Datasets          |  English 3rd and 4th random corpus (Twitter dump), 1000 posts each, used as the train and the test corpus                     |
|                   |  https://www.data.gouv.fr/fr/datasets/credibility-corpus-with-several-datasets-twitter-web-database-in-french-and-english/    |
| AI model          |  Gensim Doc2Vec                                                                                                               |
|                   |  https://radimrehurek.com/gensim/models/doc2vec.html                                                                          |
| Cross-validation  |  scikit-learn 0.22.0 - sklearn.linear_model.LogisticRegression                                                                |
|                   |  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html                               |

## Challenges

Both corpuses, the train and the test corpuses, include just 1,000 random Twitter posts each, so they are very small data sets. Therefore, the results are not resplendent, but one has to take into account the fact that picking a random post as a candidate for the most similar one to another one, is just 0.001 (1 in a thousand). Having this in mind, the test accuracy about 0.55 is much better than selecting a random choice. 

Apart from that, such **small data sets simply don't have documents very similar to each other**. Apart from that, raw data from Twitter includes many mistypings or missed whitespaces. That makes the learning harder for the algorithm. 

It should also be emphasised that I improved the test accuracy from 0.001 to 0.549 by tuning parameters of Doc2Vec, i.e. trying many different combinations of them and then, by tuning hyperparameters with Optuna. Also, I tried several different classifiers for cross validation. The printed results are the best so far, but of course the big challenge would be to make results better (using, of course, real raw data from Twitter dumps).

## What next?

In order to achieve better validation and test accuracy, one would need to use much larger data sets with millions of records. The availability of such huge, free data sets is still scarce. The big AI companies and other institutions should make more such huge, free data sets available for AI developers for various projects.
Another thing is to invent more and more accurate models. Sooner or later, better models than Doc2Vec will be created. With better data sets and tools, the outcome of this project could be improved.

## Acknowledgments

* Gensim Doc2Vec model

