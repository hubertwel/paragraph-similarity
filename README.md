<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Semantic similarity of tweets

The final project for the Building AI course.

## Summary

This project Finds **similar Twitter posts as paragraphs using Doc2Vec from Gensim API**. 
The vectors of entire paragraphs are computed together with some word vectors (dm=0, db_words=1). The printed results include the most similar post (paragraph), the second, the third, the mean and the least similar document. 

Entire posts are like paragraphs. They may include more than one sentence. The datasets are taken from the French government site ( https://www.data.gouv.fr/fr/datasets/credibility-corpus-with-several-datasets-twitter-web-database-in-french-and-english/ ), which makes them available for the purpose of machine learning projects.

The returned results are optimized by **selecting the best possible hyperparameters** and then, using **Optuna trials (optuna.create_study)**. 
At the end of the program, I do a **cross-validation using the LinearRegression classifier** as an estimator to show the validation and the test accuracy.

2021

## Background

Using the Twitter search engine (or other social media search engines) is not very good if somebody wants to find the most similar posts. The returned results reflect the **keywords** that are used and do not seem to take into account **the meaning of entire posts**. Therefore Gensim created new API with **Doc2Vec computing vectors of entire paragraphs, not just word vectors**. However, some word vectors may also be simultaneously computed like with Word2Vec. 

The idea of paragraph vectors is a few years old, but I wanted to use **real, raw data from Twitter** to see how similar results will be, not just specially prepared data for the purpose of showing good examples.

## How is it used?

### Development process

#### Dependencies
```
pip install --pre --upgrade gensim

pip install optuna
```
#### Preparing the datasets
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
```
Anybody can use this solution an many users need such a solution. In order to make it popular, the big social media platforms would need to incorporate this technology on their sites. First they would need to create the text areas, so users could paste entire paragraphs (posts) there. Right now, there are only small text boxes for keywords. Then, Twitter, facebook, etc. would need to use Gensim Doc2Vec models or their own, even better, models. For instance, facebook could combine it with their LASER, so users could search for multilingual post similarities.

Images will make your README look nice!
Once you upload an image to your repository, you can link link to it like this (replace the URL with file path, if you've uploaded an image to Github.)
![Cat](https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg)

If you need to resize images, you have to use an HTML tag, like this:
<img src="https://upload.wikimedia.org/wikipedia/commons/5/5e/Sleeping_cat_on_her_back.jpg" width="300">

This is how you create code examples:
```
def main():
   countries = ['Denmark', 'Finland', 'Iceland', 'Norway', 'Sweden']
   pop = [5615000, 5439000, 324000, 5080000, 9609000]   # not actually needed in this exercise...
   fishers = [1891, 2652, 3800, 11611, 1757]

   totPop = sum(pop)
   totFish = sum(fishers)

   # write your solution here

   for i in range(len(countries)):
      print("%s %.2f%%" % (countries[i], 100.0))    # current just prints 100%

main()
```
## Data sources and AI methods

| Syntax            | Description                                                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Datasets          |  English 3rd and 4th random corpus (Twitter dump), 1000 posts each, used as the train and the test corpus                     |
|                   |  https://www.data.gouv.fr/fr/datasets/credibility-corpus-with-several-datasets-twitter-web-database-in-french-and-english/    |                              | ----------------- |  ---------------------------------------------------------------------------------------------------------------------------- |
| AI model          |  Gensim Doc2Vec                                                                                                               |
|                   |  https://radimrehurek.com/gensim/models/doc2vec.html                                                                          |   
| ----------------- |  ---------------------------------------------------------------------------------------------------------------------------- |
| Cross-validation  |  scikit-learn 0.22.0 - sklearn.linear_model.LogisticRegression                                                                |
|                   |  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html                               |
|                   |                                                                                                                               |

## Challenges

Both corpuses, the training and the test corpuses, include just 1000 random Twitter posts each, so they are very small datasets. Therefore the results are not resplendent, but one has to take into account the fact that picking a random post as a candidate for the most similar one to another one, is just 0.001 (1 in a thousand). Having this in mind, the test accuracy about 0.55 is much better than selecting a random choice. 
It should also be emphasised that I improved the test accuracy from 0.001 to ca. 0.55 by tuning hypermarameters of Doc2Vec, i.e. trying many different combinations of them. Also, I tried several different classifiers for cross validation. The printed results are the best so far, but of course the big challenge would be to make results better (using, of course, real raw data from Twitter dumps).

## What next?

In order to achieve better validation and test accuracy, one would need to use much larger datasets with millions of records. The availability of such huge, free datasets is still scarce. The big AI companies and other institutions should make more such huge, free datasets available for AI developers for various projects.
Another thing is to invent more and more accurate models. Sooner or later, models better than Doc2Vec should be created.

## Acknowledgments

* Gensim Doc2Vec model

