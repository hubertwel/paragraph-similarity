# paragraph-similarity
<!-- This is the markdown template for the final project of the Building AI course, 
created by Reaktor Innovations and University of Helsinki. 
Copy the template, paste it to your GitHub README and edit! -->

# Project Title

Semantic similarity of tweets. The final project for the Building AI course.

## Summary

This project Finds similar Twitter posts as paragraphs using Doc2Vec from Gensim API. The vectors of entire paragraphs are computed together with some word vectors (dm=0, db_words=1). The printed results include the most similar post (paragraph), the second, the third, the mean and the least similar document. Entire posts are like paragraphs. They may include more than one sentence. The datasets are taken from the French government site, which makes them available for the purpose of machine learning projects.
The returned results are optimized by selecting the best possible hyperparameters and then, using Optuna trials. At the end of the program, I do a cross-validation using the LinearRegression classifier as an estimator to show the validation and the test accuracy.

2021

## Background

Using the Twitter search engine (or other social media search engines) is not very good if somebody wants to find the most similar posts. The returned results reflect the keywords that are used and do not seem to take into account the meaning of entire posts. Therefore Gensim created new API with Doc2Vec computing vectors of entire paragraphs, not just word vectors. However, some word vectors may also be simultaneously computed like with Word2Vec. The idea is a few years old, but I wanted to use real, raw data from Twitter to see how similar results will be, not just specially prepared data for the purpose of showing good results.

## How is it used?




Describe the process of using the solution. In what kind situations is the solution needed (environment, time, etc.)? Who are the users, what kinds of needs should be taken into account?

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

What does your project _not_ solve? Which limitations and ethical considerations should be taken into account when deploying a solution like this?

## What next?

How could your project grow and become something even more? What kind of skills, what kind of assistance would you  need to move on? 


## Acknowledgments

* list here the sources of inspiration 
* do not use code, images, data etc. from others without permission
* when you have permission to use other people's materials, always mention the original creator and the open source / Creative Commons licence they've used
  <br>For example: [Sleeping Cat on Her Back by Umberto Salvagnin](https://commons.wikimedia.org/wiki/File:Sleeping_cat_on_her_back.jpg#filelinks) / [CC BY 2.0](https://creativecommons.org/licenses/by/2.0)
* etc
