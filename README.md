# NLP_Toxic_comments_classification
Toxic Comment Classification Project

The aim of the project is to categorize the toxic comments based on the types of toxicity. Examples of toxicity types can be toxic, severely toxic, obscene, threat, insult, identity hate. Two machine learning techniques like Logistic Regression and Decision Tree are implemented to determine the 6 types of toxic comments


Data Description:
The dataset we are using for toxic comment classification is taken from Kaggle competition which can be found at Kaggle. Dataset has a large number of comments from Wikipedia talk page edits. They have been labeled by human raters for toxic behavior. Each comment can be labeled as any of the toxicity labels. Therefore, we have multiple targets for each record and we are dealing with multi-labeled classification.


Data Preprocessing:
1. Random sampling from data: due to the huge number of instances in dataset, a random sample of data containing 10000 instances are selected for data exploration and analysis. 
2. Handling missing data: no missing value. 
3. Encoding: not required because all features are numeric. 
4. Dropping unnecessary feature: two columns including the id and the data type (either test or training) are removed. Moreover, Toxicity is the summation of the other toxic comments targets, so it is removed. 
5. Scaling: not required since the data are almost in the same range of values. 
6. Text Analysis: for the test analysis we need to follow the steps as follows, a. Lowerization: all texts are already in lower case and this step is not required, b.Tokenization: all texts have been tokenized using “nltk” package been installed, the output is an RDD format c. Stop word removal: the stop words are removed from the RDD of tokenized words using “nltk” package, d. Punctuation removal: the punctuation characters are removed from the words, e. Lemmatization is done to get the data ready for the TF-IDF, finally TF-IDF is used to convert the results words to the numbers for being used by the

