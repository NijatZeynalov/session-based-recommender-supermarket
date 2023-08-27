# Session-based recommendation system - Supermarket use case

This project harnesses the power of recommendation systems to provide highly personalized and context-aware product suggestions based on users' shopping sessions without the existence of user profiles or their entire historical preferences. This report explores a simple, yet powerful, NLP-based approach (word2vec) to recommend the next item to a user.

Session-based recommendation systems rely heavily on the user’s most recent interactions, rather than on the user’s historical preferences. This is especially advantageous because a user could appear anonymously—that is, a user may not be logged in or may be browsing incognito. The method can be implemented even in the absence of historical user data, and doesn’t explicitly rely on user population statistics.

This project demonstrates Bravo supermarket website. Malik, a new customer, has been bought eggs, corn oil, and bread. Her browsing history looks like this:


![alt text](https://github-production-user-asset-6210df.s3.amazonaws.com/31247506/263505963-5a76e360-6fe4-4fba-a936-c59996315489.jpg)

## What should we recommend to him next?

We’ll consider Malik’s recent browsing history as a “session.” Formally, a session is composed of multiple user interactions that happen together in a continuous period of time—for instance, products purchased in a single transaction.

Our goal is to predict the product within Malik’s session that she will like enough to click on. __This task is called next event prediction (NEP): given a series of events (Malik’s browsing history), we want to predict the next event (Malik clicking on a product we recommend to him).__

In order to solve this problem we will use word2vec algorithm, where we will treat each session as a sentence, with each item or product in the session representing a “word.” A website’s collection of user browser histories (including Malik’s) will act as the corpus. Word2vec will crunch over the entire corpus, learning relationships between products in the context of user browsing behavior. The result will be a collection of embeddings: one for each product. 

In NEP, we consider a user’s history to recommend items for the future—but, when training models for recommendation, all the data is historical. In order to mimic “real life” behavior, we’ll pretend that we only have access to the user’s first n-1 purchased items, and use those to try to predict the nth item purchased.

To visualize this, let’s go back to Malik’s historical browsing information, collected while he was using our site. We’ll use the highlighted items as our training set to learn product representations, which will be used to generate recommendations. Recommendations are typically based on the most recent interaction by the user, called the query item. 


![alt text](https://github.com/NijatZeynalov/session-based-recommender-bravo-supermarket/assets/31247506/e3f86be1-f146-4a51-8e54-977370a7003a)


Model is trained for 100 epochs. In practice it is almost always beneficial to train word2vec for as many epochs as resources allow, or until the downstream task has reached a performance plateau—in which case, additional training does not yield an increase in the downstream metric. We shouldn’t be worried about overfitting for word2vec.

## Dataset

For this project, Data is artificially generated and can only be used for educational purposes. This dataset represents contains 34,973 purchase histories between 01/03/2022 and 31/03/2023 for 3,440 customers and 3,794 unique products. These purchase histories record transactions for each customer and detail the items that were purchased in each transaction. This is a bit different from a browsing history, as it does not contain the order of items clicked while perusing the website; it only includes the items that were eventually purchased in each transaction. However, the transactions are ordered in time, so we can treat a customer’s full transaction history as a session. Instead of predicting recommendations for what a customer might click on next, we’ll be predicting recommendations for what that customer might actually buy next. Session definitions are flexible, and care must be taken in order to properly interpret the results.

![](https://github.com/NijatZeynalov/session-based-recommender-bravo-supermarket/assets/31247506/768316fa-861f-4979-836c-4c5adbbc4300)

## Evaluation metrics

We can evaluate with the following metrics:

* Recall at K (Recall@K) defined as the proportion of cases in which the ground truth item is among the top K recommendations for all test cases (that is, a test example is assigned a score of 1 if the nth item appears in the list, and 0 otherwise.)

* Mean Reciprocal Rank (MRR@K), takes average on the reciprocal ranks of users’ desired items. This metric measures and favors higher ranks in the ordered list of recommendation results

## Conclusion
For this project, I experimented  an NLP-based algorithm—word2vec— which is known for learning low-dimensional word representations that are contextual in nature. I applied it to an supermarket sales dataset containing historical purchase transactions, to learn the structure induced by both the user’s behavior and the product’s nature to recommend the next item to be purchased. 
