# Session-based recommendation system: Supermarket use case

This project employs recommendation systems to offer highly personalized and contextually aware product suggestions during users' shopping sessions, without relying on user profiles or complete historical preferences. This report explores a simple, yet powerful, NLP-based approach (word2vec) to recommend the next item to a user.

Session-based recommendation systems prioritize recent interactions rather than a user's historical preferences. This is particularly advantageous for situations where users might be anonymous, not logged in, or browsing incognito. This method can be implemented even without access to historical user data and without explicit reliance on user population statistics.

The project centers around the supermarket context and follows Malik, a new customer, who has purchased eggs, corn oil, and bread. Here's how Malik's browsing history looks:


![alt text](https://github-production-user-asset-6210df.s3.amazonaws.com/31247506/263505963-5a76e360-6fe4-4fba-a936-c59996315489.jpg)

## What should we recommend to him next?

We treat Malik's recent browsing history as a "session." A session encompasses several user interactions occurring continuously within a specific timeframe, such as products bought in a single transaction. Our objective is to predict the product in Malik's session that he would be interested enough to click on. This task, known as next event prediction (NEP), involves predicting the next event (Malik clicking on a recommended product).


To tackle this challenge, we employ the word2vec algorithm. In this approach, each session is treated as a sentence, with individual items or products in the session treated as "words." The collection of user browser histories on the website, including Malik's, serves as the corpus. Word2vec learns relationships between products within the context of user browsing behavior, resulting in a set of embeddings, one for each product.

While NEP focuses on recommending future items based on a user's history, model training for recommendations utilizes historical data. To emulate real-life behavior, we assume access only to a user's first n-1 purchased items and use them to predict the nth item.

For visualization, let's refer back to Malik's historical browsing data collected during site usage. The highlighted items serve as our training set to learn product representations, which then generate recommendations. Recommendations are primarily based on the user's most recent interaction, referred to as the query item.


![alt text](https://github.com/NijatZeynalov/session-based-recommender-bravo-supermarket/assets/31247506/6cf0783a-2ae6-42e1-ae62-385bda418590)


The model is trained for 100 epochs. In practice, training word2vec for as long as resources allow, or until performance reaches a plateau, is beneficial. Overfitting isn't a major concern with word2vec.

## Dataset

The project employs an artificially generated dataset intended for educational purposes. This dataset comprises 34,973 purchase histories occurring between 01/03/2022 and 31/03/2023. It involves 3,440 customers and 3,794 distinct products. The dataset records transactions for each customer, detailing items bought in each transaction. Unlike browsing history, it lacks the order of clicked items; it only includes purchased items per transaction. As transactions are chronologically ordered, a customer's transaction history serves as a session. Rather than predicting clicks, the focus is on predicting actual purchases. Session definitions are adaptable, demanding careful result interpretation.

![](https://github.com/NijatZeynalov/session-based-recommender-bravo-supermarket/assets/31247506/768316fa-861f-4979-836c-4c5adbbc4300)

## Evaluation metrics

We can evaluate with the following metrics:

* Recall at K (Recall@K): it takes the proportion of cases where the true item appears among the top K recommendations for all test cases. A score of 1 indicates the nth item's presence in the list; otherwise, the score is 0.

* Mean Reciprocal Rank (MRR@K): it computes the average reciprocal rank of desired items, favoring higher ranks in the ordered recommendation list.

## Conclusion
For this project, I experimented  an NLP-based algorithm—word2vec— which is known for learning low-dimensional word representations that are contextual in nature. I applied it to an supermarket sales dataset containing historical purchase transactions, to learn the structure induced by both the user’s behavior and the product’s nature to recommend the next item to be purchased. 

A session-based recommendation system is essential due to its ability to address the limitations and complexities of user behavior in dynamic online environments. Traditional recommendation systems often rely heavily on historical user data and user profiles to generate personalized recommendations. However, these approaches encounter challenges when faced with scenarios where:

1. **Limited User History:** In cases where users are new or have minimal historical interactions, traditional systems struggle to provide accurate recommendations. Session-based systems excel here by focusing on the most recent interactions within a user's current session.

2. **Anonymous or Incognito Users:** Users who are not logged in or browsing anonymously do not have well-defined user profiles. Session-based systems capitalize on the current session's data, enabling real-time personalization without requiring complete user profiles.

3. **Changing Preferences:** Users' preferences can change over time due to various factors. Session-based recommendations, which emphasize recent interactions, capture the evolving interests of users more effectively.

4. **Temporal Dynamics:** User preferences can be influenced by temporal dynamics, where recent actions hold more significance than older ones. Session-based systems inherently adapt to these dynamics, as they prioritize recent interactions.

5. **Contextual Relevance:** Users' preferences often depend on the context of their current activity. Session-based systems take into account the immediate context of a user's ongoing session, leading to more contextually relevant recommendations.

6. **Cold-Start Problem:** Recommending items to new users or items that have just been introduced to the platform poses a challenge in traditional systems. Session-based methods leverage the current session's data to alleviate the cold-start problem.

7. **User Engagement:** By focusing on the current session's interactions, session-based systems can engage users in the present moment, potentially increasing the likelihood of conversions or interactions.

8. **Real-time Adaptability:** Online platforms often experience rapid changes in user behavior and preferences. Session-based systems can quickly adapt to these changes, ensuring up-to-date recommendations.

In essence, session-based recommendation systems are necessary to cater to the nuances of modern online user behavior. By focusing on current sessions and emphasizing real-time context, these systems provide timely and relevant recommendations, effectively addressing the challenges posed by evolving user preferences, anonymous users, and changing trends.
