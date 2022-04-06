# Capstone Food Recommender System

### Problem Statement

It is very common that we hang out with families, friends, and colleagues when comes to lunch or dinner time. There are an overwhelming number of restaurant choices in Singapore Central region. In fact, Dr. Brian Wansink from Cornell University([*source*](https://news.cornell.edu/stories/2006/12/mindless-autopilot-drives-people-underestimate-food-decisions)) claims that we make about 200 decisions about food each day.

In the past, users lack objective means of evaluating services, they typically depend upon subjective evaluations from family or friends to obtained suggestions of restaurants. Although this method is straightforward and user-friendly, it has the limitations. The recommendations from family or friends are limited to the places they have visited before. Thus, the user is not able to gain information about places that never visited by their family or friends. Besides that, there is a chance of users not liking the place recommended by their family or friends.

The aim is to create a Hybrid Recommendation System, content-collaborative model, in which user will input a restaurant name and the system will recommend us other restaurants which are more likely to be interesting to the user.

Lastly, to solve the user cold-start problem or to provide recommendations for these new users, a Location-Based Recommendation System using the K-Means Clustering Algorithm is build which takes into account a user's location when he is using the app to recommend the top 10 nearby restaurants based on location proximity.

### Background

In general, recommendation models can be divided into two categories:

- Content-Based Filtering - Model recommends based on similarity of the items and/or users using their description/metadata/profile. 
- Collaborative Filtering - Users who have similar preferences in the past are likely to have similar preferences in the future. 

The choice between the two models is largely based on the data availability. Collaborative filtering model is usually adopted and effective when sufficient ratings/feedbacks have been recorded for a group of users and items. However, if there is a lack of ratings, content based model can be used provided that the metadata of the users and items are available. This is also the common approach to address the cold-start issues, where there are insufficient historical collaborative interactions available to model new users and/or items. In fact, the best solution is a combination of both of these. This is known as a hybrid system. By combining the two systems, you can get the best of both worlds.

LightFM is a Python implementation of a hybrid recommendation algorithms for both implicit and explicit feedbacks. It is a hybrid content-collaborative model which represents users and items as linear combinations of their content features’ latent factors. The model learns embeddings or latent representations of the users and items in such a way that it encodes user preferences over items. These representations produce scores for every item for a given user; items scored highly are more likely to be interesting to the user. The user and item embeddings are estimated for every feature, and these features are then added together to be the final representations for users and items. ([*source*]( https://github.com/microsoft/recommenders/blob/main/examples/02_model_hybrid/lightfm_deep_dive.ipynb))

In addition, new users recommender platforms suffer from cold start, both in recommendation and search. To provide recommendations for these new users, many content-based methods and heuristic methods have been deployed, e.g., recommending popular items or geographically near items. 

### Contents

- Part 1 Data Collection
- Part 2 Data Cleaning & Exploratory Data Analysis(EDA)
- Part 3 Location-Based Recommender
- Part 4 Hybrid LightFM Food Recommender 

### Conclusions

|Model|Train AUC Score|Test AUC Score|Precision (k = 10)|Recall (k = 10)|Remarks|
|---|---|---|---|---|---|
|Pure Collaborative Filtering|0.955444|0.600403|0.007491|0.057937|Without Hyperparameter Tuning| 
|Hybrid LightFM|0.988701|0.635251|0.009329|0.059526|Without Hyperparameter Tuning|
|Hybrid LightFM|0.953348|0.673842|0.015671|0.098008|With Hyperparameter Tuning|

As we can see, Hybrid LightFM model test AUC score, precision score and recall score are much better than a Pure Collaborative Filtering model. It demonstrate that Hybrid LightFM methods provide more accurate recommendations than Pure Collaborative Filtering model approach. In conclusion, the Similar Restaurant Recommendation System provide individualized recommendations based on personalized.

Lastly, to solve the user cold-start problem, a Location-Based Recommendation System using the K-Means Clustering Algorithm is build which takes into account a user's location when he is using the app to recommend the top 10 nearby restaurants based on location proximity. The Location-Based Recommendation System aim to provide a quick and dirty service for passing users.
    
### Future Works

1. Scale the recommender to include all subzone area.
2. Use deep learning algorithms to predict more accurate recommendations.
3. Incorporate Graph Theory for location-based recommender to optimize travelling routes.

### References

1. https://towardsdatascience.com/sentiment-analysis-vader-or-textblob-ff25514ac540
2. https://github.com/microsoft/recommenders/blob/main/examples/02_model_hybrid/lightfm_deep_dive.ipynb
3. https://nycdatascience.com/blog/student-works/yelp-recommender-part-1/
4. https://making.lyst.com/lightfm/docs/index.html
