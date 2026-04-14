# RallyRF

Random forest ML classification applied to tennis match outcome prediction.

By Cam Bitter, Sebastian Pantzer, and Danny Smith.

## Abstract

We aim to create a custom random forest classifier. We will build up from decision trees to random forests, and explore further optimization techniques such as gradient boosting. 

We will explore the rest of the ML pipeline by performing data analysis on historical tennis match data to generate insightful player statistic features, which will be fed into our classifier to predict match outcomes. We will compare our results with a 2019 paper that achieves high match prediction accuracy using random forests, logistic regression, and support vector machines.

## Motivation and Question

The main motivation behind this project is to learn an more advanced ML technique (that still holds up in modern times) in great detail by building it from the ground up. Random forests are widely popular and effective at medium sized tabular data, and are much more efficient than deep learning models.

Tennis is a very hard sport to predict, with variables like mental state having huge implications on match outcome. By applying our custom random forest to a real-world prediction problem, we hope to learn about what affects the performance of the model we built and compare our model to research findings.

Our dataset is [JeffSackmann ATP Data Repository](https://github.com/jeffsackmann/tennis_atp). It contains ATP tennis match results spanning back to 1964, with over 40,000 matches. 

We may also use [Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject), another tennis stats repository hosted on GitHub by JeffSackmann. This dataset contains better odds of major bookmarkers such as Betting365, which we can compare against our model predictions.

## Planned Deliverables

- Python package: contains our implementations for a decision tree, random forest, and potentially a gradient boosted version
- Python package: Data analysis and feature generation/engineering
- Jupyter notebook: Implementation of our model on our collected training data
- Jupyter notebook: Comparison of our results to other models (logistic regression, support vector machines, big CNN?)


A partial success would involve implementing a decision tree and random forest to assess our research goal, and comparing them to a logistic regression model, but might not include expanding our project with more advanced features and models. Due to time constraints, we might not be able to include various visualizations of our data. A partial success will involve a comparison between a machine learning technique that is new to us and a technique that we have learned in class. 

That being said, we plan on a full success of our deliverables. A full success includes developing the models mentioned in the partial success part, but building on our learning by implementing a boosted decision tree. We would also include a deeper library of models to compare with, like support vector machines and a convolutional neural network. A full success would include helpful visualizations that outline details like differences in model performance and feature properties. A full success would encapsulate the above material in a dense but digestible manner.
## Resources Required

**Data**: https://github.com/jeffsackmann/tennis_atp
This data provides *post match* statistics for each match. Since we want to use stats to predict an unknown match outcome, we will need to perform some data analysis to collect statisics for each player prior to the match we are predicting. The dataset has the following statistics for each match:
Each match contains the following post-match player stats:

| Variable | Description |
|---|---|
| w_ace | Winner's number of aces |
| w_df | Winner's number of double faults |
| w_svpt | Winner's number of serve points |
| w_1stIn | Winner's number of first serves made |
| w_1stWon | Winner's number of first-serve points won |
| w_2ndWon | Winner's number of second-serve points won |
| w_SvGms | Winner's number of serve games |
| w_bpSaved | Winner's number of break points saved |
| w_bpFaced | Winner's number of break points faced |
 

We can calculate the average of these statistics over, say, the last 30 matches for each player and then use the historical statistics to predict a future match.

## What we will learn

We're excited to learn:
- Creation and implementation of a custom model
- Data preperation and analysis
- Effective data visualization
- Torch and Pandas
- Comparison to research

## Risk Statement

One risk of this project is time (i.e. can we finish everything we want to in time). It has a relatively large scope: custom model, data preparation, comparisons to other models, etc.

But, I believe we will not have an issue with this. The main project objective is to create a random forest classifier from scratch, not predict tennis match outcomes. If we just have enough time to complete the classifier, then we can bring in curated classification datasets to test our model against existing implementation (such as sklearn).

Another risk is that that feature generation complexity may be more difficult to implement than expected. If we have to calculate pre-match stats for every single match we run through the model (say, 20,000 matches), and we derive these stats from say a 6 month timeframe, this could become a computationally heavy task. However, we would only need to perform this task once. We could join the derived statistics with each match we run our feature generation on.

Another challenge is that amateur matchplay may have a very different set of important features than professional tennis. Thus, our model, having only been trained on professional matchplay, wouldn't generalize for amateur players. 

## Ethics Statement

Most tennis data exists in a blackbox held by commercial betting companies or the Association of Tennis Professionals (ATP). Creaing an open-source model would allow fans, journalists, coaches, and players to understand what actually drives match outcomes. Learning what aspects of a player's game are most effective against other types of players is invaluable information for players on the brink of going pro who don't have access to a fulltime coaching staff. It's also useful to recreational players who simply want better tournament outcomes. A potential caveat to our egalitarian aspirations mentioned above is that pro-level tennis is different from amateur tennis. For example, maybe the amount of aces a player has had in a match does not give as much information on who will win a game on the pro level, but would be more important for amateur level play. Amateur players would not be able to gain that insight. So, there is a potential barrier for productive use for amateurs given this hypothetical.

The world will likely not become a better place after the creation of this predictive model. We say this with some confidence as we believe virtually none of the pressing humanitarian issues in the world are affected by tennis proficiency. Also, while sports participation can improve communities in wonderful ways, the driver behind this is most likely camaraderie and teamwork rather than performance related aspects.

Problems:
Another application of our model, specifically for non-tennis players, is utilizing its predictive capability to enhance gambling.
While we do not endorse this activity, a model that can accurately predict the outcomes of sports events better than seeding
could be used to score better than gambling odds and win money. Therefore, our project could be used to justify an increase in gambling, both in frequency and amount.

## Tentative Timeline

Week 1: Implement Decision Tree classifier
Week 2: Implement Random Forest classifier

Week 2 Goal:
- Ideally by this point we will have a solid understanding of our decision tree classifier, and will have completed our random forest implementation
- Test our model on a premade classification dataset

Week 3: Feature extraction/engineering complete, mess around with gradient boosting
Week 4: Final testing and model comparison complete

Week 4 Goal:
- Final random forest classifier complete
- Data analysis pipeline complete (feature generation)
- Comparison to existing research and gambling odds