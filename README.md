# RallyRF
Random forest ML classification applied to tennis match outcome prediction.

By Cam Bitter, Sebastian Pantzer, and Danny Smith.

## Abstract
We aim to create a custom random forest classifier. We will build up from decision trees to random forests, and ideally explore further optimization techniques such as gradient boosting. We will explore the rest of the ML pipeline by applying our classifier onto historical tennis match data, where we will perform data analysis to generate helpful player statistic features. We will compare our results with a 2019 paper that achieves high match outcome prediction using random forests, logistic regression, and support vector machines.

## Motivation and Question
The main motivation behind this project is to learn an more advanced ML technique (that still holds up in modern times) in great detail by building it from the ground up. 

Tennis is a very hard sport to predict, with variables like mental state having huge implications on match outcome. By applying our custom random forest to a real-world prediction problem, we hope to learn about what affects the performance of the model we built. 

[Jeff Sackmann's ATP Data Repository](https://github.com/jeffsackmann/tennis_atp)

...
...
...

## Planned Deliverables
- Python package: contains our implementations for a decision tree, random forest, and potentially a gradient boosted version
- Python package: Data analysis and feature generation/engineering
- Jupyter notebook: Implementation of our model on our collected training data
- Jupyter notebook: Comparison of our results to other models (logistic regression, support vector machines, big CNN?)

## Resources Required

**Data**: https://github.com/jeffsackmann/tennis_atp
- This data does not come with all features we will want for each match, so we will need to implement data analysis prior to each match we predict or use an existing library

## What we will learn
- Creation and implementation of custom model
- General ML Pipeline
- Effiective data visualization
- Torch

## Risk Statement
Most of the risk of this project is time (i.e. can we finish everything we want to in time). But, I believe we will not have an issue with this. The main project objective is to create a random forest classifier from scratch, not predict tennis match outcomes. If we just have enough time to complete the classifier, then we can bring in curated classification datasets to test our model against existing implementation (such as sklearn). 

## Ethics Statement
Most tennis data exists in a blackbox held by commercial betting companies or the Association of Tennis Professionals (ATP). Creaing an open-source model would allow fans, journalists, coaches, and players to understand what actually drives match outcomes. Learning what aspects of a player's game are most effective against other types of players is invaluable information for players on the brink of going pro who don't have access to a fulltime coaching staff. It's also useful to recreational players who simply want better tournament outcomes. A potential caveat to our egalitarian aspirations mentioned above is that pro-level tennis is different from amateur tennis. For example, maybe the amount of aces a player has had in a match does not give as much information on who will win a game on the pro level, but would be more important for amateur level play. Amateur players would not be able to gain that insight. So, there is a potential barrier for productive use for amateurs given this hypothetical.

The world will likely not become a better place after the creation of this predictive model. We say this with some confidence as we believe virtually none of the pressing humanitarian issues in the world are affected by tennis proficiency. Also, while sports participation can improve communities in wonderful ways, the driver behind this is most likely camaraderie and teamwork rather than performance related aspects. 

Problems:
Our model could be misused as a gambling aid (not good) to justify increased gambling (not good). 

## Tentative Timeline
Week 1: Decision Tree implemented
Week 2: Random Forest implemented
Week 3: Feature extraction/engineering complete
Week 4: Final testing and model comparison complete
