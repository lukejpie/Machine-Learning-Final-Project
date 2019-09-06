## CS66-Machine Learning-Final Project
  Luke Pietrantonio
  lpietra1

  Project Title:
    Exploring Effect of Dataset Sizes:
      Capital Bike Share



####4/18/19:

- Initial thinking

  Amazon:

  https://snap.stanford.edu/data/web-Amazon.html

  Potential information about what features of a product provide the best reviews. Ie, most sought after elements of a product.


  Bike sharing dataset:

  Explore the usage of bikes and potentially use bike usage to predict weather or specific weather events/patterns.
  http://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset

  Storms dataset:
  https://www.ncdc.noaa.gov/stormevents/ftp.jsp

  A dataset and a goal (i.e. phoneme identification from audio signals)
  An algorithm or set of algorithms you will develop and/or apply to this dataset
  A scientific question you are trying to answer (i.e. “Will SVMs or neural networks perform better on my dataset?” or “How will pre-processing a dataset or subsampling features affect the results?”)
  A way to evaluate and interpret the results
  References


  Data Augmentation:
  https://rpmcruz.github.io/machine%20learning/2018/05/11/regression-data-augmentation.html

  Member Type – Indicates whether user was a "registered" member (Annual Member, 30-Day Member or Day Key Member) or a "casual" rider (Single Trip, 24-Hour Pass, 3-Day Pass or 5-Day Pass)




- Project Proposal:

  CS66-Final Project Proposal
  Luke Pietrantonio


  Dataset and Goal

  	The proposed data set for this final project is the a Bike Sharing Dataset from Hadi Fanaee-T of The Laboratory of Artificial Intelligence and Decision Support. It contains 17389 data points, collected from Capital Bikeshare in Washington DC, between the years 2011 and 2012. There are 16 features associated with each data point, including but not limited to: date, season, weather, temperature, number of riders etc. Not all of the features will be utilized in the modeling of the data. The dataset starts on the January 1st 2011 and ends on December 31st 2012, each data point represents one hour of each day and the corresponding weather and usage information on that day and during that hour. The goal is to predict bike share usage based on weather patterns in the DC region.

  Software and Models

  	Given the regression nature of this problem, it would be possible to build various models. Initially, a Sklearn’s linear regression will be used to predict the usage of bikes. Random forests, also from Sklearn, will then be used as a linear regression model. Finally, if time permits, a neural network with a linear regression final layer will be used to fit the data.

  Motivation and Scientific Question

  	Personal motivation for the project stems from my love for biking and witnessing first hand the growth of Capital Bikeshare in my hometown, the DC-Metro area. The scientific question that I want to address is how data set sizes can improve the predictive qualities of models. This could be achieved not only by performing training with small subsections of the data, but also by data augmentation as to increase the number of datapoints. If time permits, data augmentation may look like utilizing more Capital Bikeshare usage information, in conjunction with weather datasets, to create another dataset, similar to that of the one used for initial training.

  Results, Evaluation, and Interpretation

  	The expected result is that larger datasets will increase testing accuracies in the models and data augmentation will only continue to increase these accuracies. However, there will be a plateau where adding more training data points will no longer add benefit to the model and thus the testing accuracy will not continue to increase. Evaluation of this hypothesis will be in the form of graphs of the number of training data points versus the test accuracies of the models. There will also be comparisons of the various models’ performance at the various dataset sizes, as to explore the proficiency of the different models at the different dataset sizes.

  References


  D.F. Specht, “A general regression neural network”,  IEEE Transactions on Neural Networks, 1991

  Andy Liaw and Matthew Wiener, “Classification and Regression by randomForest”, R news, 2002

#### 4/18/19: 1 hr

- Begin work on Sklearn linear regression.
    Explore functions
    Parse dataset

#### 4/25/19: 2 hrs

- More work on Regression
  Think about ways of treating the "date". Should I just encode this as a day, rather than a full
  date, that might make regression better...?
    - I ended up just doing this and factorizing the dates
  Implemented most of the linear regression, need to check errors and actual predictions to see how the models did

  Added polynomial regression features. Need to see if this is actually good

#### 4/28/19: 1 hr

- Testing linear regression with different sizes divisions (10,100,1000)

- Trying different size polynomials

- Start to implement randomForest
  When I perform my cross validation testing, should I perform this cross validation for every single
  dataset division, or should I instead perform it once against one model and then use that for all of them?

  - Run different hyper parameter choices on each regressor

  - Potentially get a sense for hyper parameters from a small enough, but reasonable division and then use these hyper parameters for all of the divisions to
    maintain consistency.

- rfr taken from (https://medium.com/datadriveninvestor/random-forest-regression-9871bc9a25eb)

#### 4/29/19: 1 hr

- Input from Sara: Make sure I include feature analysis. For linear regression, just get the weight vector and for random forest, get the stumps.

#### 4/30/19: 1 hr

- Tried to get random forest regression working, need to still get the scores working
- Added percentages for the errors in the linear regression and polynomial regressions
  RMSE/AVE

#### 5/1/19: 1 hr

- Need to get the RMSE averages better. Still working on how to pick the best features that will allow for some sort of regression.

- May have to turn this into a classification problem by saying whether or not it was a "high" day or a  "low" day, ie bigger or smaller than average

#### 5/5/19: 1 hr

- Adding feature importance analysis for random forests and ran tests at 25, 50, 75, and 100 divisions for Random Forest Regressor
- Still need to add feature analysis for the linear/polynomial regression and potentially turn into a classification and perform a logistic regression??

#### 5/7/19: 1 hr
- Adding feature importance analysis for linear/polynomial regression, talk to Sara about how to interpret feature analysis output
- Seemingly got a logistic regression working after transforming the y variables into classification variables of "high" and "low"
- Need to add feature analysis for logistic regression and ask Sara about how to get rid of that warning from the logistic regression and how to optimize the settings for the logistic regression

#### 5/8/19: 1 hr

- Look into the polynomial piece

#### 5/12/19: 4 hrs

- Started to make presentation
- Collected Data from polynomial/linear regressions and changed error to score, to allow for better comparison versus random forests
- Also changed regularized feature importance for linear regression and I am going to ditch feature importance for polynomial regression
- Got rid of polynomial regression from presentation just because it was going to be too annoying to work in and make right, especially with feature analysis.
