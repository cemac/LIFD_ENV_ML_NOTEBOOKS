# Notes from talk

Chetan Deva gave a talk

[RandomForests overview paper](https://link.springer.com/article/10.1023/A:1010933404324)
[Leaf temp](https://esajournals.onlinelibrary.wiley.com/doi/pdf/10.1002/ecs2.2768)
Elements of statistical learning Notebook

Motivations
Plants regulate temp in extreme environment. e.g. a plant in a desert stays 18C cool than air temp or 22 C warmer than air in mountains

Engery balance SW in LE out LW

plant growth is related to leaf temp and models often use just air temp - extreme temps will create large errors

Observational data 805 obs of bean leaves multispec device measures air temp, IR temp leaf temp, RH , Photosyntheticakky activate radtion (usable light) , Photo efficency, leaf health (greeness relative chorophyll), proton conductivity, leaf thickness, leaf angle

features chosen from sciecne

temp extremes not many but data has some large differences

RF:
data not random sampled to need non parametric. easy to use

Hyperfeaters

Max features: sometimes set to 1/3 number of features

Max samples: proportion of data set used

number of trees: more trees in general better

max depth: default to unpruned - un limtted leads to overfitting

K -fold cross validation - splitting testing
1 . fold for testing amd n-1 for training repeat for all k folds

- This method is better than random selection

- chosing hyper parameters - chetan deva did a sensitivty test - test if sensitivty test and
RandomizedSearchCV gives same results

- Prediction were postitive - 0.65-0.85 r2 RMSE 1.3-1.8 - k folds gives an overview
- feature importance showed air temp, humity were biggest factor

-----
