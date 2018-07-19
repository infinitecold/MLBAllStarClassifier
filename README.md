# MLB All-Star Classifier

> Current Status: tuning random forest model hyper-parameters (07/18/18)

A machine learning classification task that predicts the players of MLB's annual All-Star Game (position players only).

The project currently implements a **random forest model**. I intend to train a neural network model in the near future.

The project is written in Python, using the `pandas` and `numpy` packages for data analysis, the `matplotlib` package for data visualization, the `pickle` library for serialization, and the `sklearn` library for the machine learning models.

To try the project out for yourself, follow: <br/>
 `data_preprocessing.py` &rarr; `cross_validation.py` &rarr; `generate_team_sizes.py` &rarr; `hyperparam_tuning.py` &rarr; `model_training.py` &rarr; `team_selection.py`

## 2018 All-Star Team

| Position | AL Prediction | AL Actual | NL Prediction | NL Actual |
| :------: | :-----------: | :-------: | :-----------: | :-------: |
| C |  [TBD] | Wilson Ramos, Rays |  [TBD] | Willson Contreras, Cubs |
| 1B |  [TBD] | Jose Abreu, White Sox |  [TBD] | Freddie Freeman, Braves |
| 2B |  [TBD] | Jose Altuve, Astros |  [TBD] | Javier Baez, Cubs |
| 3B |  [TBD] | Jose Ramirez, Indians |  [TBD] | Nolan Arenado, Rockies |
| SS |  [TBD] | Manny Machado, Orioles |  [TBD] | Brandon Crawford, Giants |
| OF |  [TBD] | Mookie Betts, Red Sox |  [TBD] | Nick Markakis, Braves |
| OF |  [TBD] | Mike Trout, Angels |  [TBD] | Matt Kemp, Dodgers |
| OF |  [TBD] | Aaron Judge, Yankees |  [TBD] | Bryce Harper, Nationals |
| Reserve |  [TBD] | J.D. Martinez, Red Sox |  [TBD] | Yadier Molina, Cardinals |
| Reserve |  [TBD] | Salvador Perez, Royals |  [TBD] | Buster Posey, Giants |
| Reserve |  [TBD] | Mitch Moreland, Red Sox |  [TBD] | J.T. Realmuto, Marlins |
| Reserve |  [TBD] | Gleyber Torres, Yankees |  [TBD] | Paul Goldschmidt, D-backs |
| Reserve |  [TBD] | Jed Lowrie, Athletics |  [TBD] | Joey Votto, Reds |
| Reserve |  [TBD] | Alex Bregman, Astros |  [TBD] | Ozzie Albies, Braves |
| Reserve |  [TBD] | Francisco Lindor, Indians |  [TBD] | Scooter Gennett, Reds |
| Reserve |  [TBD] | Michael Brantley, Indians |  [TBD] | Eugenio Suarez, Reds |
| Reserve |  [TBD] | Shin-Soo Choo, Rangers |  [TBD] | Trevor Story, Rockies |
| Reserve |  [TBD] | Mitch Haniger, Mariners |  [TBD] | Charlie Blackmon, Rockies |
| Reserve |  [TBD] | George Springer, Astros |  [TBD] | Lorenzo Cain, Brewers |
| Reserve |  [TBD] | Nelson Cruz, Mariners |  [TBD] | Christian Yelich, Brewers |
| Reserve |  [TBD] | Jean Segura, Mariners |  [TBD] | Jesus Aguilar, Brewers |

(Note that the AL's DH position and each league's Final Votes are considered reserve players for convenience.)

## Results

| Metric  | `train` (1933-2009 data) | `test` (2010-2017 data) |
| :----: | :-----------------: | :----------------: |
| Precision |  [TBD] |  [TBD] |
| Recall |  [TBD] |  [TBD] |
| F1 |  [TBD] |  [TBD] |

## Credits

Credits to [Sean Lahman's Baseball Database](http://www.seanlahman.com/baseball-archive/statistics/) for the raw baseball data excluding 2018 (files in data/raw/).

Credits to [Baseball Reference](https://www.baseball-reference.com) for the raw baseball data of 2018.

Also credits to the [MLB](https://www.mlb.com/) for, well, existing, as otherwise this project would not have existed. I guess also for selecting this year's official All-Star Team.
