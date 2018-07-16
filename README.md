# MLB All-Star Classifier

> Current Version: v0.2.0 (07/15/18)

A machine learning classification task that predicts the players of MLB's annual All-Star Game (position players only).

The project currently implements a **random forest model**. I intend to train a neural network model in the near future.

The project is written in Python, using the `pandas` and `numpy` packages for data analysis, the `matplotlib` package for data visualization, the `pickle` library for serialization, and the `sklearn` library for the machine learning models.

To try the project out for yourself, follow: <br/>
 `data_preprocessing.py` &rarr; `cross_validation.py` &rarr; `generate_team_sizes.py` &rarr; `model_training.py`/`hyperparam_tuning.py` &rarr; `team_selection.py`

## 2018 All-Star Team

| Position | AL Prediction | AL Actual | NL Prediction | NL Actual |
| :------: | :-----------: | :-------: | :-----------: | :-------: |
| C | [TBD in v1.0+] | Wilson Ramos, Rays | [TBD in v1.0+] | Willson Contreras, Cubs |
| 1B | [TBD in v1.0+] | Jose Abreu, White Sox | [TBD in v1.0+] | Freddie Freeman, Braves |
| 2B | [TBD in v1.0+] | Jose Altuve, Astros | [TBD in v1.0+] | Javier Baez, Cubs |
| 3B | [TBD in v1.0+] | Jose Ramirez, Indians | [TBD in v1.0+] | Nolan Arenado, Rockies |
| SS | [TBD in v1.0+] | Manny Machado, Orioles | [TBD in v1.0+] | Brandon Crawford, Giants |
| OF | [TBD in v1.0+] | Mookie Betts, Red Sox | [TBD in v1.0+] | Nick Markakis, Braves |
| OF | [TBD in v1.0+] | Mike Trout, Angels | [TBD in v1.0+] | Matt Kemp, Dodgers |
| OF | [TBD in v1.0+] | Aaron Judge, Yankees | [TBD in v1.0+] | Bryce Harper, Nationals |
| Reserve | [TBD in v1.0+] | J.D. Martinez, Red Sox | [TBD in v1.0+] | Yadier Molina, Cardinals |
| Reserve | [TBD in v1.0+] | Salvador Perez, Royals | [TBD in v1.0+] | Buster Posey, Giants |
| Reserve | [TBD in v1.0+] | Mitch Moreland, Red Sox | [TBD in v1.0+] | J.T. Realmuto, Marlins |
| Reserve | [TBD in v1.0+] | Gleyber Torres, Yankees | [TBD in v1.0+] | Paul Goldschmidt, D-backs |
| Reserve | [TBD in v1.0+] | Jed Lowrie, Athletics | [TBD in v1.0+] | Joey Votto, Reds |
| Reserve | [TBD in v1.0+] | Alex Bregman, Astros | [TBD in v1.0+] | Ozzie Albies, Braves |
| Reserve | [TBD in v1.0+] | Francisco Lindor, Indians | [TBD in v1.0+] | Scooter Gennett, Reds |
| Reserve | [TBD in v1.0+] | Michael Brantley, Indians | [TBD in v1.0+] | Eugenio Suarez, Reds |
| Reserve | [TBD in v1.0+] | Shin-Soo Choo, Rangers | [TBD in v1.0+] | Trevor Story, Rockies |
| Reserve | [TBD in v1.0+] | Mitch Haniger, Mariners | [TBD in v1.0+] | Charlie Blackmon, Rockies |
| Reserve | [TBD in v1.0+] | George Springer, Astros | [TBD in v1.0+] | Lorenzo Cain, Brewers |
| Reserve | [TBD in v1.0+] | Nelson Cruz, Mariners | [TBD in v1.0+] | Christian Yelich, Brewers |
| Reserve | [TBD in v1.0+] | Jean Segura, Mariners | [TBD in v1.0+] | Jesus Aguilar, Brewers |

(Note that the AL's DH position and each league's Final Votes are considered reserve players for convenience.)

## Results

| Metric  | `train` (1933-2009 data) | `test` (2010-2017 data) |
| :----: | :-----------------: | :----------------: |
| Precision | [TBD in v1.0+] | [TBD in v1.0+] |
| Recall | [TBD in v1.0+] | [TBD in v1.0+] |
| F1 | [TBD in v1.0+] | [TBD in v1.0+] |

## Credits

Credits to [Sean Lahman's Baseball Database](http://www.seanlahman.com/baseball-archive/statistics/) for the raw baseball data (files in data/raw/).

Also credits to the [MLB](https://www.mlb.com/) for, well, existing, as otherwise this project would not have existed. I guess also for selecting this year's official All-Star Team.
