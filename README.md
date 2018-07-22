# MLB All-Star Classifier

> Current Status: training new neural network model (07/22/18)

A machine learning classification task that predicts the players of MLB's annual All-Star Game (position players only).

The project currently implements a **random forest model**, with intentions to implement a neural network model in the near future.

The project is written in Python, using the `pandas` and `numpy` packages for data analysis, the `matplotlib` package for data visualization, the `pickle` library for serialization, and the `sklearn` library for the machine learning models.

To try the model out for yourself, follow: <br/>
 `data_preprocessing.py` &rarr; `cross_validation.py` &rarr; `generate_team_sizes.py` &rarr; `hyperparam_tuning.py` &rarr; `model_training.py`

To collect a season's worth of data, such as for 2018, follow: <br/>
`bbref/bbref_scraper.py` (for each table) &rarr; `bbref/bbref_parser.py` (for each table) &rarr; `bbref/bbref_normalizer.py`

## 2018 All-Star Team

| Position | AL Prediction | AL Actual | NL Prediction | NL Actual |
| :------: | :-----------: | :-------: | :-----------: | :-------: |
| C | **Wilson Ramos, Rays** | **Wilson Ramos, Rays** | **Willson Contreras, Cubs** | **Willson Contreras, Cubs** |
| 1B | Yuli Gurriel, Astros | Jose Abreu, White Sox | **Freddie Freeman, Braves** | **Freddie Freeman, Braves** |
| 2B | **Jose Altuve, Astros** | **Jose Altuve, Astros** | **Javier Baez, Cubs** | **Javier Baez, Cubs** |
| 3B | **Jose Ramirez, Indians** | **Jose Ramirez, Indians** | **Nolan Arenado, Rockies** | **Nolan Arenado, Rockies** |
| SS | **Manny Machado, Orioles** | **Manny Machado, Orioles** | Starlin Castro, Marlins | Brandon Crawford, Giants |
| OF | **Mookie Betts, Red Sox** | **Mookie Betts, Red Sox** | **Nick Markakis, Braves** | **Nick Markakis, Braves** |
| OF | **Mike Trout, Angels** | **Mike Trout, Angels** | **Matt Kemp, Dodgers** | **Matt Kemp, Dodgers** |
| OF | **Aaron Judge, Yankees** | **Aaron Judge, Yankees** | **Bryce Harper, Nationals** | **Bryce Harper, Nationals** |
| Reserve | **J.D. Martinez, Red Sox** | **J.D. Martinez, Red Sox** | Matt Carpenter, Cardinals | Yadier Molina, Cardinals |
| Reserve | Eddie Rosario, Twins | Salvador Perez, Royals | Jose Martinez, Cardinals | Buster Posey, Giants |
| Reserve | Andrew Benintendi, Red Sox | Mitch Moreland, Red Sox | **J.T. Realmuto, Marlins** | **J.T. Realmuto, Marlins** |
| Reserve | Nicholas Castellanos, Tigers | Gleyber Torres, Yankees | **Paul Goldschmidt, D-backs** | **Paul Goldschmidt, D-backs** |
| Reserve | Andrelton Simmons, Angels | Jed Lowrie, Athletics | **Joey Votto, Reds** | **Joey Votto, Reds** |
| Reserve | **Alex Bregman, Astros** | **Alex Bregman, Astros** | **Ozzie Albies, Braves** | **Ozzie Albies, Braves** |
| Reserve | **Francisco Lindor, Indians** | **Francisco Lindor, Indians** | **Scooter Gennett, Reds** | **Scooter Gennett, Reds** |
| Reserve | **Michael Brantley, Indians** | **Michael Brantley, Indians** | **Eugenio Suarez, Reds** | **Eugenio Suarez, Reds** |
| Reserve | Whit Merrifield, Royals | Shin-Soo Choo, Rangers | **Trevor Story, Rockies** | **Trevor Story, Rockies** |
| Reserve | Xander Bogaerts, Red Sox | Mitch Haniger, Mariners | **Charlie Blackmon, Rockies** | **Charlie Blackmon, Rockies** |
| Reserve | Giancarlo Stanton, Yankees | George Springer, Astros | Brandon Belt, Giants | Lorenzo Cain, Brewers |
| Reserve | **Nelson Cruz, Mariners** | **Nelson Cruz, Mariners** | Brian Anderson, Marlins | Christian Yelich, Brewers |
| Reserve | **Jean Segura, Mariners** | **Jean Segura, Mariners** | **Jesus Aguilar, Brewers** | **Jesus Aguilar, Brewers** |

(Note that the AL's DH position and each league's Final Votes are considered reserve players for convenience.)

## Results

| Metric | `train` (1933-2009 data) | `test` (2010-2017 data) | `eval` (2018 data) |
| :----: | :----------------------: | :---------------------: | :----------------: |
| F1 score | 0.983563 | 0.599455 | 0.690476 |

## Credits

Credits to [Sean Lahman's Baseball Database](http://www.seanlahman.com/baseball-archive/statistics/) for the raw baseball data excluding 2018 (files in data/raw/).

Credits to [Baseball Reference](https://www.baseball-reference.com) for the raw baseball data of 2018.

Also credits to the [MLB](https://www.mlb.com/) for, well, existing, as otherwise this project would not have existed. I guess also for selecting this year's official All-Star Team.
