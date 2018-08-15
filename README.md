# MLB All-Star Classifier

A machine learning classification task that predicts the players of MLB's annual All-Star Game (position players only). The project implements a **random forest model**.

The project is written in Python, using the `pandas` and `numpy` packages for data analysis, the `matplotlib` package for data visualization, the `pickle` library for serialization, and the `sklearn` library for the machine learning models.

To try the model out for yourself, follow: <br/>
 `data_preprocessing.py` &rarr; `cross_validation.py` &rarr; `generate_team_sizes.py` &rarr; `hyperparam_tuning.py` &rarr; `model_training.py`

To collect a season's worth of data, such as for 2018, follow: <br/>
`bbref/bbref_scraper.py` (for each table) &rarr; `bbref/bbref_parser.py` (for each table) &rarr; `bbref/bbref_normalizer.py`

## 2018 All-Star Team

| League | Position | `random_forest` Prediction | MLB Actual |
| :----: | :------: | :------------------------: | :--------: |
| AL | C | **Wilson Ramos, Rays** | **Wilson Ramos, Rays** |
| AL | 1B | Yuli Gurriel, Astros | Jose Abreu, White Sox |
| AL | 2B | **Jose Altuve, Astros** | **Jose Altuve, Astros** |
| AL | 3B | **Jose Ramirez, Indians** | **Jose Ramirez, Indians** |
| AL | SS | **Manny Machado, Orioles** | **Manny Machado, Orioles** |
| AL | OF | **Mookie Betts, Red Sox** | **Mookie Betts, Red Sox** |
| AL | OF | **Mike Trout, Angels** | **Mike Trout, Angels** |
| AL | OF | **Aaron Judge, Yankees** | **Aaron Judge, Yankees** |
| AL | Reserve | **J.D. Martinez, Red Sox** | **J.D. Martinez, Red Sox** |
| AL | Reserve | Eddie Rosario, Twins | Salvador Perez, Royals |
| AL | Reserve | Andrew Benintendi, Red Sox | Mitch Moreland, Red Sox |
| AL | Reserve | Nicholas Castellanos, Tigers | Gleyber Torres, Yankees |
| AL | Reserve | Andrelton Simmons, Angels | Jed Lowrie, Athletics |
| AL | Reserve | **Alex Bregman, Astros** | **Alex Bregman, Astros** |
| AL | Reserve | **Francisco Lindor, Indians** | **Francisco Lindor, Indians** |
| AL | Reserve | **Michael Brantley, Indians** | **Michael Brantley, Indians** |
| AL | Reserve | Whit Merrifield, Royals | Shin-Soo Choo, Rangers |
| AL | Reserve | Xander Bogaerts, Red Sox | Mitch Haniger, Mariners |
| AL | Reserve | Giancarlo Stanton, Yankees | George Springer, Astros |
| AL | Reserve | **Nelson Cruz, Mariners** | **Nelson Cruz, Mariners** |
| AL | Reserve | **Jean Segura, Mariners** | **Jean Segura, Mariners** |
| NL | C | **Willson Contreras, Cubs** | **Willson Contreras, Cubs** |
| NL | 1B | **Freddie Freeman, Braves** | **Freddie Freeman, Braves** |
| NL | 2B | **Javier Baez, Cubs** | **Javier Baez, Cubs** |
| NL | 3B | **Nolan Arenado, Rockies** | **Nolan Arenado, Rockies** |
| NL | SS | Starlin Castro, Marlins | Brandon Crawford, Giants |
| NL | OF | **Nick Markakis, Braves** | **Nick Markakis, Braves** |
| NL | OF | **Matt Kemp, Dodgers** | **Matt Kemp, Dodgers** |
| NL | OF | **Bryce Harper, Nationals** | **Bryce Harper, Nationals** |
| NL | Reserve | Matt Carpenter, Cardinals | Yadier Molina, Cardinals |
| NL | Reserve | Jose Martinez, Cardinals | Buster Posey, Giants |
| NL | Reserve | **J.T. Realmuto, Marlins** | **J.T. Realmuto, Marlins** |
| NL | Reserve | **Paul Goldschmidt, D-backs** | **Paul Goldschmidt, D-backs** |
| NL | Reserve | **Joey Votto, Reds** | **Joey Votto, Reds** |
| NL | Reserve | **Ozzie Albies, Braves** | **Ozzie Albies, Braves** |
| NL | Reserve | **Scooter Gennett, Reds** | **Scooter Gennett, Reds** |
| NL | Reserve | **Eugenio Suarez, Reds** | **Eugenio Suarez, Reds** |
| NL | Reserve | **Trevor Story, Rockies** | **Trevor Story, Rockies** |
| NL | Reserve | **Charlie Blackmon, Rockies** | **Charlie Blackmon, Rockies** |
| NL | Reserve | Brandon Belt, Giants | Lorenzo Cain, Brewers |
| NL | Reserve | Brian Anderson, Marlins | Christian Yelich, Brewers |
| NL | Reserve | **Jesus Aguilar, Brewers** | **Jesus Aguilar, Brewers** |

(Note that the AL's DH position and each league's Final Votes are considered reserve players for convenience.)

## Results

| Metric | `train` (1933-2009 data) | `test` (2010-2017 data) | `eval` (2018 data) |
| :----: | :----------------------: | :---------------------: | :----------------: |
| F1 | 0.983563 | 0.599455 | 0.690476 |

## Credits

Credits to [Sean Lahman's Baseball Database](http://www.seanlahman.com/baseball-archive/statistics/) for the raw baseball data excluding 2018 (files in data/raw/).

Credits to [Baseball Reference](https://www.baseball-reference.com) for the raw baseball data of 2018.

Also credits to the [MLB](https://www.mlb.com/) for, well, existing, as otherwise this project would not have existed. I guess also for selecting this year's official All-Star Team.
