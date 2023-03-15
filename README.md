# Trustworthy Recommender Systems via Bayesian Bandits Capsone

<h2> <u> About </u> </h2>

Team Members: Eric Song, Xiqiang Liu, Hien Bui, Vivek Saravanan

Mentor: Yuhua Zhu
<hr>

## Running Experiments
```bash
python run.py all  # run all experiments

python run.py etc  # run Explore-Then-Commit (ETC) experiments
python run.py ucb  # run UCB experiments
python run.py ts  # run Thompson Sampling experiments
python run.py optimal  # run Bayesian Optimal Policy experiments

python run.py linucb  # run LinUCB experiments
python run.py lints  # run Linear Thompson Sampling experiments
```

To run experiments related to Trustworthy Recommender Systems, run code in experiments.ipynb in TrustworthyMAB folder.

All the results are going to be saved in `results/` sub-directory.

## Visualize Results

Notebooks to visualize collected results could be found in `notebooks/` sub-directory.
