# Trustworthy Recommender Systems via Bayesian Bandits Capsone

### Team Members: Eric Song, Xiqiang Liu, Hien Bui, Vivek Saravanan

### Mentor: Yuhua Zhu

## About
Recommender systems have emerged as a simple yet powerful framework for the suggestion of relevant items to users. However, a potential issue arises when recommender systems overly recommend or spam undesired products to users in which the model loses the trust of the user. We propose a constrained bandit-based recommender system. We show this model outperforms Upper Confidence Bound (UCB) and Thompson sampling in terms of expected regret and does not lose the trust of the users.

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
