# CSC172 Association Rule Mining Project Proposal
**Student:** Josiah Raziel S. Lluch, 2022-0834  
**Date:** 12/12/2025

## 1. Project Title 
Association Rule Mining of Gaming Preferences and Habits using the Apriori Algorithm


## 2. Problem Statement
Game developers and platform owners need to understand which combinations of player characteristics, gaming habits, and preferences tend to occur together so they can design better features, target content, and tailor monetization strategies. However, survey data on gamers is high‑dimensional and contains many categorical responses (e.g., preferred genres, platforms, play styles, spending and time‑investment patterns), making it difficult to manually detect consistent preference bundles or player segments across hundreds of respondents. This project applies association rule mining with the Apriori algorithm to a gaming survey dataset to discover frequent itemsets and strong rules that reveal meaningful patterns in players’ demographics, game preferences, and play behaviors.

## 3. Objectives
- Preprocess the gaming survey dataset by cleaning responses, handling missing values, and converting all relevant features (e.g., preferred genres, platforms, competitive/casual style, spending habits, and binned playtime/hours‑per‑week) into categorical variables suitable for association rule mining.
- Transform the preprocessed data into a transaction format (one‑hot encoded items per respondent) that can be used as input to the Apriori algorithm.​
- Implement the Apriori algorithm (via mlxtend) to generate frequent itemsets and association rules that describe co‑occurring preferences and behaviors among gamers.​
- Evaluate and filter the discovered rules using support, confidence, lift, and conviction to retain only statistically meaningful and interpretable patterns.​
- Interpret and visualize key rules to highlight distinct player segments (e.g., high‑playtime competitive players, casual mobile players, multi‑genre enthusiasts) and discuss potential implications for game design and marketing.

## 4. Dataset Plan
- Source: [Gaming Preferences and Habits: Player Survey 2024 - Kaggle](https://www.kaggle.com/datasets/pranshudev/gaming-preferences-and-habits-player-survey-2024) (500 transactions, 47 items)
- Domain: Video Game Instustry Market Analysis
- Acquisition: Kaggle download to `data/Updated_Gaming_Survey_Responses.xlsx`

## 5. Technical Approach
- Preprocessing: One-hot encoding → mlxtend TransactionEncoder
- Algorithm: Apriori (min_support=0.02, min_confidence=0.6, min_lift=1.2)
- Framework: Python + pandas + mlxtend + matplotlib
- Environment: conda + Custom Python Modules + Jupyter Notebook

## 6. Expected Challenges & Mitigations
- Challenge: Missing or inconsistent survey responses
- Solution: Clean data, merge rare/ambiguous categories, and encode “No response” explicitly

- Challenge: Too many rare items and sparse matrix
- Solution: Remove items with very low support (e.g., <2–3% of respondents)

- Challenge: Long Apriori runtime and rule explosion
- Solution: Use higher min_support/confidence and limit max itemset length to 2–3

- Challenge: Trivial or obvious rules
- Solution: Apply a higher lift threshold (e.g., >1.2) and filter very common consequents

- Challenge: Redundant or hard‑to‑interpret rules
- Solution: Post‑filter similar/overlapping rules and keep only interpretable, persona‑like patterns
