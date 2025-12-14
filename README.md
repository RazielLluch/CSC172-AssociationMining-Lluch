# Association Rule Mining of Gaming Preferences and Habits using the Apriori Algorithm
**CSC172 Data Mining and Analysis Final Project**  
*Mindanao State University - Iligan Institute of Technology*  
**Student:** Josiah Raziel S. Lluch, 2022-0834  
**Semester:** AY 2025-2026 Sem 1

[![Python](https://img.shields.io/badge/Python-3.13.5-blue)](https://python.org) [![pandas](https://img.shields.io/badge/pandas-2.3.3-brightgreen)](https://pandas.pydata.org/) [![mlxtend](https://img.shields.io/badge/mlxtend-0.23.4-blue)](https://rasbt.github.io/mlxtend/)

## Abstract
This project applies association rule mining with the Apriori algorithm to a gaming survey dataset to uncover patterns in players’ preferences and habits. Each respondent’s answers (e.g., preferred genres, platforms, playstyle, location, time spent gaming, and spending behavior) are transformed into a set of binary items and analyzed as a transaction. Using Apriori, the project extracts frequent itemsets and high-quality association rules evaluated by support, confidence, and lift to reveal which combinations of characteristics tend to co‑occur among gamers. The results highlight interpretable player “personas” and preference bundles that can inform game design, content recommendations, and targeted marketing strategies.

## Table of Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
  - [1.1 Problem Statement](#11-problem-statement)
  - [1.2 Objectives](#12-objectives)
  - [1.3 Scope and Limitations](#13-scope-and-limitations)
- [2. Dataset Description](#2-dataset-description)
  - [2.1 Source and Acquisition](#21-source-and-acquisition)
  - [2.2 Data Structure](#22-data-structure)
  - [2.3 Sample Transactions](#23-sample-transactions)
- [3. Methodology](#3-methodology)
  - [3.1 Data Preprocessing](#31-data-preprocessing)
  - [3.2 Exploratory Data Analysis](#32-exploratory-data-analysis)
  - [3.3 Apriori Algorithm Implementation](#33-apriori-algorithm-implementation)
  - [3.4 Evaluation Metrics](#34-evaluation-metrics)
- [4. Results](#4-results)
  - [4.1 Top Association Rules](#41-top-association-rules)
  - [4.2 Key Visualizations](#42-key-visualizations)
  - [4.3 Performance Metrics](#43-performance-metrics)
- [5. Discussion](#5-discussion)
  - [5.1 Business Insights](#51-business-insights)
  - [5.2 Actionable Recommendations](#52-actionable-recommendations)
  - [5.3 Limitations](#53-limitations)
- [6. Conclusion](#6-conclusion)
- [7. Video Presentation](#7-video-presentation)
- [References](#references)
- [Appendix: Full Results](#appendix-full-results)



## 1. Introduction
### 1.1 Problem Statement
Game developers and platform owners need to understand which combinations of player characteristics, gaming habits, and preferences tend to occur together so they can design better features, target content, and tailor monetization strategies. However, survey data on gamers is high‑dimensional and contains many categorical responses (e.g., preferred genres, platforms, play styles, spending and time‑investment patterns), making it difficult to manually detect consistent preference bundles or player segments across hundreds of respondents. This project applies association rule mining with the Apriori algorithm to a gaming survey dataset to discover frequent itemsets and strong rules that reveal meaningful patterns in players’ demographics, game preferences, and play behaviors.

### 1.2 Objectives
- Preprocess the gaming survey dataset by cleaning responses, handling missing values, and converting all relevant features (e.g., preferred genres, platforms, competitive/casual style, spending habits, and binned playtime/hours‑per‑week) into categorical variables suitable for association rule mining.
- Transform the preprocessed data into a transaction format (one‑hot encoded items per respondent) that can be used as input to the Apriori algorithm.​
- Implement the Apriori algorithm (via mlxtend) to generate frequent itemsets and association rules that describe co‑occurring preferences and behaviors among gamers.​
- Evaluate and filter the discovered rules using support, confidence, lift, and conviction to retain only statistically meaningful and interpretable patterns.​
- Interpret and visualize key rules to highlight distinct player segments (e.g., high‑playtime competitive players, casual mobile players, multi‑genre enthusiasts) and discuss potential implications for game design and marketing.

### 1.3 Scope and Limitations
**Scope:** 
- Analyze one cleaned gaming survey dataset where each respondent is modeled as a transaction composed of binary items representing demographics, platforms, genres, playstyle, time spent gaming, and spending patterns.
- Apply association rule mining using the Apriori algorithm to generate frequent itemsets and association rules that describe co‑occurring player characteristics and habits.​
- Evaluate rules with standard metrics (support, confidence, lift, conviction) and present a curated subset of interpretable rules as gamer “personas” and preference bundles. 
**Limitations:** 
- Results are limited to the sampled population of the survey and may not generalize to all gamers (other regions, age groups, or platforms).​
- All continuous information has been discretized into bins and one‑hot encoded, so subtle numeric differences within the same bin are not modeled.​
- Apriori only captures co‑occurrence, not causality; discovered rules indicate associations, not reasons why players behave a certain way.​
- The number and quality of rules depend on chosen thresholds (support, confidence, lift) and on filtering rare items; some potentially interesting but infrequent patterns may be excluded.

## 2. Dataset Description
### 2.1 Source and Acquisition
**Source:** [Gaming Preferences and Habits: Player Survey 2024 - Kaggle](https://www.kaggle.com/datasets/pranshudev/gaming-preferences-and-habits-player-survey-2024)  
**Size:** 500 transactions, 69 unique items  
**Format:** Respondent ID + full set of survey answers → transaction of binary items (one row per player in basket format).

### 2.2 Data Structure
Raw format (one row per item):
Gender	Age_Teen	Age_Young_Adult	Age_Adult	Age_Mid_Adult	Location_India	Location_US	Location_Other	Gaming_Daily	Gaming_Weekly	Gaming_Monthly	Gaming_Rarely_Never	Gaming_Hours	Device_PC	Device_Mobile	Device_Console	Device_Handheld	Device_Tablet	Genre_Action/Adventure	Genre_FPS	Genre_RPG	Genre_Puzzle/Strategy	Genre_Simulation	Genre_MMO	Genre_Sports	Favorite_Game	Discovery_Social_Media	Discovery_Gaming_Forums	Discovery_Friends_Family	Discovery_Game_Reviews	Discovery_YouTube_Streaming	Discovery_Self_Search	Game_Mode_Single_Player	Game_Mode_Multiplayer	Game_Mode_Both	Game_Mode_Unknown	Spend_lt100	Spend_100-500	Spend_500-1000	Spend_1000plus	Spend_Unknown	Reason_Fun	Reason_Stress_Relief	Reason_Skills_Competition	Reason_Socialize	Reason_Learning	Reason_Other
Male	1	0	0	0	1	0	0	1	0	0	0	10-20 hours	0	1	0	0	0	0	0	0	0	0	0	1	FC Mobile	1	1	0	0	0	0	0	0	1	0	1	0	0	0	0	1	1	0	0	0	0
Male	0	1	0	0	1	0	0	0	0	0	1	0-1 hour	0	1	0	0	0	0	0	0	1	1	0	0	Wukong	1	0	0	0	0	0	1	0	0	0	1	0	0	0	0	0	0	1	0	0	0
Male	0	1	0	0	1	0	0	0	0	0	1	20+ hours	0	0	1	0	0	0	1	0	0	0	0	0	Call of Duty	0	1	0	0	0	0	0	1	0	0	0	1	0	0	0	1	0	0	0	0	0

Transaction format (one row per basket):
[
  ['Age_YoungAdult', 'Location_India', 'Genre_FPS', 'Genre_MOBA', 'Platform_PC', 'Platform_Mobile', 'Playtime_High'],
  ['Age_Teen', 'Location_USA', 'Genre_MOBA', 'Genre_RPG', 'Platform_Console', 'SpendsMoney_Yes']
]


### 2.3 Sample Transactions
Transaction 1: ['Age_YoungAdult', 'Location_India', 'Genre_FPS', 'Genre_MOBA', 'Platform_PC', 'Platform_Mobile', 'Playtime_High', 'Competitive_Yes']

Transaction 2: ['Age_Teen', 'Location_USA', 'Genre_RPG', 'Genre_Adventure', 'Platform_Console', 'Playtime_Medium', 'Competitive_No']

Transaction 3: ['Age_Adult', 'Location_USA', 'Genre_Sports', 'Genre_Racing', 'Platform_PC', 'Playtime_Low', 'SpendsMoney_Yes']


## 3. Methodology

### 3.1 Data Preprocessing
1. **Missing Value Handling:** Removed duplicate rows and cleaned invalid or missing values for categorical columns (e.g., Gender, Age, Monthly Spend, Favorite Game).
2. **One-Hot Encoding:** Converted categorical columns into binary columns suitable for association mining:
- Age → Age_Teen, Age_Young_Adult, Age_Adult, Age_Mid_Adult
- Location → Location_India, Location_US, Location_Other
- Gender → standardized to Male, Female, Other
- Gaming Frequency → Gaming_Daily, Gaming_Weekly, Gaming_Monthly, Gaming_Rarely_Never
- Gaming Hours → Gaming_Hours_0-1, Gaming_Hours_1-5, Gaming_Hours_5-10, Gaming_Hours_10-20, Gaming_Hours_20+, Gaming_Hours_Unknown
- Device Used → Device_PC, Device_Mobile, Device_Console, Device_Handheld, Device_Tablet
- Game Genres → Genre_Action/Adventure, Genre_FPS, Genre_RPG, Genre_Puzzle/Strategy, Genre_Simulation, Genre_MMO, Genre_Sports
- Favorite Game → standardized categorical column
- Game Discovery → Discovery_Social_Media, Discovery_Gaming_Forums, Discovery_Friends_Family, Discovery_Game_Reviews, Discovery_YouTube_Streaming, Discovery_Self_Search
- Game Mode Preference → Game_Mode_Single_Player, Game_Mode_Multiplayer, Game_Mode_Both, Game_Mode_Unknown
- Monthly Spend → Spend_lt100, Spend_100-500, Spend_500-1000, Spend_1000plus, Spend_Unknown

- Play Reason → Reason_Fun, Reason_Stress_Relief, Reason_Skills_Competition, Reason_Socialize, Reason_Learning, Reason_Other
3. **Feature Reduction / Cleaning:** Removed unnecessary columns such as timestamps, redundant text responses, and original categorical columns after encoding.
4. **Final Dataset:** Dataset ready for association mining with fully binary features suitable for Apriori algorithm.

**Before/After Statistics:**
| Metric | Raw Data | Processed Data |
|--------|----------|----------------|
| Rows (Respondents) | 500 | 500 |
| Columns (Features) | Original survey columns | Binary columns after preprocessing (47) |
| Duplicates | None | Removed |
| Missing Values | Present | Handled/Standardized |

### 3.2 Exploratory Data Analysis

### 3.3 Apriori Algorithm Implementation
**Implementation:** To be continued

### 3.4 Evaluation Metrics
- **Support:** 
- **Confidence:** 
- **Lift:** 


## 4. Results
### 4.1 Top Association Rules


### 4.2 Key Visualizations


### 4.3 Performance Metrics


## 5. Discussion

### 5.1 Business Insights

### 5.2 Actionable Recommendations

### 5.3 Limitations

## 6. Conclusion

## 7. Video Presentation

## References

## Appendix: Full Results


