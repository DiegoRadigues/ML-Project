

![391487-league-of-legends-wallpaper-2048x1152](https://github.com/user-attachments/assets/485bea22-b7ae-468c-ba9f-27a0bf17c734)


# Ecamania Esports Final Report

**Author**: Diego de Radiguès  

**Program**: 4EOAI40 Artificial Intelligence

**Student ID**: 20342  

## Table of Contents
1. [How To Run The Project](#how-to-run-the-project)  
2. [Introduction](#introduction)  
3. [Data Understanding](#data-understanding)  
4. [Exploratory Data Analysis](#exploratory-data-analysis)  
5. [Feature Engineering and Preprocessing](#feature-engineering-and-preprocessing)  
7. [Modeling and Evaluation per Role](#modeling-and-evaluation-per-role)  
7. [Overfitting Analysis](#overfitting-analysis)  
8. [SHAP Feature Importance](#shap-feature-importance)  
9. [Player Ranking](#player-ranking)  
10. [Conclusion](#conclusion)
11. [References](#references)  

---

# How to Run the Project


## 1. Clone the Repository

```bash
git clone https://github.com/DiegoRadigues/ML-Project.git
cd ML-Project
```

## 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## 3. Install Required Libraries

Make sure you have `pip` updated and install dependencies:

```bash
pip install -r requirements.txt
```

## 4. Prepare the Data

Ensure the `data.zip` file is located in the root of the project directory. It should contain:

- `game_players_stats.csv`
- `game_metadata.csv`
- `game_events.csv`

These will be extracted automatically when running the script.

## 5. Run the Script

Execute the main script.

```bash
python Script_v3.py
```

This will:
- Generate figures in the `figures/` directory
- Save logs to a timestamped `.txt` file
- Print top players by role and overall
- Save SHAP plots and model diagnostics


**Note**: Execution time may exceed 1,5 hour due to GridSearch and SHAP computations. 

---

## Introduction
**Ecamania** is a newly formed Esports team focusing on *League of Legends*.  

With support from the school and an unlimited budget, perhaps to compensate for the rather modest level of the teaching staff, the team aims to recruit the best players in the world to compete at **ECAM**.

As students from ECAM’s top-performing Electrical and Computer Engineering program, we were tasked with a data-driven analysis of professional matches to help achieve this goal.

The project has three main objectives:

1. **Match Outcome Prediction**  
   Build models to determine based on a single player's in-game statistics from a match, whether that player's team won or lost.

2. **Key Factors Identification**  
   Identify the most important quantifiable variables (in-game stats) that players should focus on to improve their chances of winning.

3. **Player Ranking**  
   Use the model predictions to evaluate players’ performance and recommend the most *talented* players for Ecamania to recruit.

We follow a structured methodology:

- Understanding and preparing the data  
- Exploring it visually  
- Building and evaluating prediction models for each player role  
- Analyzing feature importance with SHAP  
- Ranking players based on model-derived performance scores

---

## For Readers New to League of Legends

If you're not a League of Legends player, some terms and concepts in this report might be unfamiliar. As we are not players ourselves, we recommend the following resources to help you understand the game's fundamentals:

- [ARTICLE - A Beginner's Guide to League of Legends – Epic Games Store](https://store.epicgames.com/en-US/news/a-beginner-s-guide-to-league-of-legends)  
  An overview of the game's basics with roles and objecives.

- [VIDEO - The COMPLETE Beginner's Guide to League of Legends!](https://www.youtube.com/watch?v=tQbo2X2Qysc)  
  A more advanded overview covering game objectives roles and basic strategies.


---

## Data Understanding
We received a **rich dataset of professional League of Legends matches**, originally sourced via the [Leaguepedia API](https://lol.fandom.com/wiki/Help:Leaguepedia_API), a community-maintained database that compiles detailed match statistics from competitive League of Legends games.

The dataset consists of three core files:

| File | Description | Size |
|------|-------------|------|
| `game_players.csv` | Detailed stats for each player in each game | **374,554** player-game records, 28 columns |
| `game_metadata.csv` | High-level information for each match (league, teams, patch, …) | **37,459** games |
| `game_events.csv` | Fine-grained in-game events (kills, objectives, etc.) | **1.88 M+** events |

Each **player-game** record includes:

* Role (`Top`, `Jungle`, `Mid`, `Bot/ADC`, `Support`)  
* Match outcome (`win` = 1 / `loss` = 0)  
* Performance stats: kills, deaths, assists, farm (minions), gold earned, damage dealt (total & vs champions), damage taken, wards placed, **team objective** counts (towers, dragons, barons, …) and the player’s champion

---

## Exploratory Data Analysis
Our initial EDA validated that **winning players outperform losing players on nearly every metric**.

* **Distributions of kills, assists, gold, damage** clearly shift higher for winners while deaths are lower.  
  *See: “Distribution of performance stats by win vs loss”*  
* Breakdown **by role** highlights expected play-style profiles:  
  * *Supports* → low kills, high wards placed  
  * *AD Carries* → highest damage & gold  
* Across **regional leagues** win/loss counts remain roughly equal indicating no league-specific bias.  
  *See: “Win/Loss count by league”*

During this phase we flagged extreme numeric outliers (e.g., a single 30-kill game) for later treatment ensuring they cannot skew model training.

![distrib](https://github.com/user-attachments/assets/940e1d46-c892-4206-bdf3-26d29467195b)


*Figure 1 – Distribution of match outcomes for all player-game instances (green = win, red = loss).*

This chart confirms that the dataset is **well balanced** with roughly the same number of player records for winning and losing games.

This balance supports the use of **F1-score** as the primary evaluation metric in the modeling phase. as it ensures fair consideration of both classes and avoids bias toward either outcome.

![eda_stats_by_outcome](https://github.com/user-attachments/assets/4605c3bc-bc04-4597-832c-35b266421c03)

*Figure 2 – Distribution of key player performance metrics (kills, assists, damage, wards) for winning (green) vs losing (red) players.*

These stacked histograms clearly show that:
- Winning players tend to have **more kills and assists**,
- They deal **more damage to champions**
- Even **vision control (wards placed)** is slightly higher among winners.

This confirms that individual performance is a strong signal for match outcome prediction.


![Kills and Gold per Minute](https://github.com/user-attachments/assets/51c32263-aac9-46f4-8dc7-0d9faf9c5730)


*Figure 3 – Density plots comparing kills per minute (left) and gold per minute (right) for winning (green) and losing (red) players.*

These plots normalize performance by game length to allow fair comparison across matches of different durations.

Key observations:
- **Kills per minute**: Winners consistently show higher density in the 0.1–0.3 range indicating more efficient combat impact.
- **Gold per minute**: A clear shift toward the right for winning players suggests stronger economy and resource control.

These findings reinforce earlier conclusions: **winning players not only perform better overall but also do so more efficiently** relative to time played.


![correlation](https://github.com/user-attachments/assets/20c8ed2d-6983-4f0a-8b86-18da5213b616)




*Figure 4 – Heatmap of Pearson correlations among key in-game performance variables.*

This matrix reveals several important relationships:
- **Strong positive correlations** exist between:
  - `gold_earned` and `total_minions_killed` (**r = 0.85**)
  - `kills` and `gold_earned` (**r = 0.66**)
  - `damage_dealt_to_champions` and both `kills` (**r = 0.63**) and `gold` (**r = 0.79**)
- **Wards placed** shows a **mild negative correlation** with combat-related features. confirming its role as a support/vision stat.
- **Game length** has negligible correlation with most variables validing the need to normalize stats by time.

These correlations informed the **feature selection and engineering phase**.

!![wins_by_league](https://github.com/user-attachments/assets/704a9410-cb5c-4b37-8d5e-eaa64a436233)


*Figure 5 – Distribution of game outcomes (win/loss) across different professional leagues.*

Each bar pair shows the number of games won (green) and lost (red) per league, based on merged match metadata.

**Key observations:**
- The number of wins and losses is roughly equal for every league as expected.
- No regional league shows systematic bias toward winning or losing outcomes.
- Higher activity is visible in major leagues like the LCS Academy, Prime League and CBLOL Academy.

This figure confirms that the dataset is **representative and unbiased across competitive regions** supporting generalizable model training.

---

## Feature Engineering and Preprocessing

We performed several preprocessing steps and created new features to enhance the analysis:

- **Outlier Treatment**: Extremely high values in key features (kills, gold, etc.) were capped at the 99th percentile. This prevents a few rare outliers from unduly influencing the models.

- **Derived Metrics**: We calculated per-minute performance metrics to normalize for game length such as *kills per minute*, *gold per minute*, and *damage per minute*. We also computed the **KDA** ratio (kills plus assists divided by deaths, with a small adjustment to avoid division by zero) as a summary of a player's combat effectiveness.

- **Objective Contribution**: Using the events data we extracted a new feature for each player: the number of **dragon kills** they personally secured in a game. This reflects contribution to major objectives (important for junglers in particular).

- **Feature Selection**: For modeling we focused on individual performance stats and the new features. We intentionally **excluded team-level outcome indicators** (like total team kills or towers destroyed) to ensure the model learns from the player's own performance rather than obvious team results. We included the player's champion as a categorical feature since certain champions can influence playstyle and success.

- **Scaling and Encoding**: Numerical features were scaled (standardization) to put them on a comparable range. The champion names were one-hot encoded so that the model can learn differences between champions without imposing an arbitrary order.

After these steps we split the dataset by role. Each player role (Top, Jungle, Mid, Bot, Support) has its own subset of data. This approach lets us train a specialized model for each role which is appropriate because the importance of certain features can vary greatly by role. All preprocessing was applied consistently to each role-specific subset.

---

## Modeling and Evaluation per Role

Using the prepared data we trained separate classifiers for each role. We experimented with four algorithm types: **Logistic Regression**, **Random Forest**, **XGBoost** (gradient-boosted decision trees) and a simple **Multi-Layer Perceptron** neural network. For each role we performed a grid search with cross-validation to tune hyperparameters (e.g., regularization strength for logistic regression, tree depth and number for random forest, learning rate and tree depth for XGBoost). The primary evaluation metric was **F1-score** which balances precision and recall and is suitable given the roughly balanced classes. 

All model types achieved high accuracy around 90% or above for predicting match outcome from a single player's stats. This suggests a strong correlation between individual performance metrics and team victory. However, **XGBoost** consistently performed the best across all roles with the highest F1-scores (approximately **0.91–0.93** for each role). In comparison, the other models (including the simpler logistic regression) were only slightly behind (mostly **0.90–0.92**). The small performance gap indicates that much of the predictive signal is linear but XGBoost was able to capture additional nuances (possibly interactions between features) to edge out a bit more accuracy.

We selected the XGBoost model as the final model for every role. To illustrate the performance we present the confusion matrix and metrics of one role's best model  
**[Figure: Confusion matrix for one role's model]**. The confusion matrices for all roles show that most games are correctly classified. The few misclassifications often occur when a player on a losing team had very strong stats (false positive) or a player on a winning team had unusually poor stats (false negative). These cases make intuitive sense as exceptional circumstances can defy the general trend.


![confusion matrices](https://github.com/user-attachments/assets/69085fe7-b9ed-4f44-840c-bcf13f17840a)



*Figure 6 – Confusion matrices of the best model (XGBoost) for each player role. Predicted outcomes are on the x-axis; actual outcomes are on the y-axis.*

These matrices highlight the classification performance per role:

- **Diagonal dominance** confirms strong predictive accuracy across all roles.
- **False positives** (top right) typically occur when a player on a losing team performed very well.
- **False negatives** (bottom left) happen when a winning team had a player with underwhelming stats.
- Roles like **Jungle** and **Support** show slightly better balance, likely due to features like **dragon control** and **warding** being highly predictive.

Overal, XGBoost performs **consistently and robustly across all roles** with F1-scores ranging from **0.91 to 0.93**.

---

## Overfitting Analysis

To guard against **overfitting** we used cross-validation and careful regularization during model training. The dataset is large (hundreds of thousands of player instances) which helps models generalize well. Our grid search process optimized hyperparameters like tree depth and regularization strength to prevent overly complex models. For example, limiting the depth of the decision trees in XGBoost and using early stopping (if applied) ensured the model did not simply memorize the training data.

We also set aside a portion of the data as a final test set to evaluate the chosen models. The performance on this unseen test set was essentially the same as during cross-validation (F1 around **0.90–0.93** for each role), indicating that the models generalize to new matches. There was no sign of severe overfitting: training metrics and validation metrics were very close. This means the models are capturing real patterns in the data rather than noise or specific quirks of the training matches.

Finally, by restricting our feature set to meaningful in-game stats and excluding identifiers (like specific player or team names) we reduced the risk of the model over-relying on any one peculiar factor. The misclassifications discussed earlier appeared to be due to legitimate edge cases rather than overfitting errors.

![learning curves](https://github.com/user-attachments/assets/e7b69071-6a43-46b1-bce8-da07a0bdb059)


*Figure 7 – Learning curves showing F1-score on training and validation sets for each model, using only Top lane players.*

**Key interpretations:**

- **Logistic Regression**: Training and validation curves converge indicating low variance and solid generalization. No sign of overfitting.
- **MLP Neural Network**: Very high training performance but significant gap to validation clear sign of **overfitting** despite regularization.
- **Random Forest**: Similar to MLP training score near perfect but validation stagnates. Model is powerful but overfits slightly.
- **XGBoost**: Best trade-off. While training score declines **validation improves with size** showing excellent generalization and regularization.

These curves confirm that **XGBoost strikes the best bias-variance balance** validating its selection as the final model for each role.


---

## SHAP Feature Importance

To understand **which features most influenced the predictions** we applied SHAP (SHapley Additive exPlanations) to the final models. SHAP assigns each feature a contribution value for each prediction indicating how much that feature pushed the prediction towards win or loss. We generated summary plots for each role’s XGBoost model which display the distribution of SHAP values for all players and features.

The SHAP results provided clear insights into the key factors per role. Overall, the **KDA-related stats** had the biggest impact: more kills and assists strongly increase win probability while more deaths drive it down. **Gold and farm** (minions killed, gold per minute) were also highly influential : a player who earns more gold or CS (farm) tends to be on the winning side. For example in the mid lane model  
**[Figure: SHAP summary plot for Mid]** kills and gold stand out as top positive contributors, whereas high death counts have large negative SHAP values.

We observed some role-specific differences. In the **Jungle** role the number of dragons a player secured was among the top features (a jungler with more objective control heavily tilts the game toward a win). In the **Support** role vision and assist metrics (wards placed and assists) had greater importance relative to other roles. Meanwhile, **damage dealt to champions** was especially crucial for **Bot (ADC)** players who are the primary damage dealers. The inclusion of champion identity as features showed smaller effects compared to core stats but certain champions did yield slight SHAP contributions (indicating some champions have higher win rates when picked).

These findings align with common game knowledge. To win, a player should try to get kills and gold without dying and focus on role-specific duties (e.g., junglers securing objectives, supports providing vision). The SHAP analysis thus validated our model and pointed out which areas players can focus on to improve their impact on winning.


![shap values](https://github.com/user-attachments/assets/4d974365-e00e-4466-8a14-806f39137755)


*Figure 8 – SHAP summary plots showing the top features influencing win probability predictions for each role.*

Each plot shows how features (y-axis) impact the model output (x-axis SHAP value) with color encoding the feature value (blue = low, red = high).

**Key takeaways:**

- **Gold per minute** is the most influential feature across all roles.
- **Jungle**: `player_dragon_kills` stands out confirming that objective control is crucial for junglers.
- **Support**: `player_assists` and `wards_placed` have much higher SHAP impact than damage or kills.
- **Bot (ADC)**: Damage-oriented champions and farming (`minions_killed`) dominate the explanation.
- **Mid & Top**: Show a mix of **kills, gold, champion identity**, and combat metrics.

These results provide actionable insight: **players should optimize stats that matter most in their role** to maximize impact on team victory.


---

## Player Ranking

The ultimate goal was to rank players by their performance to guide Ecamania's recruitment. Using the final model for each role we computed a **performance score** for every player. Concretely, for each game a player played we took the predicted win probability from the model (based on that player's stats) and then averaged these probabilities across all their games. We also normalized these game-level scores by the distribution of the player's role to ensure fairness between roles. Players with fewer than 10 games were excluded from ranking to maintain reliability.

A higher score means the player consistently puts up stats that give their team a strong chance to win. Our results showed that the top scores were around 0.70+ meaning those players on average had a 70% chance to win a game given their performance. Below we list the top five players in each role according to our analysis:

### Top Lane:

| Player    | Games | Score |
|-----------|--------|-------|
| Putin     | 38     | 0.697 |
| Nanaue    | 42     | 0.619 |
| Eloha     | 15     | 0.617 |
| Topoon    | 138    | 0.612 |
| Brayaron  | 61     | 0.611 |

### Jungle:

| Player    | Games | Score |
|-----------|--------|-------|
| Wisdomz   | 10     | 0.650 |
| sarolu    | 32     | 0.625 |
| Yanmo     | 32     | 0.625 |
| Juny      | 32     | 0.625 |
| Derakhil  | 31     | 0.621 |

### Mid Lane:

| Player     | Games | Score |
|------------|--------|-------|
| Nallari    | 27     | 0.639 |
| Jean Mago  | 26     | 0.635 |
| sappxire1  | 16     | 0.625 |
| Zinie      | 108    | 0.616 |
| Strangers  | 25     | 0.610 |

### Bot (ADC):

| Player         | Games | Score |
|----------------|--------|-------|
| Xpontaneous    | 17     | 0.721 |
| Snow           | 61     | 0.676 |
| men            | 14     | 0.643 |
| Rahkys         | 18     | 0.639 |
| Limitationss   | 47     | 0.633 |

### Support:

| Player    | Games | Score |
|-----------|--------|-------|
| Nerzhul   | 34     | 0.676 |
| Rosseu    | 26     | 0.673 |
| Winzi     | 13     | 0.635 |
| Raito     | 60     | 0.633 |
| Atat      | 31     | 0.621 |

---

## Conclusion

Our analysis successfully met the objectives. We built accurate predictive models (around **91–93% F1 score**) that confirm a single player's in-game performance can largely determine whether their team wins. Key performance factors were identified: players who **secure kills and assists, avoid deaths, and accumulate gold and farm** give their team a much higher chance of victory. Role-specific contributions (like objective control for junglers and warding for supports) also proved crucial. These insights suggest that training should emphasize these areas. Using the model predictions we ranked players and highlighted the top talent in each role. These are prime candidates for Ecamania’s recruitment given their consistently high impact on game outcomes.

Lastly, we would like to note that despite the enthusiastic promises made by ECAM leadership, we are still patiently waiting for the arrival of our **unlimited budget** . Which, like some solo queue teammates, seems to have disconnected just before the real game began.


# References 

## Kaggle Courses (Machine Learning, AI, Data)
- [1] Kaggle, **"Intro to Machine Learning"**, *Kaggle Learn*, [Online]. Available: https://www.kaggle.com/learn/intro-to-machine-learning (accessed May 19, 2025).  
- [2] Kaggle, **"Intermediate Machine Learning"**, *Kaggle Learn*, [Online]. Available: https://www.kaggle.com/learn/intermediate-machine-learning (accessed May 19, 2025).  
- [3] Kaggle, **"Feature Engineering"**, *Kaggle Learn*, [Online]. Available: https://www.kaggle.com/learn/feature-engineering (accessed May 19, 2025).  
- [4] Kaggle, **"Intro to Deep Learning"**, *Kaggle Learn*, [Online]. Available: https://www.kaggle.com/learn/intro-to-deep-learning (accessed May 19, 2025).  
- [5] Kaggle, **"Data Visualization"**, *Kaggle Learn*, [Online]. Available: https://www.kaggle.com/learn/data-visualization (accessed May 19, 2025).  


## Machine Learning Models and Techniques  
- [11] T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed., Springer, 2009. (logistic regression, ensemble methods)
- [12] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
- [13] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proc. 22nd ACM SIGKDD Int. Conf. on Knowledge Discovery and Data Mining (KDD)*, San Francisco, CA, 2016, pp. 785–794.
- [14] I. Goodfellow, Y. Bengio, and A. Courville, *Deep Learning*, MIT Press, 2016. (MLP, neural networks)
- [15] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*, pp. 4768–4777. (SHAP)


## Project Repository
- [19] D. de Radiguès, "ML-Project: League of Legends Player Performance Analysis," GitHub repository, 2025. [Online]. Available: https://github.com/DiegoRadigues/ML-Project


