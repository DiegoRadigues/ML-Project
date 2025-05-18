# -*- coding: utf-8 -*-
"""
Ecam project from de Radiguès Diego - League of Legends Player Performance Analysis Script
This script performs data loading, exploratory data analysis (EDA), preprocessing, 
role-based modeling, SHAP interpretability, and player ranking for predicting 
victory in League of Legends matches. see all the source code and the final repport
in the GitHub repository: https://github.com/DiegoRadigues/ML-Project
"""

import os
import zipfile
import logging
import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Set up output directory for fig
output_dir = "figures"
os.makedirs(output_dir, exist_ok=True)

# Configure log to file with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
script_dir = os.path.dirname(os.path.abspath(__file__))
log_filename = os.path.join(script_dir, f"log_{timestamp}.txt")
logging.basicConfig(filename=log_filename, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def load_data():
    """Load data from CSV files (extracting from data.zip if necessary) and return dataframes."""
    data_files = ['game_players_stats.csv', 'game_metadata.csv', 'game_events.csv']
    # If data files are not pr, extract from data.zip
    if not all(os.path.exists(os.path.join(script_dir, f)) for f in data_files):
        if os.path.exists(os.path.join(script_dir, 'data.zip')):
            with zipfile.ZipFile(os.path.join(script_dir, 'data.zip'), 'r') as zip_ref:
                zip_ref.extractall(script_dir)
            logging.info("Data files extracted from data.zip.")
        else:
            logging.error("Data files not found and data.zip not found in the script directory.")
            raise FileNotFoundError("Required data files are missing.")
    # Load datasets into pd DataFrames
    df_players = pd.read_csv(os.path.join(script_dir, 'game_players_stats.csv'))
    df_metadata = pd.read_csv(os.path.join(script_dir, 'game_metadata.csv'))
    df_events = pd.read_csv(os.path.join(script_dir, 'game_events.csv'))
    # Log basic info about the datasets
    logging.info(f"Player stats dataset loaded: {df_players.shape[0]} rows, {df_players.shape[1]} columns.")
    logging.info(f"Game metadata dataset shape: {df_metadata.shape}, events dataset shape: {df_events.shape}.")
    logging.info("Columns in player stats dataset: " + ", ".join(df_players.columns))
    # Log small preview of each dataset
    logging.info("\nPreview of player_stats data:\n" + df_players.head(3).to_string(index=False))
    logging.info("\nPreview of game_metadata data:\n" + df_metadata.head(3).to_string(index=False))
    logging.info("\nPreview of game_events data:\n" + df_events.head(3).to_string(index=False))
    return df_players, df_metadata, df_events

def perform_eda(df_players: pd.DataFrame, df_metadata: pd.DataFrame):
    """Perform exploratory data analysis and generate initial visualizations."""
    # Describe player roles and key stat in the dataset
    logging.info("\nDescriptions of roles and key statistics:")
    logging.info("- Top: Top lane player, often engages in 1v1 duels and lane pushing.")
    logging.info("- Jungle: Roams the jungle, controls neutral objectives (dragons, Herald, Baron) and ganks lanes.")
    logging.info("- Mid: Mid lane player, often deals high magic damage and can roam to side lanes.")
    logging.info("- Bot (ADC): Attack damage carry, focuses on farming minions to scale into late game.")
    logging.info("- Support: Assists the ADC, places vision wards, controls vision and protects the team.")
    logging.info("\nKey statistics in the dataset:")
    logging.info("- player_kills: Number of enemy champions killed by the player.")
    logging.info("- player_assists: Number of enemy kills the player assisted with.")
    logging.info("- player_deaths: Number of times the player died.")
    logging.info("- wards_placed: Number of vision wards placed by the player.")
    logging.info("- gold_earned: Total gold earned by the player in the game.")
    logging.info("- total_damage_dealt_to_champions: Damage dealt to enemy champions.")
    logging.info("- win: Indicates if the player's team won (1) or lost (0) the game.")
    # Distrib of target variable 'win'
    logging.info("\nDistribution of game outcomes (wins vs losses):")
    logging.info(df_players['win'].value_counts().to_string())
    plt.figure()
    sns.countplot(x='win', data=df_players, hue='win', palette=['red', 'green'], legend=False)
    plt.title('Distribution of Wins (1) vs Losses (0)')
    plt.xlabel('Game Outcome')
    plt.ylabel('Count of Player-Game instances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dist_win_vs_loss.png'))
    plt.close()
    # Prepare a label for win outcome for plot
    df_players = df_players[df_players['win'].isin([0, 1])]  # drop invalid/missing labels
    df_players['outcome_label'] = df_players['win'].astype(int).map({0: 'Loss', 1: 'Win'})

    palette = {'Win': 'green', 'Loss': 'red'}
    # Distrib of key stats (kills, assists, damage, wards) by game outcome
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.histplot(data=df_players, x='player_kills', hue='outcome_label', multiple='stack',
                 palette=palette, bins=20, ax=axes[0, 0])
    axes[0, 0].set_title('Kills per game (Win vs Loss)')
    axes[0, 0].set_xlabel('Kills')
    axes[0, 0].set_ylabel('Number of players')
    axes[0, 0].legend(title='Game Outcome')
    sns.histplot(data=df_players, x='player_assists', hue='outcome_label', multiple='stack',
                 palette=palette, bins=20, ax=axes[0, 1])
    axes[0, 1].set_title('Assists per game (Win vs Loss)')
    axes[0, 1].set_xlabel('Assists')
    axes[0, 1].set_ylabel('Number of players')
    axes[0, 1].legend(title='Game Outcome')
    sns.histplot(data=df_players, x='total_damage_dealt_to_champions', hue='outcome_label', multiple='stack',
                 palette=palette, bins=20, ax=axes[1, 0])
    axes[1, 0].set_title('Damage to champions (Win vs Loss)')
    axes[1, 0].set_xlabel('Damage to Champions')
    axes[1, 0].set_ylabel('Number of players')
    axes[1, 0].legend(title='Game Outcome')
    sns.histplot(data=df_players, x='wards_placed', hue='outcome_label', multiple='stack',
                 palette=palette, bins=20, ax=axes[1, 1])
    axes[1, 1].set_title('Wards placed (Win vs Loss)')
    axes[1, 1].set_xlabel('Wards Placed')
    axes[1, 1].set_ylabel('Number of players')
    axes[1, 1].legend(title='Game Outcome')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_stats_by_outcome.png'))
    plt.close()
    logging.info("Fig. 'eda_stats_by_outcome.png': Distributions of kills, assists, damage, and wards by game outcome.")
    # Correlation matrix of key numeric stats
    features_for_corr = [
        'player_kills', 'player_deaths', 'player_assists',
        'total_minions_killed', 'gold_earned', 'level',
        'total_damage_dealt', 'total_damage_dealt_to_champions',
        'total_damage_taken', 'wards_placed', 'game_length'
    ]
    corr_matrix = df_players[features_for_corr].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix of Player Statistics")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    logging.info("Heatmap 'correlation_heatmap.png': Correlations between player stats and game length.")
    # Compute per-minute performance stats for comparison
    df_players['game_length_min'] = df_players['game_length'] / 60.0  # game length in minutes
    df_players['kills_per_min'] = df_players['player_kills'] / df_players['game_length_min']
    df_players['gold_per_min'] = df_players['gold_earned'] / df_players['game_length_min']
    df_players['damage_to_champ_per_min'] = df_players['total_damage_dealt_to_champions'] / df_players['game_length_min']
    # Compare distrib of kills/min and gold/min for wins vs losses
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.kdeplot(df_players[df_players['win'] == 1]['kills_per_min'], color='green', label='Win', ax=axes[0])
    sns.kdeplot(df_players[df_players['win'] == 0]['kills_per_min'], color='red', label='Loss', ax=axes[0])
    axes[0].set_title('Kills per minute (Win vs Loss)')
    axes[0].set_xlabel('Kills per minute')
    axes[0].legend()
    sns.kdeplot(df_players[df_players['win'] == 1]['gold_per_min'], color='green', label='Win', ax=axes[1])
    sns.kdeplot(df_players[df_players['win'] == 0]['gold_per_min'], color='red', label='Loss', ax=axes[1])
    axes[1].set_title('Gold per minute (Win vs Loss)')
    axes[1].set_xlabel('Gold per minute')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'norm_stats_win_loss.png'))
    plt.close()
    logging.info("Fig. 'norm_stats_win_loss.png': Distributions of Kills/min and Gold/min by game outcome.")
    # Cap extreme values at 99th percentile for major features to handle outliers
    selected_features = [
        'player_kills', 'player_deaths', 'player_assists',
        'total_minions_killed', 'gold_earned',
        'total_damage_dealt', 'total_damage_dealt_to_champions',
        'total_damage_taken', 'wards_placed',
        'kills_per_min', 'gold_per_min'
    ]
    for feat in selected_features:
        cap_value = df_players[feat].quantile(0.99)
        df_players[feat] = np.where(df_players[feat] > cap_value, cap_value, df_players[feat])
    logging.info("Values above the 99th percentile have been capped for key features to reduce outlier effects.")
    # Example integ of metadata: Wins vs losses by league
    df_meta_games = df_metadata[['game_id', 'league_name']].merge(
        df_players[['game_id', 'win']].drop_duplicates(), on='game_id'
    )
    plt.figure(figsize=(8, 6))
    sns.countplot(x='league_name', hue='win', data=df_meta_games, palette=['red', 'green'])
    plt.title('Wins vs Losses by League')
    plt.ylabel('Number of games')
    plt.xlabel('League')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'wins_by_league.png'))
    plt.close()
    logging.info("Fig. 'wins_by_league.png': Wins/Losses distribution across leagues.")
    # The EDA function modifies df_players (adds new columns, caps values) which will be used in preprocessing
    return df_players

def preprocess_data(df_players: pd.DataFrame, df_events: pd.DataFrame):
    """Perform feature engineering and data preprocessing (new features, derived stats, missing values)."""
    # Add derived feature: KDA (Kill+Assist / Death, with death minimum 1 to avoid division by zero)
    df_players['KDA'] = (df_players['player_kills'] + df_players['player_assists']) / df_players['player_deaths'].replace(0, 1)
    logging.info("Derived feature 'KDA' added (kill+assist divided by deaths).")
    # Extract new feature from events: number of dragons killed by each player in each game
    # Filter dragon kill events and count per game_id and killer_id
    dragon_events = df_events[df_events['event_type'] == 'drake_kill']
    dragons_per_player = dragon_events.groupby(['game_id', 'killer_id']).size().reset_index(name='player_dragon_kills')
    # Merge with players dataframe on game_id and player_id
    df_players = df_players.merge(dragons_per_player, how='left', left_on=['game_id', 'player_id'],
                                  right_on=['game_id', 'killer_id'])
    df_players['player_dragon_kills'] = df_players['player_dragon_kills'].fillna(0)
    df_players.drop(columns=['killer_id'], inplace=True, errors='ignore')  # drop merge key if present
    logging.info("New feature 'player_dragon_kills' added: count of dragons killed by the player.")
    # Define features for modeling: use numerical perf stats and the new feature
    numeric_features = [
        'player_kills', 'player_deaths', 'player_assists',
        'total_minions_killed', 'gold_earned', 'level',
        'total_damage_dealt', 'total_damage_dealt_to_champions',
        'total_damage_taken', 'wards_placed',
        'largest_killing_spree', 'largest_multi_kill',
        'kills_per_min', 'gold_per_min', 'damage_to_champ_per_min',
        'player_dragon_kills'
    ]
    # Include champion_name as a categorical feature
    categorical_features = ['champion_name']
    # Remove any rows with missing values in selected features
    missing_counts = df_players[numeric_features].isna().sum()
    logging.info("\nMissing values per selected feature:\n" + missing_counts.to_string())
    df_players.dropna(subset=numeric_features, inplace=True)
    return df_players, numeric_features, categorical_features

def train_models(df_players: pd.DataFrame, numeric_features: list, categorical_features: list):
    """Train and evaluate models for each role. Returns the best model per role and performance metrics."""
    # Define machine learning pipelines and hyperparameters for each model type
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    model_configs = {
        "Logistic Regression": {
            "pipeline": Pipeline([('preprocessor', preprocessor), ('model', LogisticRegression(max_iter=1000, random_state=42))]),
            "param_grid": {
                'model__C': [0.01, 0.1, 1.0, 10.0],
                'model__penalty': ['l1', 'l2'],
                'model__solver': ['liblinear']
            }
        },
        "Random Forest": {
            "pipeline": Pipeline([('preprocessor', preprocessor), ('model', RandomForestClassifier(random_state=42))]),
            "param_grid": {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5]
            }
        },
        "XGBoost": {
            "pipeline": Pipeline([('preprocessor', preprocessor), ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))]),
            "param_grid": {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 6, 8],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__subsample': [0.8, 1.0]
            }
        },
        "MLP Neural Network": {
            "pipeline": Pipeline([('preprocessor', preprocessor), ('model', MLPClassifier(max_iter=1000, random_state=42))]),
            "param_grid": {
                'model__hidden_layer_sizes': [(100,), (50, 50)],
                'model__alpha': [0.0001, 0.001]
            }
        }
    }
    best_models = {}
    metrics_report = {}
    roles = ['Top', 'Jungle', 'Mid', 'Bot', 'Support']
    for role in roles:
        role_data = df_players[df_players['role'] == role]
        X = role_data[numeric_features + categorical_features]
        y = role_data['win']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        logging.info(f"\n--- Modeling for role: {role} (train={len(X_train)}, test={len(X_test)}) ---")
        best_f1 = -1.0
        best_model_name = None
        best_pipeline = None
        metrics_report[role] = {}
        for model_name, cfg in model_configs.items():
            # Clone a fresh pipeline for each run to avoid reusing fitted preprocessor
            pipeline = cfg["pipeline"]
            param_grid = cfg["param_grid"]
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_
            y_pred = best_estimator.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            metrics_report[role][model_name] = (acc, f1)
            logging.info(f"{model_name}: Acc={acc:.3f}, F1={f1:.3f}, Best Params={grid_search.best_params_}")
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = model_name
                best_pipeline = best_estimator
        logging.info(f"Best model for {role}: {best_model_name} (F1={best_f1:.3f})")
        # Compute and save confusion matrix for the best model
        y_pred_best = best_pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_best)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.title(f'Confusion Matrix ({role} - {best_model_name})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{role}.png'))
        plt.close()
        logging.info(f"Confusion matrix for {role} saved as 'confusion_matrix_{role}.png'.")
        best_models[role] = best_pipeline
    # Summary of model perf by role
    logging.info("\nModel performance by role (Accuracy, F1):")
    for role, metrics in metrics_report.items():
        perf_str = ", ".join([f"{model}: Acc={metrics[model][0]:.2f}, F1={metrics[model][1]:.2f}" for model in metrics])
        logging.info(f"{role}: {perf_str}")
    # Commentary on role performance
    logging.info("\nExample observations:")
    logging.info("- Top and Mid roles often have stronger predictive metrics due to high damage and kill stats.")
    logging.info("- Support role is generally harder to predict (fewer kills, more team-oriented metrics).")
    logging.info("- Jungle performance depends on map objectives; interpretation should consider dragon/baron control.")
    return best_models, metrics_report

def plot_learning_curves(X, y, role_name='All'):
    """Plot and save learning curves for each model type to evaluate overfitting/underfitting."""
    # Define repres model pipelines with chosen hyperparam for learning curve analysis
    preprocessor_for_curve = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    model_examples = {
        "Logistic Regression": Pipeline([('preprocessor', preprocessor_for_curve),
                                         ('model', LogisticRegression(C=0.1, penalty='l1', solver='liblinear', max_iter=1000, random_state=42))]),
        "Random Forest": Pipeline([('preprocessor', preprocessor_for_curve),
                                   ('model', RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, random_state=42))]),
        "XGBoost": Pipeline([('preprocessor', preprocessor_for_curve),
                              ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                                      n_estimators=300, max_depth=3, learning_rate=0.2,
                                                      subsample=0.8, random_state=42))]),
        "MLP Neural Network": Pipeline([('preprocessor', preprocessor_for_curve),
                                        ('model', MLPClassifier(hidden_layer_sizes=(100,), alpha=0.0001, max_iter=1000, random_state=42))])
    }
    for model_name, pipeline in model_examples.items():
        # Compute learning curve
        train_sizes, train_scores, val_scores = learning_curve(pipeline, X, y, cv=3, scoring='f1', 
                                                               train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        # Plot learning curve
        plt.figure()
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training F1')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.1)
        plt.plot(train_sizes, val_mean, 'o-', color='orange', label='Cross-validation F1')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='orange', alpha=0.1)
        plt.title(f"Learning Curve ({model_name} - {role_name} role)")
        plt.xlabel("Training set size")
        plt.ylabel("F1 Score")
        plt.legend(loc="best")
        plt.tight_layout()
        filename = f"learning_curve_{model_name.replace(' ', '_')}_{role_name}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        logging.info(f"Learning curve for {model_name} saved as '{filename}'.")
    logging.info("Learning curves plotted for each model to assess overfitting/underfitting.")

def shap_analysis(best_models: dict, df_players: pd.DataFrame, numeric_features: list, categorical_features: list):
    """Perform global and local SHAP analysis for each role's best model."""
    import shap
    shap.initjs()
    logging.info("\nGlobal SHAP analysis by role:")
    for role, model_pipeline in best_models.items():
        role_data = df_players[df_players['role'] == role]
        X_role = role_data[numeric_features + categorical_features]
        # Sample up to 1000 observations for SHAP analysis to speed up
        X_sample = X_role.sample(n=min(1000, len(X_role)), random_state=42)
        # Separate preprocessor and model if pipeline
        if isinstance(model_pipeline, Pipeline):
            if 'preprocessor' in model_pipeline.named_steps:
                preproc = model_pipeline.named_steps['preprocessor']
                model_obj = model_pipeline.named_steps['model']
                X_sample_transformed = preproc.transform(X_sample)

                if hasattr(X_sample_transformed, "toarray"):
                    X_sample_dense = X_sample_transformed.toarray()
                else:
                    X_sample_dense = X_sample_transformed

                # Get feature names after preprocessing
                try:
                    # Numeric feature names (after standard scaling, same names)
                    numeric_names = numeric_features
                    cat_encoder = preproc.named_transformers_['cat']
                    cat_names = list(cat_encoder.get_feature_names_out(categorical_features)) if categorical_features else []
                    feature_names_full = numeric_names + cat_names
                except AttributeError:
                    # In case get_feature_names_out is not available, fallback to generic names
                    feature_names_full = numeric_features + categorical_features
            elif 'scaler' in model_pipeline.named_steps:  # if pipeline with only scaler
                scaler = model_pipeline.named_steps['scaler']
                model_obj = model_pipeline.named_steps['model']
                X_sample_transformed = scaler.transform(X_sample)
                feature_names_full = numeric_features  # no categorical in this branch
            else:
                model_obj = model_pipeline.named_steps.get('model', model_pipeline)
                X_sample_transformed = X_sample.values
                feature_names_full = numeric_features + categorical_features
        else:
            model_obj = model_pipeline
            X_sample_transformed = X_sample.values
            feature_names_full = numeric_features + categorical_features
        # Choose appropriate SHAP explainer based on model type
        model_type = type(model_obj).__name__
        if model_type == "LogisticRegression":
            explainer = shap.LinearExplainer(model_obj, X_sample_transformed, feature_perturbation="interventional")
            shap_values = np.array(explainer.shap_values(X_sample_transformed))
        elif model_type in ["RandomForestClassifier", "XGBClassifier"]:
            explainer = shap.TreeExplainer(model_obj)
            shap_values = explainer.shap_values(X_sample_dense)  # <- dense version of input

            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # select class 1 (win) SHAP values for binary classifier
        elif model_type == "MLPClassifier":
            # Use model's predict_proba for KernelExplainer
            explainer = shap.KernelExplainer(model_obj.predict_proba, X_sample_transformed[:50])
            shap_values = explainer.shap_values(X_sample_transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # select class 1 SHAP values
        else:
            raise ValueError(f"SHAP explainer not supported for model type {model_type}")
        # Global SHAP summary plot
        

        # Create a DataFrame with feature names to enable colored SHAP plots
        X_shap_df = pd.DataFrame(X_sample_dense, columns=feature_names_full)

        # Global SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values, features=X_shap_df, feature_names=feature_names_full, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_summary_{role}.png"))
        plt.close()


        # Identify top 3 features by mean absolute SHAP value
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top3_idx = mean_abs_shap.argsort()[::-1][:3]
        top3_features = [feature_names_full[i] for i in top3_idx]
        logging.info(f"{role}: Top SHAP features = {top3_features}")
    logging.info("Global SHAP summary plots saved (shap_summary_<Role>.png).")
    # Example local SHAP explanation for a single instance (e.g., Mid role)
    logging.info("\nExample of local SHAP explanation:")
    example_role = 'Mid'
    example_data = df_players[df_players['role'] == example_role]
    # Use the first instance (or random) for explanation
    if not example_data.empty:
        sample_idx = example_data.sample(1, random_state=42).index[0]
        sample_X = example_data.loc[[sample_idx], numeric_features + categorical_features]
        best_pipeline = best_models[example_role]
        # Get the model object and transform sample using preprocessor if needed
        if isinstance(best_pipeline, Pipeline):
            if 'preprocessor' in best_pipeline.named_steps:
                preproc = best_pipeline.named_steps['preprocessor']
                model_obj = best_pipeline.named_steps['model']
                sample_X_transformed = preproc.transform(sample_X)
                if hasattr(sample_X_transformed, 'toarray'):
                    sample_X_transformed = sample_X_transformed.toarray()

            elif 'scaler' in best_pipeline.named_steps:
                scaler = best_pipeline.named_steps['scaler']
                model_obj = best_pipeline.named_steps['model']
                sample_X_transformed = scaler.transform(sample_X)
            else:
                model_obj = best_pipeline.named_steps.get('model', best_pipeline)
                sample_X_transformed = sample_X.values
        else:
            model_obj = best_pipeline
            sample_X_transformed = sample_X.values
        # Predict probability of win for the sample
        pred_prob = model_obj.predict_proba(sample_X_transformed)[0][1]
        logging.info(f"Predicted win probability for example ({example_role}) player: {pred_prob:.3f}")
        # Compute SHAP values for the single instance
        model_type = type(model_obj).__name__
        if model_type == "LogisticRegression":
            explainer_local = shap.LinearExplainer(model_obj, sample_X_transformed, feature_perturbation="interventional")
            shap_values_local = explainer_local.shap_values(sample_X_transformed)[0]
        elif model_type in ["RandomForestClassifier", "XGBClassifier"]:
            explainer_local = shap.TreeExplainer(model_obj)
            shap_values_local = explainer_local.shap_values(sample_X_transformed)
            if isinstance(shap_values_local, list):
                shap_values_local = shap_values_local[1][0]
            else:
                shap_values_local = shap_values_local[0]
        elif model_type == "MLPClassifier":
            explainer_local = shap.KernelExplainer(model_obj.predict_proba, sample_X_transformed)
            shap_values_list = explainer_local.shap_values(sample_X_transformed)
            shap_values_local = shap_values_list[1][0] if isinstance(shap_values_list, list) else shap_values_list[0]
        else:
            raise ValueError(f"SHAP explainer not supported for model type {model_type}")
        
        
        # Construct SHAP Explanation for waterfall plot
        base_val = explainer_local.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = base_val[1] if isinstance(base_val, list) else base_val[1]

        # Adjust feature names to match the number of SHAP values
        feature_names_adjusted = feature_names_full[:len(shap_values_local)]

        shap_explanation = shap.Explanation(
            values=shap_values_local,
            base_values=base_val,
            data=sample_X_transformed[0].toarray() if hasattr(sample_X_transformed[0], 'toarray') else sample_X_transformed[0],
            feature_names=feature_names_adjusted
        )

        # Plot and save waterfall chart
        plt.figure(figsize=(8, 4))
        try:
            shap.plots.waterfall(shap_explanation, show=False)
            plt.title(f"Local SHAP Waterfall - {example_role}")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_local_example_{example_role}.png"))
            plt.close()
            logging.info(f"Local SHAP explanation saved as 'shap_local_example_{example_role}.png'.")
        except Exception as e:
            logging.warning(f"Local SHAP waterfall plot failed for {example_role}: {e}")



def rank_players(df_players: pd.DataFrame):
    """Rank players based on performance percentile and output top players overall and per role."""
    min_games = 10  # minimum games threshold for ranking
    # If performance_percentile not already present, compute it (rank win within each role as a percentile)
    if 'performance_percentile' not in df_players.columns:
        df_players['performance_percentile'] = df_players.groupby('role')['win'].rank(pct=True)
    # Aggregate statistics per player
    player_stats = df_players.groupby('player_name').agg(
        num_games=('game_id', 'count'),
        mean_score=('performance_percentile', 'mean'),
        median_score=('performance_percentile', 'median'),
        std_score=('performance_percentile', 'std')
    ).reset_index()
    # Determine each player's primary role (most frequent role played)
    primary_roles = df_players.groupby('player_name')['role'] \
        .agg(lambda x: x.value_counts().index[0]).reset_index() \
        .rename(columns={'role': 'primary_role'})
    player_stats = player_stats.merge(primary_roles, on='player_name')
    # Filter players with at least min_games
    player_stats = player_stats[player_stats['num_games'] >= min_games]
    # Sort by mean performance score
    player_stats.sort_values('mean_score', ascending=False, inplace=True)
    # Top 10 players overall
    logging.info(f"\nTop 10 players (>= {min_games} games):")
    logging.info(player_stats[['player_name', 'primary_role', 'num_games', 'mean_score']].head(10).to_string(index=False))
    # Top 5 players by primary role
    roles = ['Top', 'Jungle', 'Mid', 'Bot', 'Support']
    for role in roles:
        subset = player_stats[player_stats['primary_role'] == role].head(5)
        logging.info(f"\nTop 5 players for role {role} (>= {min_games} games):")
        if not subset.empty:
            logging.info(subset[['player_name', 'num_games', 'mean_score']].to_string(index=False))
        else:
            logging.info("No player with the minimum number of games for this role.")
    return player_stats

if __name__ == "__main__":
    # 1. Data Loading
    df_players, df_metadata, df_events = load_data()
    # 2. Exploratory Data Analysis (EDA)
    df_players = perform_eda(df_players, df_metadata)
    # 3. Preprocessing and Feature Engineering
    df_players, numeric_features, categorical_features = preprocess_data(df_players, df_events)
    # 4. Modeling (train and evaluate models per role)
    best_models, metrics_report = train_models(df_players, numeric_features, categorical_features)
    # 5. Learning Curves for model overfitting/underfitting analysis (using Top role data)
    top_data = df_players[df_players['role'] == 'Top']
    X_top = top_data[numeric_features + categorical_features]
    y_top = top_data['win']
    plot_learning_curves(X_top, y_top, role_name='Top')
    # 6. Model Interpretability with SHAP
    shap_analysis(best_models, df_players, numeric_features, categorical_features)
    # 7. Player Performance Ranking
    player_stats = rank_players(df_players)
    logging.info(f"Log file created: {log_filename}")
    logging.info(f"Script completed. All figures saved in '{os.path.abspath(output_dir)}'. See log for details.")
