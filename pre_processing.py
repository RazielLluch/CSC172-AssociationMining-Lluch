"""
preprocessing.py
-----------------
Barebones data preprocessing module for the Association Mining Project.
Uses pandas DataFrames for all dataset handling.

Dataset format: Excel (.xlsx) stored in the 'data/' directory.
"""

import pandas as pd
import numpy as np
import os


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from an Excel file.

    Parameters
    ----------
    filepath : str
        Path to the dataset file (Excel).

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] File not found: {filepath}")

    try:
        df = pd.read_excel(filepath)
        print(f"[INFO] Excel dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load Excel dataset: {e}")
        raise

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning operations such as removing duplicates
    and trimming whitespace. More steps will be added later.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    df = df.copy()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove duplicate rows
    df = df.drop(columns=["How often do you play video games?", 
                          "How many hours do you typically spend gaming in a week?", 
                          "What is your favorite game?", 
                          "Do you prefer single-player or multiplayer games?",
                          "What genres of video games do you play? (Check all that apply)",
                          "Do you prefer single-player or multiplayer games?",
                          "How much do you spend on gaming monthly (including in-game purchases, new games, etc.)?",
                          "Which device do you play games on the most?(Check all that apply)",
                          "How do you discover new games? (Check all that apply)",
                          "Why do you play video games? (Check all that apply)"])

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include=["object"]).columns
    df.columns = df.columns.str.strip()

    print("[INFO] Basic cleaning applied.")
    return df

def remove_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove the 'timestamp' column if it exists.
    """
    df = df.copy()
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])
        print("[INFO] 'timestamp' column removed.")
    else:
        print("[INFO] 'timestamp' column not found. Skipping.")
    return df

def categorize_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the numerical Age column into categorical groups and then into binary columns.
    Assumes Age values range roughly from 15 to 35.
    """
    df = df.copy()

    if "Age" not in df.columns:
        print("[WARNING] 'Age' column not found. Skipping age categorization.")
        return df

    # Define bins and labels
    bins = [14, 18, 22, 26, 35]
    labels = ["Teen", "Young_Adult", "Adult", "Mid_Adult"]

    # Categorize
    df["Age_Category"] = pd.cut(df["Age"], bins=bins, labels=labels)

    # Drop original Age column
    df = df.drop(columns=["Age"])

    # Convert to binary columns
    for label in labels:
        df[f"Age_{label}"] = (df["Age_Category"] == label).astype(int)

    # Drop intermediate categorical column
    df = df.drop(columns=["Age_Category"])

    print(f"[INFO] Age column converted to binary columns -> {', '.join(['Age_' + l for l in labels])}.")
    return df

def extract_city_state(location_value: str) -> str:
    """
    Extract the city/state from a messy location string.
    Strategy:
    - Split by comma
    - Take the last non-empty meaningful token
    - Fallback: return the string as-is if extraction fails
    """
    if not isinstance(location_value, str):
        return location_value

    # Split by comma
    parts = [p.strip() for p in location_value.split(",") if p.strip()]

    if len(parts) == 0:
        return location_value

    # Use the last component (most likely city or state)
    return parts[-1]

def clean_location(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the Location column to a binary format suitable for association mining.
    Converts all locations to either 'India', 'US', or 'Other'.
    """
    df = df.copy()

    if "Location" not in df.columns:
        print("[WARNING] 'Location' column not found. Skipping location cleaning.")
        return df

    # Replace known exceptions
    df["Location"] = df["Location"].replace({"Jain University": "Karnataka"})

    # Extract city/state as before
    def extract_city_state(location_value: str) -> str:
        if not isinstance(location_value, str):
            return "Other"
        parts = [p.strip() for p in location_value.split(",") if p.strip()]
        return parts[-1] if parts else "Other"

    df["Location_Clean"] = df["Location"].apply(extract_city_state)

    # Map to simplified countries
    india_keywords = ["Bangalore", "Karnataka", "Odisha", "Hyderabad", "Chennai", 
                      "Delhi", "Mumbai", "Pune", "Ahmedabad", "Bhubaneswar", "Kolkata"]
    us_keywords = ["California", "Florida", "Ohio", "Texas", "New York"]

    def map_to_country(location):
        if location in india_keywords:
            return "India"
        elif location in us_keywords:
            return "US"
        else:
            return "Other"

    df["Location_Clean"] = df["Location_Clean"].apply(map_to_country)

    # Drop original Location column
    df = df.drop(columns=["Location"])

    # Convert to binary columns for Apriori
    df["Location_India"] = (df["Location_Clean"] == "India").astype(int)
    df["Location_US"] = (df["Location_Clean"] == "US").astype(int)
    df["Location_Other"] = (df["Location_Clean"] == "Other").astype(int)

    # Drop the intermediate column
    df = df.drop(columns=["Location_Clean"])

    print("[INFO] Location column converted to binary columns -> 'Location_India', 'Location_US', 'Location_Other'.")
    return df

def clean_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the Gender column to only have 'Male', 'Female', or 'Other'.
    Capitalization variations are normalized.
    Any invalid or missing values are set to 'Other'.
    """
    df = df.copy()

    if "Gender" not in df.columns:
        print("[WARNING] 'Gender' column not found. Skipping gender cleaning.")
        return df

    # Normalize capitalization
    df["Gender"] = df["Gender"].str.title()

    # Define valid categories
    valid_genders = ["Male", "Female", "Other"]

    # Replace invalid or missing values with 'Other'
    df["Gender"] = df["Gender"].apply(lambda x: x if x in valid_genders else "Other")

    print("[INFO] Gender column cleaned.")
    return df

def encode_gender_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Gender column into binary columns:
    Gender_Male, Gender_Female, Gender_Other
    Suitable for Apriori association mining.
    """
    df = df.copy()

    if "Gender" not in df.columns:
        print("[WARNING] 'Gender' column not found. Skipping gender encoding.")
        return df

    # Normalize values
    series = (
        df["Gender"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "male": "Male",
        "m": "Male",
        "female": "Female",
        "f": "Female",
        "other": "Other",
        "prefer not to say": "Other"
    }

    normalized = series.replace(mapping)

    # One-hot encode
    gender_dummies = pd.get_dummies(
        normalized,
        prefix="Gender"
    )

    # Ensure all expected columns exist
    for col in ["Gender_Male", "Gender_Female", "Gender_Other"]:
        if col not in gender_dummies.columns:
            gender_dummies[col] = 0

    # Convert to boolean (important for Apriori clarity)
    gender_dummies = gender_dummies.astype(bool)

    # Drop original column and concat
    df = df.drop(columns=["Gender"])
    df = pd.concat([df, gender_dummies], axis=1)

    print("[INFO] Gender column converted to binary columns.")
    return df

def clean_gaming_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the 'How often do you play video games?' column
    and convert it into binary columns suitable for Apriori.
    """
    df = df.copy()

    original_col = "How often do you play video games?"
    new_col = "Gaming_Freq"

    if original_col not in df.columns:
        print(f"[WARNING] '{original_col}' column not found. Skipping gaming frequency cleaning.")
        return df

    # Standardize categories
    mapping = {
        "Daily": "Daily",
        "Weekly": "Weekly",
        "A few times in a week": "Weekly",
        "A few times in a month": "Monthly",
        "Rarely/Never": "Rarely/Never"
    }

    df[new_col] = df[original_col].map(mapping)
    df[new_col] = df[new_col].fillna("Rarely/Never")

    # Drop the old column
    df = df.drop(columns=[original_col])

    # Convert to binary columns
    categories = ["Daily", "Weekly", "Monthly", "Rarely/Never"]
    for cat in categories:
        col_name = f"Gaming_{cat.replace('/', '_')}"
        df[col_name] = (df[new_col] == cat).astype(int)

    # Drop the intermediate column
    df = df.drop(columns=[new_col])

    print(f"[INFO] '{original_col}' converted to binary columns -> {', '.join(['Gaming_' + c.replace('/', '_') for c in categories])}.")
    return df

def clean_gaming_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the 'How many hours do you typically spend gaming in a week?'
    column into concise categories and rename to 'Gaming_Hours'.
    """
    df = df.copy()

    original_col = "How many hours do you typically spend gaming in a week?"
    new_col = "Gaming_Hours"

    if original_col not in df.columns:
        print(f"[WARNING] '{original_col}' column not found. Skipping gaming hours cleaning.")
        return df

    # Pop the column to get a Series
    series = df.pop(original_col)

    # Mapping inputs to categories
    mapping = {
        0: "0-1 hour",
        "30mins": "0-1 hour",
        "Less than 5 hours": "1-5 hours",
        "5-10 hours": "5-10 hours",
        "10-20 hours": "10-20 hours",
        "More than 20 hours": "20+ hours"
    }

    df[new_col] = series.replace(mapping)

    # Fill any unexpected values with "Unknown"
    df[new_col] = df[new_col].fillna("Unknown")

    print(f"[INFO] '{original_col}' column cleaned and renamed -> '{new_col}'.")
    return df

def encode_gaming_hours_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'Gaming_Hours' categorical column into binary indicator columns.
    Suitable for association rule mining.
    """
    df = df.copy()

    col = "Gaming_Hours"

    if col not in df.columns:
        print(f"[WARNING] '{col}' column not found. Skipping binary encoding.")
        return df

    categories = [
        "0-1 hour",
        "1-5 hours",
        "5-10 hours",
        "10-20 hours",
        "20+ hours"
    ]

    for category in categories:
        binary_col = f"Gaming_Hours_{category.replace(' ', '_').replace('+', 'plus')}"
        df[binary_col] = (df[col] == category).astype(int)

    # Optional: drop the original categorical column
    df = df.drop(columns=[col])

    print("[INFO] 'Gaming_Hours' converted to binary columns.")
    return df

def clean_device_used(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a 'Check all that apply' devices column into multiple binary columns.
    Renames the column to 'Device_Used'.
    """
    df = df.copy()
    original_col = "Which device do you play games on the most?(Check all that apply)"
    new_col = "Device_Used"

    if original_col not in df.columns:
        print(f"[WARNING] '{original_col}' column not found. Skipping device cleaning.")
        return df

    # Pop original column
    series = df.pop(original_col)

    # Create empty DataFrame for one-hot encoding
    devices_list = []

    for val in series:
        if pd.isna(val):
            devices_list.append([])
        else:
            # Split by comma and strip whitespace
            items = [x.strip() for x in val.split(",")]
            # Normalize names
            normalized = []
            for x in items:
                if "Console" in x:
                    normalized.append("Console")
                elif "Handheld" in x:
                    normalized.append("Handheld")
                elif "PC" in x:
                    normalized.append("PC")
                elif "Mobile" in x:
                    normalized.append("Mobile")
                elif "Tablet" in x:
                    normalized.append("Tablet")
            devices_list.append(normalized)

    # Generate binary columns
    all_devices = ["PC", "Mobile", "Console", "Handheld", "Tablet"]
    for device in all_devices:
        df[f"Device_{device}"] = [1 if device in x else 0 for x in devices_list]

    print(f"[INFO] '{original_col}' column cleaned and expanded into one-hot device columns.")
    return df

def clean_game_genres(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a 'Check all that apply' genres column into multiple binary columns.
    Renames the column to 'Game_Genres' internally for processing.
    """
    df = df.copy()
    original_col = "What genres of video games do you play? (Check all that apply)"
    new_col = "Game_Genres"

    if original_col not in df.columns:
        print(f"[WARNING] '{original_col}' column not found. Skipping game genres cleaning.")
        return df

    # Pop the column to get a Series
    series = df.pop(original_col)

    genres_list = []

    for val in series:
        if pd.isna(val):
            genres_list.append([])
        else:
            # Split by comma and strip whitespace
            items = [x.strip() for x in val.split(",")]
            normalized = []
            for x in items:
                if "Action/Adventure" in x:
                    normalized.append("Action/Adventure")
                elif "FPS" in x:
                    normalized.append("FPS")
                elif "Role-Playing" in x or "RPG" in x:
                    normalized.append("RPG")
                elif "Puzzle" in x or "Strategy" in x:
                    normalized.append("Puzzle/Strategy")
                elif "Simulation" in x:
                    normalized.append("Simulation")
                elif "MMO" in x:
                    normalized.append("MMO")
                elif "Sports" in x:
                    normalized.append("Sports")
            genres_list.append(normalized)

    # Define all possible genres
    all_genres = ["Action/Adventure", "FPS", "RPG", "Puzzle/Strategy", "Simulation", "MMO", "Sports"]

    # Create one-hot encoded columns
    for genre in all_genres:
        df[f"Genre_{genre}"] = [1 if genre in x else 0 for x in genres_list]

    print(f"[INFO] '{original_col}' column cleaned and expanded into one-hot genre columns.")
    return df

def clean_favorite_game(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the 'What is your favorite game?' column as categorical
    and rename it to 'Favorite_Game'.
    """
    df = df.copy()
    original_col = "What is your favorite game?"
    new_col = "Favorite_Game"

    if original_col not in df.columns:
        print(f"[WARNING] '{original_col}' column not found. Skipping favorite game cleaning.")
        return df

    # Strip whitespace and lowercase
    series = df.pop(original_col).astype(str).str.strip().str.lower()

    # Map known duplicates to standard names
    mapping = {
        "call of duty": "Call of Duty",
        "bgmi": "BGMI",
        "bgmi, coc, chess": "BGMI / COC / Chess",
        "solo leveling arise": "Solo Leveling",
        "solo levelling": "Solo Leveling",
        "efootball": "Efootball",
        "fc mobile": "FC Mobile",
        "wukong": "Wukong",
        "fornite": "Fortnite",
        "wuthering waves": "Wuthering Waves",
        "wuther waves": "Wuthering Waves",
        "rhythm rush lite": "Rhythm Rush Lite",
        "red dead redemption 2": "Red Dead Redemption 2",
        "chess and clash of clans": "Chess / Clash of Clans",
        "god of war ragnarok": "God of War Ragnarok",
    }

    df[new_col] = series.replace(mapping)

    # Fill empty or unknown entries with 'Unknown'
    df[new_col] = df[new_col].replace({"": "Unknown"})
    df[new_col] = df[new_col].fillna("Unknown")

    print(f"[INFO] '{original_col}' column cleaned and renamed -> '{new_col}'.")
    return df

def encode_favorite_games_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'Favorite_Game' column into binary columns for each game.
    Handles combined entries (e.g., 'BGMI / COC / Chess').
    """
    df = df.copy()

    col = "Favorite_Game"

    if col not in df.columns:
        print(f"[WARNING] '{col}' column not found. Skipping favorite game encoding.")
        return df

    # Split combined values into lists
    games_series = (
        df[col]
        .astype(str)
        .str.split(r"\s*/\s*")
    )

    # Get unique game names (excluding Unknown)
    unique_games = sorted({
        game
        for games in games_series
        for game in games
        if game != "Unknown"
    })

    # Create binary columns
    for game in unique_games:
        safe_name = (
            game.lower()
            .replace(" ", "_")
            .replace("+", "plus")
        )
        binary_col = f"Favorite_Game_{safe_name}"

        df[binary_col] = games_series.apply(lambda x: int(game in x))

    # Optional: keep Unknown as its own flag
    df["Favorite_Game_unknown"] = (df[col] == "Unknown").astype(int)

    # Drop original categorical column
    df = df.drop(columns=[col])

    print("[INFO] 'Favorite_Game' converted to binary columns.")
    return df

def clean_game_discovery(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a 'Check all that apply' game discovery column into multiple binary columns.
    """
    df = df.copy()
    original_col = "How do you discover new games? (Check all that apply)"
    
    if original_col not in df.columns:
        print(f"[WARNING] '{original_col}' column not found. Skipping game discovery cleaning.")
        return df

    # Pop the column to get a Series
    series = df.pop(original_col)

    discovery_list = []

    for val in series:
        if pd.isna(val):
            discovery_list.append([])
        else:
            # Split by comma and strip whitespace
            items = [x.strip() for x in val.split(",")]
            normalized = []
            for x in items:
                if "Social Media" in x:
                    normalized.append("Social_Media")
                elif "Gaming Forums" in x:
                    normalized.append("Gaming_Forums")
                elif "Friends/Family" in x:
                    normalized.append("Friends_Family")
                elif "Game Reviews" in x or "Blogs" in x:
                    normalized.append("Game_Reviews")
                elif "YouTube" in x or "Streaming" in x or "Twitch" in x:
                    normalized.append("YouTube_Streaming")
                elif "I search" in x or "my own ways" in x:
                    normalized.append("Self_Search")
            discovery_list.append(normalized)

    # Define all possible discovery methods
    all_methods = ["Social_Media", "Gaming_Forums", "Friends_Family", "Game_Reviews", "YouTube_Streaming", "Self_Search"]

    # Create one-hot encoded columns
    for method in all_methods:
        df[f"Discovery_{method}"] = [1 if method in x else 0 for x in discovery_list]

    print(f"[INFO] '{original_col}' column cleaned and expanded into one-hot discovery columns.")
    return df

def clean_game_mode_pref(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the 'Do you prefer single-player or multiplayer games?' column
    and convert it into binary columns suitable for Apriori.
    """
    df = df.copy()
    original_col = "Do you prefer single-player or multiplayer games?"
    new_col = "Game_Mode_Pref"

    if original_col not in df.columns:
        print(f"[WARNING] '{original_col}' column not found. Skipping game mode preference cleaning.")
        return df

    # Standardize values
    series = df.pop(original_col).astype(str).str.strip().str.lower()

    mapping = {
        "single-player": "Single-Player",
        "single player": "Single-Player",
        "multiplayer": "Multiplayer",
        "multi-player": "Multiplayer",
        "both": "Both"
    }

    df[new_col] = series.replace(mapping)
    df[new_col] = df[new_col].fillna("Unknown")
    df[new_col] = df[new_col].replace({"": "Unknown"})

    # Convert to binary columns
    categories = ["Single-Player", "Multiplayer", "Both", "Unknown"]
    for cat in categories:
        col_name = f"Game_Mode_{cat.replace('-', '_')}"
        df[col_name] = (df[new_col] == cat).astype(int)

    # Drop the intermediate column
    df = df.drop(columns=[new_col])

    print(f"[INFO] '{original_col}' converted to binary columns -> {', '.join(['Game_Mode_' + c.replace('-', '_') for c in categories])}.")
    return df

def clean_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the monthly gaming spend column into discrete categories
    and convert it into binary columns suitable for Apriori.
    """
    df = df.copy()
    original_col = "How much do you spend on gaming monthly (including in-game purchases, new games, etc.)?"
    new_col = "Monthly_Spend"

    if original_col not in df.columns:
        print(f"[WARNING] '{original_col}' column not found. Skipping monthly spend cleaning.")
        return df

    # Standardize values
    series = df.pop(original_col).astype(str).str.strip()

    mapping = {
        "Less than ₹100": "<100",
        "₹100-500": "100-500",
        "₹500-1000": "500-1000",
        "₹1000 and above": "1000+",
        "More than ₹1000": "1000+"
    }

    df[new_col] = series.replace(mapping)
    df[new_col] = df[new_col].fillna("Unknown")

    # Convert to binary columns
    categories = ["<100", "100-500", "500-1000", "1000+", "Unknown"]
    for cat in categories:
        col_name = f"Spend_{cat.replace('+', 'plus').replace('<', 'lt')}"
        df[col_name] = (df[new_col] == cat).astype(int)

    # Drop intermediate column
    df = df.drop(columns=[new_col])

    print(f"[INFO] '{original_col}' converted to binary columns -> {', '.join(['Spend_' + c.replace('+', 'plus').replace('<', 'lt') for c in categories])}.")
    return df

def clean_play_reason(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a 'Check all that apply' column for reasons to play games
    into multiple binary columns.
    """
    df = df.copy()
    original_col = "Why do you play video games? (Check all that apply)"

    if original_col not in df.columns:
        print(f"[WARNING] '{original_col}' column not found. Skipping play reason cleaning.")
        return df

    # Pop the column to get a Series
    series = df.pop(original_col)

    reasons_list = []

    for val in series:
        if pd.isna(val):
            reasons_list.append([])
        else:
            # Split by comma and strip whitespace
            items = [x.strip() for x in val.split(",")]
            normalized = []
            for x in items:
                if "fun" in x.lower():
                    normalized.append("Fun")
                elif "stress" in x.lower():
                    normalized.append("Stress_Relief")
                elif "improve skills" in x.lower() or "competition" in x.lower():
                    normalized.append("Skills_Competition")
                elif "socialize" in x.lower():
                    normalized.append("Socialize")
                elif "learning" in x.lower():
                    normalized.append("Learning")
                elif "if no other better work" in x.lower():
                    normalized.append("Other")
            reasons_list.append(normalized)

    # Define all possible reasons
    all_reasons = ["Fun", "Stress_Relief", "Skills_Competition", "Socialize", "Learning", "Other"]

    # Create one-hot encoded columns
    for reason in all_reasons:
        df[f"Reason_{reason}"] = [1 if reason in x else 0 for x in reasons_list]

    print(f"[INFO] '{original_col}' column cleaned and expanded into one-hot reason columns.")
    return df


def preprocess_pipeline(filename: str) -> pd.DataFrame:
    filepath = os.path.join("data", filename)

    df = load_dataset(filepath)
    df = basic_cleaning(df)
    df = remove_timestamp(df)
    df = categorize_age(df)
    df = clean_location(df)
    df = clean_gender(df)
    df = encode_gender_binary(df)
    df = clean_gaming_frequency(df)
    df = clean_gaming_hours(df)
    df = encode_gaming_hours_binary(df)
    df = clean_device_used(df)
    df = clean_game_genres(df)
    df = clean_favorite_game(df)
    df = encode_favorite_games_binary(df)
    df = clean_game_discovery(df)
    df = clean_game_mode_pref(df)
    df = clean_monthly_spend(df)
    df = clean_play_reason(df)
    

    print("[INFO] Preprocessing pipeline complete.")
    return df


if __name__ == "__main__":
    ## Example usage ##
    dataset_path = r"Updated_Gaming_Survey_Responses.xlsx"
    df = preprocess_pipeline(dataset_path)