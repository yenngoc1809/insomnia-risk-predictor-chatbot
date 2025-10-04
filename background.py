# background.py
import pandas as pd, joblib
from pathlib import Path

CSV_PATH = "data/Student Insomnia and Educational Outcomes Dataset_version-2.csv"
ENC_PATH = "models/encoder.pkl"
OUT_PATH = "data/background.csv"
SAMPLE_N = 400

# === Mapping from original survey question headers to the model's feature names ===
# These mappings must match exactly with the survey column headers found in the CSV.
COL_MAP = {
    "1. What is your year of study?": "Year_of_Study",
    "2. What is your gender?": "Gender",
    "3. How often do you have difficulty falling asleep at night? ": "Difficulty_Falling_Asleep",  # Note: trailing space
    "4. On average, how many hours of sleep do you get on a typical day?": "Sleep_Hours",
    "5. How often do you wake up during the night and have trouble falling back asleep?": "Wakeup_Difficulty",
    "6. How would you rate the overall quality of your sleep?": "Sleep_Quality",
    "7. How often do you experience difficulty concentrating during lectures or studying due to lack of sleep?": "Concentration_Problems",
    "8. How often do you feel fatigued during the day, affecting your ability to study or attend classes?": "Daytime_Fatigue",
    "9. How often do you miss or skip classes due to sleep-related issues (e.g., insomnia, feeling tired)?": "Missed_Classes",
    "10. How would you describe the impact of insufficient sleep on your ability to complete assignments and meet deadlines?": "Impact_on_Deadlines",
    "11. How often do you use electronic devices (e.g., phone, computer) before going to sleep?": "Device_Use_Before_Sleep",
    "12. How often do you consume caffeine (coffee, energy drinks) to stay awake or alert?": "Caffeine_Consumption",
    "13. How often do you engage in physical activity or exercise?": "Physical_Activity",
    "14. How would you describe your stress levels related to academic workload?": "Stress_Levels",
    # Column 15 (GPA or grades) is not required for the model and will be dropped
}

def main():
    # Step 1: Load the encoder to retrieve the exact FEATURES order used during model training
    encoder = joblib.load(ENC_PATH)
    FEATURES = list(getattr(encoder, "feature_names_in_", []))
    if not FEATURES:
        raise RuntimeError("encoder.feature_names_in_ is empty. Verify models/encoder.pkl")

    # Step 2: Load the raw CSV file
    df_raw = pd.read_csv(CSV_PATH)
    print("CSV columns:", df_raw.columns.tolist())

    # Step 3: Rename survey columns to match the model features
    rename_dict = {}
    for old, new in COL_MAP.items():
        if old in df_raw.columns:
            rename_dict[old] = new
        else:
            # Attempt soft matching if there are trailing or leading spaces
            candidates = [c for c in df_raw.columns if c.strip() == old.strip()]
            if candidates:
                rename_dict[candidates[0]] = new

    df_renamed = df_raw.rename(columns=rename_dict)

    # Drop columns that are not used by the model
    drop_cols = [
        "Timestamp",
        "15. How would you rate your overall academic performance (GPA or grades) in the past semester?"
    ]
    for c in drop_cols:
        if c in df_renamed.columns:
            df_renamed = df_renamed.drop(columns=c)

    # Step 4: Validate that all required FEATURES are present after renaming
    missing = [f for f in FEATURES if f not in df_renamed.columns]
    if missing:
        print("Missing features after renaming, not found in CSV:", missing)
        print("Check if the survey column headers match exactly (note the trailing space in question 3).")
        return

    # Step 5: Select only the required FEATURES, drop rows with missing values,
    # and take a random sample of SAMPLE_N rows (or fewer if not enough data)
    df_bg = df_renamed[FEATURES].dropna()
    take = min(SAMPLE_N, len(df_bg))
    if take == 0:
        raise RuntimeError("After dropna() no rows remain. Verify CSV data quality.")
    df_bg = df_bg.sample(take, random_state=42)

    # Step 6: Save the background dataset
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    df_bg.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH} with shape: {df_bg.shape}")
    print("FEATURES order used:", FEATURES)

if __name__ == "__main__":
    main()
