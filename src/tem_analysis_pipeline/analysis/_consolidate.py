"""
Module for consolidating analysis results from individual CSV files into a single file.
"""
from pathlib import Path

import pandas as pd


def consolidate_study_results(
    study_name: str, model_name: str, organelle: str, output_file: str = None
) -> pd.DataFrame:
    """
    Consolidate all CSV analysis results for a study into a single DataFrame and optionally save to file.

    Args:
        study_name: Name of the study to consolidate results for
        model_name: Name of the model used for predictions
        organelle: Name of the organelle analyzed
        output_file: Optional path to save the consolidated CSV file

    Returns:
        DataFrame containing all consolidated results
    """
    studies_basedir = Path("studies")
    study_dir = studies_basedir / study_name
    
    if not study_dir.exists():
        raise FileNotFoundError(f"Study directory not found: {study_dir}")
    
    all_dfs = []
    
    # Iterate through all condition directories in the study
    for condition_dir in study_dir.iterdir():
        if not condition_dir.is_dir():
            continue
        
        condition_name = condition_dir.name
        predictions_dir = condition_dir / "prediction" / model_name / organelle
        
        if not predictions_dir.exists():
            print(f"No predictions found for condition {condition_name}")
            continue

        # Find all CSV files for this condition
        csv_files = sorted(predictions_dir.glob(f"*-{organelle}.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Add condition information
                df.insert(0, "condition", condition_name)
                all_dfs.append(df)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    if not all_dfs:
        print(f"No CSV files found for study {study_name}, model {model_name}, organelle {organelle}")
        return pd.DataFrame()
    
    # Combine all dataframes
    consolidated_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        consolidated_df.to_csv(output_path, index=False)
        print(f"Consolidated results saved to {output_path}")
    
    return consolidated_df
