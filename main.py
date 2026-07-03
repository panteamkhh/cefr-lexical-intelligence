from pathlib import Path
from src.data_cleaning import clean_dataset

base_dir = Path(__file__).resolve().parent

input_file = base_dir / "data" / "raw" / "toefl_vocabulary_cleaned_categorized.xlsx"
output_file = base_dir / "data" / "processed" / "clean_dataset.csv"

if not input_file.exists():
    raise FileNotFoundError(f"Input file not found: {input_file}")

clean_dataset(input_file, output_file)
print("Cleaning pipeline completed successfully")