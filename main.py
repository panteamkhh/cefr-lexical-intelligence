from pathlib import Path
from src.data_cleaning import clean_dataset

# Define project base directory
base_dir = Path(__file__).resolve().parent

# Define input and output paths
input_file = base_dir / "data" / "toefl_vocabulary_cleaned_categorized.xlsx"
output_file = base_dir / "data" / "clean_dataset.xlsx"

# Run data cleaning pipeline
clean_dataset(input_file, output_file)