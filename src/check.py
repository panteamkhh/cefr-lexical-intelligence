import chardet


file_path = r"D:\list of words\cefr-project\data\raw\toefl_vocabulary_cleaned_categorized.xlsx"

with open(file_path, 'rb') as f:
    print(chardet.detect(f.read(100000)))