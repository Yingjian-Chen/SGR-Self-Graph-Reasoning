import pandas as pd
from tqdm import tqdm

result_df = pd.read_csv('medqa.csv')


questions = result_df['Question'].values
answers = result_df['Label'].values


for question, answer in tqdm(zip(questions, answers), total=len(questions)):
    print("*"*100)
    print("Question:")
    print(question)
    print("Answer:")
    print(answer)