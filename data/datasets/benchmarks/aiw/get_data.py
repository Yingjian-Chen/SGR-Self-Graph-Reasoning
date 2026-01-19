import pandas as pd
from tqdm import tqdm


result_df = pd.read_pickle('./AIW_easy.pkl')

questions = result_df['questions'].values
answers = result_df['answers'].values

for question, answer in tqdm(zip(questions, answers), total=len(questions)):
    print("*"*100)
    print("Question:")
    print(question)
    print("Answer:")
    print(answer)