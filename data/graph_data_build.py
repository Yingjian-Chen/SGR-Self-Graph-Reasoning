import openai
import pandas as pd
import time
from tqdm import tqdm
import concurrent.futures
from openai import AzureOpenAI
import os

# OpenAI API
client = openai.OpenAI(api_key='')

def cot_generation(question, answer, retries=3, delay=2):
    prompt = f"""
    Question: {question}
    Answer: {answer}

    Construct a **concise reasoning graph** that explores the key reasoning paths leading to the correct answer.  
    The graph should capture multiple hypotheses or branches, but keep it **short and high-quality**: only 3-5 key steps plus the final conclusion.

    Granularity rules:
    - Each node must be an **atomic proposition**: one subject + one predicate (+ optional object/value).
    - Split sentences by meaning: if a claim can be divided into smaller independent claims, make multiple nodes.
    - No pronouns; always use the same canonical entity names.
    - Each node should be concise and only include essential information.
    - Avoid redundancy: do not repeat the same fact multiple times.

    Graph rules:
    - Start from the initial facts given in the question.
    - Allow **branching** into multiple reasoning directions only when alternative possibilities exist.
    - Branches should converge to the final conclusion that supports the correct answer.
    - Each edge represents a **small, justified inference step**, formatted as ["premise", "leads to", "next step"].
    - Keep the reasoning graph **connected from start to conclusion**.

    Output format (strictly):
    <reasoning>
    ["node1", "leads to", "node2"]
    ["node1", "leads to", "node3"]
    ["node2", "leads to", "node4"]
    ...
    ["nodeN", "leads to", "conclusion"]
    </reasoning>

    <answer>
    A/B/C/D
    </answer>

    Constraints:
    - Output only the <reasoning> graph and the <answer> block.
    - Ensure reasoning is **concise**, **atomic**, and **high-quality**.
    - Represent multiple branches when necessary, but limit the graph to key steps (3-5 + final conclusion).
"""

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a reasoning model."},
                    {"role": "user", "content": prompt},
                ],
                timeout=60,
                temperature=0.9
            )

            event = completion.choices[0].message.content.strip()

            reasoning = event.split("<answer>")[0].strip().replace('<reasoning>', '').replace('</reasoning>', '').strip()
            answer = event.split("<answer>")[1].strip().replace('</answer>', '')

            return reasoning, answer

        except Exception as e:
            print(f"Request failed: {e}. Retrying {retries - attempt - 1} more times.")
            time.sleep(delay)
    return "Request failed after multiple retries. Please check your connection or input."


def get_optimal_reasoning(question, answer, reasoning1, reasoning2, reasoning3, retries=3, delay=2):
    prompt = f"""
    Question: {question}
    Answer: {answer}

    You are given three reasoning paths that attempt to explain the answer to this question.  
    Your task is to integrate them into a **single optimal reasoning graph** that includes multiple possible branches but ultimately converges on the correct answer.

    Reasoning 1: {reasoning1}
    Reasoning 2: {reasoning2}
    Reasoning 3: {reasoning3}

    Integration rules:
    1. If two entities are similar or co-refer, merge them into one, keeping the more meaningful/specific name.
    2. Each node must be an **atomic proposition** (one subject + one predicate + optional object/value).
    3. If a sentence expresses multiple independent claims, split it into multiple nodes.
    4. Each node must represent exactly one indivisible fact/idea.
    5. Each edge must represent a **small justified inference step**, not a leap.
    6. When evaluating options, base judgments only on explicit prior nodes.
    7. At least one node must branch into ≥2 alternative reasoning directions before convergence.
    8. All branches must remain connected and converge toward the final conclusion.

    Output format:
    <reasoning>
    ["thinking1", "leads to", "thinking2"]
    ["thinking1", "leads to", "thinking3"]
    ["thinking2", "leads to", "thinking4"]
    ["thinking3", "leads to", "thinking5"]
    ...
    ["thinkingN", "leads to", "conclusion"]
    </reasoning>

    <answer>
    A/B/C/D
    </answer>

    Constraints:
    - Output only the <reasoning> graph and the <answer> block, nothing else.
    - All nodes must be atomic, unique, and quoted strings.
    - Ensure at least one explicit branching before convergence to the conclusion.
    """

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a reasoning model."},
                    {"role": "user", "content": prompt},
                ],
                timeout=60,
                temperature=0.3
            )

            event = completion.choices[0].message.content.strip()

            reasoning = event.split("<answer>")[0].strip().replace('<reasoning>', '').replace('</reasoning>',
                                                                                              '').strip()
            answer = event.split("<answer>")[1].strip().replace('</answer>', '')

            return reasoning, answer

        except Exception as e:
            print(f"Request failed: {e}. Retrying {retries - attempt - 1} more times.")
            time.sleep(delay)
    return "Request failed after multiple retries. Please check your connection or input."


def generate_reasonings_for_question(question, answer):
    """
    Generate three reasoning paths concurrently for each question.
    Returns the reasoning paths and answers as a tuple.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(cot_generation, question, answer),
            executor.submit(cot_generation, question, answer),
            executor.submit(cot_generation, question, answer)
        ]

        results = [future.result() for future in concurrent.futures.as_completed(futures)]

        reasoning1, answer1 = results[0]
        reasoning2, answer2 = results[1]
        reasoning3, answer3 = results[2]

        return reasoning1, reasoning2, reasoning3



def save_single_record(data_dict, excel_path, csv_path):
    """save single record to excel and csv"""
    # create single row DataFrame
    df = pd.DataFrame([data_dict])
    
    # if file not exists, create new file
    if not os.path.exists(excel_path):
        df.to_excel(excel_path, index=False)
        df.to_csv(csv_path, index=False, encoding='utf-8')
    else:
        # append mode write to Excel
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            # get existing row number
            existing_df = pd.read_excel(excel_path)
            start_row = len(existing_df) + 1
            df.to_excel(writer, index=False, header=False, startrow=start_row)
        
        # append mode write to CSV
        df.to_csv(csv_path, mode='a', header=False, index=False, encoding='utf-8')

def get_existing_records(excel_path):
    """get existing records from excel, for resume"""
    if os.path.exists(excel_path):
        existing_df = pd.read_excel(excel_path)
        return set(existing_df['Question'].values)
    return set()

if __name__ == '__main__':
    # set output file path
    excel_path = './graph_data/graph_reasoning_data.xlsx'
    csv_path = './graph_data/graph_reasoning_data.csv'
    
    # create output directory (if not exists)
    os.makedirs('./graph_data', exist_ok=True)
    
    # data load
    result_df = pd.read_csv('data/datasets/logiqa_train.csv')
    questions = result_df['Question'].values
    answers = result_df['Label'].values

    # get processed questions
    processed_questions = get_existing_records(excel_path)
    
    # initialize progress bar
    total_questions = len(questions)
    progress_bar = tqdm(total=total_questions, desc="Processing questions")
    
    # update progress bar initial position
    progress_bar.update(len(processed_questions))

    for question, answer in zip(questions, answers):
        # skip processed questions
        if question in processed_questions:
            continue
            
        try:
            reasoning, pre = cot_generation(question, answer)

            if str(pre).strip() == str(answer).strip():
                # prepare single record
                data_dict = {
                    'Question': question,
                    'Reasoning': reasoning,
                    'Pred': pre,
                    'Label': answer
                }
                
                # immediately save single record
                save_single_record(data_dict, excel_path, csv_path)
                processed_questions.add(question)
                
                # print success information
                print(f"\nSuccessfully processed and saved question: {question[:50]}...")
                
        except Exception as e:
            print(f"\nError processing question: {question[:50]}...")
            print(f"Error details: {str(e)}")
            continue
        
        finally:
            # update progress bar
            progress_bar.update(1)
    
    progress_bar.close()
    print("\nProcessing completed!")
