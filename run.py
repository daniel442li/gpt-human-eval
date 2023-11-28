import json
import os
import re
import tqdm
import time
from openai import OpenAI

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from dotenv import load_dotenv, find_dotenv
from concurrent.futures import ThreadPoolExecutor


load_dotenv(find_dotenv())

client = OpenAI()

HEADERS = {
    "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    "Content-Type": "application/json",
}

HUMAN_EVAL = os.environ['PWD'] + '/data/HumanEval.jsonl'
OUT_FILE = os.environ['PWD'] + '/results/results-{}.jsonl'


pattern = re.compile(r'```(?:[Pp]ython|[Pp]y)\s*([\s\S]+?)\s*```')
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt, model='gpt-3.5-turbo'):
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are an intelligent programmer. You must complete the python function given to you by the user. And you must follow the format they present when giving your answer!"},
        {"role": "user", "content": prompt}]
    )

    result = (completion.choices[0].message.content)
    match = pattern.search(result)
    
    if match:
        python_code = match.group(1)
    else:
        python_code = result

    return python_code


def iter_hval():
    all_lines = []
    with open(HUMAN_EVAL) as f:
        for line in f:
            all_lines.append(json.loads(line))

    return all_lines

def process_command(command, model):
    task_id, prompt = command
    completion = get_completion(prompt, model=model)
    return {'task_id': task_id, 'completion': completion}

def get_results(model='gpt-4'):
    out_file = OUT_FILE.format(model)

    with open(out_file, 'w') as f:
        pass
    
    batch_size = 15
    batch = []
    with tqdm.tqdm(total=len(iter_hval())) as progress_bar:  # total=expected_total_lines if known
        for line in iter_hval():
            prompt = line['prompt']
            task_id = line['task_id']
            batch.append((task_id, prompt))

            if len(batch) == batch_size:
                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(process_command, command, model) for command in batch]
                    results = [future.result() for future in futures]

                with open(out_file, 'a') as out_f:
                    for out in results:
                        out_f.write(json.dumps(out) + '\n')

                batch = []
                progress_bar.update(batch_size)  # Update progress bar for each processed batch

        # Process any remaining items in the batch after the loop
        if batch:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_command, command, model) for command in batch]
                results = [future.result() for future in futures]

            with open(out_file, 'a') as out_f:
                for out in results:
                    out_f.write(json.dumps(out) + '\n')

            progress_bar.update(len(batch))  # Update progress bar for the last batch

if __name__ == '__main__':
    model = 'gpt-4'
    get_results(model=model)

    out_f = OUT_FILE.format(model)
    print(f'Tests complete at: {out_f}')