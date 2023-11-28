### HumanEval for GPT-3.5/GPT-4 

Forked from [OpenAI's repo](https://github.com/openai/human-eval).
Should run out of the box with python3.11 (as of 11/23)

To generate the completions (after pip installing the requirements), run:
```
mkdir results
python run.py
```

Then to evaluate the completion results, run
```
python human_eval/evaluate_functional_correctness "YOUR RESULTS FILE.jsonl"
```
