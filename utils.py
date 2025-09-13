import numpy as np
from transformers import set_seed
import re

def parse_results(results, task_name):
    

    output_data = {}
    output_data['accuracies'] = {}
    output_data['profile'] = results["profile"]
    output_data['samples'] = {}

    # Extract overall accuracies
    if 'results' in results:
        for task, task_results in results['results'].items():
            output_data['accuracies'][task] = {}
            for metric, value in task_results.items():
                if metric.startswith('exact_match'):
                    output_data['accuracies'][task][metric] = value
                elif metric.startswith('pass@1'):
                    output_data['accuracies'][task][metric] = value
                    
    if task_name == "math":
        accuracies = []
        for subject in output_data['accuracies']:
            if "hendrycks_math_" in subject:
                accuracies.append(output_data["accuracies"][subject]["exact_match,flexible-extract"])
            
        output_data['accuracies']['hendrycks_math'] = {"aggregate_accuracy": np.mean(accuracies)}

    # Extract sample data
    if 'samples' in results:

        for task, sample_list in results['samples'].items():
            output_data['samples'][task] = []
            for sample in sample_list:

                if sample.get('filter') != 'flexible-extract' and sample.get('filter') != 'create_test':
                    continue
                else:
                    metric = sample.get('metrics')[0]
                
                is_correct = sample.get(metric, None)
                filtered_answer = sample.get('filtered_resps', [None])[0]
                generation = sample.get('resps', [[""]])[0][0].strip()

                # Determine if the answer is correct by comparing the filtered response with the target
                if task_name == "gsm8k":
                    target = sample['target']
                    gold_answer_match = re.search(r'####\s*([^\n]+)', target)
                    gold_answer = gold_answer_match.group(1).strip() if gold_answer_match else None
                    question = sample['doc']['question']
                elif task_name == "gpqa":
                    target = sample['doc']['Correct Answer']
                    gold_answer = sample['doc']['answer']
                    question = sample['doc']['Question']
                elif task_name == "math":
                    target = sample['doc']['solution']
                    gold_answer = sample['doc']['answer']
                    question = sample['doc']['problem']
                elif task_name == "humaneval":
                    target = sample['doc']['canonical_solution']
                    gold_answer = sample['doc']['test']
                    question = sample['doc']['prompt']


                sample_data = {
                    'is_correct': is_correct,
                    'question': question,
                    'answer': target,
                    'ground_truth': gold_answer,
                    'filtered answer': filtered_answer,
                    'generation': generation
                }
                output_data['samples'][task].append(sample_data)
                
                
    return output_data


def remove_masks(text):

    mask = "<|mask|>"
    while text.endswith(mask):
        text = text[:-len(mask)]
    return text

