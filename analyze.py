import json
from os.path import join

image_style = "anime"
compos_num = 2
method_1 = 'merge'
method_2 = 'composite'

def calculate_averages(evals):
    # Initialize sum and count dictionaries
    sum_scores = {'image 1': {'composition quality': 0, 'image quality': 0},
                  'image 2': {'composition quality': 0, 'image quality': 0}}
    count = {'image 1': 0, 'image 2': 0}

    # Sum up scores and count entries
    for entry in evals:
        for image, image_scores in entry['scores'].items():
            for dimension, value in image_scores.items():
                sum_scores[image][dimension] += value
                count[image] += 1

    # Calculate averages
    avg_scores = {}
    for image, image_scores in sum_scores.items():
        avg_scores[image] = {dimension: value / (count[image] // 2) for dimension, value in image_scores.items()}

    return avg_scores

def compare_methods(evals):
    comparison = {'composition quality': {'wins': 0, 'ties': 0, 'losses': 0},
                  'image quality': {'wins': 0, 'ties': 0, 'losses': 0}}
    total_comparisons = 0

    for entry in evals:
        total_comparisons += 1
        for dimension in ['composition quality', 'image quality']:
            if entry['scores']['image 2'][dimension] > entry['scores']['image 1'][dimension]:
                comparison[dimension]['wins'] += 1
            elif entry['scores']['image 2'][dimension] < entry['scores']['image 1'][dimension]:
                comparison[dimension]['losses'] += 1
            else:
                comparison[dimension]['ties'] += 1

    # Convert wins/losses/ties counts to win rates
    win_rates = {dim: {outcome: count / total_comparisons for outcome, count in outcomes.items()}
                 for dim, outcomes in comparison.items()}

    return win_rates

with open(join(f'eval_result/{image_style}_{compos_num}_elements_{method_1}_vs_{method_2}.json')) as f:
    eval_results = json.loads(f.read())

# Total numbers
print(f'Total {len(eval_results)} evaluation pairs!')

# Calculate averages
average_scores = calculate_averages(eval_results)
print("Average Scores:")
for method, dimensions in average_scores.items():
    if method == "image 1":
        print(f"  {method_1}:")
    else:
        print(f"  {method_2}:")
    for dimension, score in dimensions.items():
        print(f"    {dimension}: {score:.2f}")

# Compare methods (Method 2 vs Method 1)
method_comparison = compare_methods(eval_results)
print(f"\nMethod Comparison ({method_2} vs {method_1}):")
for dimension, outcomes in method_comparison.items():
    print(f"  {dimension.capitalize()}:")
    for outcome, rate in outcomes.items():
        print(f"    {outcome.capitalize()} Rate: {rate:.2f}")


