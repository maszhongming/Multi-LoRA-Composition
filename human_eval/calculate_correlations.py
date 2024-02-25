import json
from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr, kendalltau

def init_metric():
    scores = ['human_compos', 'human_image', 'clip_score', 'gpt4v_compos', 'gpt4v_image']
    methods = ['merge', 'switch', 'composite']
    return {method: {score: 0 for score in scores} for method in methods}

def print_metric(metric):
    table = PrettyTable(['Methods','Human-Composition', 'Human-Image', 'CLIPScore', 'GPT4V-Composition', 'GPT4V-Image'])
    for method, scores in metric.items():
        row = [method] + [round(scores[score], 2) for score in scores]
        table.add_row(row)
    print(table)

def calculate_correlation(pred_score, human_score):
    assert len(pred_score) == len(human_score)
    correlations = [pearsonr(pred_score, human_score)[0], spearmanr(pred_score, human_score)[0], kendalltau(pred_score, human_score)[0]]
    return correlations

def print_correlation(correlation_results):
    table = PrettyTable(['Metrics', 'Pearson', 'Spearman', 'Kendall'])
    for name, scores in correlation_results.items():
        table.add_row([name] + [round(score, 3) for score in scores])
    print(table)

def main(results):
    metric = init_metric()
    counts = {method: 0 for method in metric.keys()}

    clip_score, human_compos, human_image, gpt_compos, gpt_image = [], [], [], [], []

    for i, result in enumerate(results):
        method = ['merge', 'switch', 'composite'][i % 3]
        metric[method]['human_compos'] += result['avg_human_score']['composition']
        metric[method]['human_image'] += result['avg_human_score']['image']
        metric[method]['clip_score'] += result['clipscore']['score']
        metric[method]['gpt4v_compos'] += result['gpt4v']['composition']
        metric[method]['gpt4v_image'] += result['gpt4v']['image']
        counts[method] += 1

        human_compos.append(result['avg_human_score']['composition'])
        human_image.append(result['avg_human_score']['image'])
        clip_score.append(result['clipscore']['score'])
        gpt_compos.append(result['gpt4v']['composition'])
        gpt_image.append(result['gpt4v']['image'])

    for method in metric:
        for score in metric[method]:
            metric[method][score] /= counts[method]
    
    print('Results for each method:')
    print_metric(metric)

    correlation_results = {
        'CLIPScore --- Human Composition': calculate_correlation(clip_score, human_compos),
        'CLIPScore --- Human Image Quliaty': calculate_correlation(clip_score, human_image),
        'GPT4V Composition --- Human Composition': calculate_correlation(gpt_compos, human_compos),
        'GPT4V Image --- Human Image': calculate_correlation(gpt_image, human_image)
    }

    print('Correlations between different metrics with human judgements:')
    print_correlation(correlation_results)

if __name__ == "__main__":
    with open('results.json') as f:
        results = json.loads(f.read())
    main(results)
