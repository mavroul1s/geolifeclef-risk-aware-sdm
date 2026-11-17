"""Secondary D1 precision/recall summary from saved decisions, with no fitting."""
from pathlib import Path
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]/'results/reliability_followup_d1_d3'
    columns = ['predictor', 'decoder', 'role', 'rare_truth', 'rare_hits', 'rare_predictions',
               'predicted_richness', 'hits', 'true_richness']
    frames = []
    for fold in [0, 1]:
        f = pd.read_csv(root/f'inference_output/aligned/inference_export/fold{fold}_decisions.csv', usecols=columns)
        frames.append(f[(f.predictor == 'ensemble') & (f.role == 'assessment')])
    result = pd.concat(frames).groupby('decoder')[columns[3:]].sum()
    result['rare_recall'] = result.rare_hits/result.rare_truth
    result['rare_precision'] = result.rare_hits/result.rare_predictions
    result['micro_F1'] = 2*result.hits/(result.predicted_richness+result.true_richness)
    result.to_csv(root/'decisions_ensembles/full_coverage_rare_precision_recall.csv')


if __name__ == '__main__':
    main()
