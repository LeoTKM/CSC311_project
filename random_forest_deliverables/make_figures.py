"""Figures for methodology_random_tree.tex. Reads only frozen artifacts.

    figures/confusion_matrices.png   test and validation confusion matrices, as heatmaps
    figures/forest_structure.png     nodes per tree, and which columns the trees split on

Run from this directory:  python make_figures.py
"""
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
FIGURES = HERE / 'figures'
FINAL = ROOT / 'random_forest_deliverables' / 'forest_final_v1'

for path in (ROOT / 'train' / 'random_forest', ROOT / 'analysis', ROOT / 'data_preprocessing'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import joblib                                                        # noqa: E402
import pandas as pd                                                  # noqa: E402

CLASSES = ['Soccer', 'Baseball', 'Basketball']


def confusion_figure():
    panels = [
        ('Test', 'test_confusion.csv', '132 responses, 44 respondents, read once'),
        ('Validation', None, '132 responses, 44 respondents'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, (title, filename, subtitle) in zip(axes, panels):
        if filename is None:
            matrix = pd.read_csv(FINAL / '..' / 'forest_validation_v1' /
                                 'validation_confusion.csv', index_col=0) \
                      .reindex(index=CLASSES, columns=CLASSES).to_numpy()
        else:
            matrix = pd.read_csv(FINAL / filename, index_col=0) \
                      .reindex(index=CLASSES, columns=CLASSES).to_numpy()
        share = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap='Blues', vmin=0.0, vmax=1.0)
        for row in range(3):
            for column in range(3):
                dark = share[row, column] > 0.55
                ax.text(column, row, f'{matrix[row, column]}\n{share[row, column]:.0%}',
                        ha='center', va='center', fontsize=10,
                        color='white' if dark else '#1f2d3d',
                        fontweight='bold' if row == column else 'normal')
        ax.set_xticks(range(3), CLASSES, fontsize=9)
        ax.set_yticks(range(3), CLASSES, fontsize=9)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual', fontsize=9)
        correct = int(np.trace(matrix))
        total = int(matrix.sum())
        ax.set_title(f'{title}: {correct}/{total} correct ({correct / total:.1%})\n'
                     f'{subtitle}', fontsize=9.5)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='minor', length=0)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / 'confusion_matrices.png'
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path.relative_to(HERE))


def structure_figure():
    artifact = joblib.load(FINAL / 'final_model.joblib')
    model, preprocessor = artifact['model'], artifact['preprocessor']
    columns = list(preprocessor.columns_)
    forest = model.estimators_

    nodes = np.array([tree.tree_.node_count for tree in forest])
    splits = Counter()
    for tree in forest:
        for value in tree.tree_.feature:
            if value >= 0:
                splits[int(value)] += 1
    top = splits.most_common(12)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0),
                             gridspec_kw={'width_ratios': [1, 1.35]})
    axes[0].hist(nodes, bins=np.arange(nodes.min() - 0.5, nodes.max() + 1.5, 2),
                 color='#4c78a8', edgecolor='white')
    axes[0].axvline(nodes.mean(), color='#c0392b', linestyle='--', linewidth=1.2)
    axes[0].set_xlabel('nodes in a tree', fontsize=9)
    axes[0].set_ylabel('number of trees', fontsize=9)
    axes[0].set_title(f'300 trees, all of depth 6\naverage {nodes.mean():.1f} nodes, '
                      f'range {nodes.min()}--{nodes.max()}', fontsize=9.5)
    axes[0].tick_params(labelsize=8)

    labels = [columns[index][:30] for index, _ in top][::-1]
    values = [count for _, count in top][::-1]
    axes[1].barh(range(len(values)), values, color='#4c78a8')
    axes[1].set_yticks(range(len(values)), labels, fontsize=7.5)
    axes[1].set_xlabel('splits across the 300 trees', fontsize=9)
    axes[1].set_title('Most used columns\n'
                      f'{len(splits)} of {len(columns)} columns are used at least once',
                      fontsize=9.5)
    axes[1].tick_params(axis='x', labelsize=8)
    fig.tight_layout()
    path = FIGURES / 'forest_structure.png'
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path.relative_to(HERE))


if __name__ == '__main__':
    confusion_figure()
    structure_figure()
