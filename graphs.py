import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('lr_experiments_results.csv')

def plot_with_equal_spacing(y_column, title, save_path):
    plt.figure(figsize=(14, 8))
    for reg in sorted(df['regularization'].unique()):
        subset = df[df['regularization'] == reg].sort_values('learning_rate')
        plt.plot(subset['learning_rate'], subset[y_column], marker='o', label=f'Reg={reg:.1e}')
    plt.xscale('linear')
    plt.xticks(np.arange(0.01, 0.21, 0.01), labels=[f"{t:.2f}" for t in np.arange(0.01, 0.21, 0.01)], rotation=45)
    plt.xlim(0, 0.21)
    plt.xlabel('Learning Rate')
    plt.ylabel(y_column.replace('_', ' ').title())
    plt.title(title)
    plt.legend(title='L2 Regularization')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
plot_with_equal_spacing('val_accuracy', 'Validation Accuracy', 'lr_val_linear.png')
plot_with_equal_spacing('train_accuracy', 'Train Accuracy', 'lr_train_linear.png')