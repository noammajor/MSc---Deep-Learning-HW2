import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('top50nn.csv', skipinitialspace=True)
df['Dropout_p'] = df['Dropout_p'].replace('None', 1.0).astype(float)
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 8))
sns.stripplot(data=df, x='Activation', y='Val_Accuracy', hue='Dropout_p', 
              dodge=True, alpha=0.5, palette='Set2', size=7, marker='o')
sns.pointplot(data=df, x='Activation', y='Val_Accuracy', hue='Dropout_p', 
              dodge=0.55, palette='Set2', markers='D', scale=1.4, join=False, errorbar=None)
plt.title('Activation Function & Dropout', fontsize=15)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[len(df['Dropout_p'].unique()):], labels[len(df['Dropout_p'].unique()):], title='Dropout p', loc='lower right')
plt.tight_layout()
plt.show()
plt.close()
plt.figure(figsize=(10, 8))
sns.stripplot(data=df, x='LR', y='Val_Accuracy', hue='Activation', 
              dodge=True, alpha=0.5, palette='viridis', size=7, marker='o')
sns.pointplot(data=df, x='LR', y='Val_Accuracy', hue='Activation', 
              dodge=0.5, palette='viridis', markers='D', scale=1.4, join=False, errorbar=None)
plt.title('Learning Rate & Activation', fontsize=15)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[len(df['Activation'].unique()):], labels[len(df['Activation'].unique()):], title='Activation', loc='lower right')
plt.tight_layout()
plt.show()
plt.close()
plt.figure(figsize=(10, 8))
sns.stripplot(data=df, x='Hidden_Dim', y='Val_Accuracy', hue='Dropout_p', 
              dodge=True, alpha=0.5, palette='magma', size=7, marker='o')
sns.pointplot(data=df, x='Hidden_Dim', y='Val_Accuracy', hue='Dropout_p', 
              dodge=0.4, palette='magma', markers='D', scale=1.4, join=False, errorbar=None)
plt.title('Hidden Dimension & Dropout', fontsize=15)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[len(df['Dropout_p'].unique()):], labels[len(df['Dropout_p'].unique()):], title='Dropout p', loc='lower right')
plt.tight_layout()
plt.show()
plt.close()