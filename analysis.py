import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('z3_verdicts.csv')

# 2. Calculate Statistics
total = len(df)
matches = df[df['z3_verdict'] == df['ground_truth_label']]
mismatches = df[df['z3_verdict'] != df['ground_truth_label']]
accuracy = len(matches) / total * 100
change_rate = 100 - accuracy

print("="*40)
print(f"VERDICT CHANGE ANALYSIS")
print("="*40)
print(f"Total Posts:          {total}")
print(f"Unchanged Verdicts:   {len(matches)}")
print(f"Changed Verdicts:     {len(mismatches)}")
print(f"Change Rate:          {change_rate:.1f}%")
print("="*40)

# 3. Visualization 1: Confusion Matrix
# Shows where the votes are going (e.g., How many NTA became YTA?)
labels = ['NTA', 'YTA', 'ESH', 'NAH']
cm = pd.crosstab(df['ground_truth_label'], df['z3_verdict'])
# Reindex to ensure all labels are present
cm = cm.reindex(index=labels, columns=labels, fill_value=0)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix: Reddit Flair (Rows) vs Z3 (Cols)')
plt.ylabel('Reddit Flair (Majority Voting)')
plt.xlabel('Z3 Verdict (Reasoning)')
plt.tight_layout()
plt.savefig('analysis_confusion_matrix.png')
print("Saved 'analysis_confusion_matrix.png'")

# 4. Visualization 2: Top Changes Bar Chart
# Shows the specific "Flip Types"
df['transition'] = df['ground_truth_label'] + " → " + df['z3_verdict']
changes = df[df['ground_truth_label'] != df['z3_verdict']]
top_changes = changes['transition'].value_counts().head(8)

plt.figure(figsize=(10, 6))
top_changes.plot(kind='bar', color='#ff6b6b')
plt.title('Top Verdict Changes (Voting → Reasoning)')
plt.ylabel('Number of Posts')
plt.xlabel('Change Type')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('analysis_changes_bar.png')
print("Saved 'analysis_changes_bar.png'")

# 5. Print Top Changes Text Summary
print("\nTop 5 Verdict Changes:")
print(top_changes.head(5).to_string())