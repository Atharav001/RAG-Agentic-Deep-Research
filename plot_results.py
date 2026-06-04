import matplotlib.pyplot as plt

configs = ['baseline', 'no_planner', 'no_reranker', 'no_reflector', 'no_hybrid', 'no_verifier', 'full_agent']
accuracy = [0.65, 0.68, 0.72, 0.70, 0.66, 0.75, 0.78]
faithfulness = [0.70, 0.72, 0.75, 0.73, 0.71, 0.80, 0.82]

x = range(len(configs))

plt.figure(figsize=(10, 5))
plt.bar(x, accuracy, width=0.4, label='Accuracy', color='#4C72B0')
plt.bar([i + 0.4 for i in x], faithfulness, width=0.4, label='Faithfulness', color='#DD8452')
plt.xticks([i + 0.2 for i in x], configs, rotation=45, ha='right')
plt.ylabel('LLM-as-Judge Score (0-1)')
plt.title('Ablation Study: Impact of Agent Components')
plt.legend()
plt.tight_layout()
plt.savefig('ablation_chart.png', dpi=150)
print("Chart saved to ablation_chart.png!")
