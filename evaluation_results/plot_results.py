import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("evaluation_results/model_comparison_full.csv")

metrics = [
    "Precision@10",
    "Recall@10",
    "NDCG@10",
    "Diversity",
    "Catalogue Coverage"
]

for metric in metrics:
    plt.figure(figsize=(9, 5))

    plt.bar(df["Model"], df[metric])

    plt.title(f"Model Comparison - {metric}")
    plt.ylabel(metric)
    plt.xlabel("Recommendation Models")
    plt.xticks(rotation=10)
    plt.tight_layout()

    filename = metric.lower().replace("@", "_").replace(" ", "_")
    plt.savefig(f"evaluation_results/{filename}.png", dpi=300)
    plt.close()

print("All comparison graphs generated successfully!")