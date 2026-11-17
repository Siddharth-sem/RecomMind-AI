# Evaluation

## 1. Metrics

The project reports top-K ranking metrics at K=10:

- **Precision@10:** fraction of the top 10 recommendations that are relevant.
- **Recall@10:** fraction of the user's relevant held-out items recovered in the top 10.
- **NDCG@10:** ranking-sensitive measure that rewards relevant items appearing nearer the top.
- **Catalog coverage:** fraction of the available item catalog recommended across evaluated users. This metric is reported for the unified movie comparison.

The exact implementation is in the domain evaluation modules.

## 2. Movie results shown in the application

The Home page's unified movie table contains:

| Algorithm | Precision@10 | Recall@10 | NDCG@10 | Catalog coverage |
|---|---:|---:|---:|---:|
| User KNN | 0.1661359 | 0.1746507 | 0.2314591 | 0.0434007 |
| Item KNN | 0.1869427 | 0.2013573 | 0.2548250 | 0.1087990 |
| NMF | 0.1861996 | 0.1976627 | 0.2529811 | 0.2598098 |
| LightFM (WARP) | **0.2220807** | **0.2265795** | **0.2896272** | 0.2378121 |

The dashboard describes this comparison as a common positive-only split, relevance threshold >=4, top-10 recommendations, and previously seen items excluded.

**Displayed takeaway:** LightFM (WARP) leads the three ranking metrics; NMF has the largest catalog coverage.

The repository also contains standalone KNN/NMF CSV outputs with different values from the unified dashboard. For example, `movie_knn_evaluation_results.csv` reports 929 evaluated users, with User-based KNN precision 0.2429494 and Item-based KNN precision 0.0198062. These standalone results should not be mixed into the unified comparison without reconciling their evaluation protocols and model versions.

## 3. Book results

The book evaluation CSV reports 1,000 evaluated users.

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Collaborative Filtering | 0.0168 | 0.054102 | 0.042150 |
| Semantic Recommendations | 0.0075 | 0.035142 | 0.019753 |
| Hybrid Recommendations | **0.0193** | **0.066988** | **0.048994** |

The book evaluation implementation uses:
- an 80/20 train/test split;
- random state 42;
- relevance threshold of 7;
- top-10 evaluation;
- up to 1,000 users.

**Displayed takeaway:** Hybrid is the strongest of the three evaluated book methods on all three reported ranking metrics.

## 4. Product results

The product evaluation CSV reports 981 evaluated users.

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Collaborative | 0.009582 | 0.035264 | 0.023669 |
| Semantic | 0.007543 | 0.029307 | 0.017997 |
| Hybrid | **0.011519** | **0.043054** | **0.029310** |

The product evaluation code:
- samples eligible users with at least 10 interactions (up to 1,000; random state 42);
- splits each selected user's interactions 80/20;
- uses rating >=4 as relevant;
- evaluates top 10 recommendations;
- computes collaborative scores using a 50-component TruncatedSVD model.

**Displayed takeaway:** Hybrid is the strongest of the three evaluated product methods on all three reported ranking metrics.

## 5. Cross-domain comparison warning

Do not compare the absolute Movie, Book, and Product metric values as if they were produced by one benchmark. The domains use different datasets, relevance definitions, item spaces, and evaluation implementations. The app explicitly presents the three domain evaluations separately.

## 6. Interpretation and limitations

- A higher Precision@10, Recall@10, or NDCG@10 is better within the same evaluation setup.
- Catalog coverage measures recommendation breadth, not ranking relevance.
- Evaluation results depend on data filtering, train/test splitting, candidate generation, and relevance thresholds.
- The standalone movie KNN/NMF CSVs and the unified movie dashboard contain different result values. Keep their provenance clear in reports and presentations.
- Offline ranking metrics do not by themselves establish user satisfaction, novelty, fairness, latency, or production performance.
