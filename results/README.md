# Saved notebook output records

These CSVs preserve metrics printed in the uploaded notebook. They are not newly computed by the packaging process.

- `notebook_primary_results.csv`: output of the primary centralized and FedAvg cells (the centralized test accuracy is 0.8654; federated is 0.9167).
- `notebook_matched_comparison.csv`: output of the matched-initialization comparison cell (centralized 0.8942; FedAvg 0.8638).
- `notebook_reproducibility_runs.csv`: two additional runs, seeds 7 and 21.

Do not combine rows from different experiment tracks. The notebook's separate manually populated pilot-results cell contains inconsistent centralized values and is intentionally omitted.
