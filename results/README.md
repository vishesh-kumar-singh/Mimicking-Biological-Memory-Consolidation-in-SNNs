# MNIST with SNN - Continual Learning Experiments Results

Aggregated results for mitigating catastrophic forgetting in SNNs.

### Results for 1 Epochs per Task

**Baseline:**
- Final Task A Retention: 0.90% ± 1.75
- Combined Accuracy: 48.19%

**Freezing Experiments:**

| Percentile | Task A Retention | Task B Accuracy | Combined |
| :--- | :--- | :--- | :--- |
| 0.40 | 11.41% | 95.39% | 52.84% |
| 0.50 | 22.62% | 94.62% | 57.97% |
| 0.60 | 29.21% | 95.17% | 61.64% |
| 0.70 | 42.10% | 94.09% | 67.64% |
| 0.80 | 57.08% | 90.89% | 73.78% |

---

### Results for 2 Epochs per Task

**Baseline:**
- Final Task A Retention: 2.91% ± 4.15
- Combined Accuracy: 48.95%

**Freezing Experiments:**

| Percentile | Task A Retention | Task B Accuracy | Combined |
| :--- | :--- | :--- | :--- |
| 0.40 | 13.78% | 96.03% | 54.14% |
| 0.50 | 23.37% | 96.96% | 59.69% |
| 0.60 | 33.99% | 95.84% | 64.54% |
| 0.70 | 49.14% | 94.77% | 71.88% |
| 0.80 | 58.79% | 92.15% | 75.53% |

---

### Results for 3 Epochs per Task

**Baseline:**
- Final Task A Retention: 0.00% ± 0.00
- Combined Accuracy: 47.76%

**Freezing Experiments:**

| Percentile | Task A Retention | Task B Accuracy | Combined |
| :--- | :--- | :--- | :--- |
| 0.40 | 22.02% | 97.47% | 59.26% |
| 0.50 | 29.42% | 96.71% | 62.64% |
| 0.60 | 46.46% | 95.97% | 70.99% |
| 0.70 | 57.75% | 95.25% | 76.30% |
| 0.80 | 63.26% | 94.69% | 79.11% |

**Random Freezing Experiments:**

| Percentile | Task A Retention | Task B Accuracy | Combined |
| :--- | :--- | :--- | :--- |
| 0.40 | 11.36% | 97.43% | 53.77% |
| 0.60 | 22.33% | 97.62% | 59.54% |
| 0.80 | 41.95% | 96.63% | 69.27% |

**Index Freezing Experiments:**

| Percentile | Task A Retention | Task B Accuracy | Combined |
| :--- | :--- | :--- | :--- |
| 0.40 | 13.43% | 97.06% | 54.67% |
| 0.60 | 22.49% | 97.54% | 59.43% |
| 0.80 | 40.69% | 97.04% | 68.66% |

**NoScale Freezing Experiments:**

| Percentile | Task A Retention | Task B Accuracy | Combined |
| :--- | :--- | :--- | :--- |
| 0.40 | 9.60% | 97.47% | 53.11% |
| 0.60 | 31.42% | 97.72% | 64.34% |
| 0.80 | 51.72% | 96.67% | 74.29% |

---

### Results for 4 Epochs per Task

**Baseline:**
- Final Task A Retention: 0.00% ± 0.00
- Combined Accuracy: 48.36%

**Freezing Experiments:**

| Percentile | Task A Retention | Task B Accuracy | Combined |
| :--- | :--- | :--- | :--- |
| 0.40 | 20.45% | 97.58% | 58.50% |
| 0.50 | 33.41% | 97.52% | 65.28% |
| 0.60 | 46.48% | 97.27% | 71.91% |
| 0.70 | 55.56% | 95.66% | 75.47% |
| 0.80 | 62.08% | 95.16% | 78.65% |

---

### Results for 5 Epochs per Task

**Baseline:**
- Final Task A Retention: 0.00% ± 0.00
- Combined Accuracy: 48.28%

**Freezing Experiments:**

| Percentile | Task A Retention | Task B Accuracy | Combined |
| :--- | :--- | :--- | :--- |
| 0.40 | 13.85% | 97.53% | 55.34% |
| 0.50 | 29.40% | 97.67% | 63.29% |
| 0.60 | 44.83% | 96.84% | 70.81% |
| 0.70 | 56.24% | 96.07% | 76.34% |
| 0.80 | 62.63% | 95.04% | 78.88% |

---

