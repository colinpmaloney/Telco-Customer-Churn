# Telco Churn: Logistic Regression vs. Neural Network Showdown

### **The Mission**
The goal of this project is to move beyond "black-box" modeling. I am comparing a baseline **Logistic Regression** model against an **Advanced Neural Network** to predict customer churn.

I'm documenting the transition from Course 1 (Supervised Learning) to Course 2 (Advanced Learning Algorithms) of the Stanford Online Machine Learning Specialization by evaluating model complexity vs. performance gains.

### **Business Context**
Customer churn is a critical metric for service providers. Identifying at-risk customers allows for proactive retention strategies. This project aims to determine if the increased computational cost of a Neural Network provides a significant enough "lift" in recall to justify its deployment over a simpler linear model.

### **The Problem**
Using the **IBM Telco Customer Churn** dataset, I am predicting whether a customer will leave a service provider based on 19 features including contract type, monthly charges, and tenure.

---

## **Technical Implementation**

### **1. Data Engineering & Pre-processing**
*   **Source:** IBM Telco Customer Churn dataset (~7,000 rows, 19 features → 30 after encoding).
*   **Cleaning:** Handled empty string values in `TotalCharges` (converted to NaN, dropped).
*   **Encoding:** One-Hot Encoding for 15 nominal categorical features.
*   **Normalization:** Z-Score normalization `(x - mean) / std` for the 3 numerical features (`TotalCharges`, `MonthlyCharges`, `tenure`). Z-score was chosen over range normalization because it guarantees unit variance regardless of original scale, giving gradient descent smoother loss contours.
*   **Split:** 80/20 train/test (`random_state=817`). All normalization statistics are computed on the training set only to prevent data leakage.

### **2. Model Architectures**
*   **Baseline:** Logistic Regression with L2 Regularization (vectorized NumPy implementation).
*   **Challenger:** 3-layer Neural Network (TensorFlow/Keras).

```
Input (30 features)
├─ Dense(64, ReLU) + L2
├─ Dropout(0.1)
├─ Dense(32, ReLU) + L2
├─ Dropout(0.1)
└─ Dense(1, Sigmoid)
```

---

## **Model Tuning**

### **Hyperparameter Showdown**

| Parameter | Logistic Regression | Neural Network |
| :--- | :--- | :--- |
| **Learning Rate (Alpha)** | 1.0 | 0.001 |
| **Regularization (Lambda)** | 0.0001 | 0.001 |
| **Convergence (Epochs)** | 10,000 | 300 (early stopping, patience=15) |
| **Classification Threshold** | 0.57 | 0.57 |

### **Optimization Techniques Applied (Neural Network)**

| Technique | Result | Notes |
| :--- | :--- | :--- |
| Raised Alpha 0.0001 → 0.001 | ✅ Improved | Adam's default; previous LR was too small to converge in 100 epochs |
| EarlyStopping (patience=15) | ✅ Improved | Restores best weights; cut training time from ~12s to ~3-5s |
| L2 Regularization (λ=0.001) | ✅ Improved | Penalizes large weights; prevents overfitting |
| Dropout reduction 0.3 → 0.1 | ✅ Improved | L2 was already regularizing; stacking both over-regularized the model |
| Z-Score normalization | ✅ Improved | More stable gradient flow than range normalization |
| Threshold tuning (0.5 → 0.57) | ✅ Improved | Reduced false positives (predicting churn when customer actually stayed) |
| ReduceLROnPlateau | ❌ Reverted | Caused model to overtrain past its best checkpoint |
| BatchNormalization | ❌ Reverted | Dataset too small for stable batch statistics; compounded over-regularization |

---

## **Final Comparison: Which One Wins?**

*Run `python3 main.py` to populate with live values.*

| Metric | Logistic Regression | Neural Network |
| :--- | :--- | :--- |
| **Accuracy** | ~0.80 | ~0.78 |
| **F1-Score** | ~0.54 | ~0.60 |
| **Training Latency** | ~1.3s | ~5.1s |
| **Model Cost (J)** | ~0.42 | ~0.52 |

### **The Verdict**

The Neural Network outperforms Logistic Regression on accuracy and F1 but at significantly higher training complexity and latency. For a dataset of this size (~7K rows), the performance gap may not justify the added complexity in a production context — Logistic Regression remains a strong, interpretable baseline.

### **Performance Ceiling**
This dataset has a practical accuracy ceiling of ~82-84% for any single model. Pushing past ~0.78 requires **feature engineering** — creating interaction terms like `num_services`, `cost_per_service`, and `tenure` buckets — rather than further architecture tuning.

---

## **Key Takeaways**

**Data Pre-Processing**
- NaN values propagate through all computations — must be handled before any modeling.
- Z-Score normalization outperforms range normalization for neural networks because it standardizes variance, not just scale.
- Normalization statistics must be computed on training data only to prevent data leakage into the test set.

**Logistic Regression**
- Feature scaling allows a much higher learning rate (alpha=1 vs ~0.001), reducing the iterations needed to converge.
- Vectorization (matrix operations vs loops) speeds up computation dramatically.

**Neural Network**
- A learning rate that is too low (0.0001) will fail to converge in a reasonable number of epochs
- EarlyStopping with `restore_best_weights=True` is essential: without it, you report the last epoch's weights, not the best.
- Dropout and L2 Regularization both fight overfitting. When both are active, they can compound and *underfit* the model. Tune them together, not independently.
- Not every technique helps every model — ReduceLROnPlateau and BatchNormalization both hurt performance here. Always measure; never assume.
- Accuracy and F1 measure different things. On imbalanced data, a model can increase accuracy by predicting the majority class more aggressively, which simultaneously hurts F1. For a churn use case, F1 is the more honest metric.
- Threshold tuning is free: adjusting the classification cutoff requires no retraining and can meaningfully shift the precision/recall tradeoff based on what the confusion matrix shows.

**General**
- Vectorization significantly speeds up computations.
- Change one variable at a time — otherwise you can't isolate what caused an improvement or regression.

---

### **How to Run**
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Execute the analysis: `python3 main.py`
