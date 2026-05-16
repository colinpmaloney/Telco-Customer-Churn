# Telco Churn: Logistic Regression vs. Neural Network Showdown

### **The Mission**
The goal of this project is to move beyond "black-box" modeling. I am comparing a baseline **Logistic Regression** model against an **Advanced Neural Network** to predict customer churn. 

I’m documenting the transition from Course 1 (Supervised Learning) to Course 2 (Advanced Learning Algorithms) of the Stanford Online Machine Learning Specialization by evaluating model complexity vs. performance gains.

### **Business Context**
Customer churn is a critical metric for service providers. Identifying at-risk customers allows for proactive retention strategies. This project aims to determine if the increased computational cost of a Neural Network provides a significant enough "lift" in recall to justify its deployment over a simpler linear model.

### **The Problem**
Using the **IBM Telco Customer Churn** dataset, I am predicting whether a customer will leave a service provider based on 19 features including contract type, monthly charges, and tenure.

---

## **Technical Implementation**

This section documents the specific implementations and architecture that I will be using for the two different models.

## **1. Data Engineering & Pre-processing**
*   **Source:** IBM Telco Customer Churn dataset.
*   **Cleaning:** Handled empty string values in Total Charges.
*   **Pipeline:** Implemented *One-Hot Encoding* for nomincal categorical features and *Mean Value Normalization* for numerical inputs

### **2. Model Architectures**
*   **Baseline:** Logistic Regression with L2 Regularization.
*   **Challenger:** 3-layer Neural Network

---

## **Model Tuning**

This section documents the specific "moves" I made to optimize performance and handle over-fitting.

*Logarithmic Regression*
- Tuned the alpha parameter
- Implemented L2 Regularization

*Neural Network*
- Implemented early stopping to ensure I choose the training epoch with the best parameters
- Implemented L2 Regularization

### **Hyperparameter Showdown**

| Parameter | Logistic Regression | Neural Network |
| :--- | :--- | :--- |
| **Learning Rate (Alpha)** | [TBD] | [TBD] |
| **Regularization (Lambda)** | [TBD] | [TBD] |
| **Convergence (Epochs)** | [TBD] | [TBD] |

---

## **Final Comparison: Which One Wins?**

| Metric | Logistic Regression | Neural Network |
| :--- | :--- | :--- |
| **Accuracy** | [TBD] | [TBD] |
| **Training Latency** | [TBD] | [TBD] |
| **F1-Score** | [TBD] | [TBD] |
| **Model Cost (J)** | [TBD] | [TBD] |

### **The Verdict**

This section documents the reality of the comparison between the two models.

---

## **Key Takeaways**

This section documents the specific key takeaways that I learned during my implementation and comparison.

*Data Pre-Processing*
- Ensuring that there are no NaN values is critical before running any model because those values with propogate through the math and the whole output will be NaN

*Logistic Regression*
- Implementing Feature Scaling allowed me to drastically increase my learning rate which allowed the model to learn with less iterations ( Quicker )

*Neural Network*
- Using Non-Linear relu functions does not always make an immediate impact in the performance.
- As I increased the units per layer and layer count there was an increase in accuracy however the validation_data had a decrease in accuracy - *Over Fitting* occured.
- Using Early Stop will allow you to take the most accurate training epoch


*General*
- Using Vectorization significantly speeds up the computations.

---

### **How to Run**
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Execute the analysis: `python3 main.py`