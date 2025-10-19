# SHAP (SHapley Additive exPlanations) Tutorial with California Housing

This notebook provides a step-by-step introduction to using **SHAP** for explainable AI.  
We demonstrate how to interpret machine learning predictions using the **California Housing dataset** and a **Random Forest Regressor**.

---

## 📌 Contents
1. [What is SHAP?](#1-what-is-shap)  
2. [Installation](#installing-dependencies)  
3. [Dataset](#2-load-dataset)  
4. [Model Training](#3-train-a-model)  
5. [SHAP Explanations](#4-shap-explanations)  
   - [Global Interpretability: SHAP Summary Plot](#global-interpretability-shap-summary-plot)  
   - [Local Interpretability: SHAP Waterfall Plot](#local-interpretability-shap-waterfall-plot)  
6. [Conclusion](#5-conclusion)

---

## 1. What is SHAP?

**SHAP (SHapley Additive exPlanations)** is a method based on Shapley values from cooperative game theory.  
It explains the prediction of any machine learning model by assigning each feature an importance value for a particular prediction.

**Key advantages of SHAP:**
- **Model-agnostic:** Works with tree-based, linear, and neural network models.  
- **Fair attribution:** Distributes contributions among features fairly.  
- **Consistent explanations:** If a feature contributes more, SHAP ensures it gets higher attribution.  

👉 In simple terms, SHAP helps us answer:  
**“How much did each feature contribute to this prediction?”**

---

## Installing Dependencies

```bash
pip install shap scikit-learn matplotlib
