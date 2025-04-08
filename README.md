1st place solution for the competition.

In this competition, the task is to develop a regression model that predicts wine quality using a dataset derived from the original `Wine Quality` dataset. The provided training and testing datasets have been generated using a deep learning model; hence, their feature distributions are close to, but not exactly the same as, those of the original dataset. 

Performance is measured by the quadratic weighted kappa, which compares the predicted quality ratings with the actual ratings. The closer the kappa score is to 1, the better the model performs in terms of agreement.

**1. Evaluation Metric and Helper Functions**
**Quadratic Weighted Kappa**:
A custom function `quadratic_weighted_kappa` computes the Cohen kappa score using quadratic weights.

**Threshold Rounding**:
The `threshold_Rounder` function converts continuous predictions into discrete quality classes (assumed to be 3–8) based on optimized thresholds.

**Evaluation Function**:
The `evaluate_predictions` function is used during optimization to compute the negative quadratic weighted kappa for a set of thresholds.

**2. Model Training and Cross-Validation**

A 5-fold split is used to train the model and to validate performance across different subsets of the data using CatboostRegressor.

**3. Threshold Optimization**
**Initialization**:

On many public notebooks, thresholds are commonly initialized using fixed values, for example, [0.5, 1.5, 2.5]. However, in this training pipeline, the initialization is dynamically derived from the data itself. Instead of using static values, the initial thresholds are calculated by averaging the predictions for each quality level in the out-of-fold (OOF) predictions. 

**Optimizer**:
`scipy.optimize.minimize` is used with the `Nelder-Mead` method to optimize the rounding thresholds.

**Final Predictions**:
Once the optimized thresholds are found, they are applied to both the out-of-fold predictions and the test set predictions.

```python
oof_mask = ~np.isnan(oof_non_rounded)
oof_initial_thresholds = (
            pd.DataFrame({'target': y[oof_mask], 'prediction': oof_non_rounded[oof_mask]})
            .groupby('target')['prediction']
            .mean()
            .iloc[1:]
            .values
            .tolist()
        )

KappaOptimizer = minimize(
    evaluate_predictions,
    x0 = oof_initial_thresholds,
    args = (y, oof_non_rounded),
    method='Nelder-Mead'
)
```
