# Practical Machine Learning Course Project

This repository predicts the `classe` outcome in the Weight Lifting Exercise
Dataset using a bagged classification-tree ensemble.

## Files

- `Weight_Lifting_Exercise_Prediction.Rmd` â€” reproducible report source
- `Weight_Lifting_Exercise_Prediction.html` â€” compiled, self-contained report
- `train_model.R` â€” preprocessing, five-fold cross-validation, final model
- `results/quiz_predictions.csv` â€” predictions for the 20 quiz cases
- `results/quiz_submission_files/` â€” one answer per text file

## Reproduce

Open R in the repository root and run:

```r
source("train_model.R")
rmarkdown::render("Weight_Lifting_Exercise_Prediction.Rmd")
```

The scripts download the two course CSV files if they are not already under
`data/`. R package requirements are `rpart`, `knitr`, and `rmarkdown`.

Measured five-fold cross-validation accuracy: **99.19%**  
Estimated out-of-sample error: **0.81%**

