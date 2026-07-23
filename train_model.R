set.seed(20260723)

train_raw <- read.csv("data/pml-training.csv",
  na.strings = c("NA", "", "#DIV/0!"), check.names = TRUE)
test_raw <- read.csv("data/pml-testing.csv",
  na.strings = c("NA", "", "#DIV/0!"), check.names = TRUE)

meta <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2",
          "cvtd_timestamp", "new_window", "num_window")
candidate <- setdiff(names(train_raw), c(meta, "classe"))
keep <- candidate[colMeans(is.na(train_raw[candidate])) < 0.05]
x <- train_raw[keep]
y <- factor(train_raw$classe)
x_test <- test_raw[keep]

stratified_folds <- function(y, k = 5, seed = 1) {
  set.seed(seed)
  folds <- integer(length(y))
  for (lev in levels(y)) {
    idx <- which(y == lev)
    folds[idx] <- sample(rep(seq_len(k), length.out = length(idx)))
  }
  folds
}

ensemble_fit <- function(x, y, ntrees = 35, mtry = 35, seed = 1) {
  set.seed(seed)
  trees <- vars <- vector("list", ntrees)
  for (b in seq_len(ntrees)) {
    rows <- sample.int(nrow(x), nrow(x), replace = TRUE)
    vars[[b]] <- sample(names(x), min(mtry, ncol(x)))
    dat <- data.frame(classe = y[rows], x[rows, vars[[b]], drop = FALSE])
    trees[[b]] <- rpart::rpart(classe ~ ., dat, method = "class",
      control = rpart::rpart.control(cp = 0, minsplit = 5, minbucket = 2,
                                    maxdepth = 30, xval = 0))
  }
  structure(list(trees = trees, vars = vars, levels = levels(y)),
            class = "bagged_rpart")
}

ensemble_predict <- function(model, newdata) {
  votes <- matrix(0L, nrow(newdata), length(model$levels))
  for (b in seq_along(model$trees)) {
    p <- predict(model$trees[[b]],
      newdata[, model$vars[[b]], drop = FALSE], type = "class")
    cells <- cbind(seq_along(p), match(as.character(p), model$levels))
    votes[cells] <- votes[cells] + 1L
  }
  factor(model$levels[max.col(votes, ties.method = "first")],
         levels = model$levels)
}

fold_id <- stratified_folds(y, 5, 20260723)
cv_accuracy <- numeric(5)
all_cv_prediction <- factor(rep(NA_character_, length(y)), levels = levels(y))
for (fold in 1:5) {
  message("Cross-validation fold ", fold, " of 5")
  fit <- ensemble_fit(x[fold_id != fold, , drop = FALSE],
                      y[fold_id != fold], 35, 35, 20260723 + fold)
  pred <- ensemble_predict(fit, x[fold_id == fold, , drop = FALSE])
  all_cv_prediction[fold_id == fold] <- pred
  cv_accuracy[fold] <- mean(pred == y[fold_id == fold])
}

message("Fitting final ensemble")
final_fit <- ensemble_fit(x, y, 75, 35, 20260801)
test_prediction <- ensemble_predict(final_fit, x_test)
cv_confusion <- table(Observed = y, Predicted = all_cv_prediction)
cv_overall <- mean(all_cv_prediction == y)

dir.create("results", showWarnings = FALSE)
write.csv(data.frame(problem_id = test_raw$problem_id,
                     prediction = as.character(test_prediction)),
          "results/quiz_predictions.csv", row.names = FALSE)
dir.create("results/quiz_submission_files", showWarnings = FALSE)
for (i in seq_along(test_prediction)) {
  writeLines(as.character(test_prediction[i]),
    sprintf("results/quiz_submission_files/problem_id_%02d.txt", i))
}
saveRDS(list(keep = keep, folds = fold_id, cv_accuracy = cv_accuracy,
             cv_overall = cv_overall, cv_confusion = cv_confusion,
             predictions = test_prediction, model = final_fit),
        "results/model_results.rds")

cat("Predictors:", length(keep), "\n")
cat("Fold accuracy:", paste(sprintf("%.5f", cv_accuracy), collapse = ", "), "\n")
cat("Overall CV accuracy:", sprintf("%.5f", cv_overall), "\n")
cat("Predictions:", paste(test_prediction, collapse = " "), "\n")
