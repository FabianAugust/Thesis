# Load packages ---------------------------------------------------------------------

library(MASS) # For lm.ridge
library(e1071) # For svm
library(mlbench) # For BostonHousing data

# Data partition functions ----------------------------------------------------------

PartitionData <- function(data, n_part = 2){
  # Function: Partition a dataset 'data' in 'n_part' partitions
  # Returns: list of data.frames for each partition. If n_part = 1 returns the 
  #          same input data.frame in a list
  
  id_part <- sample(rep(1:n_part, length.out = dim(data)[1]))
  l_data <- lapply(1:n_part, FUN = function(x) data[id_part == x, ])
  
  return(l_data)
}


# Regression/Knowledge transfer functions -------------------------------------------

RegressionTrain <- function(l_data, features, privileged){
  # Performs regression for KT on training data
  # Returns: list of lists of fitted regressions
  
  # fm = list of regression formulas
  fm <- paste0(privileged, " ~ ", paste0(features, collapse = " + "))
  fm <- sapply(fm, FUN = as.formula)
  
  n <- length(l_data)
  id_part <- 1:n
  
  if(n == 1){
    l_data2 <- l_data
  } else {
    l_data2 <- lapply(id_part, FUN = function(x){
      Reduce('rbind', l_data[-x])
    })
  }

  
  fits <- lapply(l_data2, FUN = function(x){
    lapply(fm, FUN = function(y){
      lm.ridge(y, data = x)
    })
  })
  
  return(fits)
}

PrivilegedInformation <- function(df, fit, features){
  l <- list()
  if(is.data.frame(df)) df <- list(df)
  for(i in 1:length(df)){
    fit2 <- fit[[i]]
    dat <- df[[i]]
    for(j in 1:length(fit2)){
      nm <- paste0("P", j)
      fits <- fit2[[j]]
      preds <- as.matrix(cbind(const=1,dat[,features])) %*% coef(fits)
      dat[,nm] <- preds
    }
    l[[i]] <- dat
  }
  return(l)
}


# Tune and fit SVM models -----------------------------------------------------------

TuneModels <- function(data, vars){
  C <- 2^seq(-5, 5, 0.5)
  G <- 2^seq(-6, 6, 0.5)
  l <- list()
  for(i in 1:length(data)){
    a <- data[[i]]
    l[[i]] <- tune.svm(a[,vars], y = a[,"medv"], cost = C, gamma = G, kernel = "radial",
                       tunecontrol = tune.control(cross = 6))$best.parameters
  }
  return(l)
}

SVM <- function(data, param, vars){
  l <- list()
  formul <- as.formula(paste0("medv ~", paste0(vars, collapse = " + "))) 
  for(i in 1:length(data)){
    a <- data[[i]]
    b <- param[[i]]
    
    l[[i]] <- svm(formul, data = a, kernel='radial',
                  cost = b[1,2], gamma = b[1,1])
    
  }
  return(l)
}


# Create predictions for SVM --------------------------------------------------------

SVMpred <- function(data, mod){
  
  l <- length(data)
  
  l2 <- list()
  for(i in 1:l){
    a <- data[[i]]
    b <- mod[[i]]
    l2[[i]] <- predict(b, a)
    
  }
  return(l2)
  
}


# Evaluation ------------------------------------------------------------------------

# Get data
data(BostonHousing)
df <- BostonHousing
df <- df[,-4] # For simplicity remove factor variable

# Choose decision/privileged features
mat <- cor(df)
privileged <- names(mat[-13,13])[abs(mat[-13,13]) > 0.4]
features <- names(mat[-13,13])[abs(mat[-13,13]) <= 0.4]

#features <- colnames(df)[c(1:2,5,7:9,11,12)]
#privileged <- colnames(df)[c(6,10, 13)]


iter <- 10
n_test <- 100 # number of test features
res_matrix <- matrix(NA, ncol = 5, nrow = iter) # Empty result matrix

for(i in 1:iter){
  
  # 1. Create data
  ID <- sample(1:dim(df)[1], n_test) # Sample rougly 20% for test set
  
  train <- df[-ID, ] # Create original training set
  test <- df[ID,] # Create original test set
  
  train_standard <- PartitionData(train, n_part = 1) # Standard training set
  train_partition <- PartitionData(train, n_part = 2) # Partition training set
  
  
  # 2. Perform regressions for KT
  fit_standard <- RegressionTrain(train_standard, features = features, privileged = privileged)
  fit_partition <- RegressionTrain(train_partition, features = features, privileged = privileged)
  
  # 3. Perform knowledge transfers for training data
  train_standard_PI <- PrivilegedInformation(train_standard, fit_standard, features)
  train_partition_PI <- PrivilegedInformation(train_partition, fit_partition, features)
  
  # 4. perform knowledge transfer for test data
  test_standard <- PrivilegedInformation(test, fit_standard, features)
  test_partition <- PrivilegedInformation(list(test, test), fit_partition, features)
  
  # 5. Tune models
  parameters_standard <- TuneModels(train_standard, features)
  parameters_standard_PI <- TuneModels(train_standard_PI, paste0("P", 1:length(privileged)))
  parameters_partition_PI <- TuneModels(train_partition_PI, paste0("P", 1:length(privileged)))
  
  # 6. Create SVM models
  svm_standard <- SVM(train_standard, parameters_standard, features)
  svm_standard_PI <- SVM(train_standard_PI, parameters_standard_PI, paste0("P", 1:length(privileged)))
  svm_partition_PI <- SVM(train_partition_PI, parameters_partition_PI, paste0("P", 1:length(privileged)))
  
  
  # 7. Create predictions from SVM
  pred_standard <- SVMpred(test_standard, svm_standard)
  pred_standard_PI <- SVMpred(test_standard, svm_standard_PI)
  pred_partition_PI <- SVMpred(test_partition, svm_partition_PI)
  
  # 8. Add predictions to data
  test$pred_standard <- pred_standard[[1]]
  test$pred_standard_PI <- pred_standard_PI[[1]]
  test$pred_partition_PI_1 <- pred_partition_PI[[1]]
  test$pred_partition_PI_2 <- pred_partition_PI[[2]]
  
  test$pred_partition_PI <- as.numeric(apply(test[,c("pred_partition_PI_1", "pred_partition_PI_2")],
                                             1, mean))
  
  # 9. Evaluate model
  res_matrix[i,1] <- sum(abs(test$medv - test$pred_standard)) / n_test
  res_matrix[i,2] <- sum(abs(test$medv - test$pred_standard_PI)) / n_test
  res_matrix[i,3] <- sum(abs(test$medv - test$pred_partition_PI)) / n_test
  res_matrix[i,4] <- sum(abs(test$medv - test$pred_partition_PI_1)) / n_test
  res_matrix[i,5] <- sum(abs(test$medv - test$pred_partition_PI_2)) / n_test
  
  print(i)
}

Result <- colMeans(res_matrix, na.rm = T) # Mean Absolute Error (MAE) vector
names(Result) <- c("Standard", "Standard PI", "Partition PI", "Partition PI 1", "Partition PI 2")
sort(Result)
