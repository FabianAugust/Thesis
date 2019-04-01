library(MASS)
library(e1071)

# Data simulation functions ---------------------------------------------------------

SimulateTrain <- function(l, noise){
  # Simulate training data
  Z <- data.frame('X1' = runif(l, -1, 1), 'X2' = runif(l, -1, 1))
  Z$Y <- sign(Z$X1 + Z$X2)
  Z$X3 <- Z$X1 + Z$X2 + noise * rnorm(l, 0, 1)
  
  fit <- lm.ridge(X3 ~ X1 + X2, data = Z)
  preds <- as.matrix(cbind(const=1,Z[,c("X1", "X2")])) %*% coef(fit)
  Z$Xstar <- preds
  
  return(list(Z[,c("Y", "X1", "X2", "X3", "Xstar")], fit))
}

SimulateTest <- function(l = 10000, noise, fit){
  # Simulate test data
  Z <- data.frame('X1' = runif(l, -1, 1), 'X2' = runif(l, -1, 1))
  Z$Y <- sign(Z$X1 + Z$X2)
  Z$X3 <- Z$X1 + Z$X2 + noise * rnorm(l, 0, 1)
  
  preds <- as.matrix(cbind(const=1,Z[,c("X1", "X2")])) %*% coef(fit)
  Z$Xstar <- preds
  
  return(Z[,c("Y", "X1", "X2", "X3", "Xstar")])
}


# SVM functions ---------------------------------------------------------------------

svm_run2 <- function(dat, cost1, gamma1, cost2, gamma2){
  
  svm.model1 <- svm(Y ~ X1 + X2, data = dat,
                    type='C-classification', kernel='radial',
                    cost = cost1, gamma = gamma1)
  
  svm.model2 <- svm(Y ~ Xstar, data = dat,
                    type='C-classification', kernel='radial', 
                    cost = cost2, gamma = gamma2)
  
  return(list(svm.model1, svm.model2))
}

svm_pred <- function(W, mod){
  l <- dim(W)[1]
  
  pred1 <- predict(mod[[1]], W)
  pred2 <- predict(mod[[2]], W)
  
  res_standard <- table(W$Y, pred1)
  res_knowtran <- table(W$Y, pred2)
  
  standard <- ((l - sum(diag(res_standard))) / l)*100 # Error rate for standard method
  knowtran <- ((l - sum(diag(res_knowtran))) / l)*100 # Error rate for knowledge transfer
  
  c(standard, knowtran)
}


# Evaluation ------------------------------------------------------------------------

# Cost and gamma values
C <- 2^seq(-5, 5, 0.5)
G <- 2^seq(-6, 6, 0.5)

res_mat <- matrix(NA, nrow = 10, ncol = 2)
for(i in 1:50){
  # Simulate data
  out <- SimulateTrain(40, 0.01)
  df <- out[[1]]
  W <- SimulateTest(10000, 0.01, out[[2]])
  
  param1 <- tune.svm(df[,c("X1", "X2")], y = df[,"Y"], cost = C, gamma = G,
                         tunecontrol = tune.control(cross = 6))
  
  param2 <- tune.svm(df[,c("Xstar")], y = df[,"Y"], cost = C, gamma = G,
                          tunecontrol = tune.control(cross = 6))
  
  mods <- svm_run2(df, cost1 = param1$best.parameters[2],
                   gamma1 = param1$best.parameters[1], 
                   cost2 = param2$best.parameters[2],
                   gamma2 = param2$best.parameters[1])
  
  res_mat[i,] <- svm_pred(W, mods)
}
colMeans(res_mat)





