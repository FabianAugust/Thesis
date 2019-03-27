library(e1071)


# Load and prepare iris dataset -----------------------------------------------------

df <- iris
df <- df[df$Species != "setosa", ] # Remove one Species to get binary classification
df$Species <- droplevels(df$Species)

# View separation of variables
plot(df$Petal.Length, df$Petal.Width, col = c("red", "blue")[df$Species])
plot(df$Sepal.Length, df$Sepal.Width, col = c("red", "blue")[df$Species])

df$Species <- ifelse(as.numeric(df$Species) == 2, 1, -1) # Relabel outcome


# SVM -------------------------------------------------------------------------------

rand <- sample(1:100, 20) # Randomly choose 20 individuals for evaluation data

# Create test and training datasets
test <- df[rand,]
train <- df[-rand, ]

# Perform knowledge transfer regressions
res1 <- lm(Petal.Length ~ Sepal.Length + Sepal.Width, data = train)
train$Petal.Length_p <- predict(res1)
  
res2 <- lm(Petal.Width ~ Sepal.Length + Sepal.Width, data = train)
train$Petal.Width_p <- predict(res2)
  
test$Petal.Length_p <- predict(res1, newdata = test)
  
test$Petal.Width_p <- predict(res2, newdata = test)

# Create a standard dataset with only decision features and knowledge transfer set with
# transfered privileged features
standard_dat <- train[,c(1,2,5)]
knowtran_dat <- train[,c(1,2,5,6,7)]

# perform svm
svm.model1 <- svm(Species ~ ., data = standard_dat, type='C-classification', kernel='linear', 
                 scale=T)

svm.model2 <- svm(Species ~ ., data = knowtran_dat, type='C-classification', kernel='linear', 
                 scale=T)

# Predict on test data
pred1 <- predict(svm.model1, test[,-c(3,4,5,6)])
pred2 <- predict(svm.model2, test[,-c(3,4,5)])

# Evaulate result
res_standard <- table(test$Species, pred1)
res_knowtran <- table(test$Species, pred2)

(20 - sum(diag(res_standard))) / 20 # Error rate for standard method
(20 - sum(diag(res_knowtran))) / 20 # Error rate for knowledge transfer

# Add the code into a function and do a simple evaluation ---------------------------

svm_eval <- function(n){
  rand <- sample(1:100, n)
  
  test <- df[rand,]
  train <- df[-rand, ]
  
  res1 <- lm(Petal.Length ~ Sepal.Length + Sepal.Width, data = train)
  train$Petal.Length_p <- predict(res1)
  
  res2 <- lm(Petal.Width ~ Sepal.Length + Sepal.Width, data = train)
  train$Petal.Width_p <- predict(res2)
  
  test$Petal.Length_p <- predict(res1, newdata = test)
  
  test$Petal.Width_p <- predict(res2, newdata = test)
  
  
  standard_dat <- train[,c(1,2,5)]
  knowtran_dat <- train[,c(1,2,5,6,7)]
  
  svm.model1 <- svm(Species ~ ., data = standard_dat, type='C-classification', kernel='linear', 
                    scale=T)
  
  svm.model2 <- svm(Species ~ ., data = knowtran_dat, type='C-classification', kernel='linear', 
                    scale=T)
  
  pred1 <- predict(svm.model1, test[,-c(3,4,5,6)])
  pred2 <- predict(svm.model2, test[,-c(3,4,5)])
  
  res_standard <- table(test$Species, pred1)
  res_knowtran <- table(test$Species, pred2)
  
  standard <- (n - sum(diag(res_standard))) / n # Error rate for standard method
  knowtran <- (n - sum(diag(res_knowtran))) / n # Error rate for knowledge transfer
  
  return(c(standard, knowtran))
  
  
}

# Simple evaluation
res <- list()
for(i in 1:1000){
  res[[i]] <- svm_eval(20)
}
res <- Reduce('rbind', res)
res2 <- colMeans(res)
names(res2) <- c("Standard", "Knowledge transfer")
res2*100 # Average error rate

