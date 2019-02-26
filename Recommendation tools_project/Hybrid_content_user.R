#####################
### Hybrid RecSys ###
#####################
library("recommenderlab")
library("tm")
library("SnowballC")
library("dbscan")
library("proxy")





### Split train - Test ###
set.seed(2)


train_rows = sample(1:nrow(user_artists_transform_matrix), 0.7*nrow(user_artists_transform_matrix))

train <- as(user_artists_transform_matrix, "matrix")[train_rows,]
test <- as(user_artists_transform_matrix, "matrix")[-train_rows,]
test1 <- test[1:281, ]
test2 <- test[282:563, ]

### Compute individual models ###
CBTFIDF <- ContentBased(fin_matrix, test1, 3, 10, onlyNew=F)
IB <- UserBasedCF(train, test1, 3, 10, onlyNew=F)

### Transform results to lists (to be able to use the rowMeans function) ###
CBTFIDF_list <- as.list(CBTFIDF$prediction)
IB_list <- as.list(IB$prediction)

####################
### Compute Mean ###
####################
hybrid <- rowMeans(cbind(as.numeric(CBTFIDF_list), as.numeric(IB_list)), na.rm=T)

### Transform list back to matrix with correct number of dimensions ###
Hybrid_prediction <- matrix(hybrid, nrow=nrow(test1), ncol=ncol(test1))
rownames(Hybrid_prediction) <- rownames(test1)
colnames(Hybrid_prediction) <- colnames(test1)

### Evaluate ###
# RSME
RSME(CBTFIDF$prediction, test1)
RSME(IB$prediction, test1)
RSME(Hybrid_prediction, test1)

# Classification
Classification(CBTFIDF$prediction, test1, threshold=5)
Classification(IB$prediction, test1, threshold=5)
Classification(Hybrid_prediction, test1, threshold=5)

#########################
### Linear Regression ###
#########################

# Train a linear Regression
### flatten test1 dataset
test_list <- as.list(test1)

### Transform list and matrices to dataframe
test_df <- data.frame(matrix(unlist(test_list), byrow=T))
CBTFIDF_df <- data.frame(matrix(unlist(CBTFIDF_list), byrow=T))
IB_df <- data.frame(matrix(unlist(IB_list), byrow=T))

### Combine created dataframes
input <- cbind(test_df, CBTFIDF_df, IB_df)
colnames(input) <- c('TARGET', 'CBTFIDF', 'IB')

### Train the linear regression
fit <- lm(TARGET ~ CBTFIDF + IB, data=input)
summary(fit)

### Score Models
CBTFIDF2 <- ContentBased(fin_matrix, test2, 3, 10, onlyNew=F)
IB2 <- UserBasedCF(train, test2, 3, 10, onlyNew=F)

### Matrix to list
test_list2 <- as.list(test2)
CBTFIDF_list2 <- as.list(CBTFIDF2$prediction)
IB_list2 <- as.list(IB2$prediction)

### List to dataframe
test_df2 <- data.frame(matrix(unlist(test_list2), byrow=T))
CBTFIDF_df2 <- data.frame(matrix(unlist(CBTFIDF_list2), byrow=T))
IB_df2 <- data.frame(matrix(unlist(IB_list2), byrow=T))

### combine dataframes to have an input dataset for linear regression
input2 <- cbind(test_df2, CBTFIDF_df2, IB_df2)
colnames(input2) <- c('TARGET', 'CBTFIDF', 'IB')

### Predict using the model calculated on test2 dataset
Hybrid_lin_reg <- predict(fit, input2)

### Transform the list of results to matrix with the correct dimensions
Hybrid_lin_reg <- matrix(Hybrid_lin_reg, nrow=nrow(test2), ncol=ncol(test2))
rownames(Hybrid_lin_reg) <- rownames(test2)
colnames(Hybrid_lin_reg) <- colnames(test2)

# Evaluation
# RSME
RSME(CBTFIDF2$prediction, test2)
RSME(IB2$prediction, test2)
RSME(Hybrid_lin_reg, test2)

# Classification
Classification(CBTFIDF2$prediction, test2, threshold=5)
Classification(IB2$prediction, test2, threshold=5)
Classification(Hybrid_lin_reg, test2, threshold=5)
