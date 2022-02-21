#### Our goal will be to predict whether a person has heart failure.

library("caret")
library("class")
library("faraway")
library("InformationValue")
library("leaps")
library("randomForest")
library("caTools")
library("tidyverse")
library("pROC")

set.seed(9999)

## Load the dataset.

heart_disease <- read.csv("~/Documents/Second Semester/Stat Mod & Comp/Final Project/heart_failure_clinical_records_dataset.csv")


## Data initialization and split to train/test.

trainIndex <- createDataPartition(heart_disease$DEATH_EVENT, p=0.70, list=FALSE, times=1)

heart_disease_train <- heart_disease[trainIndex,]
heart_disease_test <- heart_disease[-trainIndex,]

## BIC Model Selection
heart_disease_train_BIC <- regsubsets(DEATH_EVENT ~ ., data=heart_disease_train)
heart_disease_train_BIC
heart_disease_train_BIC_sum <- summary(heart_disease_train_BIC)
heart_disease_train_BIC_sum
heart_disease_train_BIC_sum$which

plot(heart_disease_train_BIC_sum$bic, ylab="BIC", xlab="Number of Predictors")

#### Through BIC model selection, we picked the parameters with most importance?


# Save File

jpeg(file="BIC_ModelSelection.jpeg")
plot(heart_disease_train_BIC_sum$bic, ylab="BIC", xlab="Number of Predictors")
dev.off()

## Logistic Regression
lmod_heart_disease <- glm(DEATH_EVENT ~ ejection_fraction + serum_creatinine + time,
                          family=binomial, heart_disease_train)
summary(lmod_heart_disease)

#### Prediction
lmod_heart_disease_prob <- predict(lmod_heart_disease, heart_disease_test, type="response")
lmod_heart_disease_prob
optCutoff <- optimalCutoff(heart_disease_test$DEATH_EVENT,lmod_heart_disease_prob)


#optCutoff. what is this?

heart_disease_test_pred_lmod <- heart_disease_test %>%
  mutate(predict=1*(lmod_heart_disease_prob > optCutoff)) %>%
  mutate(accurate=1*(predict==DEATH_EVENT))
lmod_acc <- sum(heart_disease_test_pred_lmod$accurate)/nrow(heart_disease_test_pred_lmod)

#lmod_acc. what is this?

#### Matrix Analysis]
confusion_matrix_lmod <- as.data.frame(table(heart_disease_test_pred_lmod$DEATH_EVENT,heart_disease_test_pred_lmod$predict))
confusion_matrix_lmod
confusion_matrix_lmod$Var1 <- as.character(confusion_matrix_lmod$Var1)
confusion_matrix_lmod$Var2 <- as.character(confusion_matrix_lmod$Var2)
confusion_matrix_lmod$Var1[confusion_matrix_lmod$Var1 == 0] <- "Survived"
confusion_matrix_lmod$Var1[confusion_matrix_lmod$Var1 == 1] <- "Died"
confusion_matrix_lmod$Var2[confusion_matrix_lmod$Var2 == 0] <- "Survived"
confusion_matrix_lmod$Var2[confusion_matrix_lmod$Var2 == 1] <- "Died"

ggplot(data=confusion_matrix_lmod, mapping=aes(x=Var1,y=Var2)) +
  geom_tile(aes(fill=Freq), color = "white") +
  geom_text(aes(label=sprintf("%1.0f", Freq)), vjust=1) +
  scale_fill_gradient(low="steelblue", high="red") +
  theme_bw() + theme(legend.position="none") +
  xlab("Predicted") + ylab("Actual") + ggtitle("Predicted versus Actual - Logistic Regression")

#Save File
jpeg(file="confusionMatrix_logisticRegression.jpeg")
ggplot(data=confusion_matrix_lmod, mapping=aes(x=Var1,y=Var2)) +
  geom_tile(aes(fill=Freq), color = "white") +
  geom_text(aes(label=sprintf("%1.0f", Freq)), vjust=1) +
  scale_fill_gradient(low="steelblue", high="red") +
  theme_bw() + theme(legend.position="none") +
  xlab("Predicted") + ylab("Actual") + ggtitle("Predicted versus Actual - Logistic Regression")
dev.off()

## K-Nearest Neighbors

#### Selecting the same parameters as the logistic model and train and test

heart_disease_train_filtered_x <- heart_disease_train %>%
  select(ejection_fraction,serum_creatinine,time)
heart_disease_train_filtered_y <- heart_disease_train$DEATH_EVENT
heart_disease_test_filtered_x <- heart_disease_test %>%
  select(ejection_fraction,serum_creatinine,time)
heart_disease_test_filtered_y <- heart_disease_test$DEATH_EVENT

#### Calculating the error.
calc_error <- function(actual, predicted){
  error <- mean(actual != predicted)
  return(error)
}

#### Determine k.
ks = 1:20
errors <- rep(x=0, times=length(ks))

for (i in seq_along(ks)) {
  prediction <- knn(train=heart_disease_train_filtered_x,
                    test=heart_disease_test_filtered_x,
                    cl=heart_disease_train_filtered_y,
                    k=ks[i])
  errors[i] <- calc_error(heart_disease_test_filtered_y, prediction)
}

plot(errors, type = "b", col = "dodgerblue", cex = 1, pch = 20, 
     xlab = "k, Number of Neighbors", ylab = "Classification Error",
     main = "(Test) Error Rate vs Neighbors")
abline(h = min(errors), col = "darkorange", lty = 3)

#### Saving the file
jpeg(file="KNN_ErrorRate.jpeg")
plot(errors, type = "b", col = "dodgerblue", cex = 1, pch = 20, 
     xlab = "k, Number of Neighbors", ylab = "Classification Error",
     main = "(Test) Error Rate vs Neighbors")
abline(h = min(errors), col = "darkorange", lty = 3)
dev.off()

#### Finding the minimum error
min(errors)
k <- which(errors == min(errors))
k

#### Final KNN
heart_disease_knn <- knn(train=heart_disease_train_filtered_x,
                         test=heart_disease_test_filtered_x,
                         cl=heart_disease_train_filtered_y,
                         k=k)
knn_acc <- 1-calc_error(heart_disease_test_filtered_y,heart_disease_knn)

#Analysis
confusion_matrix_knn <- as.data.frame(table(heart_disease_test_filtered_y,heart_disease_knn))
confusion_matrix_knn$heart_disease_test_filtered_y <- as.character(confusion_matrix_knn$heart_disease_test_filtered_y)
confusion_matrix_knn$heart_disease_knn <- as.character(confusion_matrix_knn$heart_disease_knn)
confusion_matrix_knn$heart_disease_test_filtered_y[confusion_matrix_knn$heart_disease_test_filtered_y == 0] <- "Survived"
confusion_matrix_knn$heart_disease_test_filtered_y[confusion_matrix_knn$heart_disease_test_filtered_y == 1] <- "Died"
confusion_matrix_knn$heart_disease_knn[confusion_matrix_knn$heart_disease_knn == 0] <- "Survived"
confusion_matrix_knn$heart_disease_knn[confusion_matrix_knn$heart_disease_knn == 1] <- "Died"

ggplot(data=confusion_matrix_knn, mapping=aes(x=heart_disease_knn,y=heart_disease_test_filtered_y)) +
  geom_tile(aes(fill=Freq), color = "white") +
  geom_text(aes(label=sprintf("%1.0f", Freq)), vjust=1) +
  scale_fill_gradient(low="steelblue", high="red") +
  theme_bw() + theme(legend.position="none") +
  xlab("Predicted") + ylab("Actual") + ggtitle("Predicted versus Actual - KNN")

#Save File
jpeg("confusionMatrix_KNN.jpeg")
ggplot(data=confusion_matrix_knn, mapping=aes(x=heart_disease_knn,y=heart_disease_test_filtered_y)) +
  geom_tile(aes(fill=Freq), color = "white") +
  geom_text(aes(label=sprintf("%1.0f", Freq)), vjust=1) +
  scale_fill_gradient(low="steelblue", high="red") +
  theme_bw() + theme(legend.position="none") +
  xlab("Predicted") + ylab("Actual") + ggtitle("Predicted versus Actual - KNN")
dev.off()

## Random Forest 

#Random Forest Modeling
random_forest_model <- randomForest(formula=as.factor(DEATH_EVENT) ~ ejection_fraction + serum_creatinine + time, data=heart_disease_train)

#Prediction
heart_disease_test_pred_rf <- heart_disease_test %>%
  mutate(pred = predict(random_forest_model, heart_disease_test))%>%
  mutate(accurate=1*(pred==DEATH_EVENT))
rf_acc <- sum(heart_disease_test_pred_rf$accurate)/nrow(heart_disease_test_pred_rf)

#Analysis
confusion_matrix_rf <- as.data.frame(table(heart_disease_test_pred_rf$DEATH_EVENT,heart_disease_test_pred_rf$pred))
confusion_matrix_rf$Var1 <- as.character(confusion_matrix_rf$Var1)
confusion_matrix_rf$Var2 <- as.character(confusion_matrix_rf$Var2)
confusion_matrix_rf$Var1[confusion_matrix_rf$Var1 == 0] <- "Survived"
confusion_matrix_rf$Var1[confusion_matrix_rf$Var1 == 1] <- "Died"
confusion_matrix_rf$Var2[confusion_matrix_rf$Var2 == 0] <- "Survived"
confusion_matrix_rf$Var2[confusion_matrix_rf$Var2 == 1] <- "Died"

ggplot(data=confusion_matrix_rf, mapping=aes(x=Var1,y=Var2)) +
  geom_tile(aes(fill=Freq), color = "white") +
  geom_text(aes(label=sprintf("%1.0f", Freq)), vjust=1) +
  scale_fill_gradient(low="steelblue", high="red") +
  theme_bw() + theme(legend.position="none") +
  xlab("Predicted") + ylab("Actual") + ggtitle("Predicted versus Actual - Random Forest")

#Save File
jpeg("confusionMatrix_RandomForest.jpeg")
ggplot(data=confusion_matrix_rf, mapping=aes(x=Var1,y=Var2)) +
  geom_tile(aes(fill=Freq), color = "white") +
  geom_text(aes(label=sprintf("%1.0f", Freq)), vjust=1) +
  scale_fill_gradient(low="steelblue", high="red") +
  theme_bw() + theme(legend.position="none") +
  xlab("Predicted") + ylab("Actual") + ggtitle("Predicted versus Actual - Random Forest")
dev.off()

###############
heart_disease
dim(heart_disease)
head(heart_disease)

#### Use the summary function for viewing metrics related to our data.

summary(heart_disease)

# Upon closer analysis, an issue with the column types is detected. 
# In R, a categorical variable is a factor. However, summary(heart_disease) revealed that "sex" was incorrectly treated as a number, as opposed to a factor (0 = Female, 1 = Male).

#### Use the sapply function to view each column type.

sapply(heart_disease, class)

#Sex - Gender of patient Male = 1, Female =0
#Age - Age of patient
#Diabetes - 0 = No, 1 = Yes
#Anaemia - 0 = No, 1 = Yes
#High_blood_pressure - 0 = No, 1 = Yes
#Smoking - 0 = No, 1 = Yes
#DEATH_EVENT - 0 = No, 1 = Yes

#### Use the transform function to change the in built type of each feature.
heart_disease <- transform(
  heart_disease,
  age=as.integer(age),
  sex=as.factor(sex),
  diabetes=as.factor(diabetes),
  anaemia=as.factor(anaemia),
  high_blood_pressure=as.factor(high_blood_pressure),
  smoking=as.factor(smoking),
  DEATH_EVENT=as.factor(DEATH_EVENT)
)

#creatinine_phosphokinase
#ejection_fraction
#platelets
#serum_creatinine
#serum_sodium
#time

#### View the corrected column types.
sapply(heart_disease, class)
summary(heart_disease)

#### Split the data into train and test.

#### Initialize an instance of Random Forest.


lrROC <- roc(heart_disease_test_pred_lmod$DEATH_EVENT ~ heart_disease_test_pred_lmod$predict, plot=TRUE, print.auc=TRUE, col="green", lwd=3, legacy.axes=TRUE, main="ROC Curves")
knnROC <- roc(heart_disease_test_filtered_y ~ as.integer(heart_disease_knn), plot=TRUE, print.auc=TRUE, col="blue", lwd=3, legacy.axes=TRUE, main="ROC Curves", add=TRUE, print.auc.y = .4)
rfROC <- roc(heart_disease_test_pred_rf$DEATH_EVENT ~ as.integer(heart_disease_test_pred_rf$pred), plot=TRUE, print.auc=TRUE, col="red", lwd=3, legacy.axes=TRUE, main="ROC Curves", add=TRUE, print.auc.y=.8)


heart_disease_test_pred_lmod
lmod_acc <- sum(heart_disease_test_pred_lmod$accurate)/nrow(heart_disease_test_pred_lmod)
dim(heart_disease)
heart_disease_knn
knn_acc

