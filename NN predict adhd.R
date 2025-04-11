# Set working directory
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load required libraries
library(keras)
library(caret)
library(dplyr)
library(tensorflow)
library(torch)

# Read the data set
data <- read.csv("nn_data.csv")

# Remove the first column (index) and Sex_F column
data <- data[, -c(1, ncol(data))]

# Convert TRUE/FALSE character strings to numeric (0/1)
data <- as.data.frame(lapply(data, function(x) {
  if(is.character(x)) {
    ifelse(toupper(x) == "TRUE", 1, 0)
  } else {
    as.numeric(x)
  }
}))

# Convert ADHD_Outcome to numeric (0 and 1)
data$ADHD_Outcome <- as.numeric(data$ADHD_Outcome)

# Calculate class weights
class_counts <- table(data$ADHD_Outcome)
total_samples <- sum(class_counts)
weight_0 <- total_samples / (2 * class_counts[1])  # Weight for class 0
weight_1 <- total_samples / (2 * class_counts[2])  # Weight for class 1

# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(data$ADHD_Outcome, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Scale the numeric features - excluding ADHD_Outcome which is y
preprocess_params <- preProcess(train_data[, 
                                           -which(names(train_data) == "ADHD_Outcome")], 
                                method = c("center", "scale"))
train_scaled <- predict(preprocess_params, train_data)
test_scaled <- predict(preprocess_params, test_data)

# Prepare data for torch
X_train <- as.matrix(train_scaled[, -ncol(train_scaled)])
y_train <- as.matrix(train_scaled[, ncol(train_scaled)])
X_test <- as.matrix(test_scaled[, -ncol(test_scaled)])
y_test <- as.matrix(test_scaled[, ncol(test_scaled)])

# Convert to torch tensors
X_train_tensor <- torch_tensor(X_train, dtype = torch_float32())
y_train_tensor <- torch_tensor(y_train, dtype = torch_float32())
X_test_tensor <- torch_tensor(X_test, dtype = torch_float32())
y_test_tensor <- torch_tensor(y_test, dtype = torch_float32())

# Create class weights tensor -- help to address the imbalance in 0 and 1 class
class_weights <- torch_tensor(c(weight_0, weight_1), dtype = torch_float32())

# Define weighted binary cross entropy loss
weighted_bce_loss <- function(pred, target) {
  # Create a tensor of weights for each sample
  weights <- torch_zeros_like(target)
  weights[target == 0] <- weight_0
  weights[target == 1] <- weight_1
  
  # Calculate weighted loss
  loss <- nnf_binary_cross_entropy(pred, target, weight = weights)
  return(loss)
}

# Define the neural network architecture
nn_model <- nn_module(
  initialize = function(input_size) {
    self$fc1 <- nn_linear(input_size, 128)
    self$fc2 <- nn_linear(128,64)
    self$fc3 <- nn_linear(64,32)
    self$fc4 <- nn_linear(32, 1)
  },
  
  forward = function(x) {
    x %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$fc2() %>%
      nnf_relu() %>%
      self$fc3() %>%
      nnf_relu() %>%
      self$fc4() %>%
      torch_sigmoid()
  }
)

# Create model instance
model <- nn_model(ncol(X_train))

# Create data loader for mini-batches
train_ds <- tensor_dataset(X_train_tensor, y_train_tensor)
train_dl <- dataloader(train_ds, batch_size = 16, shuffle = TRUE)

# Define optimizer
optimizer <- optim_adam(model$parameters, lr = 0.03)

# Training loop
for (epoch in 1:100) {
  model$train()
  train_loss <- 0
  n_batches <- 0
  
  coro::loop(for (batch in train_dl) {
    optimizer$zero_grad()
    
    # Get input and target tensors
    x <- batch[[1]]
    y <- batch[[2]]
    
    # Forward pass
    pred <- model(x)
    
    # Compute weighted loss
    loss <- weighted_bce_loss(pred, y)
    
    # Backward pass and optimization
    loss$backward()
    optimizer$step()
    
    train_loss <- train_loss + loss$item()
    n_batches <- n_batches + 1
  })
  
  # Print epoch statistics
  cat(sprintf("Epoch: %d, Loss: %.4f\n", epoch, train_loss / n_batches))
}

# Make predictions
model$eval()
with_no_grad({
  test_pred <- model(X_test_tensor)
  predicted_classes <- as.numeric(test_pred > 0.5)
})

# Calculate accuracy
accuracy <- mean(predicted_classes == as.numeric(y_test))
print(paste("Accuracy:", accuracy))

# Create confusion matrix
confusion_matrix <- table(Predicted = predicted_classes,
                          Actual = as.numeric(y_test))
print("Confusion Matrix:")
print(confusion_matrix)

# Calculate additional metrics
precision <- confusion_matrix[2,2] / sum(confusion_matrix[2,])
recall <- confusion_matrix[2,2] / sum(confusion_matrix[,2])
f1_score <- 2 * (precision * recall) / (precision + recall)

print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1 Score:", f1_score))

