#####################################
## Chapter 3: Getting started with ##
## neural networks                 ##
#####################################



# 1. Binary classification: IMDB ------------------------------------------
library(keras)

## 1.1 Load in data
imdb = dataset_imdb(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

#%<-% multi-assignment operator to unpack a list into a set of dinstance variables#

# ## Convert index back to english words ##
# word_index <- dataset_imdb_word_index()
# reverse_word_index <- names(word_index)
# names(reverse_word_index) <- word_index
# decoded_review <- sapply(train_data[[1]], function(index) {
#   word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
#   if (!is.null(word)) word else "?" })

## 1.2 Encoding the integer sequences into a binary matrix
vectorize_sequences = function(sequences, dimension = 10000) {
  results = matrix(0, nrow = length(sequences), ncol = dimension) 
  for (i in 1:length(sequences)){
    results[i, sequences[[i]]] = 1 
  }
  results
}
x_train = vectorize_sequences(train_data) 
x_test = vectorize_sequences(test_data)
y_train = as.numeric(train_labels)
y_test = as.numeric(test_labels)

## 1.3 Define models
model = keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

## 1.4 Complie the model
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy"))

# ## or passing them as function objects
# model %>% compile(
#   optimizer = optimizer_rmsprop(lr = 0.001), loss = loss_binary_crossentropy,
#   metrics = metric_binary_accuracy
# )

## 1.5 Validation
val_indices =1:10000
x_val = x_train[val_indices,] 
partial_x_train = x_train[-val_indices,]
y_val = y_train[val_indices]
partial_y_train = y_train[-val_indices]

history = model %>% fit(
  partial_x_train, partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

plot(history)


## 1.6 Battle the overfitting
## Due to overfitting, start to build from strach. 
model <- keras_model_sequential() %>%
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)
model %>% fit(x_train, y_train, epochs = 4, batch_size = 512)
results = model %>% evaluate(x_test, y_test)


## 1.7 Prediction
model %>% predict(x_test[1:10,])


# 2. Multiclass classification --------------------------------------------

## 2.1 load the data
library(keras)
reuters = dataset_reuters(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters #labels 0 to 45

## 2.2 Data preparation
vectorize_sequences = function(sequences, dimension = 10000) {
  results = matrix(0, nrow = length(sequences), ncol = dimension) 
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1 
  results
}
x_train = vectorize_sequences(train_data) 
x_test = vectorize_sequences(test_data)

one_hot_train_labels <- to_categorical(train_labels)
one_hot_test_labels <- to_categorical(test_labels)

### 2.3 Build model 

model = keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

### 2.4 Validation
val_indices = 1:1000
x_val = x_train[val_indices,] 
partial_x_train = x_train[-val_indices,]
y_val = one_hot_train_labels[val_indices,] 
partial_y_train = one_hot_train_labels[-val_indices,]

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

plot(history)

### 2.5 retrain
model = keras_model_sequential() %>%
layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax")
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
history = model %>% fit( partial_x_train, partial_y_train,
                          epochs = 9,
                          batch_size = 512,
                          validation_data = list(x_val, y_val)
)
results = model %>% evaluate(x_test, one_hot_test_labels)

### 2.6 Prediction
predictions = model %>% predict(x_test)



# 3. Regression -----------------------------------------------------------

### 3.1 Load the data
library(keras)

dataset = dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset
train_data = scale(train_data)
test_data = scale(test_data)

### 3.2 build the model
build_model = function(){
  model = keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu",input_shape = dim(train_data)[[2]]) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)
  model %>% compile(
    optimizer = "rmsprop",
    loss = "mse",
    metrics = c("mae")
  )
}

### 3.3 K fold cross validation

k = 4
indices = sample(1:nrow(train_data))
folds = cut(indices, breaks = k, labels = F)

num_epochs = 100
all_scores = c()

for(i in 1:k){
  cat("processing fold #", i, "\n")
  
  val_indices = which(folds == i, arr.ind = T)
  val_data = train_data[val_indices,]
  val_targets = train_targets[val_indices]
  
  partial_train_data = train_data[-val_indices,] 
  partial_train_targets = train_targets[-val_indices]
  
  model = build_model()
  
  model %>% fit(partial_train_data, partial_train_targets,
                epochs = num_epochs, batch_size = 1, verbose = 0)
  results = model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores = c(all_scores, results[2])
}

all_scores

### Another training with more epochs. 

num_epochs = 500
all_mae_histories = NULL

for(i in 1:k){
  cat("processing fold #", i, "\n")
  
  val_indices = which(folds == i, arr.ind = T)
  val_data = train_data[val_indices,]
  val_targets = train_targets[val_indices]
  
  partial_train_data = train_data[-val_indices,] 
  partial_train_targets = train_targets[-val_indices]
  
  model = build_model()
  
  history = model %>% fit(
    partial_train_data, partial_train_targets,
    validaton_data = list(val_data, val_targets),
    epochs = num_epochs, batch_size = 1, verbose = 0)
  
  mae_history = history$metrics$mae
  all_mae_histories = rbind(all_mae_histories, mae_history)
}

average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)
ggplot2::ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_line()

### Final model
model = build_model()
model %>% fit(train_data, train_targets,
              epochs = 80, batch_size = 16, verbose = 0)
result = model %>% evaluate(test_data, test_targets)

