##############################################
## Chapter 2: Mathematical Building Blocks  ##
## of neural networks                       ##
##############################################



# 2.1 Structure of neural network -----------------------------------------
### Through an example of MNIST classification ###

## 1) Loading the data
library(keras)
mnist = dataset_mnist()
train_images = mnist$train$x
train_labels = mnist$train$y
test_images = mnist$test$x
test_labels = mnist$test$y

## 2) Network architecture
network = keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>%
  layer_dense(units = 10, activation = "softmax")
summary(network)

## 3) Compile
network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

## 4) Prepare the data (vectorization as we are using ANN)
train_images = array_reshape(train_images, c(60000, 28 * 28))
train_images = train_images / 255
test_images = array_reshape(test_images, c(10000, 28 * 28)) 
test_images = test_images / 255

## 5) Prepare the labels
## to_categorical function created the 0/1 vector indicating the class
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

## 6) Fit
model_fit = network %>% fit(train_images, train_labels, 
                epochs = 5, batch_size = 128,
                validation_split = 0.2)

## 7) Check the performance
network %>% evaluate(test_images, test_labels)
plot(model_fit)

## 8) Predict
network %>% predict_classes(test_images[1:10,])


# 2.2 Data representation for NN ------------------------------------------

rm(list = ls())
library(keras)
mnist = dataset_mnist() 
train_images = mnist$train$x 
train_labels = mnist$train$y 
test_images = mnist$test$x 
test_labels <- mnist$test$y

# ranks
length(dim(train_images))

# shapes
dim(train_images)

# datatype
typeof(train_images)

plot(as.raster(train_images[5,,], max = 255))


# 2.3 Tensor Operation ----------------------------------------------------

## 1) element wise operation
x = matrix(rnorm(12,3,4),ncol = 4)
y = matrix(rnorm(12,3,4),ncol = 4)

z = x+y ## Elementwise addition
z_relu = pmax(z,0) ## Elementwise relu

## 2) tensors with different dimension
x = array(round(runif(1000,0,9)), dim = c(64, 3, 32, 10))
y = array(-1, dim = c(32,10))
z = sweep(x, c(3,4), y, pmin)


# 2.4 Gradient based method -----------------------------------------------


