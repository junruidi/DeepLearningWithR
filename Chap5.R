##################################################
## Chapter 5. Deep Learning for Computer Vision ##
## Di, Junrui                                   ##
##################################################


# 1. Convnet for MNIST Data -----------------------------------------------

library(keras)


model = keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu")

model = model %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

mnist =dataset_mnist()
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist

train_images = array_reshape(train_images, c(60000, 28, 28, 1)) 
train_images = train_images / 255

test_images = array_reshape(test_images, c(10000, 28, 28, 1))
test_images = test_images / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(
  train_images, train_labels,
  epochs = 5, batch_size=64
)

 model %>% evaluate(test_images, test_labels)

 
test_data = array_reshape(test_images[1:20,,,] , dim = c(20,28,28,1))
 apply(model %>% predict(test_data), MARGIN = 1, function(x) which.max(x) - 1)
 


# 2. Cat/Dog Classification -----------------------------------------------
rm(list = ls())
library(keras) 
setwd("~/OneDrive - Pfizer/Learning/Deep Learning with R/cats_and_dogs_small/")
 
train_cats_dir = "train/cats/"
train_dogs_dir = "train/dogs/"
validation_cats_dir = "validation/cats/"
validation_dogs_dir = "validation/dogs/"
test_cats_dir = "test/cats/"
test_dogs_dir = "test/dogs/"

train_dir = "train/"
validation_dir = "validation/"
test_dir = "test/"
## This is a balanced binary classification problem --> accuracy can be used as a meausre of success

## 2.1 Define the network

model = keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")
summary(model)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4), metrics = c("acc")
)

## 2.2 Data preprocessing

train_datagen = image_data_generator(rescale = 1/255) 
validation_datagen = image_data_generator(rescale = 1/255)

train_generator = flow_images_from_directory( train_dir,
                                               train_datagen,
                                               target_size = c(150, 150), # an arbitrary choice
                                               batch_size = 20,
                                               class_mode = "binary")

validation_generator  = flow_images_from_directory( validation_dir,
                                                    validation_datagen,
                                                    target_size = c(150, 150),
                                                    batch_size = 20,
                                                    class_mode = "binary")
batch = generator_next(train_generator)

## 2.3 Fitting the model
history = model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50
)

# ## 2.4 Save the model (since they are large)
# model %>% save_model_hdf5("cats_and_dogs_small_1.h5")

## 2.5 Data augumentation
datagen = image_data_generator(
  rescale = 1/255, 
  rotation_range = 40, 
  width_shift_range = 0.2, 
  height_shift_range = 0.2, 
  shear_range = 0.2, 
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)
fnames = list.files(train_cats_dir, full.names = TRUE)
img_path = fnames[[3]]


img = image_load(img_path, target_size = c(150, 150)) 
img_array = image_to_array(img)
img_array = array_reshape(img_array, c(1, 150, 150, 3))

augmentation_generator = flow_images_from_data(  img_array,
                                                 generator = datagen,
                                                 batch_size = 1)
for (i in 1:4) {
  batch <- generator_next(augmentation_generator)
  plot(as.raster(batch[1,,,]))
  }

## 2.6 Convnet with dropout

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4), metrics = c("acc")
)


# 3. Use a pretrained network ---------------------------------------------

## 3.1 Feature extraction

library(keras)

conv_base = application_vgg16(
  weights = "imagenet",
  include_top = FALSE, #refers to including (or not) the densely 
                       #connected classifier on top of the network
  input_shape = c(150,150,3)
)

