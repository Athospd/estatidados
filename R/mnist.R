library(torch)
library(torchvision)
library(mestrado) # remotes::install_github("athospd/mestrado")
library(tidymodels)

# dados -------------------------------------------------------------------
mnist_train <- mnist_dataset(
  root = "/media/athos/DATA/OneDrive/Documents/datasets/",
  train = TRUE,
  download = TRUE
)

mnist_test <- mnist_dataset(
  root = "/media/athos/DATA/OneDrive/Documents/datasets/",
  train = FALSE
)

# EDA ---------------------------------------------------------------------
amostrinha <- mnist_train[1]
dim(amostrinha$x)
amostrinha$y

map_dfr(1:18, ~ {
  img <- mnist_train[.x]
  torch_tensor(img$x) %>%
    image_tensors_to_tbl() %>%
    mutate(i = .x)
}) %>%
  ggpixelgrid()


# dataloader --------------------------------------------------------------
ds_train <- dataloader(
  mnist_train,
  batch_size = 32,
  shuffle = TRUE,
  num_workers = 1
)

ds_test <- dataloader(
  mnist_test,
  batch_size = 32,
  num_workers = 1
)

teste <- enumerate(ds_train)

batch_de_teste <- teste[[1]][[1]]*1.
dim(batch_de_teste) # (32, 28, 28)

y_de_teste <- teste[[1]][[2]]
dim(y_de_teste) # (32)

# exemplinho de operacoes em imagens --------------------------------------
img <- t(mnist_train[3]$x[28:1,])
image(img, col = gray.colors(257, 0, 1))

img_torch <- torch_tensor(img*1.)
img_torch <- img_torch$unsqueeze(1)$unsqueeze(1)


##

conv1 <- nn_conv2d(1, 1, c(3,3)) # <--------- operador de CNN!

##

# imagem convolucionada
img_convolucionada <- conv1(img_torch)
image(as.matrix(img_convolucionada$squeeze()), col = gray.colors(257, 0, 1))

# filtro
image(as.matrix(conv1$parameters$weight$squeeze()), col = gray.colors(257, 0, 1))


# modelo ------------------------------------------------------------------
# meu_modelo <- torch::nn_module(
#   "MeuModelo",
#   initialize = function() {
#     # parametros
#     # self$conv1 <- nn_conv2d(...)
#   },
#   forward = function() {
#     # o deep learninzinho
#     # o encadeamento de camadas
#   }
# )

cnn_do_athos <- nn_module(
  "CNNdoAthos",
  initialize = function() {
    self$conv1 <- nn_conv2d( 1,  32, kernel_size = c(3,3))
    self$conv2 <- nn_conv2d(32,  64, kernel_size = c(3,3))
    self$conv3 <- nn_conv2d(64, 128, kernel_size = c(3,3))

    self$linear1 <- nn_linear(128, out_features = 64)
    self$linear2 <- nn_linear(64, out_features = 10)

  },
  forward = function(batch_de_imagens_em_forma_de_tensors) {
    batch_de_imagens_em_forma_de_tensors$unsqueeze(2) %>%
      self$conv1() %>%
      torch_relu() %>%
      nnf_avg_pool2d(c(2,2)) %>%

      self$conv2() %>%
      torch_relu() %>%
      nnf_avg_pool2d(c(2,2)) %>%

      self$conv3() %>%
      torch_relu() %>%
      nnf_avg_pool2d(c(2,2)) %>%

      torch_flatten(start_dim = 2) %>%
      self$linear1() %>%
      torch_relu() %>%
      self$linear2()
  }
)

modelinho <- cnn_do_athos()

# teste
modelinho(batch_de_teste)

# otimizador --------------------------------------------------------------
otimizador <- optim_adam(modelinho$parameters)

# funcao de custo ---------------------------------------------------------
funcao_de_custo <- nn_cross_entropy_loss() # binary cross entropy

# ajuste do modelo --------------------------------------------------------
library(progress)

for(epoch in 1:10) {
  pb <- progress_bar$new(total = length(ds_train))
  for(batch in enumerate(ds_train)) {
    otimizador$zero_grad()

    esperado <- modelinho(batch[[1]]*1.)
    observado <- batch[[2]]
    custo <- funcao_de_custo(esperado, observado)
    custo$backward()

    otimizador$step()
    pb$tick()
  }
}

# predicao ----------------------------------------------------------------
predicoes <- tibble(
  observado = numeric(0),
  esperado = numeric(0)
)

for(batch in enumerate(ds_test)) {
  esperado <- modelinho(batch[[1]]*1.) %>% torch_argmax(dim = 2)

  predicoes_batch <- tibble(
    esperado = as.numeric(esperado),
    observado = as.numeric(batch[[2]])
  )

  predicoes <- bind_rows(predicoes, predicoes_batch)
}

# matriz de confusao ------------------------------------------------------

predicoes %>%
  mutate_all(as.factor) %>%
  conf_mat(esperado, observado)
