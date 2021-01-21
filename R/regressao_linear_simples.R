library(torch)
library(tidyverse)

# dados -------------------------------------------------------------------
plot(cars)

# modelo ------------------------------------------------------------------
rmse <- function(y, pred) {
  torch_sqrt(torch_mean((y - pred)^2))
}

# b0 e b1: queremos otimizá-los, por isso requires_grad = TRUE
b0 <- torch_rand(1, requires_grad = TRUE)
b1 <- torch_rand(1, requires_grad = TRUE)

x <- torch_tensor(cars$speed)
y <- torch_tensor(cars$dist)

for(passo in 1:100) {

  predito <- b0 + b1 * x # regressão
  esperado <- y

  custo <- rmse(predito, esperado)
  custo$backward()

  with_no_grad(b0$subtract_(0.0001 * b0$grad)) # b0 - 0.001 * df/db0
  with_no_grad(b1$subtract_(0.0001 * b1$grad)) # b1 - 0.001 * df/db1


  Sys.sleep(0.05)
  plot(cars)
  lines(x, predito, col = "red", lwd = 5)
}




# Exemplo menos manual ----------------------------------------------------
lin <- nn_linear(1,1)
mse <- nn_mse_loss()
opt <- torch::optim_adam(lin$parameters, lr = 0.5)

x <- torch_tensor(cars$speed)
y <- torch_tensor(cars$dist)

x <- x$unsqueeze(2)
y <- y$unsqueeze(2)

custos <- c()
for(i in 1:100) {
  opt$zero_grad()
  pred <- lin(x)
  custo <- mse(y, pred)
  custo$backward()
  opt$step()

  custos <- c(custos, as.numeric(custo))
  Sys.sleep(0.05)
  plot(cars)
  lines(x, pred, col = "red", lwd = 5)
}

lin$parameters

b <- coef(lm(dist ~ speed, data = cars))
lines(x, b[1] + b[2]*x, col = "royalblue", lwd = 5)
