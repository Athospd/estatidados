library(torch)

# regressão logística ----------------------------------------------------
x <- torch_tensor(as.matrix(mtcars[, -8]))
x <- (x - torch_mean(x, dim = 1L))/torch_std(x, dim = 1L) # normalizar!
y <- torch_tensor(mtcars$vs)

preditor_linear <- nn_linear(dim(x)[2], 1) # XB (xis beta, preditor linear)

bce <- nn_bce_loss() # binary cross entropy
opt <- torch::optim_adam(preditor_linear$parameters, lr = 0.05)

# 1/(1 + e^(XB))

custos <- c()
for(i in 1:200) {
  opt$zero_grad()
  pred <- x %>% preditor_linear() %>% nnf_sigmoid() # g(XB)
  custo <- bce(pred, y)
  custo$backward()
  opt$step()

  custos <- c(custos, as.numeric(custo))
  Sys.sleep(0.05)
  plot(custos, col = "royalblue", type = "l", xlim = c(1, 200), ylim = c(0,0.6))
}

preditor_linear$parameters

y = as.numeric(y)
yh = x %>% preditor_linear() %>% nnf_sigmoid() %>% as.numeric()

table(y, yh > 0.5)
