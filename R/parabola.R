library(torch)

f <- function(x) x * (x - 1)

plot(f)

# OBJETIVO: queremos achar o x que minimiza f
x <- torch_tensor(0, requires_grad = TRUE)

x$grad

# gradient descent
# x[t+1] = x[t] - a * df/dx[t]

# quem faz o gradient descent é o otimizador
otimizador <- optim_adam(x)


fx <- f(x) # calcula f(x)
fx$backward() # encontra a direção pro fundo da f (calcula as derivadas)
x$grad # o gradiente de x está mudado!


# dá um passo em direção ao fundo de f(x)
# atualiza o valor de x
otimizador$step()
x

points(x, fx, col = "red")

# agora vamos repetir o procedimento até chegar ao fundo
for(passo in 2:100) {
  otimizador$zero_grad()
  fx <- f(x)
  fx$backward()
  otimizador$step()
  points(x, fx, col = "red")
  Sys.sleep(0.1)
}

# o passo tá muito pequeno! temos que mudar a "learning rate"

