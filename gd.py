from random import randint

def f(x):
    return 3*x + 10

alpha = 10**-6
w = 1
b = 1

inpt = [709, 942, 603, 508, 420, 846, 835, -98, -900, 243, -106, -878, -557, -460, 197, -382, 758, 966, -221, -913, -98, 145, 990, 841,
        707, -829, -419, -739, 534, 256, -858, -275, 126, 680, 540, -917, -339, 276, -173, 698, -822, 365, -880, -638, 882, 34, 80, 626,
        121, 623, -515, 956, 497, 986, 471, -785, -491, 681, 209, 54, -465, 347, -742, -794, 475, -24, -414, 880, -896, 184, 724, -447,
        558, 733, 56, 528, -942, 221, -827, -318, -975, -440, 415, 273, 794, 343, 319, -201, -683, -790, 187, -39, -317, -551, 229, 711,
        -285, 287, 624, 53]

weight_gradients = []
bias_gradients = []

def train(inpt, w, b, alpha):
    for i in range(len(inpt)):
        x = inpt[i]
        y_hat = x*w+b
        y = f(x)
        z_grad = 2*(y_hat - y)
        b_grad = z_grad
        w_grad = z_grad * x
        w = w - (alpha * w_grad)
        b = b - (alpha * b_grad)
        weight_gradients.append(w_grad)
        bias_gradients.append(b_grad)
    return w, b

nw = 1
nb = 1

for epoch in range(50_000):
    nw, nb = train(inpt, nw, nb, 10**-6)
    if epoch % 500 == 0:
        #print(f"Epoch: {epoch}\nw: {nw}, b: {nb}\n")
        print(nw, nb)