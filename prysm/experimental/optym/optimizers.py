"""Various optimization algorithms."""
from prysm.mathops import np

class GradientDescent:
    def __init__(self, fg, x0, alpha):
        self.fg = fg
        self.x0 = x0
        self.alpha = alpha
        self.x = x0.copy()
        self.iter = 0

    def step(self):
        f, g = self.fg(self.x)
        self.x -= self.alpha*g
        self.iter += 1
        return self.x, f, g

    def runN(self, N):
        for _ in range(N):
            yield self.step()


class ADAGrad:
    def __init__(self, fg, x0, alpha):
        self.fg = fg
        self.x0 = x0
        self.alpha = alpha
        self.x = x0.copy()
        self.accumulator = np.zeros_like(self.x)
        self.eps = np.finfo(x0.dtype).eps
        self.iter = 0

    def step(self):
        f, g = self.fg(self.x)
        self.accumulator += (g*g)
        self.x -= self.alpha * g / np.sqrt(self.accumulator+self.eps)
        self.iter += 1
        return self.x, f, g

    def runN(self, N):
        for _ in range(N):
            yield self.step()


class RMSProp:
    def __init__(self, fg, x0, alpha, gamma=0.9):
        self.fg = fg
        self.x0 = x0
        self.alpha = alpha
        self.gamma = gamma
        self.x = x0.copy()
        self.accumulator = np.zeros_like(self.x)
        self.eps = np.finfo(x0.dtype).eps
        self.iter = 0

    def step(self):
        gamma = self.gamma
        f, g = self.fg(self.x)
        self.accumulator = gamma*self.accumulator + (1-gamma)*(g*g)
        self.x -= self.alpha * g / np.sqrt(self.accumulator+self.eps)
        self.iter += 1
        return self.x, f, g

    def runN(self, N):
        for _ in range(N):
            yield self.step()


class ADAM:
    def __init__(self, fg, x0, alpha=0.1, beta1=0.9, beta2=0.999):
        self.fg = fg
        self.x0 = x0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.x = x0.copy()
        self.m = np.zeros_like(x0)
        self.v = np.zeros_like(x0)
        self.eps = np.finfo(x0.dtype).eps
        self.iter = 0

    def step(self):
        self.iter += 1
        beta1 = self.beta1
        beta2 = self.beta2
        f, g = self.fg(self.x)
        # update momentum estimates
        self.m = beta1*self.m + (1-beta1) * g
        self.v = beta2*self.v + (1-beta2) * (g*g)

        mhat = self.m / (1 - beta1**self.iter)
        vhat = self.v / (1 - beta2**self.iter)

        self.x -= self.alpha * mhat/(np.sqrt(vhat)+self.eps)
        return self.x, f, g

    def runN(self, N):
        for _ in range(N):
            yield self.step()
