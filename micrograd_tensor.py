import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, data=\n{self.data})"

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, _children=(self, other), _op='@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
            
        out._backward = _backward
        return out

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            # Handle broadcasting for Self
            if self.grad.shape != out.grad.shape:
                # If self is scalar (shape ()), sum everything
                if self.grad.shape == ():
                    self.grad += out.grad.sum()
                else:
                    # Otherwise, sum out the batch dimension (axis 0)
                    # NOTE: This assumes (1, N) broadcasting into (Batch, N)
                    self.grad += out.grad.sum(axis=0, keepdims=True)
            else:
                self.grad += out.grad
            
            # Handle broadcasting for Other
            if other.grad.shape != out.grad.shape:
                if other.grad.shape == ():
                    other.grad += out.grad.sum()
                else:
                    other.grad += out.grad.sum(axis=0, keepdims=True)
            else:
                other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')
        
        def _backward():
            grad_self = other.data * out.grad
            grad_other = self.data * out.grad
            
            # Handle broadcasting for Self
            if self.grad.shape != grad_self.shape:
                if self.grad.shape == ():
                    self.grad += grad_self.sum()
                else:
                    self.grad += grad_self.sum(axis=0, keepdims=True)
            else:
                self.grad += grad_self
            
            # Handle broadcasting for Other
            if other.grad.shape != grad_other.shape:
                if other.grad.shape == ():
                    other.grad += grad_other.sum()
                else:
                    other.grad += grad_other.sum(axis=0, keepdims=True)
            else:
                other.grad += grad_other
                
        out._backward = _backward
        return out
    
    def mean(self):
        out = Tensor(np.array([self.data.mean()]), _children=(self,), _op='mean')
        
        def _backward():
            n = self.data.size
            self.grad += out.grad * (np.ones_like(self.data) / n)
        out._backward = _backward
        return out
        
    def __neg__(self): return self * Tensor(np.array(-1.0))
    def __sub__(self, other): return self + (-other)
    def __matmul__(self, other): return self.matmul(other)
    def __radd__(self, other): return self + Tensor(other)
    def __rmul__(self, other): return self * Tensor(other)

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = np.ones_like(self.data)
        
        for node in reversed(topo):
            node._backward()

class Linear:
    def __init__(self, nin, nout):
        scale = 1.0 / np.sqrt(nin)
        self.w = Tensor(np.random.uniform(-1, 1, (nin, nout)) * scale)
        self.b = Tensor(np.zeros((1, nout)))
    
    def __call__(self, x):
        return (x @ self.w) + self.b
    
    def parameters(self):
        return [self.w, self.b]

def gelu(x):
    sqrt_2_pi = np.sqrt(2 / np.pi)
    coeff = 0.044715
    x_val = x.data
    tanh_arg = sqrt_2_pi * (x_val + coeff * x_val**3)
    tanh_out = np.tanh(tanh_arg)
    out_data = 0.5 * x_val * (1 + tanh_out)
    
    out = Tensor(out_data, _children=(x,), _op='gelu')
    
    def _backward():
        sech2 = 1 - tanh_out**2
        grad_part = 0.5 * (1 + tanh_out + (x_val * sech2 * (sqrt_2_pi * (1 + 3 * coeff * x_val**2))))
        x.grad += grad_part * out.grad
    out._backward = _backward
    return out

if __name__ == "__main__":
    np.random.seed(1337)

    X_data = np.array([
        [2.0/3.0, 3.0/3.0, -1.0/3.0],
        [3.0/3.0, -1.0/3.0, 0.5/3.0],
        [0.5/3.0, 1.0/3.0, 1.0/3.0],
        [1.0/3.0, 1.0/3.0, -1.0/3.0]
    ])
    
    Y_data = np.array([[1.0], [-1.0], [-1.0], [1.0]])
    
    inputs = Tensor(X_data)
    targets = Tensor(Y_data)
    
    layer1 = Linear(3, 8)
    layer2 = Linear(8, 8)
    layer3 = Linear(8, 1)
    
    params = layer1.parameters() + layer2.parameters() + layer3.parameters()
    print(f"Starting Tensor Training...")
    
    for k in range(200):
        
        x = inputs
        x = gelu(layer1(x))
        x = gelu(layer2(x))
        x = layer3(x)
        
        diff = x - targets
        loss = (diff * diff).mean()
        
        for p in params:
            p.grad = np.zeros_like(p.data)
            
        loss.backward()
        
        lr = 0.1
        for p in params:
            p.data -= lr * p.grad
            
        if k % 20 == 0:
            print(f"Step {k}: Loss = {loss.data[0]:.6f}")

    print("\nFinal Predictions:\n", x.data)
    print("Targets:\n", Y_data)