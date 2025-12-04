import math
import random
from graphviz import Digraph

# --- 1. The Autograd Engine (Scalar) ---
class Value:
    def __init__(self, data: float, _op: str = '', _children: tuple = ()):
        self.grad = 0.0
        self._children = set(_children)
        self.data = data
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _children=(self, other), _op="+")
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _children=(self, other), _op="*")
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    # Robustness wrappers
    def __rmul__(self, other): return self * other
    def __radd__(self, other): return self + other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)

    def gelu(self):
        # Approximate GELU (Tanh version)
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        x = self.data
        coeff = 0.044715
        sqrt_2_pi = math.sqrt(2 / math.pi)
        
        # Forward
        tanh_arg = sqrt_2_pi * (x + coeff * x**3)
        tanh_out = math.tanh(tanh_arg)
        out_data = 0.5 * x * (1 + tanh_out)
        
        out = Value(out_data, _children=(self,), _op="gelu")
        
        def _backward():
            # Derivative logic
            tanh_out = math.tanh(sqrt_2_pi * (self.data + coeff * self.data**3))
            sech2 = 1 - tanh_out**2
            grad_gelu = 0.5 * (1 + tanh_out + (self.data * sech2 * (sqrt_2_pi * (1 + 3 * coeff * self.data**2))))
            self.grad += grad_gelu * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# --- 2. The Neural Network Architecture ---
class Neuron:
    def __init__(self, nin):
        # Initialize weights between -1 and 1
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
    
    def __call__(self, x):
        # Linear only: w*x + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act 
    
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin) for _ in range(nout)]
        self.nonlin = nonlin
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        if self.nonlin:
            outs = [out.gelu() for out in outs]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            # Use GELU for all layers EXCEPT the last one (Linear Output)
            is_last_layer = (i == len(nouts) - 1)
            self.layers.append(Layer(sz[i], sz[i+1], nonlin=not is_last_layer))
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# --- 3. Visualization Tools (Graphviz) ---
def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = Left to Right
    
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # Create a rectangular node for the value
        dot.node(name=uid, label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        
        if n._op:
            # Create a small oval node for the operation
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot

# --- 4. Main Execution ---
if __name__ == "__main__":
    # 1. Dataset (Normalized)
    xs = [
        [Value(2.0/3.0), Value(3.0/3.0), Value(-1.0/3.0)],
        [Value(3.0/3.0), Value(-1.0/3.0), Value(0.5/3.0)],
        [Value(0.5/3.0), Value(1.0/3.0), Value(1.0/3.0)],
        [Value(1.0/3.0), Value(1.0/3.0), Value(-1.0/3.0)],
    ]
    ys = [Value(1.0), Value(-1.0), Value(-1.0), Value(1.0)] # Targets

    # 2. Initialize Neural Net
    model = MLP(3, [4, 4, 1])

    # 3. Training Loop
    print(f"Starting Scalar Training...")
    epochs = 200
    
    for k in range(epochs):
        
        # Forward Pass
        ypred = [model(x) for x in xs]
        
        # Loss (MSE)
        loss = Value(0.0)
        for ygt, yout in zip(ys, ypred):
            loss = loss + (yout - ygt) * (yout - ygt)
            
        # Zero Gradients
        for p in model.parameters():
            p.grad = 0.0
            
        # Backward Pass
        loss.backward()
        
        # Update (SGD)
        learning_rate = 0.1 - (0.09 * k / epochs)
        for p in model.parameters():
            p.data += -learning_rate * p.grad
            
        if k % 20 == 0:
            print(f"Step {k}: Loss = {loss.data:.5f}")

    print("\nFinal Predictions vs Targets:")
    for pred, target in zip(ypred, ys):
        print(f"Pred: {pred.data:.4f} | Target: {target.data}")

    # 4. Generate Graph
    print("\nGenerating computational graph of the final loss...")
    try:
        dot = draw_dot(loss)
        dot.render('scalar_graph', view=True)
        print("Graph rendered to 'scalar_graph.svg'")
    except Exception as e:
        print(f"\nCould not render graph locally: {e}")
        print("Copy the source below and paste into https://dreampuf.github.io/GraphvizOnline/")
        print("-" * 20)
        print(dot.source)
        print("-" * 20)