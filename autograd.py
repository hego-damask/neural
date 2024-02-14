import math

class Var:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []
        self.grad = 0
    def __add__(self, other):
        return Var(self.value+other.value, [(1, self), (1, other)])
    def __radd__(self, other):
        return Var(self.value+other, [(1, self)])
    def __sub__(self, other):
        return Var(self.value-other.value, [(1, self), (-1, other)])
    def __mul__(self, other):
        return Var(self.value*other.value, [(other.value, self), (self.value, other)])
    def __truediv__(self, other):
        return Var(self.value/other.value, [(1/other.value, self), ((-1*self.value)/other.value**2, other)])
    def __neg__(self):
        return Var(-1*self.value, [(-1, self)])
    def sin(self):
        return Var(math.sin(self.value), [(math.cos(self.value), self)])
    def cos(self):
        return Var(math.cos(self.value), [(-1*math.sin(self.value), self)])
    def exp(self):
        return Var(math.exp(self.value), [(math.exp(self.value), self)])
    
    def backward(self, grad=1):
        self.grad += grad
        for coeff, child in self.children:
            child.backward(grad*coeff)




print('with automatic differentiation')
print('e^(sin(x^2))')
x = Var(10)
f = (x*x).sin().exp()
f.backward()
print(f.value)
print(x.grad)

#sigmoid
def sigmoid(x):
    one = Var(1)
    ans = one/(one+(-x).exp())
    return ans
print('sigmoid')
x = Var(1)	
f = sigmoid(x)
f.backward()
print(f.value)
print(x.grad)

#tanh
def tanh(x):
    e1 = x.exp()
    e2 = (-x).exp()
    return (e1-e2)/(e1+e2)

print('tanh')
x = Var(1)
f = tanh(x)
f.backward()
print(f.value)
print(x.grad)

print('without automatic differentiation')
print('e^(sin(x^2))')
x = 10
f = math.exp(math.sin(x**2))
df = math.exp(math.sin(x**2))*math.cos(x**2)*2*x 
print(f)
print(df)

print('sigmoid')
x = 1
s = 1/(1+math.exp(-x))
ds = s*(1-s)
print(s)
print(ds)

print('tanh')
x = 1
t = (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
dt = 1-t**2
print(t)
print(dt)

