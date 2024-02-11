from random import uniform
import math
class Tensor:
    def __init__(self, data = None, size = None, tree = []):
        self.tree = tree
        if size is not None:
            if data is not None:
                assert(size == len(data))
                self.size = size
                self.data = data
            else:
                self.size = size
                self.data = [uniform(-1,1) for _ in range(size)]
        else:
            if data is not None:
                assert type(data) is list
                self.data = data
                self.size = len(data)
            else:
                raise ValueError('provide values for size and or data')

    def __repr__(self):
        return f'Tensor:\n(size: {self.size}) (data: {self.data})'
    
    def __add__(self, tensor):
        assert(self.size == tensor.size)
        result = Tensor([x + y for x, y in zip(self.data, tensor.data)], self.size, tree = ['ADD'])
        return result
    
    def __mul__(self, tensor):
        assert(self.size == tensor.size)
        result = Tensor([x * y for x, y in zip(self.data, tensor.data)], self.size, tree = ['MUL'])
        return result
    
    def dot(self, tensor):
        result = [sum([x * y for x, y in zip(self.data, tensor.data)])]
        result = Tensor(result, len(result), tree = ['DOT'])
        return result

    @staticmethod
    def sin(tensor):
        return Tensor(list(map(math.sin, tensor.data)), tensor.size, tree = ['SIN'])
    @staticmethod
    def cos(tensor):
        return Tensor(list(map(math.cos, tensor.data)), tensor.size, tree = ['COS'])
    @staticmethod
    def sigmoid(tensor):
        return Tensor(list(map(lambda x: 1/(1+math.exp(-x)), tensor.data)), tree = ['SIGMOID'])
    @staticmethod
    def relu(tensor):
        return Tensor(list(map(lambda x: max(0, x), tensor.data)), tree = ['RELU'])
