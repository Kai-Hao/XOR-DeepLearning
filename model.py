import torch
import torch.nn as nn
import torch.nn.functional as F

class STANDARD_MODEL(nn.Module):
    def __init__(self, activation = nn.Sigmoid, output_activation = nn.Sigmoid, hidden_layer_bias = True, one_hot_encoding = False, init_weight = None, seed = None):
        super(STANDARD_MODEL, self).__init__()
        # FIX SEED
        if seed is not None:
            torch.manual_seed(seed)
        # TWO LAYER HIDDEN LAYER
        self.fc1 = nn.Linear(2, 2, bias = hidden_layer_bias)
        if one_hot_encoding:
            self.fc2 = nn.Linear(2, 2, bias = hidden_layer_bias)
        else:
            self.fc2 = nn.Linear(2, 1, bias = hidden_layer_bias)
        # INITIAL ACTIVATION FUNCTION
        if activation is not None:
            self.activation = activation
            self.output_activation = activation
        else:
            self.activation = None
            self.output_activation = None
        # INITIAL WEIGHT
        if init_weight is not None:
            self._init_weights(init_weight)
    
    def _init_weights(self, init_weight_method):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init_weight_method(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        if self.activation is not None:
            x = self.activation(x)
        x = self.fc2(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x
    
    
# TESTING
if __name__ == "__main__":
    model = STANDARD_MODEL(activation = None, seed = 42, one_hot_encoding = True)
    print(model)
    # Test forward pass with dummy data
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    fc1 = model.fc1(x)
    print("fc1 output:", fc1)
    fc2 = model.fc2(fc1)
    print("fc2 output:", fc2)
    output = nn.Sigmoid()(fc2)
    print("Output:", output)
