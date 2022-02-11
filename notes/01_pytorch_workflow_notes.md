# Notes

1. In `LinearRegressionModel`, the `requires_grad` and `dtype` parameters to `torch.randn` are redundant, as `nn.Parameter` takes care of setting them.

2. Again in `LinearRegressionModel`, the `forward` function calculates $w \cdot x + b$, but in the previous notebook, you introduced the linear function in its general form, $x \cdot A^{T} + b$. Since this example is using vectors, it doesn't matter, but it might be worth to note this difference.
