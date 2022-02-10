# Notes

1. In the `plot_decision_boundary` function, why use `X = X.to("cpu")` instead of `X.cpu()`?

2. Since softmax maps large numbers to large probabilities and small numbers to small probabilities, to my understanding, it is better to predict directly using `torch.argmax` without going through softmax, as softmax is a relatively costly operation. Of course, if we're interested in the probabilities (to show confidence or top-k predictions), softmax should be used.