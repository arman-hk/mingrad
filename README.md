# mingrad
![](/pom.jpg)

mingrad is a lean, but mean autograd engine that extends [micrograd](https://github.com/karpathy/micrograd) into a mini deep learning framework.

#### Features
- switching from scalars to numpy: numpy supports vectorized operations, broadcasting, and in-place array operations. but, the most important aspect is that numpy uses a C backend.

#### Examples
- [linear neural net prediction](/examples/linear.ipynb) on [diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes) 👇🏻

![](/figures/linear.png)

---
this project is licensed under the terms of the [MIT License](/LICENSE).