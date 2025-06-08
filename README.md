# mingrad
![mingrad logo](/pom.jpg)

`mingrad` is a lean, but mean autograd engine that extends [micrograd](https://github.com/karpathy/micrograd) into a mini deep learning framework.

#### Features
- **Vectorized Operations:** Execution of advanced mathematical tasks through efficient array-based processing.
- **NumPy Integration:** NumPy's `C` backend for enhanced numerical computation efficiency.
- **Scalability:** Batch processing and management of high-dimensional data arrays.
- **NN architecture into three key operation types:**
    - *Unary Operations:* Single-array manipulations using element-wise operations such as `RELU`.
    - *Binary Operations:* Combining two arrays to produce a single output, operations like `add` and `mul`.
    - *Reduction Operations:* Streamlining data by reducing larger arrays into a summarized form through operations such as `mean` and `argmax`.
- **Modularity:** Straightforward expansions and customizations.

#### Examples
- [linear neural net prediction](/examples/linear.ipynb) on [diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes) 👇🏻

![](/figures/linear.png)

#### (⚠️ not active anymore, since this was more of a educational project for me.)