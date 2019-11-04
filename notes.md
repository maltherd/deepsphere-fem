# Bachelor Project Notes
## History
### Phase 1
After having learnt the relevant theory, we tried to implement that in the existing DeepSphere codebase. So, we need a way to efficiently solve a linear system of the shape :

$$
\textbf{Ax}_i = \textbf{Bx}_{i-1}
$$

We solve it for $\textbf{x}_i$ Knowing that $\textbf{A}$ and $\textbf{B}$ are hermitian semi-definite positive, and sparse. This system resolution will be applied $K$ times at every layer of the convolutional network, $K$ being the degree of our filter's polynomial. A priori, the necessary functions exist in `numpy`, so we thought about using that.

However, this will not work alone, because we need to keep the gradients of every layer in order to train the network. Tensorflow functions compute these automatically, and so, we decided to use Tensorflow functions after all.

### Phase 2
Because Tensorflow 1 (as used in DeepSphere) does not have any way of leveraging the properties of our system directly, as in a `solve_sparse` function, we have to find an workaround.

Computing $\textbf{Bx}_{i-1}$ efficiently is easy, it is a simple `tf.sparse.sparse_dense_matmul` call. Then, we thought about using the Cholesky decomposition of $\textbf{B}$, in order to optimize the resolution. Doing this leverages the properties of $\textbf{B}$.

There exist `tf.linalg.cholesky` and `tf.linalg.cholesky_solve` for this purpose. Notice that we do not need to compute the decomposition at every layer ! We can just use $chol(\textbf{B})$ as input for DeepSphere instead of $\textbf{B}$.

### Phase 3
The issue with `tf.linalg.cholesky_solve` is that it is very slow. It also does not leverage the sparsity of $\textbf{B}$.

So, we found a python package to address that : `scikit-sparse`. For our purposes, it is just a wrapper around CHOLMOD, a C library for sparse Cholesky decomposition and resolution. The resolution is ~4 orders of magnitude faster than the Tensorflow function, as can be seen in the accompanying `Cholesky.ipynb` notebook.
