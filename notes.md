# Bachelor Project Notes
## Scope
The goal of this project is to adapt DeepSphere, a graph convolutional neural network designed for spherical data.

In the existing version of DeepSphere, the sphere is approximated by a graph, the nodes of which are given by the HEALPix pixellisation of the sphere. We would like to use an FEM tesselisation of the sphere instead.

## Theory
### Problem
A convolution is usually performed by taking advantage of its simplicity in Fourier frequency space. Fourier transforms on the sphere are possible, but expensive, because the Fourier basis on the sphere (the so-called Spherical Harmonics) is hard to compute.

This is why we look for an approximation of this Fourier basis.

We can add that, for a given manifold, there exists an operator $L$, called the Laplacian operator, which eigenfunctions coincide with that manifold's Fourier basis. From now on, we will just look for an approximation of $L$. That is fine, because its eigendecomposition still gives us the Fourier Basis, and also we will see that doing this is simpler for our purposes.

### Graph Laplacian
It has been shown that, by building a graph which nodes are sampled from a manifold, this graph's *Graph Laplacian*, defined as follows, converge to that manifold's Laplacian under certain conditions.

A graph $G$'s Laplacian is a matrix defined as :
$$ \textbf{L} = \textbf{D} - \textbf{W} $$
It's the difference between the (diagonal) degrees matrix, and the (symmetric) edge weights matrix, and so it is symmetric, and as dense as the graph is connected (so it is usually sparse). It is $n \times n$, with $n$ the number of nodes in the graph.

This graph laplacian can be a reasonably good approximation, however, its correctness stops growing with $n$ after some point, which is why we are looking elsewhere.

### FEM Laplacian
This FEM explanation is drawn quite largely from Martino's paper.

The Laplacian eigenvalue problem on a given manifold $\mathcal{M}$ is defined as :
$$ \Delta_\mathcal{M}f = -\lambda f $$
This type of equation can be approximated using the Finite Elements Method, as follows.
First, multiply the equation by a sufficiently regular function $v$ and integrate on the sphere. Since the sphere has no borders, integrating by parts yields :
$$ \int_{\mathbb{S}_2} \nabla f(x) \cdot \nabla v(x)\mathrm{d}x = \lambda \int_{\mathbb{S}_2}  f(x) \cdot v(x)\mathrm{d}x $$
For all $v$ in our sufficiently regular function space.

We then approximate $\mathbb{S}_2$ by a triangulation $\mathcal{T}$ which contains $n$ points. We also restrict our sufficiently regular function space to be $X^1_\mathcal{T}$, the space of all continuous piecewise linear functions on $\mathbb{S}_2$. All functions in that space are linear combinations of its basis functions :
$$ \phi_i(x_j) = \delta_{ij} ~~ \forall x_j \in \mathcal{T} ~~ \forall i \in [0, n-1] $$

That means that the previous integral equation can be solved for all $v$ by solving it for all $\phi_i$. So, by writing the equation $n$ times, we can rephrase it with matrices :

Find ($f$,$\lambda$) such that $\textbf{Af} = \lambda\textbf{Bf}$, where :

$$\left\{\begin{array}{lll}
(\textbf{A})_{ij}&=&\int_{\mathbb{S}_2}\nabla\phi_{i}(\textbf{x})\cdot\nabla\phi_{j}(\textbf{x})\mathrm{d}\textbf{x}\\
(\textbf{B})_{ij}&=&\int_{\mathbb{S}_2}\phi_{i}(\textbf{x})\phi_{j}(\textbf{x})\mathrm{d}\textbf{x}\\
(\textbf{f})_{i}&=& f_i : f(\textbf{x}) = \sum_{k=0}^{n-1} f_k\phi_k(\textbf{x})
\end{array}\right. $$

$\textbf{A}$ is called the *stiffness* matrix. It is symmetric and sparse. $\textbf{B}$ is called the *mass* matrix. It is also symmetric and sparse. By reordering the equation, we get:

$$\textbf{B}^{-1}\textbf{Af} = \lambda\textbf{f}$$

Which can be re-labeled as follows, by using an educated analogy to the first equation:

$$\textbf{B}^{-1}\textbf{Af} = \textbf{Lf} = \lambda\textbf{f} \Rightarrow \textbf{L} = \textbf{B}^{-1}\textbf{A}$$

Giving us a brand new FEM Laplacian operator. We will use it as we did the Graph Laplacian, with graph convolutions. The pixellisation will thus not change, at least for now. A possible next step would be to use the FEM triangulation, for our graph. (It would be much more complicated in terms of implementation, though, because the FEM libraries tend to be heavy).

### Graph Convolution
For a graph $G$, its graph Laplacian $\textbf{L}$ can be eigen-decomposed as follows:

$$ \textbf{L} = \mathbf{U\Lambda U}^\mathrm{T} $$

With $\mathbf{U}$ being the matrix of eigenvectors (our Fourier basis), and $\mathbf{\Lambda}$ the diagonal matrix of eigenvalues. Thanks to this, we can define the Fourier transform on $G$ as follows:

$$ \mathbf{\hat x} = \mathbf{U}^\mathrm{T}\mathbf{x} $$

And also the inverse transform : $\mathbf{x} = \mathbf{U}\mathbf{\hat x}$. So, by using the well-known definition of convolution as *the product of Fourier transforms*, we can derive a formula for the convolution $y$ of a signal $x$ by a filter $h$ as follows :

$$ \mathbf{y} = \mathcal{F}_G^{-1}(\mathbf{K\hat{f}}) = \mathbf{UK\hat{f}} = \mathbf{UKU}^\mathrm{T}\mathbf{f} = \mathbf{U}h(\mathbf{\Lambda})\mathbf{U}^\mathrm{T}\mathbf{f} $$
with a filter on frequencies :
$$ h : \lambda \mapsto h(\lambda) $$
Note that our FEM Laplacian does not have orthogonal eigenvectors like the standard graph Laplacian. This is fine, we just need to replace every $\mathbf{U}^\mathrm{T}$ with a $\mathbf{U}^\mathrm{-1}$.
Also this all works, but we usually don't want to compute the eigendecomposition of $\mathbf{L}$. Indeed, since we will need many different $\mathbf{L}$'s (one for each layer), this decomposition would need to be computed multiple times.

So, in order to increase performance, we limit ourselves to filters that are polynomial :

$$ h(\lambda) = \sum_{k=0}^{K-1} \theta_k \lambda^k $$

It can be shown that filters of this shape are localized to the K-nearest neighnbours of a node, which allows for some optimizations. Also, we can use a nice iterative algorithm to find the K summands. Here is the algorithm :

$$
\mathbf{y} = \mathbf{U}h(\mathbf{\Lambda})\mathbf{U}^\mathrm{T}\mathbf{f} = \mathbf{U}\left(\sum_{k=0}^{K-1} \theta_k \lambda^k\right)\mathbf{U}^\mathrm{T}\mathbf{f} = \sum_{k=0}^{K-1} \theta_k \mathbf{L}^k\mathbf{f}
$$
Where :
$$
\begin{array}{lll}
\mathbf{y}_1 = \mathbf{L}\mathbf{x}& \Rightarrow & \mathbf{B}\mathbf{y}_1 = \mathbf{A}\mathbf{x} \\
\mathbf{y}_2 = \mathbf{L}^2\mathbf{x} = \mathbf{L}\mathbf{y}_1& \Rightarrow & \mathbf{B}\mathbf{y}_2 = \mathbf{A}\mathbf{y}_1 \\
...&&...\\
\mathbf{y}_K = \mathbf{L}^K\mathbf{x} = \mathbf{L}\mathbf{y}_{K-1}& \Rightarrow & \mathbf{B}\mathbf{y}_K = \mathbf{A}\mathbf{y}_{K-1}
\end{array}
$$
This series of system resolutions can be done efficiently, since $\mathbf{A}$ and $\mathbf{B}$ are sparse and symmetric. We will discuss how later.

## History
### Phase 1
After having learnt the relevant theory, we tried to implement that in the existing DeepSphere codebase. So, we need a way to efficiently solve a linear system of the shape :
$$ \textbf{By}_i = \textbf{Ay}_{i-1} $$

We solve it for $\textbf{y}_i$ Knowing that $\textbf{A}$ and $\textbf{B}$ are hermitian semi-definite positive, and sparse. This system resolution will be applied $K$ times at every layer of the convolutional network, $K$ being the degree of our filter's polynomial. A priori, the necessary functions exist in `numpy`, so we thought about using that.

However, this will not work alone, because we need to keep the gradients of every layer in order to train the network. Tensorflow functions compute these automatically, and so, we decided to use Tensorflow functions after all.

### Phase 2
Because Tensorflow 1 (as used in DeepSphere) does not have any way of leveraging the properties of our system directly, as in a `solve_sparse` function, we have to find an workaround.

Computing $\textbf{By}_{i-1}$ efficiently is easy, it is a simple `tf.sparse.sparse_dense_matmul` call. Then, we thought about using the Cholesky decomposition of $\textbf{B}$, in order to optimize the resolution. Doing this leverages the properties of $\textbf{B}$.

There exist `tf.linalg.cholesky` and `tf.linalg.cholesky_solve` for this purpose. Notice that we do not need to compute the decomposition at every layer ! We can just use $\mathrm{chol}(\textbf{B})$ as input for DeepSphere instead of $\textbf{B}$.

### Phase 3
The issue with these functions is that they do not leverage the sparsity of $\textbf{B}$.

So, we found a python package to address that : `scikit-sparse`. For our purposes, it is just a wrapper around CHOLMOD, a C library for sparse Cholesky decomposition and resolution.
The decomposition is ~4 orders of magnitude faster than `tf.linalg.cholesky`, as can be seen in the accompanying `Cholesky.ipynb` notebook, and the resolution is (slightly) faster than Tensorflow too.

We still need the gradients, so we'll still use `tf.linalg.cholesky_solve` in every layer, but the previous step of decomposing $\textbf{B}$ will be much faster.
