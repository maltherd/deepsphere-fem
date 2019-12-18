# 4-5
## Without approximations
* Δ’s eigenfunctions = “spherical harmonics”
* The Spherical Harmonic Fourier Transform (SHT) is very expensive to compute...
* Convolutions become expensive ⇒ no good.

# 7
## Polynomial filters
$$ h(\lambda) = \sum_{k=0}^{K-1} \theta_k \lambda^k $$

$$
\mathbf{y} = \mathbf{U}h(\mathbf{\Lambda})\mathbf{U}^\mathrm{T}\mathbf{f} = \mathbf{U}\left(\sum_{k=0}^{K-1} \theta_k \mathbf{\Lambda}^k\right)\mathbf{U}^\mathrm{T}\mathbf{f} = \sum_{k=0}^{K-1} \theta_k \mathbf{L}^k\mathbf{f}
$$

# 15
## Powers of L
$$
\left\{~\begin{array}{lll}
\mathbf{L}^0\mathbf{f}& =& \mathbf{f} \\
\mathbf{L}^{k+1}\mathbf{f}& =& \mathbf{L} ~ \cdot ~ \mathbf{L}^k\mathbf{f}
\end{array}\right.
$$

# 16
## Powers of FEM L
$$
\left\{\begin{array}{lllll}
\mathbf{L}^0\mathbf{f}& =& \mathbf{f} \\
\mathbf{L}^{k+1}\mathbf{f}& =& \mathbf{L} ~ \cdot ~ \mathbf{L}^{k}\mathbf{f}& \Rightarrow & \mathbf{B}\cdot(\mathbf{L}^{k+1}\mathbf{f}) = (\mathbf{A}\mathbf{L}^{k}\mathbf{f})
\end{array}\right.
$$
