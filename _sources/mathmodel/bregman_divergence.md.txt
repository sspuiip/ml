# Bregman divergence

The surprising result - due to [Banerjee, Gou, and Wang 2005](https://ieeexplore.ieee.org/document/1459065) is the following,

> If you have some abstract way of measuring the "distance" between any two points and, for any choice of distribution over points the mean point minimises the average distance to all the others, then your distance measure must be a Bregman divergence.


## Squared Euclidean Distance

One member of the Bregman divergence family is the squared Euclidean distance (SED), i.e.,

$$
d^2(\pmb{x},\pmb{y})=\sum_{i=1}^n(x_i-y_i)^2
$$

SED can be represented as follow,

$$
d^2(\pmb{x},\pmb{y})=\Vert \pmb{x}-\pmb{y}\Vert^2=\Vert \pmb{x}\Vert^2-\Vert \pmb{y}\Vert^2-\langle 2\pmb{y},\pmb{x}-\pmb{y}\rangle
$$

Obviously, the derivative of $\Vert \pmb{y}\Vert^2$ is $2\pmb{y}$. Now, we take a look on the term $\Vert \pmb{y}\Vert^2+\langle 2\pmb{y},\pmb{x}-\pmb{y}\rangle$ which is the value of the tangent line to $\Vert \pmb{y}\Vert^2$ at $\pmb{y}$ evaluated at $\pmb{x}$. This means the whole expression is just the difference between the function $f(\pmb{x})=\Vert \pmb{x}\Vert^2$ at $\pmb{x}$ and the value of $f$'s tangent at $\pmb{y}$ evaluated at $\pmb{x}$. That is,

$$
d^2(\pmb{x},\pmb{y})=f(\pmb{x})-\underbrace{(f(\pmb{y})+\langle 2\pmb{y},\pmb{x}-\pmb{y}\rangle)}_{\textrm{Tangent of $f$ at $\pmb{y}$ evaluated at $\pmb{x}$}}
$$

![Bregman Divergence](../img/bregman.png)

## Bregman divergences

So far, the only one constraint which is placed on the distance measure $d^2$ is that it be non-negative for all possible choices of point $x$ and $y$. This is equivalent to the function $f$ always sitting above its tangent, that is,

$$
f(x)\ge f(y)+\langle \nabla f(y),x-y\rangle, \quad \forall x,y\in\mathbb{R}^n.
$$

Obviously, the above condition placed on the function $f$ is equivalent to functions $f$ being convex. This means that we can derive a distance measure $d_f$ that has a similar stucture to the SED by simply choosing a convex function $f$ and defining,

$$
\boxed{d_f(x,y)=f(x)-f(y)-\langle \nabla f(y), x-y\rangle}
$$

Distance defined like this are precisely the **Bregman divergence** and the convexity of $f$ guarantees they are non-negative for all $x,y\in\mathbb{R}^n$.

- Kullback-Leibler(KL) divergence can be expressed as a Bregman divergence using the convex function,

$$
f_{KL}(p)=\sum_{i=1}^np_i\log p_i
$$

![Bregman divergences](../img/bregmanfun.png)

## Reference

[Meet the Bregman Divergences](https://mark.reid.name/blog/meet-the-bregman-divergences.html)