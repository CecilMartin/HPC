#HPC Hw4 (Zhe Chen)

##Machine configuration

Results are obtained by running on CIMS's gpu machines cuda{1-5}.cims.nyu.edu. Configurations of these servers can be found on https://cims.nyu.edu/webapps/content/systems/resources/computeservers.

Software environment is gcc-4.9.2, cuda-9.3. 

####Notice that one must use cuda 9 or higher verision to support \_\_sync\_warp in P1.


## P1: Matrix-vector operations on a GPU




## P1: Approximating Sine Function with Taylor Series & Vectorization

1.First, I improve sin4_vec to be 12 digit accuracy just by adding higher Taylor terms up to order 11. The performance is shown as below. 

### Extra credit

2.To evaluate sine function outside of $[-\pi/4,\pi/4]$ efficiently, I develop scheme as following.

2.a). Move any 'x' to $[-\pi/4,7\pi/4]$ since it's $2\pi$ circulant by $x=x-2\pi floor(x+\pi/4)$.

2.b). By $sin(x)=-sin(x-\pi)$, move $x \in [3\pi/4,7\pi/4]$ to $[-\pi/4,3\pi/4]$.

2.c). We can already compute $sin(x)$ in $[-\pi/4,\pi/4]$. To evaluate it in $[\pi/4,3\pi/4]$, we use formula $sin(x)=cos(x-\pi/2)$. Thus, the problem is converted to compute $cos(x),\ x \in [-\pi/4,\pi/4]$. We could just use cosine function's taylor expansion up to $x^{12}$ to get 12 digits accuracy.

The reslt is shown in below as "(extended range)".


```
Reference time: 22.9906
Reference time (extended range): 32.7627
Taylor time:    4.4044      Error: 6.928125e-12
Taylor time (extended range):    19.5440      Error: 6.928014e-12
Intrin time:    1.3712      Error: 2.454130e-03
Vector time:    1.5247      Error: 6.928125e-12
```

## P2: Parrallel Scan in OpenMP

To parallelize scan operation, I divide n into nthreads batches with each batch's length=n/nthreads. Generally, it's better to have nthreads less or equal to max core number, which is 4 in my machine.

In each batch, we calculate prefix-sum locally and record the last term $batch[i]$. Then we scan $batch[i]$ to get $prefix\_sum\_batch[i]$, which is the value  that should be added into batch i+1.

Importantly, we should add $prefix\_sum\_batch[i]$ to each batch parallelingly too. Or we lose the point of paralleling scan operator. It's also a compete scan to do this operation!

The result is shown below and we can see good scaling.


| Number of Threads               | 1      | 2      | 3      | 4      |
|---------------------------------|--------|--------|--------|--------|
| Parallel Scan / Sequential Scan | 1.4336 | 0.8748 | 0.6125 | 0.5047 |