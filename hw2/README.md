#HPC Hw2 (Zhe Chen)

##Machine configuration

All results are got by running on my office's machine 'blob'. It has a 4 core CPU, Intel Xeon E5-1603, 2.80GHz. Max memory size is 375GB, max memory bandwidth is 31.4GB/s. It uses Sandy Bridge EP, so operations per cycle is 8. Thus, theoretic FLOPS/s is 4 Cores * 2.80GHz *  8 operations/cycle = 89.6 GFLOPs/s

##Dubug
Problem 1 and 3 are both to debug the codes. All are solved and renamed as required. 

All the bugs I found and solved is commented around with `//BUG:` and explained how that is a bug and why that solution works. Just search the codes with `BUG`.

##Makefile

##Problem 2: Optimization Matrix-Matrix Multiplication

### Rearrange loops

Intutively, we need to make cache hits more and flop-rate bigger to make it faster. As for loops order, this means we need to read data sequentially as much as possible to lower the cost of reading. Thus, for problem $C=A*B$, it's easy to get the idea that we should loop rows of $A$ first, say i, then rows of B, say p, then column of C, say j. 


i,p,j

 Dimension       Time    Gflop/s       GB/s        Error
       500   0.521711   4.312734  17.319940 0.000000e+00
      1000   0.938626   4.261547  17.080280 0.000000e+00
      1500   2.285623   2.953243  11.828724 0.000000e+00
     
p,i,j
Dimension       Time    Gflop/s       GB/s        Error
       500   1.560516   1.441831   5.790394 0.000000e+00
      1000   2.824688   1.416086   5.675671 0.000000e+00
      1500   7.559161   0.892956   3.576587 0.000000e+00
      
i,j,p
 Dimension       Time    Gflop/s       GB/s        Error
       500   0.617410   3.644257  14.635335 0.000000e+00
      1000   1.106277   3.615732  14.491854 0.000000e+00
      1500   3.096304   2.180019   8.731701 0.000000e+00
      

Decrease after 2048;

BLOCK_SIZE=16
 Dimension       Time    Gflop/s       GB/s        Error
      2048   5.948961   2.887877  11.562790 0.000000e+00
      
      
64
 Dimension       Time    Gflop/s       GB/s        Error
      2048   4.718624   3.640864  14.577679 0.000000e+00
      
      

128

 Dimension       Time    Gflop/s       GB/s        Error
      2048   4.387855   3.915323  15.676585 0.000000e+00
      
256

 Dimension       Time    Gflop/s       GB/s        Error
      2048   4.051400   4.240477  16.978472 0.000000e+00
            
512
 Dimension       Time    Gflop/s       GB/s        Error
      2048   3.927520   4.374228  17.514000 0.000000e+00
      
      
1024
 Dimension       Time    Gflop/s       GB/s        Error
      2048   5.858231   2.932603  11.741869 0.000000e+00
      
      
parallel:
 Dimension       Time    Gflop/s       GB/s        Error
       512   0.497402   4.317403  17.337072 0.000000e+00
      1024   0.518087   4.145024  16.612479 0.000000e+00
      1536   2.462841   2.942844  11.786702 0.000000e+00
      2048   5.850661   2.936398  11.757062 0.000000e+00
      
paraller+blocking

 Dimension       Time    Gflop/s       GB/s        Error
       512   0.556471   3.859109  15.496734 0.000000e+00
      1024   0.283655   7.570757  30.342173 0.000000e+00
      1536   0.639638  11.331023  45.383108 0.000000e+00
      2048   1.195404  14.371598  57.542530 0.000000e+00
      2560   3.525609   9.517343  38.099114 0.000000e+00

