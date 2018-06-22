---
layout: default
title: "ND4J's Benchmarking Tool"
description: ""
---

## Benchmarking ND4J

Nd4j makes extensive use of vectorized C++ code for all numerical operations (utilizing [JavaCPP](https://github.com/bytedeco/javacpp)), off-heap data storage and optimized native BLAS libraries (OpenBLAS, Intel MKL, cuBLAS etc). 

It requires F-ordering of arrays due to limitations in cuBLAS. 

Deeplearning4j is therefore written around F-ordering, and ND4J supports this. 

[Here's an example test](https://gist.github.com/raver119/92b615704ca1bf169aa23a6a6e7d9880) you can run yourself that demonstrates ordering:

```
  o.n.i.TensorFlowImportTest - Orders: CCC; Time: 11532 ns;
  o.n.i.TensorFlowImportTest - Orders: CCF; Time: 2101 ns;
  o.n.i.TensorFlowImportTest - Orders: CFC; Time: 10202 ns;
  o.n.i.TensorFlowImportTest - Orders: CFF; Time: 1960 ns;
  o.n.i.TensorFlowImportTest - Orders: FCC; Time: 10744 ns;
  o.n.i.TensorFlowImportTest - Orders: FCF; Time: 1717 ns;
  o.n.i.TensorFlowImportTest - Orders: FFC; Time: 10097 ns;
  o.n.i.TensorFlowImportTest - Orders: FFF; Time: 1716 ns;
```  

## C -> F ordering

`mmuli` returns an F-ordered array. If you specify C-ordered result, you'll have F -> C conversion underneath, and C -> F assign takes time.

ND4J returns the `mmuli` result in F order, and DL4J algorithms are designed with that in mind. It is possible to get the `mmuli` result in C order, but it'll cause an additional conversion operation call, which is expensive. 

ND4J `mmuli` always returns F because of Cuda: The cuBLAS implementation has no option for C-ordered output.

`mmuli` CCC always introduces a dup in ND4J. We expect the gemm result to always be F, so if result is not F, each operation allocates a `tempResult` array, F-ordered, and does `result.assign(tempResult)`.
