# Thrust 简单使用

## vector 相关

运算符”=”可以用来复制host_vector到

device_vector（反之亦然）。运算符”=”也可以用来复制 host_vector到host_vector或device_vector到device_vector。

同样device_vector访问单个元素可以使用标准的括号表示法。但是每次访问需要调用cudaMemcpy.

## algorithm

### transformation

Transformations会对指定输入范围内的元素进行特定操作，并将其结果存储到指定位置。如thrust::fill

此外transformations还包括thrust::sequence、thrust::replace、thrust::transform（衍生于C++的for_each）

### Reductions
见注释

### Prefix-Sums
见注释

### Reordering
见注释

### Sorting
见注释

