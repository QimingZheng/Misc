### CUDA流

一系列异步CUDA操作,如：cudaMalloc,cudaMemcpy，kernel函数执行
流中的操作相对于主机来说是异步的

CUDA流中排队的操作和主机都是异步的，所以排队的过程中并不耽误主机运行其他指令，所以这就隐藏了执行这些操作的开销。

CUDA的API分为同步和异步的两种：

1. 同步行为的函数会阻塞主机端线程直到其完成

2. 异步行为的函数在调用后会立刻把控制权返还给主机。

多个cuda 流中的调度情况:

```
for (int i = 0; i < nStreams; i++) {
   int offset = i * bytesPerStream;
   cudaMemcpyAsync(&d_a[offset], &a[offset], bytePerStream, streams[i]);
   kernel<<grid, block, 0, streams[i]>>(&d_a[offset]);
   cudaMemcpyAsync(&a[offset], &d_a[offset], bytesPerStream, streams[i]);
   }
```

![image][./multi-stream.png]


Ｆｅｒｍｉ架构上的单队列中存在虚拟依赖关系,减弱了流之间的并行度
![image][./single-Q.png]

Hyper-Q技术通过引入多队列解决的上述问题
![image][./Hyper-Q.png]

