cudaHostMalloc 比 malloc 性能上可以快一倍左右 
这是因为cudaHostMalloc用的是页锁定机制，而malloc要
遵守操作系统的分页机制，就有可能出现页替换的问题;
同时，malloc得到的内存在copy到显存时要过两步，分页内存-> 锁定内存 -> 显存
而

零拷贝内存：
cudaHostMallocMapped: GPU可以直接访问主机内存
