softmax操作的cuda实现：

设矩阵大小为M X N，则分为M个block每个block的thread处理一行

线程号 curIdx 表示在block内部的编号,NextIdx则表示该线程所要处理的元素位置

