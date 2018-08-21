PIN  API文档查询：
https://software.intel.com/sites/landingpage/pintool/docs/97619/Pin/html/group__PIN__CONTROL.html

6.823 的教学流程是先讲几个主要的体系结构概念　然后几个概念循环着深入讲

*LAB*  0 是有关PIN的基本使用方法和ISA的入门概念

*LAB*  1 涉及cache的相关知识,可以参考下 csg.csail.mit.edu/6.823/lectures/L03.pdf 复习cache的结构以及运行方式

| addr tag  | data 0 - data 1 -data 2 .... data n ("cache block") | "cache line"

cache 算法：（实际就是一个ｍｅｍｏｒｙ　ｂｌｏｃｋ应该ｃａｃｈｅ到哪一个ｃａｃｈｅ　ｌｉｎｅ）

1. 直接映射 cache

假设有2^k条cache line, 把物理地址 按域划分为:| Tag(t bit) | Index(k bit) | Offset(b bit) |

对于一条访存指令中的绝对地址，先定位Index-cache line，比较addr tag和Tag是否一致，一致则hit，然后在cache的Index行取offset处的数

扩展后的直接映射 cache算法可以把ｔａｇ＋ｉｎｄｅｘ进行ｈａｓｈ,只不过这样做考验hash的性能以及扩大了addr tag的位数(k->t+k)

2. 两(多)路组关联 cache

和1 很像, 主要的目标是解决1中可能造成的映射到同一cache line的内存数据竞争现象

相当于同意Index 对应的cache line有两(多)条,在这种情况下tag要分别比较这两条的addr tag是否命中,有一条命中即hit

两路组关联会涉及cache line替换算法问题,LRU/FIFO/NLRU/RANDOM

