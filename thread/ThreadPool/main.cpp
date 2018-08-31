#include <iostream>
#include <vector>
#include <chrono>

#include "ThreadPool.h"

int F(int x, int y){
    return x*y;
}

int main()
{

    ThreadPool pool(4);
    std::vector< std::future<int> > results;
/*
    for(int i = 0; i < 80000; ++i) {
        results.emplace_back(
          pool.enqueue([i] {
            std::cout << "hello " << i << std::endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "world " << i << std::endl;
            return i*i;
        })
      );
    }
*/
    for(int i = 0; i < 64; ++i) {
        results.emplace_back(pool.enqueue(F,i,i*i));
    }
    for(auto && result: results)    //通过future.get()获取返回值
        std::cout << result.get() << '\t';
    std::cout << std::endl;

    return 0;
}
