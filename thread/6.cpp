#include <iostream>       // std::cout
#include <functional>     // std::ref
#include <thread>         // std::thread
#include <future>         // std::promise, std::future

/* Future Promise相关
 * Promise实现了父子线程间返回值
**/

// 多线程计算 \sum_n{\frac{1}{n^n}}

void partial_sum(std::promise<double>& prom, int i) {
	double ret = 0.0;
	for (int j= i*1000+1;j<=(i+1)*1000;j++)
		ret += 1.00/(j*j);
	prom.set_value(ret);
}

int main ()
{
	std::promise<double> prom[10]; // 生成一个 std::promise<int> 对象.
	std::future<double> fut[10];
	for(int i=0;i<10;i++)	{
		fut[i] = prom[i].get_future(); // 和 future 关联.
	}
	std::thread t[10];
	for (int i=0;i<10;i++)
		t[i] = std::thread(partial_sum, std::ref(prom[i]), i); // 将 future 交给另外一个线程t.
	double x,sum =0.0;
       	for(int i=0;i<10;i++){
		x = fut[i].get();
		std::cout<<x<<'\n';//
		sum+=x;
	}
	std::cout << "value: " << sum << '\n'; // 打印 value: 10.
       	for(int i=0;i<10;i++)
	t[i].join();
	return 0;
}
