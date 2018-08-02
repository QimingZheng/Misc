#include <iostream>
#include <string>
#include <boost/assign.hpp>
#include <boost/typeof/typeof.hpp>
#include <boost/assign.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/ref.hpp>
#include <boost/bind.hpp>
using namespace boost;
using namespace std;

#define BOOST_DATE_TIME_SOURCE
#define BOOST_THREAD_NO_LIB

template <typename T>
class basic_atom: noncopyable {
private:
    T n;
    typedef boost::mutex mutex_t;
    mutex_t mu;

public:
    basic_atom(T x = T()):n(x) {}
    T operator++(){
        mutex_t::scoped_lock lock(mu);
        return ++n;
    }
    operator T()    {return n;}
};

boost::mutex io_mu;

typedef basic_atom<int> atom_int;

void printing(atom_int& x, const string& str){
    for (int i = 0;  i < 5; ++i) {
        boost::mutex::scoped_lock lock(io_mu);
        cout << str << ++x <<endl;
    }
}

void to_interrupt(atom_int& x, const string& str){
    try {
        for (int i = 0;  i < 5; ++i) {
            this_thread::sleep(posix_time::seconds(1));
            boost::mutex::scoped_lock lock(io_mu);
            cout << str << ++x <<endl;
        }
    } catch (thread_interrupted& ) {
        cout << "thread_interrupted" <<endl;
    }
}

int main(int argc, const char * argv[]) {

    cout << "start" <<endl;
    this_thread::sleep(posix_time::seconds(2));
    cout << "sleep 2 seconds" <<endl;

    boost::mutex mu;
    try {
        mu.lock();
        cout << "do some operations" << endl;
        mu.unlock();
    } catch (...) {
        mu.unlock();
    }

    boost::mutex mu1;
    boost::mutex::scoped_lock lock(mu1);
    cout << "some operations" <<endl;

    atom_int x;
    
    thread t1(printing, boost::ref(x), "hello");

    if (t1.joinable()) {
        t1.join();          // 等待t1 线程结束再返回，不管执行多长时间
    }

    thread t2(printing, boost::ref(x), "boost");
    t2.timed_join(posix_time::seconds(2));      // 最多等待1秒返回
    thread t(to_interrupt, boost::ref(x), "interrupt");
    this_thread::sleep(posix_time::seconds(4));
    t.interrupt();
    t.join();

    return 0;
}
