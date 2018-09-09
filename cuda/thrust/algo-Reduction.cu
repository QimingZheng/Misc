#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>

// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
  __host__ __device__
  T operator()(const T& x) const
  { 
    return x * x;
  }
};

int main(void)
{
  // initialize host array
  float x[4] = {1.0, 2.0, 3.0, 4.0};

  // transfer to device
  thrust::device_vector<float> d_x(x, x + 4); //利用x来初始化d_x

  // setup arguments
  square<float>        unary_op;   //一元运算
  thrust::plus<float> binary_op;  //二元运算
  float init = 0;

  // compute norm
  float norm = std::sqrt( thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op) );

  std::cout << norm << std::endl;

  return 0;
}
