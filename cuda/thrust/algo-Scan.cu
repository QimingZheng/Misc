#include <thrust/scan.h>

int main(){
int data[6] = {1, 0, 2, 2, 1, 3};
thrust::inclusive_scan(data, data + 6, data);  // in-place scan ，in-place指的是输入输出的位置一样的
// data is now {1, 1, 3, 5, 6, 9}

thrust::exclusive_scan(data, data + 6, data); // in-place scan
// data is now {0, 1, 1, 3, 5, 6}
}
