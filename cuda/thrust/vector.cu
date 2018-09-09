#include <thrust/host_vector.h>  
#include <thrust/device_vector.h>  
# include <thrust/copy.h>  
# include <thrust/fill.h>  
# include <thrust/sequence.h> 
#include <iostream>  
int fun_1( void )  
{  
// H has storage for 4 integers  
thrust :: host_vector <int > H (4);  
// initialize individual elements  
H [0] = 14;  
H [1] = 20;  
H [2] = 38;  
H [3] = 46;  
// H. size () returns the size of vector H  
std :: cout << "H has size " << H. size () << std :: endl ;  
// print contents of H  
for ( int i = 0; i < H. size (); i ++)  
std :: cout << "H[" << i << "] = " << H[i] << std :: endl ;  
// resize H  
H. resize (2) ;  
std :: cout << "H now has size " << H. size () << std :: endl ;  
// Copy host_vector H to device_vector D  
thrust :: device_vector <int > D = H;  
// elements of D can be modified  
D [0] = 99;  
D [1] = 88;  
// print contents of D  
for ( int i = 0; i < D. size (); i ++)  
std :: cout << "D[" << i << "] = " << D[i] << std :: endl ;  
// H and D are automatically deleted when the function returns  
return 0;  
}

int fun_2(void)
{
// initialize all ten integers of a device_vector to 1  
thrust :: device_vector <int > D(10 , 1);  
// set the first seven elements of a vector to 9  
thrust :: fill (D. begin () , D. begin () + 7, 9);  
// initialize a host_vector with the first five elements of D  
thrust :: host_vector <int > H(D. begin () , D. begin () + 5);  
// set the elements of H to 0, 1, 2, 3, ...  
thrust :: sequence (H. begin () , H. end ());  
// copy all of H back to the beginning of D  
thrust :: copy (H. begin () , H. end () , D. begin ());  
// print D  
for ( int i = 0; i < D. size (); i ++)  
std :: cout << "D[" << i << "] = " << D[i] << std :: endl ;  
return 0;  
}


int main(){
std::cout<<"==== F-1 ====\n";
fun_1();
std::cout<<"==== F-2 ====\n";
fun_2();
return 0;
}
