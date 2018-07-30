#include<boost/lexical_cast.hpp>
    #include<string.h>
using namespace boost;

int main(){

{
using boost::lexical_cast;
int a = lexical_cast<int>("12345");
long b = lexical_cast<long>("43274697812518482");
double c = lexical_cast<double>("123432.54353");
float pi = lexical_cast<float>("3.1415");

std::cout<< a << " " << b << " " << c << " " << pi << std::endl;

}

{

std::string str = lexical_cast<std::string>(123);             // int -> string
std::cout << str << std::endl;
std::cout << lexical_cast<std::string>(1.234) << std::endl;        // float -> string
std::cout << lexical_cast<std::string>(0x11) << std::endl;         // 16进制 -> string

// lexical_cast can cast 0 & 1 to bool, but only support 0 & 1, not support True or False string
bool bl = lexical_cast<bool>("1");                   // string -> bool, only support 1 & 0
}



}
