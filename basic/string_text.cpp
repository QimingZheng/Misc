#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <vector>
using namespace boost;
using namespace std;

void test(){

string str("readme.txt");
    if (ends_with(str, ".txt")) {
        cout << to_upper_copy(str) + "UPPER" << endl;              // upper case
    }

    replace_first(str, "readme", "followme");                       // replace
    cout << str << endl;

    vector<char> v(str.begin(), str.end());
    vector<char> v2 = to_upper_copy(erase_first_copy(v, "txt")); // delete sub string
    for (int i = 0; i < v2.size(); ++i) {
        cout << v2[i];
    }

}

int main(){

cout << format ("%s:%d+%d=%d\n" )% "sum" % 1 % 2 % (1+2);
format fmt("(%1% + %2%) * %2% = %3%\n" );
fmt % 2 % 5 % ((2+5) * 5);
cout << fmt.str();

test();

}
