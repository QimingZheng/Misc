#include <boost/date_time/gregorian/gregorian.hpp> 
#include<iostream>
using namespace std;
using namespace boost::gregorian;
int main(){
date d1;                    // 无效日期
date d2(2010,1,1);
date d3(2000, Jan, 1);
date d4(d2);                // 拷贝构造
date d0 = from_string("1999-12-31");
date d5 ( from_string("2005/1/1") );
date d6 = from_undelimited_string("20001109");

cout << day_clock::local_day() << endl;
cout << day_clock::universal_day() << endl;

date neg(neg_infin);            //negative infinite time
date pos(pos_infin);            //positive infinite time
date notdate(not_a_date_time);  //not a date time
date maxdate(max_date_time);    //max date
date mindate(min_date_time);    //min date
cout << neg << endl << pos << endl << maxdate << endl << mindate <<endl;


}
