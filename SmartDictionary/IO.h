#include "util.h"
#include <fstream>
/**
 * Smart-Dict文件IO：
 * 很简单的格式： word \n  中文解释
 **/

vector<WordEntry> ReadF(){
    ifstream fin("wordlist",ios::in);
    string wd,ep;
    char tmp[10000];
    int ind=0;
    vector<WordEntry> ret;
    while(fin.getline(tmp,10000)){
        wd = tmp;
        fin.getline(tmp,10000);
        ep = tmp;
        WordEntry we;
        we.index= ind;
        we.word = wd;
        we.explain = ep;
        ret.push_back(we);
    }
    fin.close();
    return ret;
}

