#include<iostream>
#include<stdio.h>
#include<string.h>
#include<algorithm>
#include<vector>
#include<map>
#include<assert.h>

using namespace std;
#define DEPTH 4

map<char,int> CharTable = {
    {'a', 0},    {'b', 1},    {'c', 2},    {'d', 3},    {'e', 4},    {'f', 5},
    {'g', 6},    {'h', 7},    {'i', 8},    {'j', 9},    {'k', 10},    {'l', 11},
    {'m', 12},    {'n', 13},    {'o', 14},    {'p', 15},    {'q', 16},    {'r', 17},
    {'s', 18},    {'t', 19},    {'u', 20},    {'v', 21},    {'w', 22},    {'x', 23},
    {'y', 24},    {'z', 25},    
    {'A', 26},    {'B', 27},    {'C', 28},    {'D', 29},    {'E', 30},    {'F', 31},
    {'G', 32},    {'H', 33},    {'I', 34},    {'J', 35},    {'K', 36},    {'L', 37},
    {'M', 38},    {'N', 39},    {'O', 40},    {'P', 41},    {'Q', 42},    {'R', 43},
    {'S', 44},    {'T', 45},    {'U', 46},    {'V', 47},    {'W', 48},    {'X', 49},
    {'Y', 50},    {'Z', 51}
};

struct WordEntry{
    string word;
    string explain;
    int index;
};

struct TrieNode{
    //char ch;
    TrieNode *son[52];
    bool end;
    WordEntry *we;
    /*
    TrieNode(char c) : ch(c){
        end=false;
        for(int i=0;i<26;i++) 
            son[i]=NULL;
        we=NULL;
    }
    */
    TrieNode(){
        //ch='';
        end=false;
        for(int i=0;i<52;i++) 
            son[i]=NULL;
        we=NULL;
    }
};

WordEntry* Search(TrieNode *cur,string word){
    if (cur == NULL) return NULL;
    if(word == "" && cur->end !=true) {
        return NULL;
    }
    if(word == "" && cur->end == true) {
        return cur->we;
    }
    int ind = CharTable[word[0]];
    word = word.substr(1,word.size()-1);
    return Search(cur->son[ind],word);
}

bool InsertWE(TrieNode *rt,string word, WordEntry we){
    assert(rt!=NULL);
    if (word=="") {
        rt->end=true;
        rt->we = new WordEntry;
        *rt->we=we;
        return true;
    }
    int ind = CharTable[word[0]];
    word = word.substr(1,word.size()-1);
    if (rt->son[ind]==NULL) {
        rt->son[ind] = new TrieNode();
    }
    return InsertWE(rt->son[ind], word, we); 
}

bool Insert(TrieNode *rt, string word, WordEntry we){
    assert(rt!=NULL);
    if(Search(rt,word)!=NULL) {cout<<word<<" already inserted!\n"; return false;}
    return InsertWE(rt, word, we);
}


bool Remove(TrieNode *rt, string word){
    assert(rt!=NULL);
    if(Search(rt,word)==NULL) {cout<<"No "<<word<<" in the dictionary!\n"; return false;}
    if (word=="") {
        rt->end=false;
        delete rt->we;
        rt->we = NULL;
        return true;
    }
    int ind = CharTable[word[0]];
    word = word.substr(1,word.size()-1);
    return Remove(rt->son[ind], word);
}

void Partially_Search(TrieNode *rt, vector<WordEntry> &ret, int search_depth){
    if(rt==NULL) return;
    if(search_depth <= 0) return;
//    cout<<"1Hello"<<endl;
    if (rt->end==true) {
//    cout<<"*Hello*"<<endl;
        ret.push_back(*rt->we);
    }
//    cout<<"2Hello"<<endl;
    for(int i=0;i<52;i++){
        Partially_Search(rt->son[i],ret,search_depth-1);
    }
//    cout<<"3Hello"<<endl;
}


vector<WordEntry> Match(TrieNode *rt, string word, int search_depth=DEPTH){
    assert(rt!=NULL);
    vector<WordEntry> ret;
    if(word==""){ cout<<"Empty string!"; return ret;}
    TrieNode *p = rt;
    while(p!=NULL && word!=""){
        int ind = CharTable[word[0]];
        word = word.substr(1,word.size()-1);
        p = p->son[ind];
    }
    if(p==NULL) return ret;
    Partially_Search(p,ret,search_depth);
    return ret;
}

void Traverse(TrieNode *rt){
    if(rt!=NULL){
        if(rt->end==true) cout<<"------------\t------------\n"<<rt->we->word<<"\t"<<rt->we->explain<<endl;
        for (int i=0;i<52;i++)
            Traverse(rt->son[i]);
    }
}
