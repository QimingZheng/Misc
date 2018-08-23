#include "IO.h"
#include<random>

void ManualTest(){
    TrieNode *root = new TrieNode;
    while(true){
       string a;
       cin>>a;
       WordEntry b;
       b.word = a;
       b.explain =a;
       b.index = 0;
       cout<< Insert(root,a,b) <<endl;
       cout<< Search(root,a)->explain <<"\n";
        auto vec=Match(root,a.substr(0,a.size()-1));
        for(auto i = vec.begin(); i < vec.end();i++)
        {
            cout<<(*i).explain<<" ; ";
        }
        cout<<endl;
        Traverse(root);
        if(rand()%2) Remove(root,a);
    }
}

int main(){
    TrieNode *root = new TrieNode;
    vector<WordEntry> wordlist = ReadF();
    for(auto i = wordlist.begin();i<wordlist.end();i++)
        Insert(root,(*i).word,*i);
    while(true){
        cout<<">> ";
        string a;
        cin>>a;
        auto vec=Match(root,a);
        cout<<"====你可能在找=====\n";
        for(auto i = vec.begin(); i < vec.end();i++)
        {
            cout<<(*i).word<<" : "<<(*i).explain<<"\n";
        }
        cout<<endl;
    //    Traverse(root);
    }

}
