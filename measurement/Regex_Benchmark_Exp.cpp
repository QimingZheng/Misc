#include <iostream>
#include <string>
#include <stdio.h>
#include <fstream>
#include <vector>
#include "pcre.h"
#include "re2/re2.h"
#include "hs.h"
#include "measurement.h"

#define ERROR_LOG 0

using namespace std;
using namespace re2;

struct pcre_input_scratch{
    vector<string> packets;
    vector<pcre *> compiled_rules;
    int size(){
        int re=0;
        for(int i=0; i<packets.size() ;i++){
            re += packets[i].size() * sizeof(packets[i][0]);
        }
        return re;
    }
};

struct pcre_output_scratch{
    int size(){return 0;}
};

struct re2_input_scratch{
    vector<string> packets;
    vector<RE2 *> compiled_rules;
    int size(){
        int re=0;
        for(int i=0; i<packets.size() ;i++){
            re += packets[i].size() * sizeof(packets[i][0]);
        }
        return re;
    }
};

struct re2_output_scratch{
    int size(){return 0;}
};

struct hs_input_scratch{
    vector<string> packets;
    vector<pair<hs_database_t *, hs_scratch_t *>> compiled_rules;
    int size(){
        int re=0;
        for(int i=0; i<packets.size() ;i++){
            re += packets[i].size() * sizeof(packets[i][0]);
        }
        return re;
    }
};

struct hs_output_scratch{
    int size(){return 0;}
};

vector<string> read_rules(string filename){
    ifstream fin = ifstream(filename);
    vector<string> ruleset;
    string temp;
    while(fin>>temp){
        ruleset.push_back(temp);
    }
    fin.close();
    return ruleset;
}

vector<string> read_packet(string filename){
    ifstream fin = ifstream(filename);
    vector<string> packetset;
    string temp;
    while(fin>>temp){
        packetset.push_back(temp);
    }
    fin.close();
    return packetset;
}

vector<pcre *> compile_pcre_rules(vector<string> &rules){
    vector<pcre *> compiled_rules;
    for(int i=0; i<rules.size(); i++){
        const char *error;
        int erroffset;
        pcre *regex = pcre_compile(rules[i].data(), 0, &error, &erroffset, NULL);
        if (regex!=NULL) compiled_rules.push_back(regex);
        else {
            if (ERROR_LOG) cout<<"PCRE Compilation Error: "<< rules[i] <<endl;
        }
    }
    cout<<"PCRE compiled "<<compiled_rules.size()<<" rules\n";
    return compiled_rules;
}

vector<RE2 *> compile_re2_rules(vector<string> &rules){
    vector<RE2 *> compiled_rules;
    for(int i=0; i<rules.size(); i++){
        RE2 *regex = new RE2(rules[i].data(), RE2::Quiet);
        if (regex->ok()) compiled_rules.push_back(regex);
        else {
            if (ERROR_LOG) cout<<"RE2 Compilation Error: "<< rules[i] <<endl;
        }
    }
    cout<<"RE2 compiled "<<compiled_rules.size()<<" rules\n";
    return compiled_rules;
}

vector<pair<hs_database_t *, hs_scratch_t *>> compile_hs_rules(vector<string> &rules){
    vector<pair<hs_database_t *, hs_scratch_t *>> compiled_rules;
    for(int i=0; i<rules.size(); i++){
        hs_database_t *database;
        hs_compile_error_t *compile_err;
        hs_scratch_t *scratch = NULL;
        if (hs_compile(rules[i].data(), HS_FLAG_DOTALL, HS_MODE_BLOCK, NULL, &database,
                    &compile_err) != HS_SUCCESS) {
            if (ERROR_LOG) cout<<"HS Compilation Error: "<< rules[i] <<endl;
            hs_free_compile_error(compile_err);
            continue;
        }
        if (hs_alloc_scratch(database, &scratch) != HS_SUCCESS) {
            if (ERROR_LOG) cout<<"HS allocation Error: Unable to allocate scratch space.\n";
            hs_free_database(database);
            continue;
        }
        compiled_rules.push_back(make_pair(database, scratch));
    }
    cout<<"HS compiled "<<compiled_rules.size()<<" rules\n";
    return compiled_rules;
}

void PCRE_Handler(event<pcre_input_scratch, pcre_output_scratch> &tracker){
    int OVECCOUNT = 30;
    int *ovector = new int[OVECCOUNT];
    for(int i = 0; i<tracker.input->compiled_rules.size(); i++){
        for(int j=0; j<tracker.input->packets.size(); j++){
            pcre_exec(tracker.input->compiled_rules[i], NULL, tracker.input->packets[j].data(), 
            tracker.input->packets[j].size(), 0, 0, ovector, OVECCOUNT);
        }
    }
}

void RE2_Handler(event<re2_input_scratch, re2_output_scratch> &tracker){
    for(int i = 0; i<tracker.input->compiled_rules.size(); i++){
        for(int j=0; j<tracker.input->packets.size(); j++){
            int rc;
            RE2::FullMatch(tracker.input->packets[j], *(tracker.input->compiled_rules[i]), (void*)NULL, &rc);
        }
    }
}

void HS_Handler(event<hs_input_scratch, hs_output_scratch> &tracker){
    for (int i=0; i<tracker.input->compiled_rules.size(); i++){
        for (int j=0;j<tracker.input->packets.size();j++){
            if (hs_scan(tracker.input->compiled_rules[i].first, tracker.input->packets[j].data(), 
            tracker.input->packets[j].size(), 0, tracker.input->compiled_rules[i].second, NULL, NULL) != HS_SUCCESS) {
                cout<<"Matching Error\n";
            }
        }
    }
    
    for (int i=0; i<tracker.input->compiled_rules.size(); i++){
        hs_free_database(tracker.input->compiled_rules[i].first);
        hs_free_scratch(tracker.input->compiled_rules[i].second);
    }
}

vector<pcre_input_scratch> PCRE_Sample_Input;
vector<pcre_output_scratch> PCRE_Sample_Output;

vector<re2_input_scratch> RE2_Sample_Input;
vector<re2_output_scratch> RE2_Sample_Output;

vector<hs_input_scratch> HS_Sample_Input;
vector<hs_output_scratch> HS_Sample_Output;

int main(){
    vector<string> rules = read_rules("snort.rules");
    vector<string> packets = read_packet("large_request.txt");


    for (int batch = 0; batch< 10; batch++){
        vector<pcre *> compiled_pcre_rules = compile_pcre_rules(rules);
        vector<RE2 *> compiled_re2_rules = compile_re2_rules(rules);
        vector<pair<hs_database_t *, hs_scratch_t *>> compiled_hs_rules = compile_hs_rules(rules);

        pcre_input_scratch *new_pcre_input = new pcre_input_scratch;
        pcre_output_scratch *new_pcre_output = new pcre_output_scratch;
        re2_input_scratch *new_re2_input = new re2_input_scratch;
        re2_output_scratch *new_re2_output = new re2_output_scratch;
        hs_input_scratch *new_hs_input = new hs_input_scratch;
        hs_output_scratch *new_hs_output = new hs_output_scratch;

        for (int i = batch * packets.size()/10; i<(batch+1) * packets.size()/10; i++){
            new_pcre_input->packets.push_back(packets[i]);
            new_re2_input->packets.push_back(packets[i]);
            new_hs_input->packets.push_back(packets[i]);
        }
        for (int j=0; j<compiled_pcre_rules.size(); j++){
            new_pcre_input->compiled_rules.push_back(compiled_pcre_rules[j]);
        }
        for (int j=0; j<compiled_re2_rules.size(); j++){
            new_re2_input->compiled_rules.push_back(compiled_re2_rules[j]);
        }
        for (int j=0; j<compiled_hs_rules.size(); j++){
            new_hs_input->compiled_rules.push_back(compiled_hs_rules[j]);
        }
        PCRE_Sample_Input.push_back(*new_pcre_input);
        PCRE_Sample_Output.push_back(*new_pcre_output);
        RE2_Sample_Input.push_back(*new_re2_input);
        RE2_Sample_Output.push_back(*new_re2_output);
        HS_Sample_Input.push_back(*new_hs_input);
        HS_Sample_Output.push_back(*new_hs_output);
    }

    measurement_framework<pcre_input_scratch, pcre_output_scratch>(&PCRE_Handler, PCRE_Sample_Input, PCRE_Sample_Output);
    measurement_framework<re2_input_scratch, re2_output_scratch>(&RE2_Handler, RE2_Sample_Input, RE2_Sample_Output);
    measurement_framework<hs_input_scratch, hs_output_scratch>(&HS_Handler, HS_Sample_Input, HS_Sample_Output);

    return 0;
}