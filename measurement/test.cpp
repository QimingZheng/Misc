#include "measurement.h"

struct input_scratch{
    int in[100000];
};

struct output_scratch{
    int out[100000];
};


void my_handler(event<input_scratch, output_scratch> &tracker){
    for(int i=0; i<100000; i++) {
        tracker.input->in[i] = i;
        tracker.output->out[i]=0;
    }
    int last = 0;
    for(int i=0; i<100000; i++) {
        tracker.output->out[i] = tracker.input->in[i] + last;
        last = tracker.output->out[i];
    }
}

vector<input_scratch> Sample_Input;
vector<output_scratch> Sample_Output;

int main(){
    for(int i=0; i<100; i++) {
        input_scratch *new_input = new input_scratch;
        output_scratch *new_output = new output_scratch;
        Sample_Input.push_back(*new_input);
        Sample_Output.push_back(*new_output);
    }

    measurement_framework<input_scratch, output_scratch>(&my_handler, Sample_Input, Sample_Output);
}