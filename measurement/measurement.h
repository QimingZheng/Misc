#ifndef MEASUREMENT_HEADER
#define MEASUREMENT_HEADER

#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <stdio.h>
#include <float.h>
#include <vector>
#include <thread>
#include <assert.h>
#include <math.h>
#include <iomanip>

using namespace std;

template<typename T, typename R>
struct event{
    struct timeval start;
    struct timeval stop;
    int id;
    T *input;
    R *output;
};

struct result{
    double mean;
    double std;
    double max;
    double min;
    double sample_size;
};

template<typename T, typename R>
double input_size(event<T,R> &Event){
    return sizeof(*(Event.input));
}

template<typename T, typename R>
double output_size(event<T,R> &Event){
    return sizeof(*(Event.output));
}

template<typename T, typename R>
double total_size(event<T,R> *EventSeries, int len){
    double re = 0.0;
    for (int iter = 0; iter<len; iter++){
        re += (input_size(EventSeries[iter]) + output_size(EventSeries[iter]));
    }
    return re;
}

template<typename T, typename R>
struct result throughput(event<T,R> *EventSeries, int len){
    struct result re;
    re.sample_size = len;
    re.mean = 0.0;
    re.std = 0.0;
    re.max = DBL_MIN;
    re.min = DBL_MAX;
    double *_throughput_ = new double [len];
    for (int iter = 0; iter<len; iter++){
        double start_time = EventSeries[iter].start.tv_sec + EventSeries[iter].start.tv_usec/1000000.0;
        double stop_time = EventSeries[iter].stop.tv_sec + EventSeries[iter].stop.tv_usec/1000000.0;
        double workload = input_size(EventSeries[iter])+output_size(EventSeries[iter]);
        _throughput_[iter] = workload/(stop_time- start_time);
    }
    for (int iter = 0; iter<len; iter++){
        re.mean += _throughput_[iter];
        re.max = max(re.max, _throughput_[iter]);
        re.min = min(re.min, _throughput_[iter]);
    }
    re.mean /= len;
    for (int iter = 0; iter<len; iter++){
        re.std += (_throughput_[iter]-re.mean)*(_throughput_[iter]-re.mean);
    }
    re.std /= len;
    re.std = sqrt(re.std);
    return re;
}

template<typename T, typename R>
struct result latency (event<T,R> *EventSeries, int len){
    struct result re;
    re.sample_size = len;
    re.mean = 0.0;
    re.std = 0.0;
    re.max = DBL_MIN;
    re.min = DBL_MAX;
    double *_latency_ = new double [len];
    for (int iter = 0; iter<len; iter++){
        double start_time = EventSeries[iter].start.tv_sec + EventSeries[iter].start.tv_usec/1000000.0;
        double stop_time = EventSeries[iter].stop.tv_sec + EventSeries[iter].stop.tv_usec/1000000.0;
        _latency_[iter] = stop_time-start_time;
    }
    for (int iter = 0; iter<len; iter++){
        re.mean += _latency_[iter];
        re.max = max(re.max, _latency_[iter]);
        re.min = min(re.min, _latency_[iter]);
    }
    re.mean /= len;
    for (int iter = 0; iter<len; iter++){
        re.std += (_latency_[iter]-re.mean)*(_latency_[iter]-re.mean);
    }
    re.std /= len;
    re.std = sqrt(re.std);
    return re;
}

template <typename T, typename R>
void Handler(void (*SystemHandler)(event<T, R> &), event<T, R> &tracker_event){
    gettimeofday(&(tracker_event.start), NULL);
    SystemHandler(tracker_event);
    gettimeofday(&(tracker_event.stop), NULL);
}

template <typename T, typename R>
void measurement_framework(void (*SystemHandler)(event<T, R> &), vector<T> &InputSeries, vector<R> &OutputSeries){
    int len = InputSeries.size();
    thread *t = new thread[len];
    event<T,R> EventSeries[len];
    for(int iter = 0; iter < len; iter ++){
        event<T, R> *this_invoke = new event<T,R>;
        EventSeries[iter] = (*this_invoke);
        EventSeries[iter].input = &InputSeries[iter];
        EventSeries[iter].output = &OutputSeries[iter];
        t[iter] = thread(Handler<T, R>, SystemHandler, std::ref(EventSeries[iter]));
    }

    for(int iter = 0; iter < len; iter ++){
        t[iter].join();
    }

    cout.setf(ios::left);
    cout.setf(ios::showpoint);
    cout.precision(10);
    struct result throughput_re = throughput(EventSeries, len);
    struct result latency_re = latency(EventSeries, len);
    cout<<setw(20)<<left<<"statistic"<<setw(20)<<left<<"mean"<<setw(20)<<left<<"std"<<setw(20)<<left<<"min"<<setw(20)<< left<<"max"<<endl;
    cout<<setw(20)<<left<<"Throughput"<<setw(20)<<left<< throughput_re.mean<<setw(20)<<left<<throughput_re.std<<setw(20)<<left
    <<setw(20)<<left<<throughput_re.min<<setw(20)<<left<<throughput_re.max <<endl;
    cout<<setw(20)<<left<<"Latency"<<setw(20)<<left<< latency_re.mean<<setw(20)<<left<<latency_re.std<<setw(20)<<left
    <<setw(20)<<left<<latency_re.min<<setw(20)<<left<<latency_re.max <<endl;

}

#endif
