#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <iostream> 
#include <queue> 
#include <thread> 
#include <mutex> 
#include <vector> 
#include <functional> 
#include <atomic> 
#include <string> 
using namespace std; 

#include "thread_safe_queue.hpp"

class ThreadPool 
{
private: 
    atomic_bool done; 
    atomic_int total_tasks; 
    vector<thread> threads; 
    ThreadSafeQueue<function<void()>> work_queue; 
    void work(); 
public: 
    ThreadPool(const int num_threads = thread::hardware_concurrency());
    ~ThreadPool();
    template <typename F> void submit_job(const F &func);
    template <typename F, typename ...A> void submit_job(const F &func, const A &...args);
    void wait_for_jobs();
    int get_num_threads() const { return threads.size(); }
};

ThreadPool::ThreadPool(const int num_threads) :
    done(false), total_tasks(0)
{
    for (int i = 0; i < num_threads; i++)
        threads.push_back(thread(&ThreadPool::work, this)); 
}

ThreadPool::~ThreadPool()
{
    done = true; 
    for (auto &t : threads) 
        t.join(); 
}

void ThreadPool::work()
{
    function<void()> func; 
    while (!done)
    {
        if (work_queue.try_pop(func))
        {
            func(); 
            total_tasks--;
        }
        this_thread::sleep_for(chrono::microseconds(10)); 
    }
}

template <typename F> 
void ThreadPool::submit_job(const F &func) 
{
    total_tasks++; 
    work_queue.push(func); 
}

template <typename F, typename ...A> 
void ThreadPool::submit_job(const F &func, const A &...args)
{
    total_tasks++; 
    work_queue.push([func, args...](){ func(args...); });
}

void ThreadPool::wait_for_jobs()
{
    while (total_tasks > 0)
    {
        this_thread::sleep_for(chrono::microseconds(10)); 
    }
}

#endif 
