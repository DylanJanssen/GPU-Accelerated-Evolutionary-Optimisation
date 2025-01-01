#ifndef THREAD_SAFE_QUEUE_H
#define THREAD_SAFE_QUEUE_H

#include <iostream>
#include <queue>
#include <thread>
#include <mutex>
#include <vector>
#include <functional>
#include <atomic>
#include <string>
using namespace std;

template <typename T>
class ThreadSafeQueue
{
private:
    mutex m;
    queue<T> data;

public:
    ThreadSafeQueue() {}
    void push(const T &value);
    bool try_pop(T &value);
    bool empty();
    int size();
};

template <typename T>
void ThreadSafeQueue<T>::push(const T &value)
{
    lock_guard lk(m);
    data.push(std::move(value));
}

template <typename T>
bool ThreadSafeQueue<T>::try_pop(T &value)
{
    lock_guard lk(m);
    if (data.empty())
        return false;
    value = std::move(data.front());
    data.pop();
    return true;
}

template <typename T>
bool ThreadSafeQueue<T>::empty()
{
    lock_guard lk(m);
    return data.empty();
}

template <typename T>
int ThreadSafeQueue<T>::size()
{
    lock_guard lk(m);
    return data.size();
}

#endif 