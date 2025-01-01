#ifndef CUVECTOR_H 
#define CUVECTOR_H 
#include "cuda_err.cuh"

template <class T>
class cuvector
{
private: 
    T *h_ptr, *d_ptr; 
    const int sz; 
public: 
    cuvector(int sz); 
    ~cuvector();
    void cpu(); 
    void cuda(); 
    T* get_device_ptr(); 
    T& operator[](int index); 
    int size() const { return sz; }
    T* begin() { return h_ptr; }
    T* end() { return h_ptr + sz; }
};

template <class T> 
cuvector<T>::cuvector(int size) : sz(size)
{
    h_ptr = new T[sz];
    cuda_error_check(cudaMalloc((void **)&d_ptr, sz * sizeof(T)));
}

template <class T> 
cuvector<T>::~cuvector()
{
    delete[] h_ptr;
    cuda_error_check(cudaFree(d_ptr)); 
}

template <class T> 
void cuvector<T>::cuda()
{
    cuda_error_check(cudaMemcpy(d_ptr, h_ptr, sz * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T> 
void cuvector<T>::cpu()
{
    cuda_error_check(cudaMemcpy(h_ptr, d_ptr, sz * sizeof(T), cudaMemcpyDeviceToHost));
}

template <class T> 
T* cuvector<T>::get_device_ptr()
{
    return d_ptr; 
}

template <class T> 
T& cuvector<T>::operator[](int index)
{
    return h_ptr[index]; 
}
#endif 