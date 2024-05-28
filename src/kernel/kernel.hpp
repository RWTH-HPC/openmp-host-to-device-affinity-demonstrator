#ifndef __KERNEL_KERNEL__
#define __KERNEL_KERNEL__

#include "../util/define.hpp"

#include <cstddef>
#if (USE_OMP_TARGET == 0)
#include <cuda_runtime.h>
#endif

namespace kernel
{
class MatrixMultiplyDevice
{
  protected:
    int device;

  public:
    MatrixMultiplyDevice(int device) : device(device)
    {
    }
    virtual ~MatrixMultiplyDevice() = default;
    virtual void execute(double const *a, double const *b, double *c, size_t const n) const = 0;
    virtual void executeAsync(double const *a, double const *b, double *c, size_t const n, int const stream_id) const = 0;
};

#if (USE_OMP_TARGET == 0)
class MatrixMultiplyCUDA : public MatrixMultiplyDevice
{
  private:
    cudaStream_t *streams;

  public:
    MatrixMultiplyCUDA(int device, int num_streams = 0);
    ~MatrixMultiplyCUDA() override;
    void execute(double const *a, double const *b, double *c, size_t const n) const override;
    void executeAsync(double const *a, double const *b, double *c, size_t const n, int const stream_id) const override;
    void syncronizeStream(int const stream_id) const;
};
#else
class MatrixMultiplyOMP : public MatrixMultiplyDevice
{
  public:
    MatrixMultiplyOMP(int device);
    void execute(double const *a, double const *b, double *c, size_t const n) const override;
    void executeAsync(double const *a, double const *b, double *c, size_t const n, int const stream_id) const override;
};
#endif
} // namespace kernel

#endif // __KERNEL_KERNEL__
