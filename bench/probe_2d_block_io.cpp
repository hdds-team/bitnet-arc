/*
 * bench/probe_2d_block_io.cpp -- Path 8 minimal probe.
 *
 * Per docs/research-notes-paths.md Path 8: test if cl_intel_subgroup_
 * 2d_block_io extension delivers measurable memory throughput speedup
 * over scalar global reads, for a workload-relevant tile shape (8 rows
 * × 32 cols of int8 = matches typical A_slab read in kv4 ternary).
 *
 * Spec line 97 (saved in docs/specs/cl_intel_subgroup_2d_block_io.asciidoc):
 *   void intel_sub_group_2d_block_read_8b_8r32x1c(
 *       __global void* base_address, int width, int height, int pitch,
 *       int2 coord, ushort *destination);
 *
 * Reads 8 rows × 32 cols × 1 byte per col = 256 bytes, distributed
 * across the SG=16 lanes (each lane gets 256/16 = 16 bytes = 8 ushorts).
 *
 * Compares N_REPS reads of an 8x32 block via:
 *   (a) intel_sub_group_2d_block_read_8b_8r32x1c (Path 8 builtin)
 *   (b) scalar per-lane reads in a loop (baseline)
 *
 * Acceptance: ratio (a) / (b) >= 2x = 2D block IO worth integrating
 * in kv4 ternary kernel (Phase 2 v4 optim or Phase 1 v4 if simple).
 */

#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

const char* g_kernel_src = R"CLC(
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable

// Kernel A: 2D block read 8 rows x 32 cols x 1 byte (= 256 bytes
// distributed over 16 lanes = 16 bytes/lane = 8 ushorts/lane).
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void block_read_2d(const int n_reps,
                            __global const uchar* base,
                            const int width_bytes,
                            const int height,
                            const int pitch_bytes,
                            __global int* output)
{
    ushort accum = 0;
    int2 coord = (int2)(0, 0);
    for (int i = 0; i < n_reps; ++i) {
        ushort dst[8];
        intel_sub_group_2d_block_read_8b_8r32x1c(
            (__global void*)base, width_bytes, height, pitch_bytes,
            coord, dst);
        // XOR-accumulate to prevent DCE
        accum ^= dst[0] ^ dst[1] ^ dst[2] ^ dst[3]
              ^  dst[4] ^ dst[5] ^ dst[6] ^ dst[7];
    }
    const int gid = get_global_id(0);
    output[gid] = (int)accum;
}

// Kernel B: scalar per-lane reads of equivalent 256-byte block.
// Each of 16 lanes reads 16 bytes = 4 uints from the 256-byte block.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void scalar_read(const int n_reps,
                          __global const uchar* base,
                          const int /*width_bytes*/,
                          const int /*height*/,
                          const int pitch_bytes,
                          __global int* output)
{
    const int lane = get_sub_group_local_id();
    int accum = 0;
    for (int i = 0; i < n_reps; ++i) {
        // Each lane reads 16 bytes (= 4 uints) from offset (lane*16)
        // within the 8x32 = 256-byte block. Loop over 8 rows for full
        // coverage of the block.
        for (int row = 0; row < 8; ++row) {
            __global const uint* p = (__global const uint*)(
                base + row * pitch_bytes + lane);
            accum ^= p[0];
        }
    }
    output[get_global_id(0)] = accum;
}
)CLC";

void check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "OpenCL error %d in %s\n", err, what);
        std::exit(1);
    }
}

cl_device_id find_arc_device() {
    cl_uint np = 0; clGetPlatformIDs(0, nullptr, &np);
    std::vector<cl_platform_id> ps(np); clGetPlatformIDs(np, ps.data(), nullptr);
    for (cl_platform_id p : ps) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || !nd) continue;
        std::vector<cl_device_id> ds(nd); clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, ds.data(), nullptr);
        for (cl_device_id d : ds) {
            char n[256] = {}; clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(n), n, nullptr);
            if (std::strstr(n, "Arc")) return d;
        }
    }
    std::fprintf(stderr, "no Arc device\n"); std::exit(1);
}

double event_ms(cl_event e) {
    cl_ulong t0=0, t1=0;
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
    return double(t1-t0) / 1.0e6;
}

bool device_has_extension(cl_device_id dev, const char* ext) {
    std::size_t sz=0; clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, nullptr, &sz);
    std::vector<char> b(sz+1, 0); clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sz, b.data(), nullptr);
    return std::strstr(b.data(), ext) != nullptr;
}

} /* namespace */

int main() {
    cl_device_id dev = find_arc_device();
    char name[256]={}; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::fprintf(stderr, "device: %s\n", name);

    if (!device_has_extension(dev, "cl_intel_subgroup_2d_block_io")) {
        std::fprintf(stderr, "FAIL: cl_intel_subgroup_2d_block_io not exposed.\n");
        return 1;
    }
    std::fprintf(stderr, "OK: cl_intel_subgroup_2d_block_io exposed.\n");

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err); check(err, "ctx");
    cl_command_queue_properties qp[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, qp, &err); check(err, "q");

    cl_program prog = clCreateProgramWithSource(ctx, 1, &g_kernel_src, nullptr, &err); check(err, "prog");
    err = clBuildProgram(prog, 1, &dev, "-cl-std=CL3.0", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::size_t lsz=0; clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &lsz);
        std::vector<char> log(lsz+1, 0); clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, lsz, log.data(), nullptr);
        std::fprintf(stderr, "BUILD FAIL:\n%s\n", log.data());
        return 1;
    }
    std::fprintf(stderr, "OK: kernels compiled.\n");

    cl_kernel k_block = clCreateKernel(prog, "block_read_2d", &err); check(err, "k_block");
    cl_kernel k_scalar = clCreateKernel(prog, "scalar_read", &err); check(err, "k_scalar");

    /* Source buffer: bigger than the block to avoid pure cache hits.
     * 4 KB = 1024 rows of 32 bytes (we only read 8 rows but tile may
     * stride; this also gives realistic memory bandwidth signal). */
    constexpr int WIDTH_BYTES = 32;
    constexpr int HEIGHT      = 1024;
    constexpr int PITCH_BYTES = WIDTH_BYTES;
    cl_mem dsrc = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
                                 WIDTH_BYTES * HEIGHT, nullptr, &err); check(err, "dsrc");
    std::vector<std::uint8_t> hsrc(WIDTH_BYTES * HEIGHT);
    for (std::size_t i = 0; i < hsrc.size(); ++i) hsrc[i] = static_cast<std::uint8_t>(i & 0xff);
    clEnqueueWriteBuffer(q, dsrc, CL_TRUE, 0, hsrc.size(), hsrc.data(), 0, nullptr, nullptr);

    constexpr std::size_t NUM_LANES = 16;
    cl_mem dout = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                 NUM_LANES * sizeof(cl_int), nullptr, &err); check(err, "dout");

    auto run = [&](cl_kernel kern, int n_reps) -> double {
        check(clSetKernelArg(kern, 0, sizeof(int), &n_reps), "arg0");
        check(clSetKernelArg(kern, 1, sizeof(cl_mem), &dsrc), "arg1");
        const int wb = WIDTH_BYTES, h = HEIGHT, pb = PITCH_BYTES;
        check(clSetKernelArg(kern, 2, sizeof(int), &wb), "arg2");
        check(clSetKernelArg(kern, 3, sizeof(int), &h),  "arg3");
        check(clSetKernelArg(kern, 4, sizeof(int), &pb), "arg4");
        check(clSetKernelArg(kern, 5, sizeof(cl_mem), &dout), "arg5");
        std::size_t global = NUM_LANES, local = NUM_LANES;
        cl_event evt;
        check(clEnqueueNDRangeKernel(q, kern, 1, nullptr, &global, &local, 0, nullptr, &evt), "enq");
        check(clWaitForEvents(1, &evt), "wait");
        const double ms = event_ms(evt);
        clReleaseEvent(evt);
        return ms;
    };

    constexpr unsigned warmups = 5, timed_n = 25;
    std::vector<int> reps_list = {1000, 5000, 10000};
    std::printf("variant,N_REPS,t_ms_med,bytes_per_call,GB_per_s\n");

    double med_block_at_5k = 0, med_scalar_at_5k = 0;
    for (int n_reps : reps_list) {
        for (unsigned w = 0; w < warmups; ++w) { run(k_block, n_reps); run(k_scalar, n_reps); }
        std::vector<double> tb, ts;
        tb.reserve(timed_n); ts.reserve(timed_n);
        for (unsigned t = 0; t < timed_n; ++t) {
            tb.push_back(run(k_block, n_reps));
            ts.push_back(run(k_scalar, n_reps));
        }
        std::sort(tb.begin(), tb.end()); std::sort(ts.begin(), ts.end());
        const double mb = tb[tb.size()/2], ms = ts[ts.size()/2];
        const double gb_block  = (n_reps * 256.0 / 1e9) / (mb * 1e-3);
        const double gb_scalar = (n_reps * 256.0 / 1e9) / (ms * 1e-3);
        std::printf("2d_block_8b_8r32x1c,%d,%.5f,%.0f,%.2f\n", n_reps, mb, 256.0, gb_block);
        std::printf("scalar_8x32,%d,%.5f,%.0f,%.2f\n", n_reps, ms, 256.0, gb_scalar);
        std::fprintf(stderr,
                     "  N_REPS=%5d : block=%.4f ms (%.1f GB/s) | scalar=%.4f ms (%.1f GB/s) | speedup=%.3fx\n",
                     n_reps, mb, gb_block, ms, gb_scalar, ms / mb);
        if (n_reps == 5000) { med_block_at_5k = mb; med_scalar_at_5k = ms; }
    }

    clReleaseMemObject(dsrc); clReleaseMemObject(dout);
    clReleaseKernel(k_block); clReleaseKernel(k_scalar);
    clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx);

    if (med_block_at_5k > 0 && med_scalar_at_5k > 0) {
        const double ratio = med_scalar_at_5k / med_block_at_5k;
        std::fprintf(stderr,
                     "\nblock_read vs scalar_read (5000 reps): %.3fx\n",
                     ratio);
        if (ratio >= 2.0) {
            std::fprintf(stderr, "PASS: 2D block IO >= 2x. Worth integrating in kv4.\n");
            return 0;
        } else if (ratio >= 1.5) {
            std::fprintf(stderr, "MARGINAL: 1.5-2x. Worth Phase 2 v4 optim.\n");
            return 0;
        } else {
            std::fprintf(stderr, "MODEST: <1.5x. Defer to Phase 3 if needed.\n");
            return 0;
        }
    }
    return 1;
}
