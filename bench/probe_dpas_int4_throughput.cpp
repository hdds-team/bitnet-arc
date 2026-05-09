/*
 * bench/probe_dpas_int4_throughput.cpp -- Path 6 (INT4 K=64).
 *
 * Per docs/research-notes-paths.md Path 6: test if the INT4 K=64
 * builtin (`intel_sub_group_i4_i4_matrix_mad_k64`) delivers
 * meaningful speedup over both FP16 wrapper baseline and INT8 K=32
 * direct.
 *
 * Spec line 162 (docs/specs/cl_intel_subgroup_matrix_multiply_accumulate.asciidoc):
 *   int8 intel_sub_group_i4_i4_matrix_mad_k64(short8 a, int8 b, int8 acc);  // M=8 SG=16
 *
 * Per-lane storage (SG=16, M=8, K=64, N=16):
 *   - short8 a: 8 shorts × 2 bytes = 16 bytes/lane = 32 INT4/lane
 *               (each byte holds 2 INT4 nibbles)
 *               SG=16 × 32 = 512 INT4 = M*K = 8*64 ✓
 *   - int8 b:   8 ints × 4 bytes = 32 bytes/lane = 64 INT4/lane
 *               SG=16 × 64 = 1024 INT4 = K*N = 64*16 ✓
 *   - int8 acc: 8 int32/lane × 16 lanes = 128 int32 = M*N ✓
 *
 * All-nibble=1 init (each INT4 = 1):
 *   short = 0x1111 (4 nibbles all = 1)
 *   int   = 0x11111111 (8 nibbles all = 1)
 *
 * Validation:
 *   each acc[m, n] += sum_{k=0..63}(1*1) = K = 64 per MMA
 *   after n_reps: acc[m, n] = 64 * n_reps
 *   per-lane sum (.s0..s7) = 8 * 64 * n_reps = 512 * n_reps
 *
 * Acceptance gate (per design v3 sec6.0 spirit):
 *   PASS     : ratio >= 2.0× FP16 wrapper baseline (= structural
 *              advantage, justifies kv4 INT4 ternary path)
 *   MARGINAL : 1.5× to 2× -> kv4 may proceed, lower expectation
 *   FAIL     : <1.5× -> INT4 path not winning enough; consider
 *              Path 7 (INT2) or Path 9 (inline asm) next
 *
 * Build: bench/Makefile target probe_dpas_int4_throughput.
 * Standalone OpenCL, no SYCL.
 */

#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

const char* g_kernel_src = R"CLC(
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

// SG=16, M=8, K=64, N=16. Per spec line 162 (saved in docs/specs/).
// All nibbles = 1 init: each MMA adds K = 64 to each acc element.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void dpas_int4_throughput(const int n_reps,
                                   __global int* output)
{
    const short s = 0x1111;   // 4 nibbles all = 1
    const int   p = 0x11111111;   // 8 nibbles all = 1
    short8 a   = (short8)(s, s, s, s, s, s, s, s);
    int8   b   = (int8)(p, p, p, p, p, p, p, p);
    int8   acc = (int8)(0, 0, 0, 0, 0, 0, 0, 0);

    for (int i = 0; i < n_reps; ++i) {
        acc = intel_sub_group_i4_i4_matrix_mad_k64(a, b, acc);
    }

    const int gid = get_global_id(0);
    output[gid] = acc.s0 + acc.s1 + acc.s2 + acc.s3
                + acc.s4 + acc.s5 + acc.s6 + acc.s7;
}
)CLC";

void check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "OpenCL error %d in %s\n", err, what);
        std::exit(1);
    }
}

cl_device_id find_arc_device() {
    cl_uint n_platforms = 0;
    check(clGetPlatformIDs(0, nullptr, &n_platforms), "clGetPlatformIDs");
    std::vector<cl_platform_id> platforms(n_platforms);
    clGetPlatformIDs(n_platforms, platforms.data(), nullptr);
    for (cl_platform_id p : platforms) {
        cl_uint n_devs = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &n_devs)
                != CL_SUCCESS || n_devs == 0)
            continue;
        std::vector<cl_device_id> devs(n_devs);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, n_devs, devs.data(), nullptr);
        for (cl_device_id d : devs) {
            char name[256] = {};
            clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name), name, nullptr);
            if (std::strstr(name, "Arc") != nullptr) return d;
        }
    }
    std::fprintf(stderr, "no Intel Arc OpenCL device found\n");
    std::exit(1);
}

double event_ms(cl_event e) {
    cl_ulong t0 = 0, t1 = 0;
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START,
                            sizeof(t0), &t0, nullptr);
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END,
                            sizeof(t1), &t1, nullptr);
    return double(t1 - t0) / 1.0e6;
}

} /* anonymous namespace */

int main() {
    cl_device_id dev = find_arc_device();
    char name[256] = {}, drv[256] = {};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(drv), drv, nullptr);
    std::fprintf(stderr, "device: %s (driver: %s)\n", name, drv);

    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    check(err, "clCreateContext");
    cl_command_queue_properties qprops[] =
        {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue q = clCreateCommandQueueWithProperties(
        ctx, dev, qprops, &err);
    check(err, "clCreateCommandQueueWithProperties");

    cl_program prog = clCreateProgramWithSource(
        ctx, 1, &g_kernel_src, nullptr, &err);
    check(err, "clCreateProgramWithSource");
    err = clBuildProgram(prog, 1, &dev, "-cl-std=CL3.0",
                         nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::size_t log_sz = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
                              0, nullptr, &log_sz);
        std::vector<char> log(log_sz + 1, 0);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG,
                              log_sz, log.data(), nullptr);
        std::fprintf(stderr, "BUILD FAIL:\n%s\n", log.data());
        std::fprintf(stderr,
                     "FAIL: INT4 K=64 builtin not exposed or "
                     "different signature on this driver. Path 6 blocked.\n");
        return 1;
    }
    std::fprintf(stderr, "OK: kernel compiled (INT4 K=64 builtin found).\n");

    cl_kernel kern = clCreateKernel(prog, "dpas_int4_throughput", &err);
    check(err, "clCreateKernel");

    constexpr std::size_t NUM_LANES = 16;
    cl_mem dout = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                 NUM_LANES * sizeof(cl_int),
                                 nullptr, &err);
    check(err, "clCreateBuffer");

    auto run_one = [&](int n_reps, bool validate) -> double {
        const cl_int zero = 0;
        check(clEnqueueFillBuffer(q, dout, &zero, sizeof(cl_int),
                                  0, NUM_LANES * sizeof(cl_int),
                                  0, nullptr, nullptr),
              "clEnqueueFillBuffer");
        check(clSetKernelArg(kern, 0, sizeof(int), &n_reps),
              "clSetKernelArg(n_reps)");
        check(clSetKernelArg(kern, 1, sizeof(cl_mem), &dout),
              "clSetKernelArg(dout)");
        std::size_t global = NUM_LANES, local = NUM_LANES;
        cl_event evt;
        check(clEnqueueNDRangeKernel(q, kern, 1, nullptr, &global,
                                     &local, 0, nullptr, &evt),
              "clEnqueueNDRangeKernel");
        check(clWaitForEvents(1, &evt), "clWaitForEvents");
        cl_int exec_status = 0;
        clGetEventInfo(evt, CL_EVENT_COMMAND_EXECUTION_STATUS,
                       sizeof(exec_status), &exec_status, nullptr);
        if (exec_status != CL_COMPLETE) {
            std::fprintf(stderr,
                "kernel did not COMPLETE (status=%d, n_reps=%d)\n",
                exec_status, n_reps);
            clReleaseEvent(evt);
            std::exit(4);
        }
        const double ms = event_ms(evt);
        clReleaseEvent(evt);

        /* Validation: M=8, K=64, all-nibble=1
         *   each acc[m, n] = K * n_reps = 64 * n_reps
         *   per-lane sum (.s0..s7) = 8 * 64 * n_reps = 512 * n_reps */
        if (validate) {
            std::vector<cl_int> host_out(NUM_LANES, -1);
            check(clEnqueueReadBuffer(q, dout, CL_TRUE, 0,
                                      NUM_LANES * sizeof(cl_int),
                                      host_out.data(), 0, nullptr, nullptr),
                  "clEnqueueReadBuffer");
            const cl_int expected = 512 * n_reps;
            int ok = 0, bad = 0;
            for (cl_int v : host_out) {
                if (v == expected) ++ok;
                else               ++bad;
            }
            if (bad > 0) {
                std::fprintf(stderr,
                    "VALIDATION FAIL n_reps=%d: expected %d in all %zu "
                    "lanes, got ok=%d bad=%d (sample: %d, %d, %d, %d)\n",
                    n_reps, expected, NUM_LANES, ok, bad,
                    host_out[0], host_out[1], host_out[2], host_out[3]);
                std::exit(5);
            }
        }
        return ms;
    };

    /* Validate first */
    {
        const double validate_ms = run_one(50, true);
        std::fprintf(stderr,
                     "VALIDATION OK at n_reps=50, t=%.4f ms "
                     "(kernel wrote 512*50=25600 in all %zu lanes, MMA confirmed)\n",
                     validate_ms, NUM_LANES);
    }

    std::vector<int> reps_list = {500, 1000, 2000, 5000};
    constexpr unsigned warmups = 5, timed = 25;
    /* M=8, K=64, N=16: 8*16*64*2 = 16384 ops/MMA */
    constexpr double OPS_PER_MMA = 8.0 * 16.0 * 64.0 * 2.0;

    std::printf("variant,N_REPS,t_ms_med,ops_per_mma,gops_per_s\n");

    double med_at_2k = 0;
    for (int n_reps : reps_list) {
        for (unsigned w = 0; w < warmups; ++w) run_one(n_reps, false);
        std::vector<double> ts;
        ts.reserve(timed);
        for (unsigned t = 0; t < timed; ++t)
            ts.push_back(run_one(n_reps, false));
        std::sort(ts.begin(), ts.end());
        const double med = ts[ts.size() / 2];
        const double gops = (n_reps * OPS_PER_MMA) / (med * 1e6);
        std::printf("dpas_opencl_int4_8x16x64,%d,%.5f,%.0f,%.2f\n",
                    n_reps, med, OPS_PER_MMA, gops);
        std::fprintf(stderr,
                     "  N_REPS=%5d : INT4 DPAS K=64 t_med=%.4f ms (%.1f GOps/s)\n",
                     n_reps, med, gops);
        if (n_reps == 2000) med_at_2k = med;
    }
    /* End-of-run validation: confirm no degradation */
    {
        const double validate_ms = run_one(reps_list.back(), true);
        std::fprintf(stderr,
                     "VALIDATION OK at n_reps=%d (end-of-run), t=%.4f ms\n",
                     reps_list.back(), validate_ms);
    }

    clReleaseMemObject(dout);
    clReleaseKernel(kern);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    /* Reference baselines (from earlier validated probes):
     *   FP16 SYCL wrapper (16x16x16, 4096 ops/MMA): 93.6 GOps/s
     *   INT8 DPAS direct (8x16x32, 8192 ops/MMA):  122.1 GOps/s */
    constexpr double BASELINE_FP16_GOPS = 93.6;
    constexpr double BASELINE_INT8_DPAS_GOPS = 122.1;
    if (med_at_2k <= 0.0) {
        std::fprintf(stderr, "\nINVALID: missing 2000-rep timing.\n");
        return 1;
    }
    const double gops_at_2k = (2000.0 * OPS_PER_MMA) / (med_at_2k * 1e6);
    const double ratio_fp16 = gops_at_2k / BASELINE_FP16_GOPS;
    const double ratio_int8 = gops_at_2k / BASELINE_INT8_DPAS_GOPS;
    std::fprintf(stderr,
                 "\nINT4 DPAS K=64 at N_REPS=2000: %.1f GOps/s\n"
                 "  vs FP16 wrapper baseline   (93.6 GOps/s):   %.3fx\n"
                 "  vs INT8 DPAS direct        (122.1 GOps/s):  %.3fx\n",
                 gops_at_2k, ratio_fp16, ratio_int8);
    std::fprintf(stderr,
                 "  Note: INT4 K=64 packs 4x more ops/MMA than FP16 16x16x16,\n"
                 "  so a ratio = 4.0x means same wall-clock per call.\n"
                 "  INT4 vs INT8 packs 2x more ops/MMA (16384 vs 8192).\n");
    if (ratio_fp16 >= 2.0) {
        std::fprintf(stderr,
                     "PASS: INT4 K=64 >= 2x FP16 wrapper. Strong signal\n"
                     "for kv4 ternary INT4 path. Path 6 unlocked.\n");
        return 0;
    } else if (ratio_fp16 >= 1.5) {
        std::fprintf(stderr,
                     "MARGINAL: 1.5x <= ratio < 2x. Worth trying kv4 ternary\n"
                     "INT4 path with measured expectations.\n");
        return 0;
    } else {
        std::fprintf(stderr,
                     "FAIL: ratio < 1.5x. INT4 K=64 doesn't deliver\n"
                     "structural speedup. Try Path 7 (INT2) or 9 (inline asm).\n");
        return 1;
    }
}
