//! Synthetic single-dispatch-per-pipeline workload used to reverse-engineer
//! per-command counter records in the .gpuprofiler_raw format.
//!
//! Each kernel runs a different fma-loop length so that "Kernel ALU
//! Instructions" and "Kernel Invocations" are unique per command. With one
//! dispatch per pipeline Xcode has nothing to aggregate, so any per-command
//! integer we see in the GPU Commands tab maps to exactly one stored record.

use std::path::PathBuf;

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct SynthBenchOptions {
    pub output: PathBuf,
    pub iterations: Vec<u32>,
    pub threadgroup_counts: Vec<u32>,
    pub threads_per_group: u32,
}

/// Distinct loop-iteration counts. Mostly primes, all unique.
pub const DEFAULT_ITERATIONS: &[u32] = &[
    101, 197, 379, 547, 769, 1097, 1543, 2179, 3079, 4337, 6121, 8629, 12161, 17137, 24151, 34057,
];

/// Distinct threadgroup counts so `Kernel Invocations` is unique per command.
pub const DEFAULT_THREADGROUP_COUNTS: &[u32] = &[
    16, 23, 30, 37, 44, 51, 58, 65, 72, 79, 86, 93, 100, 107, 114, 121,
];

pub const DEFAULT_THREADS_PER_GROUP: u32 = 32;

impl Default for SynthBenchOptions {
    fn default() -> Self {
        Self {
            output: PathBuf::from("/tmp/synth.gputrace"),
            iterations: DEFAULT_ITERATIONS.to_vec(),
            threadgroup_counts: DEFAULT_THREADGROUP_COUNTS.to_vec(),
            threads_per_group: DEFAULT_THREADS_PER_GROUP,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SynthBenchPlanRow {
    pub index: usize,
    pub function_name: String,
    pub iterations: u32,
    pub threadgroups: u32,
    pub threads_per_group: u32,
    pub invocations: u64,
}

#[derive(Debug, Clone)]
pub struct SynthBenchPlan {
    pub rows: Vec<SynthBenchPlanRow>,
}

pub fn plan(options: &SynthBenchOptions) -> Result<SynthBenchPlan> {
    if options.iterations.len() != options.threadgroup_counts.len() {
        return Err(Error::InvalidInput(
            "iterations and threadgroup_counts must have the same length".to_owned(),
        ));
    }
    let rows = options
        .iterations
        .iter()
        .zip(options.threadgroup_counts.iter())
        .enumerate()
        .map(|(index, (iterations, groups))| SynthBenchPlanRow {
            index,
            function_name: format!("synth_k{:02}", index),
            iterations: *iterations,
            threadgroups: *groups,
            threads_per_group: options.threads_per_group,
            invocations: u64::from(*groups) * u64::from(options.threads_per_group),
        })
        .collect();
    Ok(SynthBenchPlan { rows })
}

pub fn metal_source(iterations: &[u32]) -> String {
    let mut src = String::from(
        "#include <metal_stdlib>\n\
         using namespace metal;\n\n\
         template<uint N>\n\
         inline float synth_work(float seed) {\n\
         \x20   float x = seed;\n\
         \x20   for (uint j = 0; j < N; ++j) {\n\
         \x20       float c = 1.0f + float(j) * 1e-7f;\n\
         \x20       x = fma(x, c, 0.001f);\n\
         \x20   }\n\
         \x20   return x;\n\
         }\n\n",
    );
    for (i, n) in iterations.iter().enumerate() {
        src.push_str(&format!(
            "kernel void synth_k{i:02}(device float* out [[buffer(0)]],\n\
             \x20                     uint tid [[thread_position_in_grid]]) {{\n\
             \x20   out[tid] = synth_work<{n}>(float(tid) * 0.001f);\n\
             }}\n\n"
        ));
    }
    src
}

#[cfg(target_os = "macos")]
pub fn run(options: &SynthBenchOptions) -> Result<SynthBenchPlan> {
    use objc2::rc::autoreleasepool;
    use objc2::runtime::{AnyObject, ProtocolObject};
    use objc2_foundation::{NSString, NSURL};
    use objc2_metal::{
        MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager, MTLCommandBuffer,
        MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice,
        MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
    };

    let plan_value = plan(options)?;

    autoreleasepool(|_pool| {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or(Error::Unsupported("no Metal device available"))?;

        let source = metal_source(&options.iterations);
        let ns_source = NSString::from_str(&source);
        let library = device
            .newLibraryWithSource_options_error(&ns_source, None)
            .map_err(|error| {
                Error::InvalidInput(format!(
                    "failed to compile synth-bench Metal source: {}",
                    error.localizedDescription()
                ))
            })?;

        let mut pipelines = Vec::with_capacity(plan_value.rows.len());
        for row in &plan_value.rows {
            let function_name = NSString::from_str(&row.function_name);
            let function = library.newFunctionWithName(&function_name).ok_or_else(|| {
                Error::InvalidInput(format!("missing kernel function {}", row.function_name))
            })?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| {
                    Error::InvalidInput(format!(
                        "failed to create pipeline for {}: {}",
                        row.function_name,
                        error.localizedDescription()
                    ))
                })?;
            pipelines.push(pipeline);
        }

        let queue = device
            .newCommandQueue()
            .ok_or(Error::Unsupported("failed to create MTLCommandQueue"))?;

        let max_invocations = plan_value
            .rows
            .iter()
            .map(|row| row.invocations as usize)
            .max()
            .unwrap_or(1);
        let buffer_len = max_invocations.max(1) * std::mem::size_of::<f32>();
        let out_buf = device
            .newBufferWithLength_options(buffer_len, MTLResourceOptions::StorageModeShared)
            .ok_or(Error::Unsupported("failed to allocate output MTLBuffer"))?;

        let capture_manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
        if !capture_manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
            return Err(Error::Unsupported(
                "MTLCaptureDestinationGPUTraceDocument is not supported (set METAL_CAPTURE_ENABLED=1)",
            ));
        }
        if options.output.exists() {
            std::fs::remove_dir_all(&options.output)?;
        }
        let output_path = options
            .output
            .to_str()
            .ok_or_else(|| Error::InvalidInput("output path is not valid UTF-8".to_owned()))?;
        let url = NSURL::fileURLWithPath(&NSString::from_str(output_path));

        let descriptor = MTLCaptureDescriptor::new();
        let device_proto: &ProtocolObject<dyn MTLDevice> = device.as_ref();
        let device_object: &AnyObject = device_proto.as_ref();
        unsafe { descriptor.setCaptureObject(Some(device_object)) };
        descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
        descriptor.setOutputURL(Some(&url));

        capture_manager
            .startCaptureWithDescriptor_error(&descriptor)
            .map_err(|error| {
                Error::InvalidInput(format!(
                    "MTLCaptureManager.startCapture failed: {}",
                    error.localizedDescription()
                ))
            })?;

        let cmd_buf = queue
            .commandBuffer()
            .ok_or(Error::Unsupported("failed to create MTLCommandBuffer"))?;
        let encoder = cmd_buf.computeCommandEncoder().ok_or(Error::Unsupported(
            "failed to create MTLComputeCommandEncoder",
        ))?;

        for (row, pipeline) in plan_value.rows.iter().zip(pipelines.iter()) {
            encoder.setComputePipelineState(pipeline);
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&out_buf), 0, 0);
            }
            let threadgroups = MTLSize {
                width: row.threadgroups as usize,
                height: 1,
                depth: 1,
            };
            let threads = MTLSize {
                width: row.threads_per_group as usize,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads);
        }
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        capture_manager.stopCapture();

        Ok(plan_value)
    })
}

#[cfg(not(target_os = "macos"))]
pub fn run(_options: &SynthBenchOptions) -> Result<SynthBenchPlan> {
    Err(Error::Unsupported("synth-bench requires macOS"))
}

pub fn format_plan(plan: &SynthBenchPlan) -> String {
    let mut out = String::new();
    out.push_str("Synthetic dispatch plan\n");
    out.push_str("idx  function    iter   tg  threads  invocations\n");
    for row in &plan.rows {
        out.push_str(&format!(
            "{idx:>3}  {name:<10} {iter:>5} {tg:>4}  {threads:>7}  {invocations:>11}\n",
            idx = row.index,
            name = row.function_name,
            iter = row.iterations,
            tg = row.threadgroups,
            threads = row.threads_per_group,
            invocations = row.invocations,
        ));
    }
    out
}

/// One kernel of the counter oracle. Each runs alone in its own compute
/// encoder, so per-encoder hardware counters are per-kernel counters, and the
/// expected behaviour is known by construction (see `docs/COUNTERS_M4.md`).
#[derive(Debug, Clone)]
pub struct CounterOracleKernel {
    pub encoder_label: &'static str,
    pub function_name: &'static str,
    /// What the kernel is designed to stress.
    pub intent: &'static str,
    pub threadgroups: u32,
    pub threads_per_group: u32,
    /// Device-memory bytes the kernel must read (lower bound: every byte is
    /// read exactly once, no reuse).
    pub device_bytes_read: u64,
    /// Device-memory bytes the kernel must write.
    pub device_bytes_written: u64,
    /// F32 FMA instructions issued per thread by the kernel's main loop.
    pub fma_per_thread: u64,
}

impl CounterOracleKernel {
    pub fn threads(&self) -> u64 {
        u64::from(self.threadgroups) * u64::from(self.threads_per_group)
    }
}

/// float4 elements per 64 MiB stream buffer.
pub const ORACLE_STREAM_FLOAT4S: u64 = 4 * 1024 * 1024;
const ORACLE_STREAM_BYTES: u64 = ORACLE_STREAM_FLOAT4S * 16;
const ORACLE_ALU_ITERS: u64 = 4096;
const ORACLE_ALU_CHAINS: u64 = 8;
const ORACLE_READ_PER_THREAD: u64 = 16;
const ORACLE_TG_ITERS: u64 = 4096;
const ORACLE_LOWOCC_ITERS: u64 = 16384;

pub fn counter_oracle_kernels() -> Vec<CounterOracleKernel> {
    vec![
        CounterOracleKernel {
            encoder_label: "oracle_alu",
            function_name: "oracle_alu",
            intent: "F32 FMA bound: 8 independent FMA chains, no device traffic beyond one store per thread",
            threadgroups: 160,
            threads_per_group: 1024,
            device_bytes_read: 0,
            device_bytes_written: 160 * 1024 * 4,
            fma_per_thread: ORACLE_ALU_ITERS * ORACLE_ALU_CHAINS,
        },
        CounterOracleKernel {
            encoder_label: "oracle_copy",
            function_name: "oracle_copy",
            intent: "DRAM bandwidth bound: copy 64 MiB to 64 MiB, one float4 per thread",
            threadgroups: (ORACLE_STREAM_FLOAT4S / 1024) as u32,
            threads_per_group: 1024,
            device_bytes_read: ORACLE_STREAM_BYTES,
            device_bytes_written: ORACLE_STREAM_BYTES,
            fma_per_thread: 0,
        },
        CounterOracleKernel {
            encoder_label: "oracle_read",
            function_name: "oracle_read",
            intent: "DRAM read bound: reduce a separate 64 MiB buffer, 16 float4 per thread, 1 float out",
            threadgroups: (ORACLE_STREAM_FLOAT4S / ORACLE_READ_PER_THREAD / 1024) as u32,
            threads_per_group: 1024,
            device_bytes_read: ORACLE_STREAM_BYTES,
            device_bytes_written: ORACLE_STREAM_FLOAT4S / ORACLE_READ_PER_THREAD * 4,
            fma_per_thread: 0,
        },
        CounterOracleKernel {
            encoder_label: "oracle_tgmem",
            function_name: "oracle_tgmem",
            intent: "threadgroup-memory bound: 4096 dependent threadgroup loads per thread, no device traffic",
            threadgroups: 160,
            threads_per_group: 1024,
            device_bytes_read: 0,
            device_bytes_written: 160 * 1024 * 4,
            fma_per_thread: ORACLE_TG_ITERS,
        },
        CounterOracleKernel {
            encoder_label: "oracle_lowocc",
            function_name: "oracle_lowocc",
            intent: "occupancy limited: one 32-thread simdgroup per threadgroup holding 32 KiB of threadgroup memory",
            threadgroups: 80,
            threads_per_group: 32,
            device_bytes_read: 0,
            device_bytes_written: 80 * 32 * 4,
            fma_per_thread: ORACLE_LOWOCC_ITERS * 2,
        },
    ]
}

pub fn counter_oracle_metal_source() -> String {
    format!(
        r#"#include <metal_stdlib>
using namespace metal;

kernel void oracle_alu(device float* out [[buffer(0)]],
                       constant float& k [[buffer(1)]],
                       uint tid [[thread_position_in_grid]]) {{
    float a0 = float(tid) * 1e-7f, a1 = a0 + 1.0f, a2 = a0 + 2.0f, a3 = a0 + 3.0f;
    float a4 = a0 + 4.0f, a5 = a0 + 5.0f, a6 = a0 + 6.0f, a7 = a0 + 7.0f;
    for (uint i = 0; i < {alu_iters}u; ++i) {{
        a0 = fma(a0, k, 1e-4f); a1 = fma(a1, k, 1e-4f);
        a2 = fma(a2, k, 1e-4f); a3 = fma(a3, k, 1e-4f);
        a4 = fma(a4, k, 1e-4f); a5 = fma(a5, k, 1e-4f);
        a6 = fma(a6, k, 1e-4f); a7 = fma(a7, k, 1e-4f);
    }}
    out[tid] = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
}}

kernel void oracle_copy(device const float4* src [[buffer(0)]],
                        device float4* dst [[buffer(1)]],
                        uint tid [[thread_position_in_grid]]) {{
    dst[tid] = src[tid];
}}

kernel void oracle_read(device const float4* src [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint threads [[threads_per_grid]]) {{
    float4 acc = 0.0f;
    for (uint i = 0; i < {read_per_thread}u; ++i) {{
        acc += src[tid + i * threads];
    }}
    out[tid] = acc.x + acc.y + acc.z + acc.w;
}}

kernel void oracle_tgmem(device float* out [[buffer(0)]],
                         uint tid [[thread_position_in_grid]],
                         uint lid [[thread_position_in_threadgroup]]) {{
    threadgroup float table[4096];
    for (uint i = lid; i < 4096u; i += 1024u) {{
        table[i] = float(i) * 1e-3f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = float(lid);
    uint idx = lid;
    for (uint i = 0; i < {tg_iters}u; ++i) {{
        float v = table[idx];
        acc = fma(acc, 0.999f, v);
        idx = (idx + uint(v) + 33u) & 4095u;
    }}
    out[tid] = acc;
}}

kernel void oracle_lowocc(device float* out [[buffer(0)]],
                          constant float& k [[buffer(1)]],
                          uint tid [[thread_position_in_grid]],
                          uint lid [[thread_position_in_threadgroup]]) {{
    threadgroup float hog[8192];
    for (uint i = lid; i < 8192u; i += 32u) {{
        hog[i] = float(i);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float a0 = hog[lid], a1 = hog[lid + 32u];
    for (uint i = 0; i < {lowocc_iters}u; ++i) {{
        a0 = fma(a0, k, 1e-4f);
        a1 = fma(a1, k, 1e-4f);
    }}
    out[tid] = a0 + a1;
}}
"#,
        alu_iters = ORACLE_ALU_ITERS,
        read_per_thread = ORACLE_READ_PER_THREAD,
        tg_iters = ORACLE_TG_ITERS,
        lowocc_iters = ORACLE_LOWOCC_ITERS,
    )
}

/// Capture the counter oracle: every kernel of [`counter_oracle_kernels`] in
/// its own labelled compute encoder of a single command buffer.
#[cfg(target_os = "macos")]
pub fn run_counter_oracle(output: &std::path::Path) -> Result<Vec<CounterOracleKernel>> {
    use objc2::rc::autoreleasepool;
    use objc2::runtime::{AnyObject, ProtocolObject};
    use objc2_foundation::{NSString, NSURL};
    use objc2_metal::{
        MTLBuffer, MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager,
        MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
        MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
    };

    let kernels = counter_oracle_kernels();
    autoreleasepool(|_pool| {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or(Error::Unsupported("no Metal device available"))?;
        let source = NSString::from_str(&counter_oracle_metal_source());
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|error| {
                Error::InvalidInput(format!(
                    "failed to compile counter-oracle Metal source: {}",
                    error.localizedDescription()
                ))
            })?;
        let mut pipelines = Vec::with_capacity(kernels.len());
        for kernel in &kernels {
            let function = library
                .newFunctionWithName(&NSString::from_str(kernel.function_name))
                .ok_or_else(|| {
                    Error::InvalidInput(format!("missing kernel {}", kernel.function_name))
                })?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| {
                    Error::InvalidInput(format!(
                        "failed to create pipeline for {}: {}",
                        kernel.function_name,
                        error.localizedDescription()
                    ))
                })?;
            pipelines.push(pipeline);
        }

        // Every buffer is filled with distinct non-zero data on the CPU before
        // capture. Zero-filled buffers are a trap: untouched pages alias the
        // zero page, reads hit in cache and the DRAM counters read ~1/30 of
        // the logical traffic.
        let alloc = |bytes: u64, seed: u32| {
            let buffer = device
                .newBufferWithLength_options(bytes as usize, MTLResourceOptions::StorageModeShared)
                .ok_or(Error::Unsupported("failed to allocate MTLBuffer"))?;
            let words = unsafe {
                std::slice::from_raw_parts_mut(
                    buffer.contents().as_ptr().cast::<u32>(),
                    bytes as usize / 4,
                )
            };
            let mut state = seed.wrapping_mul(2_654_435_761).max(1);
            for word in words {
                // xorshift32; keep the exponent bits in [2^-2, 2^1) so the
                // floats are finite and normal.
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                *word = 0x3E80_0000 | (state & 0x007F_FFFF) | ((state >> 8) & 0x0100_0000);
            }
            Ok::<_, Error>(buffer)
        };
        let copy_src = alloc(ORACLE_STREAM_BYTES, 1)?;
        let copy_dst = alloc(ORACLE_STREAM_BYTES, 2)?;
        let read_src = alloc(ORACLE_STREAM_BYTES, 3)?;
        let small_out = alloc(160 * 1024 * 4, 4)?;
        let read_out = alloc(ORACLE_STREAM_FLOAT4S / ORACLE_READ_PER_THREAD * 4, 5)?;
        let k: f32 = 0.9999;

        let queue = device
            .newCommandQueue()
            .ok_or(Error::Unsupported("failed to create MTLCommandQueue"))?;
        let capture_manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
        if !capture_manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
            return Err(Error::Unsupported(
                "MTLCaptureDestinationGPUTraceDocument is not supported (set METAL_CAPTURE_ENABLED=1)",
            ));
        }
        if output.exists() {
            std::fs::remove_dir_all(output)?;
        }
        let output_path = output
            .to_str()
            .ok_or_else(|| Error::InvalidInput("output path is not valid UTF-8".to_owned()))?;
        let descriptor = MTLCaptureDescriptor::new();
        let device_proto: &ProtocolObject<dyn MTLDevice> = device.as_ref();
        let device_object: &AnyObject = device_proto.as_ref();
        unsafe { descriptor.setCaptureObject(Some(device_object)) };
        descriptor.setDestination(MTLCaptureDestination::GPUTraceDocument);
        descriptor.setOutputURL(Some(&NSURL::fileURLWithPath(&NSString::from_str(
            output_path,
        ))));
        capture_manager
            .startCaptureWithDescriptor_error(&descriptor)
            .map_err(|error| {
                Error::InvalidInput(format!(
                    "MTLCaptureManager.startCapture failed: {}",
                    error.localizedDescription()
                ))
            })?;

        let cmd_buf = queue
            .commandBuffer()
            .ok_or(Error::Unsupported("failed to create MTLCommandBuffer"))?;
        for (kernel, pipeline) in kernels.iter().zip(pipelines.iter()) {
            let encoder = cmd_buf.computeCommandEncoder().ok_or(Error::Unsupported(
                "failed to create MTLComputeCommandEncoder",
            ))?;
            encoder.setLabel(Some(&NSString::from_str(kernel.encoder_label)));
            encoder.setComputePipelineState(pipeline);
            let set = |buffer: &ProtocolObject<dyn MTLBuffer>, index: usize| unsafe {
                encoder.setBuffer_offset_atIndex(Some(buffer), 0, index);
            };
            match kernel.function_name {
                "oracle_copy" => {
                    set(&copy_src, 0);
                    set(&copy_dst, 1);
                }
                "oracle_read" => {
                    set(&read_src, 0);
                    set(&read_out, 1);
                }
                _ => set(&small_out, 0),
            }
            if matches!(kernel.function_name, "oracle_alu" | "oracle_lowocc") {
                unsafe {
                    encoder.setBytes_length_atIndex(
                        std::ptr::NonNull::from(&k).cast(),
                        std::mem::size_of::<f32>(),
                        1,
                    );
                }
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: kernel.threadgroups as usize,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: kernel.threads_per_group as usize,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.endEncoding();
        }
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();
        capture_manager.stopCapture();
        Ok(kernels)
    })
}

#[cfg(not(target_os = "macos"))]
pub fn run_counter_oracle(_output: &std::path::Path) -> Result<Vec<CounterOracleKernel>> {
    Err(Error::Unsupported("counter oracle requires macOS"))
}

pub fn format_counter_oracle_plan(kernels: &[CounterOracleKernel]) -> String {
    let mut out = String::from(
        "Counter oracle plan (one kernel per compute encoder)\n\
         encoder        threads    dev_read_B  dev_write_B  fma/thread  intent\n",
    );
    for kernel in kernels {
        out.push_str(&format!(
            "{:<13} {:>8} {:>13} {:>12} {:>11}  {}\n",
            kernel.encoder_label,
            kernel.threads(),
            kernel.device_bytes_read,
            kernel.device_bytes_written,
            kernel.fma_per_thread,
            kernel.intent,
        ));
    }
    out
}
