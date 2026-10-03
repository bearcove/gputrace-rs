use std::collections::BTreeMap;

use serde::Serialize;

use crate::error::Result;
use crate::profiler;
use crate::trace::TraceBundle;
use crate::xcode_mio;

#[derive(Debug, Clone, Serialize)]
pub struct TimingReport {
    pub synthetic: bool,
    pub source: String,
    pub total_duration_ns: u64,
    pub command_buffer_count: usize,
    pub encoder_count: usize,
    pub dispatch_count: usize,
    pub command_buffers: Vec<CommandBufferTiming>,
    pub encoders: Vec<EncoderTiming>,
    pub kernels: Vec<KernelTiming>,
    pub agxps_pipeline_costs: Vec<AgxpsPipelineTimingCost>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgxpsPipelineTimingCost {
    pub name: String,
    pub command_count: usize,
    pub analyzer_weighted_cost: u64,
    pub analyzer_percent: f64,
    pub instruction_cost: u64,
    pub instruction_percent: Option<f64>,
    pub execution_events: u64,
    pub matched_work_cliques: usize,
    pub record_cliques: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CommandBufferTiming {
    pub index: usize,
    /// Capture-relative start time (zero-anchored at the first CB's start so
    /// dispatch-spacing math stays simple). Always 0 for the first CB.
    pub timestamp_ns: u64,
    /// Boot-relative start time in nanoseconds, derived from the GPU monotonic
    /// tick counter (APSTimelineData `start_ticks` × `timebase_numer` /
    /// `timebase_denom`). `0` when the trace doesn't carry a profiler
    /// timeline. Useful for single-CB traces where `timestamp_ns` is always 0.
    pub absolute_timestamp_ns: u64,
    pub duration_ns: Option<u64>,
    pub encoder_count: usize,
    pub dispatch_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EncoderTiming {
    pub label: String,
    pub address: u64,
    pub dispatch_count: usize,
    pub synthetic_duration_ns: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct KernelTiming {
    pub name: String,
    pub dispatch_count: usize,
    pub synthetic_duration_ns: u64,
    pub percent_of_total: f64,
    pub min_duration_ns: u64,
    pub max_duration_ns: u64,
    pub avg_duration_ns: f64,
    pub p50_duration_ns: u64,
    pub p95_duration_ns: u64,
    pub p99_duration_ns: u64,
}

fn push_kernel_duration(
    kernel_stats: &mut BTreeMap<String, KernelTiming>,
    kernel_samples: &mut BTreeMap<String, Vec<u64>>,
    name: String,
    duration_ns: u64,
) {
    let kernel = kernel_stats
        .entry(name.clone())
        .or_insert_with(|| KernelTiming {
            name: name.clone(),
            dispatch_count: 0,
            synthetic_duration_ns: 0,
            percent_of_total: 0.0,
            min_duration_ns: 0,
            max_duration_ns: 0,
            avg_duration_ns: 0.0,
            p50_duration_ns: 0,
            p95_duration_ns: 0,
            p99_duration_ns: 0,
        });
    kernel.dispatch_count += 1;
    kernel.synthetic_duration_ns = kernel.synthetic_duration_ns.saturating_add(duration_ns);
    kernel_samples.entry(name).or_default().push(duration_ns);
}

fn finalize_kernel_timings(
    kernels: &mut [KernelTiming],
    samples_by_name: &BTreeMap<String, Vec<u64>>,
    total_duration_ns: u64,
) {
    let percent_denominator_ns = kernel_percent_denominator_ns(kernels, total_duration_ns);
    for kernel in kernels {
        if percent_denominator_ns > 0 {
            kernel.percent_of_total =
                (kernel.synthetic_duration_ns as f64 / percent_denominator_ns as f64) * 100.0;
        }

        let Some(samples) = samples_by_name.get(&kernel.name) else {
            continue;
        };
        if samples.is_empty() {
            continue;
        }

        let mut sorted = samples.clone();
        sorted.sort_unstable();
        kernel.min_duration_ns = sorted[0];
        kernel.max_duration_ns = sorted[sorted.len() - 1];
        kernel.avg_duration_ns = sorted.iter().sum::<u64>() as f64 / sorted.len() as f64;
        kernel.p50_duration_ns = percentile_nearest_rank(&sorted, 50);
        kernel.p95_duration_ns = percentile_nearest_rank(&sorted, 95);
        kernel.p99_duration_ns = percentile_nearest_rank(&sorted, 99);
    }
}

fn kernel_percent_denominator_ns(kernels: &[KernelTiming], total_duration_ns: u64) -> u64 {
    kernels
        .iter()
        .fold(0u64, |sum, kernel| {
            sum.saturating_add(kernel.synthetic_duration_ns)
        })
        .max(total_duration_ns)
}

fn percentile_nearest_rank(sorted_samples: &[u64], percentile: u64) -> u64 {
    if sorted_samples.is_empty() {
        return 0;
    }
    let percentile = percentile.clamp(1, 100);
    let rank = (percentile as usize * sorted_samples.len()).div_ceil(100);
    sorted_samples[rank.saturating_sub(1).min(sorted_samples.len() - 1)]
}

pub fn report(trace: &TraceBundle) -> Result<TimingReport> {
    let profiler_summary = profiler::stream_data_summary(&trace.path).ok();
    report_with_profiler_summary(trace, profiler_summary.as_ref())
}

pub fn report_with_profiler_summary(
    trace: &TraceBundle,
    profiler_summary: Option<&profiler::ProfilerStreamDataSummary>,
) -> Result<TimingReport> {
    report_with_context(trace, profiler_summary, None)
}

pub fn report_with_context(
    trace: &TraceBundle,
    profiler_summary: Option<&profiler::ProfilerStreamDataSummary>,
    xcode_mio_report: Option<&xcode_mio::XcodeMioAnalysisReport>,
) -> Result<TimingReport> {
    let mut report = report_without_agxps(trace, profiler_summary)?;
    report.agxps_pipeline_costs = xcode_mio_report
        .map(agxps_costs_from_xcode_mio)
        .unwrap_or_default();
    Ok(report)
}

fn report_without_agxps(
    trace: &TraceBundle,
    profiler_summary: Option<&profiler::ProfilerStreamDataSummary>,
) -> Result<TimingReport> {
    if let Some(summary) = profiler_summary {
        return Ok(report_from_profiler(trace, summary));
    }
    if let Ok(raw_timings) = profiler::raw_encoder_timings(&trace.path)
        && !raw_timings.is_empty()
    {
        return report_from_raw_profiler(trace, &raw_timings);
    }

    let command_buffers = trace.command_buffers()?;
    let regions = trace.command_buffer_regions()?;

    let mut command_buffer_timings = Vec::new();
    let mut encoder_stats: BTreeMap<u64, EncoderTiming> = BTreeMap::new();
    let mut kernel_stats: BTreeMap<String, KernelTiming> = BTreeMap::new();
    let mut kernel_samples: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    let mut total_duration_ns = 0u64;
    let mut total_dispatch_count = 0usize;

    for (index, region) in regions.iter().enumerate() {
        let timestamp_ns = command_buffers
            .get(index)
            .map_or(region.command_buffer.timestamp, |cb| cb.timestamp);
        let duration_ns = command_buffers
            .get(index + 1)
            .and_then(|next| next.timestamp.checked_sub(timestamp_ns));
        if let Some(duration_ns) = duration_ns {
            total_duration_ns = total_duration_ns.saturating_add(duration_ns);
        }
        total_dispatch_count += region.dispatches.len();
        command_buffer_timings.push(CommandBufferTiming {
            index: region.command_buffer.index,
            timestamp_ns,
            absolute_timestamp_ns: 0,
            duration_ns,
            encoder_count: region.encoders.len(),
            dispatch_count: region.dispatches.len(),
        });

        let per_dispatch_duration = if let Some(duration_ns) = duration_ns {
            if region.dispatches.is_empty() {
                0
            } else {
                duration_ns / region.dispatches.len() as u64
            }
        } else {
            0
        };

        let mut encoder_dispatch_counts: BTreeMap<u64, usize> = BTreeMap::new();
        for dispatch in &region.dispatches {
            if let Some(encoder_id) = dispatch.encoder_id {
                *encoder_dispatch_counts.entry(encoder_id).or_default() += 1;
            }
            let kernel_name = dispatch
                .kernel_name
                .clone()
                .unwrap_or_else(|| "unknown".to_owned());
            push_kernel_duration(
                &mut kernel_stats,
                &mut kernel_samples,
                kernel_name,
                per_dispatch_duration,
            );
        }

        for encoder in &region.encoders {
            let entry = encoder_stats
                .entry(encoder.address)
                .or_insert_with(|| EncoderTiming {
                    label: encoder.label.clone(),
                    address: encoder.address,
                    dispatch_count: 0,
                    synthetic_duration_ns: 0,
                });
            let dispatch_count = encoder_dispatch_counts
                .get(&encoder.address)
                .copied()
                .unwrap_or(0);
            entry.dispatch_count += dispatch_count;
            entry.synthetic_duration_ns = entry
                .synthetic_duration_ns
                .saturating_add(per_dispatch_duration.saturating_mul(dispatch_count as u64));
        }
    }

    let mut encoders: Vec<_> = encoder_stats.into_values().collect();
    encoders.sort_by(|left, right| {
        right
            .synthetic_duration_ns
            .cmp(&left.synthetic_duration_ns)
            .then_with(|| left.label.cmp(&right.label))
    });

    let mut kernels: Vec<_> = kernel_stats.into_values().collect();
    finalize_kernel_timings(&mut kernels, &kernel_samples, total_duration_ns);
    kernels.sort_by(|left, right| {
        right
            .synthetic_duration_ns
            .cmp(&left.synthetic_duration_ns)
            .then_with(|| left.name.cmp(&right.name))
    });

    Ok(TimingReport {
        synthetic: true,
        source: "synthetic".to_owned(),
        total_duration_ns,
        command_buffer_count: command_buffer_timings.len(),
        encoder_count: encoders.len(),
        dispatch_count: total_dispatch_count,
        command_buffers: command_buffer_timings,
        encoders,
        kernels,
        agxps_pipeline_costs: Vec::new(),
    })
}

fn agxps_costs_from_xcode_mio(
    report: &xcode_mio::XcodeMioAnalysisReport,
) -> Vec<AgxpsPipelineTimingCost> {
    let mut rows = report
        .top_pipelines
        .iter()
        .filter(|pipeline| pipeline.agxps_analyzer_cost > 0)
        .map(|pipeline| AgxpsPipelineTimingCost {
            name: pipeline
                .function_name
                .clone()
                .unwrap_or_else(|| "<unknown function>".to_owned()),
            command_count: pipeline.command_count,
            analyzer_weighted_cost: pipeline.agxps_analyzer_cost,
            analyzer_percent: pipeline.agxps_analyzer_cost_percent.unwrap_or(0.0),
            instruction_cost: pipeline.agxps_trace_cost,
            instruction_percent: pipeline.agxps_trace_cost_percent,
            execution_events: pipeline.agxps_trace_events,
            matched_work_cliques: pipeline.agxps_trace_matched_work_cliques,
            record_cliques: pipeline.agxps_analyzer_record_cliques,
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .analyzer_weighted_cost
            .cmp(&left.analyzer_weighted_cost)
            .then_with(|| left.name.cmp(&right.name))
    });
    rows
}

fn report_from_profiler(
    trace: &TraceBundle,
    summary: &profiler::ProfilerStreamDataSummary,
) -> TimingReport {
    let command_buffers = trace.command_buffers().unwrap_or_default();
    let command_buffer_regions = trace.command_buffer_regions().unwrap_or_default();
    let profiler_command_buffers = summary
        .timeline
        .as_ref()
        .map(command_buffer_timings_from_timeline)
        .unwrap_or_default();

    // streamData typically attributes far more dispatches than the MTLB byte-scan
    // surfaces. With only one CB we can confidently distribute streamData's count
    // to it; with multiple CBs we fall back to the per-region count to avoid
    // double-counting.
    let region_dispatches: usize = command_buffer_regions
        .iter()
        .map(|region| region.dispatches.len())
        .sum();
    let single_cb_streamdata_total =
        if command_buffers.len() == 1 && summary.dispatches.len() > region_dispatches {
            Some(summary.dispatches.len())
        } else {
            None
        };

    let command_buffer_timings = command_buffers
        .iter()
        .enumerate()
        .map(|(index, cb)| CommandBufferTiming {
            index: cb.index,
            timestamp_ns: profiler_command_buffers
                .get(index)
                .map(|profiler| profiler.timestamp_ns)
                .unwrap_or(cb.timestamp),
            absolute_timestamp_ns: profiler_command_buffers
                .get(index)
                .map(|profiler| profiler.absolute_timestamp_ns)
                .unwrap_or(0),
            duration_ns: profiler_command_buffers
                .get(index)
                .and_then(|profiler| profiler.duration_ns)
                .or_else(|| {
                    command_buffers
                        .get(index + 1)
                        .and_then(|next| next.timestamp.checked_sub(cb.timestamp))
                }),
            encoder_count: command_buffer_regions
                .get(index)
                .map(|region| region.encoders.len())
                .unwrap_or(0),
            dispatch_count: single_cb_streamdata_total.unwrap_or_else(|| {
                command_buffer_regions
                    .get(index)
                    .map(|region| region.dispatches.len())
                    .unwrap_or(0)
            }),
        })
        .collect::<Vec<_>>();

    let encoders = summary
        .encoder_timings
        .iter()
        .map(|encoder| EncoderTiming {
            label: format!("encoder {}", encoder.index),
            address: encoder.index as u64,
            dispatch_count: summary
                .dispatches
                .iter()
                .filter(|dispatch| dispatch.encoder_index == encoder.index)
                .count(),
            synthetic_duration_ns: encoder.duration_micros.saturating_mul(1_000),
        })
        .collect::<Vec<_>>();

    let mut kernel_stats = BTreeMap::<String, KernelTiming>::new();
    let mut kernel_samples = BTreeMap::<String, Vec<u64>>::new();
    for dispatch in &summary.dispatches {
        let name = dispatch
            .function_name
            .clone()
            .unwrap_or_else(|| format!("pipeline_{}", dispatch.pipeline_index));
        push_kernel_duration(
            &mut kernel_stats,
            &mut kernel_samples,
            name,
            dispatch.duration_us.saturating_mul(1_000),
        );
    }

    let total_duration_ns = summary.total_time_us.saturating_mul(1_000);
    let mut kernels = kernel_stats.into_values().collect::<Vec<_>>();
    finalize_kernel_timings(&mut kernels, &kernel_samples, total_duration_ns);
    kernels.sort_by(|left, right| {
        right
            .synthetic_duration_ns
            .cmp(&left.synthetic_duration_ns)
            .then_with(|| left.name.cmp(&right.name))
    });

    let mut encoders = encoders;
    encoders.sort_by(|left, right| {
        right
            .synthetic_duration_ns
            .cmp(&left.synthetic_duration_ns)
            .then_with(|| left.label.cmp(&right.label))
    });

    TimingReport {
        synthetic: false,
        source: "streamData".to_owned(),
        total_duration_ns,
        command_buffer_count: command_buffer_timings.len(),
        encoder_count: encoders.len(),
        dispatch_count: summary.dispatches.len(),
        command_buffers: command_buffer_timings,
        encoders,
        kernels,
        agxps_pipeline_costs: Vec::new(),
    }
}

fn report_from_raw_profiler(
    trace: &TraceBundle,
    raw_timings: &[profiler::ProfilerRawEncoderTiming],
) -> Result<TimingReport> {
    let regions = trace.command_buffer_regions()?;
    let encoders = trace.compute_encoders()?;

    let mut encoder_rows = Vec::new();
    let mut total_duration_ns = 0u64;
    for timing in raw_timings {
        total_duration_ns = total_duration_ns.saturating_add(timing.duration_ns);
        let encoder = encoders
            .iter()
            .find(|encoder| encoder.index == timing.index);
        let dispatch_count = regions
            .iter()
            .flat_map(|region| region.dispatches.iter())
            .filter(|dispatch| dispatch.encoder_id == encoder.map(|encoder| encoder.address))
            .count();
        encoder_rows.push(EncoderTiming {
            label: encoder
                .map(|encoder| encoder.label.clone())
                .unwrap_or_else(|| format!("encoder {}", timing.index)),
            address: encoder
                .map(|encoder| encoder.address)
                .unwrap_or(timing.index as u64),
            dispatch_count,
            synthetic_duration_ns: timing.duration_ns,
        });
    }

    let mut kernel_stats = BTreeMap::<String, KernelTiming>::new();
    let mut kernel_samples = BTreeMap::<String, Vec<u64>>::new();
    let encoder_duration_by_addr = encoder_rows
        .iter()
        .map(|encoder| (encoder.address, encoder.synthetic_duration_ns))
        .collect::<BTreeMap<_, _>>();
    for region in &regions {
        let mut dispatches_by_encoder = BTreeMap::<u64, usize>::new();
        for dispatch in &region.dispatches {
            if let Some(encoder_id) = dispatch.encoder_id {
                *dispatches_by_encoder.entry(encoder_id).or_default() += 1;
            }
        }
        for dispatch in &region.dispatches {
            let kernel_name = dispatch
                .kernel_name
                .clone()
                .unwrap_or_else(|| "unknown".to_owned());
            let per_dispatch_duration = dispatch
                .encoder_id
                .and_then(|encoder_id| encoder_duration_by_addr.get(&encoder_id).copied())
                .map(|duration| {
                    duration
                        / dispatches_by_encoder
                            .get(&dispatch.encoder_id.unwrap_or_default())
                            .copied()
                            .unwrap_or(1) as u64
                })
                .unwrap_or(0);
            push_kernel_duration(
                &mut kernel_stats,
                &mut kernel_samples,
                kernel_name,
                per_dispatch_duration,
            );
        }
    }

    let mut kernels: Vec<_> = kernel_stats.into_values().collect();
    finalize_kernel_timings(&mut kernels, &kernel_samples, total_duration_ns);
    kernels.sort_by(|left, right| {
        right
            .synthetic_duration_ns
            .cmp(&left.synthetic_duration_ns)
            .then_with(|| left.name.cmp(&right.name))
    });

    let command_buffers = trace.command_buffers()?;
    let command_buffer_rows = command_buffers
        .iter()
        .enumerate()
        .map(|(index, cb)| CommandBufferTiming {
            index: cb.index,
            timestamp_ns: cb.timestamp,
            absolute_timestamp_ns: 0,
            duration_ns: command_buffers
                .get(index + 1)
                .and_then(|next| next.timestamp.checked_sub(cb.timestamp)),
            encoder_count: regions
                .get(index)
                .map(|region| region.encoders.len())
                .unwrap_or(0),
            dispatch_count: regions
                .get(index)
                .map(|region| region.dispatches.len())
                .unwrap_or(0),
        })
        .collect::<Vec<_>>();

    encoder_rows.sort_by(|left, right| {
        right
            .synthetic_duration_ns
            .cmp(&left.synthetic_duration_ns)
            .then_with(|| left.label.cmp(&right.label))
    });

    Ok(TimingReport {
        synthetic: false,
        source: "raw-profiler-heuristic".to_owned(),
        total_duration_ns,
        command_buffer_count: command_buffer_rows.len(),
        encoder_count: encoder_rows.len(),
        dispatch_count: regions.iter().map(|region| region.dispatches.len()).sum(),
        command_buffers: command_buffer_rows,
        encoders: encoder_rows,
        kernels,
        agxps_pipeline_costs: Vec::new(),
    })
}

fn command_buffer_timings_from_timeline(
    timeline: &profiler::ProfilerTimelineInfo,
) -> Vec<CommandBufferTiming> {
    let first_start = timeline
        .command_buffer_timestamps
        .first()
        .map(|entry| entry.start_ticks)
        .unwrap_or_default();

    timeline
        .command_buffer_timestamps
        .iter()
        .map(|entry| CommandBufferTiming {
            index: entry.index,
            timestamp_ns: ticks_to_ns(
                entry.start_ticks.saturating_sub(first_start),
                timeline.timebase_numer,
                timeline.timebase_denom,
            ),
            absolute_timestamp_ns: ticks_to_ns(
                entry.start_ticks,
                timeline.timebase_numer,
                timeline.timebase_denom,
            ),
            duration_ns: Some(ticks_to_ns(
                entry.end_ticks.saturating_sub(entry.start_ticks),
                timeline.timebase_numer,
                timeline.timebase_denom,
            )),
            encoder_count: 0,
            dispatch_count: 0,
        })
        .collect()
}

fn ticks_to_ns(ticks: u64, numer: u64, denom: u64) -> u64 {
    ticks.saturating_mul(numer.max(1)) / denom.max(1)
}

pub fn format_report(report: &TimingReport) -> String {
    let mut out = String::new();
    let duration_label = if report.synthetic {
        "Synthetic ns"
    } else {
        "Duration ns"
    };
    if report.synthetic {
        out.push_str("Synthetic timing report\n");
        out.push_str("Derived from command-buffer timestamps and dispatch attribution. No profiler bundle was available, so per-dispatch durations are computed by dividing the command-buffer wall time evenly across dispatches — kernel percentages here are weighted by *dispatch count*, not by actual GPU time spent in each kernel. Capture with the GPU profiler enabled to get streamData-backed per-dispatch durations.\n\n");
    } else if report.source == "streamData" {
        out.push_str("Profiler-backed timing report\n");
        out.push_str(
            "Kernel, dispatch, and encoder timing come from streamData; command-buffer rows prefer APSTimelineData spans when present.\n",
        );
        if report.agxps_pipeline_costs.is_empty() {
            out.push_str(
                "Caveat: streamData's `cumulative_us` is the *dispatch-issue cadence* (the CPU-side clock that ticks each time a dispatch is enqueued), not actual GPU compute time. Apple's profiler advances it by a fixed ~µs per dispatch regardless of how long the dispatch actually runs on the GPU. So the kernel %% column below reflects dispatch *count* weighted by issue cadence, not real cost. No AGXPS analyzer-weighted pipeline costs were available for this run.\n\n",
            );
        } else {
            out.push_str(
                "AGXPS pipeline candidates below come from Xcode's private timing analyzer over Profiling_f_*.raw. `Ana %` is analyzer-weighted clique duration; `W1 %` is the instruction-stats word1 aggregate. Neither is exact Xcode UI parity on the validated non-synthetic trace yet. The streamData kernel table remains dispatch-cadence timing, not real GPU cost.\n\n",
            );
        }
        if kernel_percent_denominator_ns(&report.kernels, report.total_duration_ns)
            > report.total_duration_ns
        {
            out.push_str(
                "Kernel percentages are shares of summed profiler dispatch time because attributed dispatch durations exceed wall-clock GPU time.\n\n",
            );
        }
    } else {
        out.push_str("Raw-profiler timing report\n");
        out.push_str(
            "Encoder timing comes from Counters_f_* heuristic aggregation; command-buffer rows still come from trace timestamps when available.\n\n",
        );
    }
    out.push_str(&format!(
        "total={} ns, command_buffers={}, encoders={}, dispatches={}\n\n",
        report.total_duration_ns,
        report.command_buffer_count,
        report.encoder_count,
        report.dispatch_count
    ));
    if !report.agxps_pipeline_costs.is_empty() {
        out.push_str("AGXPS pipeline cost candidates:\n");
        out.push_str(&format!(
            "{:<42} {:>5} {:>9} {:>9} {:>14} {:>10} {:>10}\n",
            "Name", "Cmds", "Ana %", "W1 %", "Weighted Cost", "Events", "Cliques"
        ));
        for row in report.agxps_pipeline_costs.iter().take(20) {
            out.push_str(&format!(
                "{:<42} {:>5} {:>8.3}% {:>8} {:>14} {:>10} {:>10}\n",
                truncate(&row.name, 42),
                row.command_count,
                row.analyzer_percent,
                format_optional_percent(row.instruction_percent),
                row.analyzer_weighted_cost,
                row.execution_events,
                row.record_cliques,
            ));
        }
        out.push('\n');
    }
    if !report.kernels.is_empty() {
        out.push_str("Kernels:\n");
        out.push_str(&format!(
            "{:<30} {:>10} {:>14} {:>12} {:>12} {:>12} {:>8}\n",
            "Name", "Dispatches", duration_label, "Avg ns", "P95 ns", "P99 ns", "%"
        ));
        for kernel in report.kernels.iter().take(20) {
            out.push_str(&format!(
                "{:<30} {:>10} {:>14} {:>12.0} {:>12} {:>12} {:>7.2}\n",
                truncate(&kernel.name, 30),
                kernel.dispatch_count,
                kernel.synthetic_duration_ns,
                kernel.avg_duration_ns,
                kernel.p95_duration_ns,
                kernel.p99_duration_ns,
                kernel.percent_of_total
            ));
        }
        out.push('\n');
    }
    if !report.encoders.is_empty() {
        out.push_str("Encoders:\n");
        out.push_str(&format!(
            "{:<32} {:>10} {:>16}\n",
            "Label", "Dispatches", duration_label
        ));
        for encoder in report.encoders.iter().take(20) {
            let label = if encoder.label.is_empty() {
                format!("0x{:x}", encoder.address)
            } else {
                encoder.label.clone()
            };
            out.push_str(&format!(
                "{:<32} {:>10} {:>16}\n",
                truncate(&label, 32),
                encoder.dispatch_count,
                encoder.synthetic_duration_ns
            ));
        }
        out.push('\n');
    }
    if !report.command_buffers.is_empty() {
        out.push_str("Command buffers:\n");
        for cb in &report.command_buffers {
            // Prefer the absolute (boot-relative) timestamp when the profiler
            // decoded one — it's meaningful even for single-CB traces where
            // the capture-relative `timestamp_ns` is always 0.
            let ts = if cb.absolute_timestamp_ns != 0 {
                cb.absolute_timestamp_ns
            } else {
                cb.timestamp_ns
            };
            out.push_str(&format!(
                "  CB {}: ts={} ns duration={} encoders={} dispatches={}\n",
                cb.index,
                ts,
                cb.duration_ns
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "?".to_owned()),
                cb.encoder_count,
                cb.dispatch_count
            ));
        }
    }
    out
}

pub fn format_csv(report: &TimingReport) -> String {
    let mut out = String::new();
    out.push_str("kind,name,dispatch_count,synthetic_duration_ns,percent_of_total,min_duration_ns,max_duration_ns,avg_duration_ns,p50_duration_ns,p95_duration_ns,p99_duration_ns\n");
    for kernel in &report.kernels {
        out.push_str(&format!(
            "kernel,{},{},{},{},{},{},{},{},{},{}\n",
            escape_csv(&kernel.name),
            kernel.dispatch_count,
            kernel.synthetic_duration_ns,
            kernel.percent_of_total,
            kernel.min_duration_ns,
            kernel.max_duration_ns,
            kernel.avg_duration_ns,
            kernel.p50_duration_ns,
            kernel.p95_duration_ns,
            kernel.p99_duration_ns
        ));
    }
    for encoder in &report.encoders {
        let label = if encoder.label.is_empty() {
            format!("0x{:x}", encoder.address)
        } else {
            encoder.label.clone()
        };
        out.push_str(&format!(
            "encoder,{},{},{},,,,,,,\n",
            escape_csv(&label),
            encoder.dispatch_count,
            encoder.synthetic_duration_ns
        ));
    }
    out
}

fn truncate(value: &str, width: usize) -> String {
    if value.len() <= width {
        return value.to_owned();
    }
    let keep = width.saturating_sub(3);
    format!("{}...", &value[..keep])
}

fn format_optional_percent(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.3}%"))
        .unwrap_or_else(|| "-".to_owned())
}

fn escape_csv(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_csv() {
        let report = TimingReport {
            synthetic: true,
            source: "synthetic".into(),
            total_duration_ns: 100,
            command_buffer_count: 1,
            encoder_count: 1,
            dispatch_count: 2,
            command_buffers: vec![],
            encoders: vec![EncoderTiming {
                label: "enc".into(),
                address: 1,
                dispatch_count: 2,
                synthetic_duration_ns: 100,
            }],
            kernels: vec![KernelTiming {
                name: "kernel".into(),
                dispatch_count: 2,
                synthetic_duration_ns: 100,
                percent_of_total: 100.0,
                min_duration_ns: 40,
                max_duration_ns: 60,
                avg_duration_ns: 50.0,
                p50_duration_ns: 40,
                p95_duration_ns: 60,
                p99_duration_ns: 60,
            }],
            agxps_pipeline_costs: Vec::new(),
        };
        let csv = format_csv(&report);
        assert!(csv.contains("p50_duration_ns"));
        assert!(csv.contains("kernel,kernel,2,100,100,40,60,50,40,60,60"));
        assert!(csv.contains("encoder,enc,2,100"));
    }

    #[test]
    fn computes_nearest_rank_percentiles() {
        assert_eq!(percentile_nearest_rank(&[10, 20, 30, 40], 50), 20);
        assert_eq!(percentile_nearest_rank(&[10, 20, 30, 40], 95), 40);
        assert_eq!(percentile_nearest_rank(&[10], 99), 10);
    }

    #[test]
    fn kernel_percent_denominator_uses_summed_dispatch_time_when_larger() {
        let mut kernels = vec![
            KernelTiming {
                name: "a".into(),
                dispatch_count: 1,
                synthetic_duration_ns: 1_500,
                percent_of_total: 0.0,
                min_duration_ns: 0,
                max_duration_ns: 0,
                avg_duration_ns: 0.0,
                p50_duration_ns: 0,
                p95_duration_ns: 0,
                p99_duration_ns: 0,
            },
            KernelTiming {
                name: "b".into(),
                dispatch_count: 1,
                synthetic_duration_ns: 500,
                percent_of_total: 0.0,
                min_duration_ns: 0,
                max_duration_ns: 0,
                avg_duration_ns: 0.0,
                p50_duration_ns: 0,
                p95_duration_ns: 0,
                p99_duration_ns: 0,
            },
        ];

        finalize_kernel_timings(&mut kernels, &BTreeMap::new(), 1_000);

        assert_eq!(kernels[0].percent_of_total, 75.0);
        assert_eq!(kernels[1].percent_of_total, 25.0);
    }

    #[test]
    fn formats_profiler_backed_report() {
        let report = TimingReport {
            synthetic: false,
            source: "streamData".into(),
            total_duration_ns: 1_500,
            command_buffer_count: 1,
            encoder_count: 1,
            dispatch_count: 2,
            command_buffers: vec![CommandBufferTiming {
                index: 0,
                timestamp_ns: 0,
                absolute_timestamp_ns: 0,
                duration_ns: Some(1_500),
                encoder_count: 1,
                dispatch_count: 2,
            }],
            encoders: vec![EncoderTiming {
                label: "enc".into(),
                address: 1,
                dispatch_count: 2,
                synthetic_duration_ns: 1_500,
            }],
            kernels: vec![KernelTiming {
                name: "kernel".into(),
                dispatch_count: 2,
                synthetic_duration_ns: 1_500,
                percent_of_total: 100.0,
                min_duration_ns: 500,
                max_duration_ns: 1_000,
                avg_duration_ns: 750.0,
                p50_duration_ns: 500,
                p95_duration_ns: 1_000,
                p99_duration_ns: 1_000,
            }],
            agxps_pipeline_costs: Vec::new(),
        };

        let text = format_report(&report);
        assert!(text.contains("Profiler-backed timing report"));
        assert!(text.contains("APSTimelineData"));
        assert!(text.contains("Duration ns"));
        assert!(text.contains("P95 ns"));
    }

    #[test]
    fn formats_agxps_pipeline_costs() {
        let report = TimingReport {
            synthetic: false,
            source: "streamData".into(),
            total_duration_ns: 1_500,
            command_buffer_count: 1,
            encoder_count: 0,
            dispatch_count: 2,
            command_buffers: Vec::new(),
            encoders: Vec::new(),
            kernels: Vec::new(),
            agxps_pipeline_costs: vec![AgxpsPipelineTimingCost {
                name: "hot_kernel".into(),
                command_count: 2,
                analyzer_weighted_cost: 90,
                analyzer_percent: 90.0,
                instruction_cost: 50,
                instruction_percent: Some(50.0),
                execution_events: 123,
                matched_work_cliques: 7,
                record_cliques: 11,
            }],
        };

        let text = format_report(&report);
        assert!(text.contains("AGXPS pipeline cost candidates"));
        assert!(text.contains("hot_kernel"));
        assert!(text.contains("90.000%"));
    }

    #[test]
    fn formats_raw_profiler_report() {
        let report = TimingReport {
            synthetic: false,
            source: "raw-profiler-heuristic".into(),
            total_duration_ns: 900,
            command_buffer_count: 1,
            encoder_count: 1,
            dispatch_count: 2,
            command_buffers: vec![CommandBufferTiming {
                index: 0,
                timestamp_ns: 0,
                absolute_timestamp_ns: 0,
                duration_ns: Some(900),
                encoder_count: 1,
                dispatch_count: 2,
            }],
            encoders: vec![EncoderTiming {
                label: "enc".into(),
                address: 1,
                dispatch_count: 2,
                synthetic_duration_ns: 900,
            }],
            kernels: vec![KernelTiming {
                name: "kernel".into(),
                dispatch_count: 2,
                synthetic_duration_ns: 900,
                percent_of_total: 100.0,
                min_duration_ns: 450,
                max_duration_ns: 450,
                avg_duration_ns: 450.0,
                p50_duration_ns: 450,
                p95_duration_ns: 450,
                p99_duration_ns: 450,
            }],
            agxps_pipeline_costs: Vec::new(),
        };

        let text = format_report(&report);
        assert!(text.contains("Raw-profiler timing report"));
        assert!(text.contains("Counters_f_* heuristic"));
    }
}
