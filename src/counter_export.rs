use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::Serialize;

use crate::counter;
use crate::error::Result;
use crate::profiler;
use crate::timeline;
use crate::trace::TraceBundle;
#[cfg(test)]
use crate::xcode_counters;

#[derive(Debug, Clone, Serialize)]
pub struct CounterExportReport {
    pub trace_source: PathBuf,
    pub source: String,
    pub total_rows: usize,
    pub rows: Vec<CounterExportRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CounterExportRow {
    pub row_index: usize,
    pub command_buffer_index: usize,
    pub encoder_index: usize,
    pub encoder_label: String,
    pub kernel_name: Option<String>,
    pub pipeline_addr: Option<u64>,
    pub start_time_ns: u64,
    pub end_time_ns: u64,
    pub duration_ns: Option<u64>,
    pub dispatch_count: usize,
    pub kernel_invocations: usize,
    pub metric_source: String,
    pub execution_cost_percent: Option<f64>,
    pub execution_cost_samples: usize,
    pub sample_count: usize,
    pub avg_sampling_density: Option<f64>,
    pub occupancy_percent: Option<f64>,
    pub alu_utilization_percent: Option<f64>,
    pub shader_launch_limiter_percent: Option<f64>,
    pub instruction_throughput_percent: Option<f64>,
    pub integer_complex_percent: Option<f64>,
    pub f32_limiter_percent: Option<f64>,
    pub l1_cache_percent: Option<f64>,
    pub last_level_cache_percent: Option<f64>,
    pub control_flow_percent: Option<f64>,
    pub device_memory_bandwidth_gbps: Option<f64>,
    pub gpu_read_bandwidth_gbps: Option<f64>,
    pub gpu_write_bandwidth_gbps: Option<f64>,
    pub buffer_device_memory_bytes_read: Option<f64>,
    pub buffer_device_memory_bytes_written: Option<f64>,
    pub bytes_read_from_device_memory: Option<f64>,
    pub bytes_written_to_device_memory: Option<f64>,
    pub buffer_l1_miss_rate_percent: Option<f64>,
    pub buffer_l1_read_accesses: Option<f64>,
    pub buffer_l1_read_bandwidth_gbps: Option<f64>,
    pub buffer_l1_write_accesses: Option<f64>,
    pub buffer_l1_write_bandwidth_gbps: Option<f64>,
    pub compute_shader_launch_utilization_percent: Option<f64>,
    pub control_flow_utilization_percent: Option<f64>,
    pub instruction_throughput_utilization_percent: Option<f64>,
    pub integer_complex_utilization_percent: Option<f64>,
    pub integer_conditional_utilization_percent: Option<f64>,
    pub f32_utilization_percent: Option<f64>,
    pub temporary_register_count: Option<i64>,
    pub uniform_register_count: Option<i64>,
    pub spilled_bytes: Option<i64>,
    pub threadgroup_memory: Option<i64>,
    pub instruction_count: Option<i64>,
    pub alu_instruction_count: Option<i64>,
    pub branch_instruction_count: Option<i64>,
    pub compilation_time_ms: Option<f64>,
    pub metrics: BTreeMap<String, f64>,
    pub metric_metadata: BTreeMap<String, CounterExportMetricMetadata>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CounterExportMetricMetadata {
    pub key: String,
    pub counter_type: Option<String>,
    pub description: Option<String>,
    pub unit: Option<String>,
    pub groups: Vec<String>,
    pub timeline_groups: Vec<String>,
    pub visible: Option<bool>,
    pub batch_filtered: Option<bool>,
}

pub fn report(trace: &TraceBundle) -> Result<CounterExportReport> {
    let profiler_summary = profiler::stream_data_summary(&trace.path).ok();
    report_with_context(trace, profiler_summary.as_ref(), None)
}

pub fn report_with_context(
    trace: &TraceBundle,
    profiler_summary: Option<&profiler::ProfilerStreamDataSummary>,
    precomputed_raw_counter_report: Option<&counter::RawCountersReport>,
) -> Result<CounterExportReport> {
    let timeline = match timeline::report_with_profiler_summary(trace, profiler_summary) {
        Ok(timeline) => timeline,
        Err(_) => empty_timeline_report(
            trace,
            if profiler_summary.is_some() {
                "streamData"
            } else {
                "timeline-unavailable"
            },
        ),
    };
    let limiters = counter::extract_limiters_for_trace(&trace.path);
    let raw_counter_report_owned;
    let raw_counter_report = if let Some(report) = precomputed_raw_counter_report {
        Some(report)
    } else {
        raw_counter_report_owned = counter::raw_counters_report(trace).ok();
        raw_counter_report_owned.as_ref()
    };
    let mut aps_rows = raw_counter_report
        .and_then(|raw_counter_report| {
            aps_counter_rows(trace, &timeline, profiler_summary, raw_counter_report)
        })
        .unwrap_or_default();
    let has_profiler_dispatches = profiler_summary
        .as_ref()
        .is_some_and(|summary| !summary.dispatches.is_empty());
    let suppress_legacy_limiter_rows = has_profiler_dispatches || !aps_rows.is_empty();

    let limiters_by_encoder = limiters
        .into_iter()
        .map(|limiter| (limiter.encoder_index, limiter))
        .collect::<BTreeMap<_, _>>();
    let occupancy_by_encoder = limiters_by_encoder
        .iter()
        .filter_map(|(encoder, limiter)| Some((*encoder, limiter.occupancy?)))
        .collect::<BTreeMap<_, _>>();

    let mut execution_cost_by_name = BTreeMap::<String, (f64, usize)>::new();
    let mut sample_stats_by_name = BTreeMap::<String, (usize, f64, usize)>::new();
    let mut pipeline_stats_by_name = BTreeMap::<String, profiler::ProfilerPipelineStats>::new();
    let mut profiler_dispatches_by_name =
        BTreeMap::<String, Vec<&profiler::ProfilerDispatch>>::new();
    if let Some(summary) = profiler_summary {
        for cost in &summary.execution_costs {
            let name = cost
                .function_name
                .clone()
                .unwrap_or_else(|| format!("pipeline_{}", cost.pipeline_id));
            let entry = execution_cost_by_name.entry(name).or_default();
            entry.0 += cost.cost_percent;
            entry.1 += cost.sample_count;
        }
        for dispatch in &summary.dispatches {
            let name = dispatch
                .function_name
                .clone()
                .unwrap_or_else(|| format!("pipeline_{}", dispatch.pipeline_index));
            let entry = sample_stats_by_name.entry(name.clone()).or_default();
            entry.0 += dispatch.sample_count;
            entry.1 += dispatch.sampling_density;
            entry.2 += 1;
            profiler_dispatches_by_name
                .entry(name)
                .or_default()
                .push(dispatch);
        }
        for pipeline in &summary.pipelines {
            if let (Some(name), Some(stats)) = (&pipeline.function_name, &pipeline.stats) {
                pipeline_stats_by_name
                    .entry(name.clone())
                    .or_insert_with(|| stats.clone());
            }
        }
    }

    let mut dispatches_by_encoder = BTreeMap::<usize, Vec<&timeline::TimelineDispatch>>::new();
    for dispatch in &timeline.dispatches {
        if let Some(encoder_index) = dispatch.encoder_index {
            dispatches_by_encoder
                .entry(encoder_index)
                .or_default()
                .push(dispatch);
        }
    }

    let mut rows = Vec::new();
    let mut represented_kernels = BTreeSet::new();
    for encoder in &timeline.encoders {
        let encoder_dispatches = dispatches_by_encoder
            .get(&encoder.index)
            .cloned()
            .unwrap_or_default();
        let mut kernels = BTreeMap::<String, usize>::new();
        let mut pipeline_addr = None;
        for dispatch in &encoder_dispatches {
            if let Some(kernel_name) = &dispatch.kernel_name {
                *kernels.entry(kernel_name.clone()).or_default() += 1;
            }
            if pipeline_addr.is_none() {
                pipeline_addr = dispatch.encoder_address;
            }
        }
        let kernel_name = kernels
            .into_iter()
            .max_by(|left, right| left.1.cmp(&right.1).then_with(|| left.0.cmp(&right.0)))
            .map(|(name, _)| name);
        let (execution_cost_percent, execution_cost_samples) = kernel_name
            .as_ref()
            .and_then(|name| execution_cost_by_name.get(name).copied())
            .map(|(percent, samples)| (Some(percent), samples))
            .unwrap_or((None, 0));
        if execution_cost_percent.is_some()
            && let Some(kernel_name) = &kernel_name
        {
            represented_kernels.insert(kernel_name.clone());
        }
        let (sample_count, avg_sampling_density) = kernel_name
            .as_ref()
            .and_then(|name| sample_stats_by_name.get(name).copied())
            .map(|(samples, density_sum, density_count)| {
                let avg = if density_count == 0 {
                    None
                } else {
                    Some(density_sum / density_count as f64)
                };
                (samples, avg)
            })
            .unwrap_or((0, None));
        let pipeline_stats = kernel_name
            .as_ref()
            .and_then(|name| pipeline_stats_by_name.get(name));
        let limiter = limiters_by_encoder.get(&encoder.index);
        let row_metrics = BTreeMap::<String, f64>::new();
        let row_metric = |name: &str| row_metrics.get(name).copied();
        let kernel_invocations = row_metric("Kernel Invocations")
            .map(|value| value.round().max(0.0) as usize)
            .unwrap_or(encoder.dispatch_count);
        let occupancy_percent =
            row_metric("Kernel Occupancy").or_else(|| limiter.and_then(|limiter| limiter.occupancy));
        let alu_utilization_percent = row_metric("ALU Utilization")
            .or_else(|| limiter.and_then(|limiter| limiter.alu_utilization));
        let device_memory_bandwidth_gbps = row_metric("Device Memory Bandwidth")
            .or_else(|| limiter.and_then(|limiter| limiter.device_memory_bandwidth_gbps));
        let buffer_l1_read_bandwidth_gbps =
            row_metric("Buffer L1 Read Bandwidth").or_else(|| row_metric("L1 Read Bandwidth"));
        let buffer_l1_write_bandwidth_gbps =
            row_metric("Buffer L1 Write Bandwidth").or_else(|| row_metric("L1 Write Bandwidth"));
        let buffer_l1_read_bandwidth_gbps = buffer_l1_read_bandwidth_gbps
            .or_else(|| limiter.and_then(|limiter| limiter.buffer_l1_read_bandwidth_gbps));
        let buffer_l1_write_bandwidth_gbps = buffer_l1_write_bandwidth_gbps
            .or_else(|| limiter.and_then(|limiter| limiter.buffer_l1_write_bandwidth_gbps));
        let gpu_read_bandwidth_gbps = row_metric("GPU Read Bandwidth");
        let gpu_write_bandwidth_gbps = row_metric("GPU Write Bandwidth");
        let buffer_device_memory_bytes_read = row_metric("Buffer Device Memory Bytes Read");
        let buffer_device_memory_bytes_written = row_metric("Buffer Device Memory Bytes Written");
        let bytes_read_from_device_memory = row_metric("Bytes Read From Device Memory");
        let bytes_written_to_device_memory = row_metric("Bytes Written To Device Memory");
        let buffer_l1_miss_rate_percent = row_metric("Buffer L1 Miss Rate");
        let buffer_l1_read_accesses = row_metric("Buffer L1 Read Accesses");
        let buffer_l1_write_accesses = row_metric("Buffer L1 Write Accesses");
        let compute_shader_launch_utilization_percent =
            row_metric("Compute Shader Launch Utilization");
        let control_flow_utilization_percent = row_metric("Control Flow Utilization");
        let instruction_throughput_utilization_percent =
            row_metric("Instruction Throughput Utilization");
        let integer_complex_utilization_percent = row_metric("Integer and Complex Utilization");
        let integer_conditional_utilization_percent =
            row_metric("Integer and Conditional Utilization");
        let f32_utilization_percent = row_metric("F32 Utilization");
        let metric_source = if execution_cost_percent.is_some() {
            "execution-cost".to_owned()
        } else if sample_count > 0 {
            "streamData".to_owned()
        } else if limiter.is_some() {
            "raw-counter".to_owned()
        } else {
            timeline.source.clone()
        };
        if suppress_legacy_limiter_rows
            && metric_source == "raw-counter"
            && encoder_dispatches.is_empty()
            && kernel_name.is_none()
        {
            continue;
        }

        rows.push(CounterExportRow {
            row_index: rows.len(),
            command_buffer_index: encoder.command_buffer_index,
            encoder_index: encoder.index,
            encoder_label: encoder.label.clone(),
            kernel_name,
            pipeline_addr,
            start_time_ns: encoder.start_time_ns,
            end_time_ns: encoder.end_time_ns,
            duration_ns: encoder.duration_ns,
            dispatch_count: encoder.dispatch_count,
            kernel_invocations,
            metric_source,
            execution_cost_percent,
            execution_cost_samples,
            sample_count,
            avg_sampling_density,
            occupancy_percent,
            alu_utilization_percent,
            shader_launch_limiter_percent: row_metric("Compute Shader Launch Limiter")
                .or_else(|| limiter.and_then(|limiter| limiter.compute_shader_launch)),
            instruction_throughput_percent: row_metric("Instruction Throughput Limiter")
                .or_else(|| limiter.and_then(|limiter| limiter.instruction_throughput)),
            integer_complex_percent: row_metric("Integer and Complex Limiter")
                .or_else(|| limiter.and_then(|limiter| limiter.integer_complex)),
            f32_limiter_percent: row_metric("F32 Limiter")
                .or_else(|| limiter.and_then(|limiter| limiter.f32_limiter)),
            l1_cache_percent: row_metric("L1 Cache Limiter")
                .or_else(|| limiter.and_then(|limiter| limiter.l1_cache)),
            last_level_cache_percent: row_metric("Last Level Cache Limiter")
                .or_else(|| limiter.and_then(|limiter| limiter.last_level_cache)),
            control_flow_percent: row_metric("Control Flow Limiter")
                .or_else(|| limiter.and_then(|limiter| limiter.control_flow)),
            device_memory_bandwidth_gbps,
            gpu_read_bandwidth_gbps,
            gpu_write_bandwidth_gbps,
            buffer_device_memory_bytes_read,
            buffer_device_memory_bytes_written,
            bytes_read_from_device_memory,
            bytes_written_to_device_memory,
            buffer_l1_miss_rate_percent,
            buffer_l1_read_accesses,
            buffer_l1_read_bandwidth_gbps,
            buffer_l1_write_accesses,
            buffer_l1_write_bandwidth_gbps,
            compute_shader_launch_utilization_percent,
            control_flow_utilization_percent,
            instruction_throughput_utilization_percent,
            integer_complex_utilization_percent,
            integer_conditional_utilization_percent,
            f32_utilization_percent,
            temporary_register_count: pipeline_stats.map(|stats| stats.temporary_register_count),
            uniform_register_count: pipeline_stats.map(|stats| stats.uniform_register_count),
            spilled_bytes: pipeline_stats.map(|stats| stats.spilled_bytes),
            threadgroup_memory: pipeline_stats.map(|stats| stats.threadgroup_memory),
            instruction_count: pipeline_stats.map(|stats| stats.instruction_count),
            alu_instruction_count: pipeline_stats.map(|stats| stats.alu_instruction_count),
            branch_instruction_count: pipeline_stats.map(|stats| stats.branch_instruction_count),
            compilation_time_ms: pipeline_stats.map(|stats| stats.compilation_time_ms),
            metrics: row_metrics,
            metric_metadata: BTreeMap::new(),
        });
    }

    append_profiler_rows(
        &mut rows,
        &execution_cost_by_name,
        &sample_stats_by_name,
        &pipeline_stats_by_name,
        &profiler_dispatches_by_name,
        &occupancy_by_encoder,
        &represented_kernels,
    );

    let source = if aps_rows.is_empty() {
        timeline.source
    } else {
        format!("{}+aps-counter-samples", timeline.source)
    };
    for mut row in aps_rows.drain(..) {
        row.row_index = rows.len();
        rows.push(row);
    }

    Ok(CounterExportReport {
        trace_source: trace.path.clone(),
        source,
        total_rows: rows.len(),
        rows,
    })
}

fn empty_timeline_report(trace: &TraceBundle, source: &str) -> timeline::TimelineReport {
    timeline::TimelineReport {
        synthetic: true,
        source: source.to_owned(),
        command_buffers_profiler_backed: false,
        start_time_ns: 0,
        end_time_ns: 0,
        duration_ns: 0,
        command_buffer_count: 0,
        encoder_count: 0,
        dispatch_count: 0,
        counter_track_count: 0,
        command_buffers: Vec::new(),
        encoders: Vec::new(),
        dispatches: Vec::new(),
        counter_tracks: Vec::new(),
        events: vec![timeline::TimelineEvent {
            name: "structural timeline unavailable".to_owned(),
            category: Some("counter-export".to_owned()),
            phase: "i".to_owned(),
            timestamp_us: 0,
            duration_us: None,
            process_id: 0,
            thread_id: 0,
            args: BTreeMap::from([(
                "trace".to_owned(),
                serde_json::Value::String(trace.path.display().to_string()),
            )]),
        }],
    }
}

fn append_profiler_rows(
    rows: &mut Vec<CounterExportRow>,
    execution_cost_by_name: &BTreeMap<String, (f64, usize)>,
    sample_stats_by_name: &BTreeMap<String, (usize, f64, usize)>,
    pipeline_stats_by_name: &BTreeMap<String, profiler::ProfilerPipelineStats>,
    profiler_dispatches_by_name: &BTreeMap<String, Vec<&profiler::ProfilerDispatch>>,
    occupancy_by_encoder: &BTreeMap<usize, f64>,
    represented_kernels: &BTreeSet<String>,
) {
    let total_dispatch_time_us = profiler_dispatches_by_name
        .values()
        .flatten()
        .map(|dispatch| dispatch.duration_us)
        .sum::<u64>();
    let mut costs = profiler_dispatches_by_name
        .keys()
        .chain(execution_cost_by_name.keys())
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|kernel_name| {
            let dispatch_time_us = profiler_dispatches_by_name
                .get(&kernel_name)
                .into_iter()
                .flatten()
                .map(|dispatch| dispatch.duration_us)
                .sum::<u64>();
            let cost_percent = execution_cost_by_name
                .get(&kernel_name)
                .map(|(percent, _)| *percent)
                .or_else(|| {
                    (total_dispatch_time_us != 0)
                        .then(|| dispatch_time_us as f64 / total_dispatch_time_us as f64 * 100.0)
                })
                .unwrap_or_default();
            (kernel_name, cost_percent)
        })
        .collect::<Vec<_>>();
    costs.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });

    for (kernel_name, execution_cost_percent) in costs {
        if represented_kernels.contains(&kernel_name) {
            continue;
        }
        let dispatches = profiler_dispatches_by_name
            .get(&kernel_name)
            .cloned()
            .unwrap_or_default();
        let first_dispatch = dispatches.first().copied();
        let command_buffer_index = first_dispatch
            .map(|dispatch| dispatch.encoder_index)
            .unwrap_or_default();
        let encoder_index = first_dispatch
            .map(|dispatch| dispatch.encoder_index)
            .unwrap_or_default();
        let start_time_ns = first_dispatch
            .map(|dispatch| {
                dispatch
                    .cumulative_us
                    .saturating_sub(dispatch.duration_us)
                    .saturating_mul(1000)
            })
            .unwrap_or_default();
        let end_time_ns = first_dispatch
            .map(|dispatch| dispatch.cumulative_us.saturating_mul(1000))
            .unwrap_or(start_time_ns);
        let duration_ns = first_dispatch.map(|dispatch| dispatch.duration_us.saturating_mul(1000));
        let (sample_count, avg_sampling_density, dispatch_count) = sample_stats_by_name
            .get(&kernel_name)
            .copied()
            .map(|(samples, density_sum, density_count)| {
                let avg = (density_count != 0).then(|| density_sum / density_count as f64);
                (samples, avg, density_count)
            })
            .unwrap_or((0, None, dispatches.len()));
        let execution_cost_samples = execution_cost_by_name
            .get(&kernel_name)
            .map(|(_, samples)| *samples)
            .unwrap_or(sample_count);
        let pipeline_stats = pipeline_stats_by_name.get(&kernel_name);
        let occupancy = occupancy_by_encoder.get(&encoder_index).copied();
        let metric_source = if execution_cost_by_name.contains_key(&kernel_name) {
            "profile-execution-cost"
        } else {
            "profile-dispatch-time"
        };

        rows.push(CounterExportRow {
            row_index: rows.len(),
            command_buffer_index,
            encoder_index,
            encoder_label: kernel_name.clone(),
            kernel_name: Some(kernel_name.clone()),
            pipeline_addr: None,
            start_time_ns,
            end_time_ns,
            duration_ns,
            dispatch_count,
            kernel_invocations: dispatch_count,
            metric_source: metric_source.to_owned(),
            execution_cost_percent: Some(execution_cost_percent),
            execution_cost_samples,
            sample_count,
            avg_sampling_density,
            occupancy_percent: occupancy,
            alu_utilization_percent: None,
            shader_launch_limiter_percent: None,
            instruction_throughput_percent: None,
            integer_complex_percent: None,
            f32_limiter_percent: None,
            l1_cache_percent: None,
            last_level_cache_percent: None,
            control_flow_percent: None,
            device_memory_bandwidth_gbps: None,
            gpu_read_bandwidth_gbps: None,
            gpu_write_bandwidth_gbps: None,
            buffer_device_memory_bytes_read: None,
            buffer_device_memory_bytes_written: None,
            bytes_read_from_device_memory: None,
            bytes_written_to_device_memory: None,
            buffer_l1_miss_rate_percent: None,
            buffer_l1_read_accesses: None,
            buffer_l1_read_bandwidth_gbps: None,
            buffer_l1_write_accesses: None,
            buffer_l1_write_bandwidth_gbps: None,
            compute_shader_launch_utilization_percent: None,
            control_flow_utilization_percent: None,
            instruction_throughput_utilization_percent: None,
            integer_complex_utilization_percent: None,
            integer_conditional_utilization_percent: None,
            f32_utilization_percent: None,
            temporary_register_count: pipeline_stats.map(|stats| stats.temporary_register_count),
            uniform_register_count: pipeline_stats.map(|stats| stats.uniform_register_count),
            spilled_bytes: pipeline_stats.map(|stats| stats.spilled_bytes),
            threadgroup_memory: pipeline_stats.map(|stats| stats.threadgroup_memory),
            instruction_count: pipeline_stats.map(|stats| stats.instruction_count),
            alu_instruction_count: pipeline_stats.map(|stats| stats.alu_instruction_count),
            branch_instruction_count: pipeline_stats.map(|stats| stats.branch_instruction_count),
            compilation_time_ms: pipeline_stats.map(|stats| stats.compilation_time_ms),
            metrics: BTreeMap::new(),
            metric_metadata: BTreeMap::new(),
        });
    }
}

fn aps_counter_rows(
    _trace: &TraceBundle,
    timeline: &timeline::TimelineReport,
    profiler_summary: Option<&profiler::ProfilerStreamDataSummary>,
    raw_counter_report: &counter::RawCountersReport,
) -> Option<Vec<CounterExportRow>> {
    let mut groups_by_row = raw_counter_report
        .grouped_derived_metrics
        .iter()
        .filter(|group| group.group_kind == "encoder_sample")
        .filter_map(|group| group.encoder_sample_row_index.map(|row| (row, group)))
        .collect::<Vec<_>>();
    if groups_by_row.is_empty() {
        return None;
    }
    groups_by_row.sort_by_key(|(row, _)| *row);

    let sample_index_by_row = raw_counter_report
        .aggregate_metadata
        .iter()
        .flat_map(|metadata| metadata.encoder_sample_indices.iter())
        .map(|index| (index.row_index, index.sample_index))
        .collect::<BTreeMap<_, _>>();
    let timing_by_sample_index = profiler_summary
        .map(|summary| {
            summary
                .encoder_timings
                .iter()
                .map(|timing| (timing.end_offset_micros as u32, timing))
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();

    let mut rows = Vec::new();
    for (row_index, group) in groups_by_row {
        let mut metrics = BTreeMap::new();
        let mut metric_metadata = BTreeMap::new();
        for metric in &group.derived_metrics {
            insert_metric(&mut metrics, &metric.name, metric.value);
            metric_metadata
                .entry(metric.name.clone())
                .or_insert_with(|| CounterExportMetricMetadata {
                    key: metric.key.clone(),
                    counter_type: metric.counter_type.clone(),
                    description: metric.description.clone(),
                    unit: metric.unit.clone(),
                    groups: metric.groups.clone(),
                    timeline_groups: metric.timeline_groups.clone(),
                    visible: metric.visible,
                    batch_filtered: metric.batch_filtered,
                });
        }

        let sample_index = group
            .encoder_sample_index
            .or_else(|| sample_index_by_row.get(&row_index).copied())
            .unwrap_or(row_index as u32);
        let timing = timing_by_sample_index.get(&sample_index).copied();
        let command_buffer_index = row_index;
        let timeline_encoder = preferred_timeline_encoder(timeline, command_buffer_index);
        let encoder_label = timeline_encoder
            .map(|encoder| encoder.label.clone())
            .unwrap_or_else(|| format!("encoder_sample_{row_index}"));
        let pipeline_addr = timeline_encoder.map(|encoder| encoder.address);
        let kernel_name = kernel_name_from_label(&encoder_label);
        let start_time_ns = timing
            .map(|timing| {
                timing
                    .end_offset_micros
                    .saturating_sub(timing.duration_micros)
                    .saturating_mul(1000)
            })
            .or_else(|| timeline_encoder.map(|encoder| encoder.start_time_ns))
            .unwrap_or_default();
        let end_time_ns = timing
            .map(|timing| timing.end_offset_micros.saturating_mul(1000))
            .or_else(|| timeline_encoder.map(|encoder| encoder.end_time_ns))
            .unwrap_or(start_time_ns);
        let duration_ns = timing
            .map(|timing| timing.duration_micros.saturating_mul(1000))
            .or_else(|| timeline_encoder.and_then(|encoder| encoder.duration_ns));
        let kernel_invocations = first_metric(&metrics, &["Kernel Invocations"])
            .map(|value| value.round().max(0.0) as usize)
            .unwrap_or_default();

        rows.push(CounterExportRow {
            row_index: rows.len(),
            command_buffer_index,
            encoder_index: sample_index as usize,
            encoder_label,
            kernel_name,
            pipeline_addr,
            start_time_ns,
            end_time_ns,
            duration_ns,
            dispatch_count: kernel_invocations,
            kernel_invocations,
            metric_source: "aps-counter-samples".to_owned(),
            execution_cost_percent: None,
            execution_cost_samples: 0,
            sample_count: group.record_count,
            avg_sampling_density: None,
            occupancy_percent: first_metric(&metrics, &["Kernel Occupancy", "CS Occupancy"]),
            alu_utilization_percent: first_metric(
                &metrics,
                &[
                    "ALU Utilization",
                    "CS ALU Performance",
                    "Kernel ALU Performance",
                ],
            ),
            shader_launch_limiter_percent: first_metric(
                &metrics,
                &[
                    "Compute Shader Launch Limiter",
                    "Shader Launch Limiter",
                    "Vertex Shader Launch Limiter",
                ],
            ),
            instruction_throughput_percent: first_metric(
                &metrics,
                &["Instruction Throughput Limiter"],
            ),
            integer_complex_percent: first_metric(&metrics, &["Integer and Complex Limiter"]),
            f32_limiter_percent: first_metric(&metrics, &["F32 Limiter"]),
            l1_cache_percent: first_metric(
                &metrics,
                &[
                    "Texture Cache Limiter",
                    "Texture Read Cache Limiter",
                    "L1 Cache Limiter",
                ],
            ),
            last_level_cache_percent: first_metric(&metrics, &["Last Level Cache Limiter"]),
            control_flow_percent: first_metric(&metrics, &["Control Flow Limiter"]),
            device_memory_bandwidth_gbps: first_metric(
                &metrics,
                &["Device Memory Bandwidth", "Main Memory Throughput"],
            ),
            gpu_read_bandwidth_gbps: first_metric(&metrics, &["GPU Read Bandwidth"]),
            gpu_write_bandwidth_gbps: first_metric(&metrics, &["GPU Write Bandwidth"]),
            buffer_device_memory_bytes_read: first_metric(
                &metrics,
                &["Buffer Device Memory Bytes Read"],
            ),
            buffer_device_memory_bytes_written: first_metric(
                &metrics,
                &["Buffer Device Memory Bytes Written"],
            ),
            bytes_read_from_device_memory: first_metric(
                &metrics,
                &["Bytes Read From Device Memory"],
            ),
            bytes_written_to_device_memory: first_metric(
                &metrics,
                &["Bytes Written To Device Memory"],
            ),
            buffer_l1_miss_rate_percent: first_metric(
                &metrics,
                &[
                    "Buffer L1 Miss Rate",
                    "Texture Cache Read Miss Rate",
                    "Texture Cache Miss Rate",
                ],
            ),
            buffer_l1_read_accesses: first_metric(&metrics, &["Buffer L1 Read Accesses"]),
            buffer_l1_read_bandwidth_gbps: first_metric(
                &metrics,
                &["Buffer L1 Read Bandwidth", "L1 Read Bandwidth"],
            ),
            buffer_l1_write_accesses: first_metric(&metrics, &["Buffer L1 Write Accesses"]),
            buffer_l1_write_bandwidth_gbps: first_metric(
                &metrics,
                &["Buffer L1 Write Bandwidth", "L1 Write Bandwidth"],
            ),
            compute_shader_launch_utilization_percent: first_metric(
                &metrics,
                &["Compute Shader Launch Utilization"],
            ),
            control_flow_utilization_percent: first_metric(&metrics, &["Control Flow Utilization"]),
            instruction_throughput_utilization_percent: first_metric(
                &metrics,
                &["Instruction Throughput Utilization"],
            ),
            integer_complex_utilization_percent: first_metric(
                &metrics,
                &["Integer and Complex Utilization"],
            ),
            integer_conditional_utilization_percent: first_metric(
                &metrics,
                &["Integer and Conditional Utilization"],
            ),
            f32_utilization_percent: first_metric(&metrics, &["F32 Utilization"]),
            temporary_register_count: None,
            uniform_register_count: None,
            spilled_bytes: None,
            threadgroup_memory: None,
            instruction_count: None,
            alu_instruction_count: None,
            branch_instruction_count: None,
            compilation_time_ms: None,
            metrics,
            metric_metadata,
        });
    }
    Some(rows)
}

fn insert_metric(metrics: &mut BTreeMap<String, f64>, name: &str, value: f64) {
    if !value.is_finite() {
        return;
    }
    match metrics.get(name).copied() {
        Some(existing) if existing.abs() >= value.abs() => {}
        _ => {
            metrics.insert(name.to_owned(), value);
        }
    }
}

fn first_metric(metrics: &BTreeMap<String, f64>, names: &[&str]) -> Option<f64> {
    names.iter().find_map(|name| metrics.get(*name).copied())
}

fn preferred_timeline_encoder(
    timeline: &timeline::TimelineReport,
    command_buffer_index: usize,
) -> Option<&timeline::TimelineEncoder> {
    timeline
        .encoders
        .iter()
        .filter(|encoder| encoder.command_buffer_index == command_buffer_index)
        .find(|encoder| !encoder.label.ends_with(".command_buffer"))
        .or_else(|| {
            timeline
                .encoders
                .iter()
                .find(|encoder| encoder.command_buffer_index == command_buffer_index)
        })
}

fn kernel_name_from_label(label: &str) -> Option<String> {
    (!label.contains('.') && !label.is_empty()).then(|| label.to_owned())
}

fn top_metrics_summary(metrics: &BTreeMap<String, f64>, limit: usize) -> String {
    let mut values = metrics.iter().collect::<Vec<_>>();
    values.sort_by(|left, right| {
        right
            .1
            .abs()
            .partial_cmp(&left.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(right.0))
    });
    values
        .into_iter()
        .take(limit)
        .map(|(name, value)| format!("{name}={value:.2}"))
        .collect::<Vec<_>>()
        .join("; ")
}

#[cfg(test)]
fn match_xcode_encoder<'a>(
    encoder_index: usize,
    encoder_label: &str,
    data: &'a xcode_counters::XcodeCounterData,
) -> Option<&'a xcode_counters::XcodeEncoderCounters> {
    if let Some(exact) = data
        .encoders
        .iter()
        .find(|encoder| encoder.index == encoder_index)
    {
        return Some(exact);
    }

    let normalized_label = normalize_for_matching(encoder_label);
    if !normalized_label.is_empty() {
        if let Some(exact) = data
            .encoders
            .iter()
            .find(|encoder| normalize_for_matching(&encoder.encoder_label) == normalized_label)
        {
            return Some(exact);
        }

        if let Some(fuzzy) = data.encoders.iter().find(|encoder| {
            let normalized_encoder = normalize_for_matching(&encoder.encoder_label);
            !normalized_encoder.is_empty()
                && (normalized_encoder.contains(&normalized_label)
                    || normalized_label.contains(&normalized_encoder))
        }) {
            return Some(fuzzy);
        }
    }

    data.encoders.get(encoder_index)
}

pub fn format_report(report: &CounterExportReport) -> String {
    let mut out = String::new();
    out.push_str("Counter export report\n");
    out.push_str(&format!(
        "trace={} source={} rows={}\n\n",
        report.trace_source.display(),
        report.source,
        report.total_rows
    ));
    out.push_str(
        "row cb enc label kernel source duration_ns dispatches invocations exec% samples occ alu llc dev_bw regs spills inst\n",
    );
    for row in &report.rows {
        out.push_str(&format!(
            "{:>3} {:>2} {:>3} {:<16} {:<20} {:<14} {:>12} {:>10} {:>10} {:>6} {:>7} {:>7} {:>7} {:>7} {:>8} {:>6} {:>6} {:>6}\n",
            row.row_index,
            row.command_buffer_index,
            row.encoder_index,
            truncate(&row.encoder_label, 16),
            truncate(row.kernel_name.as_deref().unwrap_or("-"), 20),
            row.metric_source,
            row.duration_ns
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned()),
            row.dispatch_count,
            row.kernel_invocations,
            row.execution_cost_percent
                .map(|value| format!("{value:.2}"))
                .unwrap_or_else(|| "-".to_owned()),
            row.sample_count,
            row.occupancy_percent
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "-".to_owned()),
            row.alu_utilization_percent
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "-".to_owned()),
            row.last_level_cache_percent
                .map(|value| format!("{value:.1}"))
                .unwrap_or_else(|| "-".to_owned()),
            row.device_memory_bandwidth_gbps
                .map(|value| format!("{value:.2}"))
                .unwrap_or_else(|| "-".to_owned()),
            row.temporary_register_count
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned()),
            row.spilled_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned()),
            row.instruction_count
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned()),
        ));
    }
    if report.rows.iter().any(|row| !row.metrics.is_empty()) {
        out.push_str("\ntrace-derived metrics\n");
        for row in &report.rows {
            if row.metrics.is_empty() {
                continue;
            }
            out.push_str(&format!(
                "row {} {}: {}\n",
                row.row_index,
                truncate(&row.encoder_label, 32),
                top_metrics_summary(&row.metrics, 8)
            ));
        }
    }
    out
}

pub fn format_csv(report: &CounterExportReport) -> String {
    let mut out = String::new();
    out.push_str("row_index,command_buffer_index,encoder_index,encoder_label,kernel_name,pipeline_addr,start_time_ns,end_time_ns,duration_ns,dispatch_count,kernel_invocations,metric_source,execution_cost_percent,execution_cost_samples,sample_count,avg_sampling_density,occupancy_percent,alu_utilization_percent,shader_launch_limiter_percent,instruction_throughput_percent,integer_complex_percent,f32_limiter_percent,l1_cache_percent,last_level_cache_percent,control_flow_percent,device_memory_bandwidth_gbps,gpu_read_bandwidth_gbps,gpu_write_bandwidth_gbps,buffer_device_memory_bytes_read,buffer_device_memory_bytes_written,bytes_read_from_device_memory,bytes_written_to_device_memory,buffer_l1_miss_rate_percent,buffer_l1_read_accesses,buffer_l1_read_bandwidth_gbps,buffer_l1_write_accesses,buffer_l1_write_bandwidth_gbps,compute_shader_launch_utilization_percent,control_flow_utilization_percent,instruction_throughput_utilization_percent,integer_complex_utilization_percent,integer_conditional_utilization_percent,f32_utilization_percent,temporary_register_count,uniform_register_count,spilled_bytes,threadgroup_memory,instruction_count,alu_instruction_count,branch_instruction_count,compilation_time_ms\n");
    for row in &report.rows {
        let columns = vec![
            row.row_index.to_string(),
            row.command_buffer_index.to_string(),
            row.encoder_index.to_string(),
            csv_string(&row.encoder_label),
            csv_string(row.kernel_name.as_deref().unwrap_or("")),
            row.pipeline_addr
                .map(|value| format!("0x{value:x}"))
                .unwrap_or_default(),
            row.start_time_ns.to_string(),
            row.end_time_ns.to_string(),
            option_csv(row.duration_ns),
            row.dispatch_count.to_string(),
            row.kernel_invocations.to_string(),
            csv_string(&row.metric_source),
            option_csv(row.execution_cost_percent),
            row.execution_cost_samples.to_string(),
            row.sample_count.to_string(),
            option_csv(row.avg_sampling_density),
            option_csv(row.occupancy_percent),
            option_csv(row.alu_utilization_percent),
            option_csv(row.shader_launch_limiter_percent),
            option_csv(row.instruction_throughput_percent),
            option_csv(row.integer_complex_percent),
            option_csv(row.f32_limiter_percent),
            option_csv(row.l1_cache_percent),
            option_csv(row.last_level_cache_percent),
            option_csv(row.control_flow_percent),
            option_csv(row.device_memory_bandwidth_gbps),
            option_csv(row.gpu_read_bandwidth_gbps),
            option_csv(row.gpu_write_bandwidth_gbps),
            option_csv(row.buffer_device_memory_bytes_read),
            option_csv(row.buffer_device_memory_bytes_written),
            option_csv(row.bytes_read_from_device_memory),
            option_csv(row.bytes_written_to_device_memory),
            option_csv(row.buffer_l1_miss_rate_percent),
            option_csv(row.buffer_l1_read_accesses),
            option_csv(row.buffer_l1_read_bandwidth_gbps),
            option_csv(row.buffer_l1_write_accesses),
            option_csv(row.buffer_l1_write_bandwidth_gbps),
            option_csv(row.compute_shader_launch_utilization_percent),
            option_csv(row.control_flow_utilization_percent),
            option_csv(row.instruction_throughput_utilization_percent),
            option_csv(row.integer_complex_utilization_percent),
            option_csv(row.integer_conditional_utilization_percent),
            option_csv(row.f32_utilization_percent),
            option_csv(row.temporary_register_count),
            option_csv(row.uniform_register_count),
            option_csv(row.spilled_bytes),
            option_csv(row.threadgroup_memory),
            option_csv(row.instruction_count),
            option_csv(row.alu_instruction_count),
            option_csv(row.branch_instruction_count),
            option_csv(row.compilation_time_ms),
        ];
        out.push_str(&columns.join(","));
        out.push('\n');
    }
    out
}

pub fn format_xcode_csv(report: &CounterExportReport) -> String {
    let mut out = String::new();
    let mut headers = vec![
        "Index".to_owned(),
        "Encoder FunctionIndex".to_owned(),
        "CommandBuffer Label".to_owned(),
        "Debug Group".to_owned(),
        "Encoder Label".to_owned(),
        String::new(),
    ];
    headers.extend(XCODE_COUNTER_NAMES.iter().map(|name| (*name).to_owned()));
    out.push_str(&headers.join(","));
    out.push('\n');

    for row in &report.rows {
        let mut columns = vec![
            row.row_index.to_string(),
            row.encoder_index.to_string(),
            csv_string(&format!("Command Buffer {}", row.command_buffer_index)),
            String::new(),
            csv_string(&row.encoder_label),
            String::new(),
        ];
        columns.extend(
            XCODE_COUNTER_NAMES
                .iter()
                .map(|name| xcode_metric_csv(row, name)),
        );
        out.push_str(&columns.join(","));
        out.push('\n');
    }

    out
}

fn xcode_metric_csv(row: &CounterExportRow, name: &str) -> String {
    match name {
        "ALU Utilization" => option_csv(row.alu_utilization_percent),
        "Kernel Invocations" => row.kernel_invocations.to_string(),
        "Kernel Occupancy" => option_csv(row.occupancy_percent),
        "Compute Shader Launch Limiter" => option_csv(row.shader_launch_limiter_percent),
        "Instruction Throughput Limiter" => option_csv(row.instruction_throughput_percent),
        "Integer and Complex Limiter" => option_csv(row.integer_complex_percent),
        "F32 Limiter" => option_csv(row.f32_limiter_percent),
        "L1 Cache Limiter" => option_csv(row.l1_cache_percent),
        "Last Level Cache Limiter" => option_csv(row.last_level_cache_percent),
        "Control Flow Limiter" => option_csv(row.control_flow_percent),
        "Device Memory Bandwidth" => option_csv(row.device_memory_bandwidth_gbps),
        "GPU Read Bandwidth" => option_csv(row.gpu_read_bandwidth_gbps),
        "GPU Write Bandwidth" => option_csv(row.gpu_write_bandwidth_gbps),
        "Buffer Device Memory Bytes Read" => option_csv(row.buffer_device_memory_bytes_read),
        "Buffer Device Memory Bytes Written" => option_csv(row.buffer_device_memory_bytes_written),
        "Bytes Read From Device Memory" => option_csv(row.bytes_read_from_device_memory),
        "Bytes Written To Device Memory" => option_csv(row.bytes_written_to_device_memory),
        "Buffer L1 Miss Rate" => option_csv(row.buffer_l1_miss_rate_percent),
        "Buffer L1 Read Accesses" => option_csv(row.buffer_l1_read_accesses),
        "Buffer L1 Read Bandwidth" => option_csv(row.buffer_l1_read_bandwidth_gbps),
        "L1 Read Bandwidth" => option_csv(row.buffer_l1_read_bandwidth_gbps),
        "Buffer L1 Write Accesses" => option_csv(row.buffer_l1_write_accesses),
        "Buffer L1 Write Bandwidth" => option_csv(row.buffer_l1_write_bandwidth_gbps),
        "L1 Write Bandwidth" => option_csv(row.buffer_l1_write_bandwidth_gbps),
        "Compute Shader Launch Utilization" => {
            option_csv(row.compute_shader_launch_utilization_percent)
        }
        "Control Flow Utilization" => option_csv(row.control_flow_utilization_percent),
        "Instruction Throughput Utilization" => {
            option_csv(row.instruction_throughput_utilization_percent)
        }
        "Integer and Complex Utilization" => option_csv(row.integer_complex_utilization_percent),
        "Integer and Conditional Utilization" => {
            option_csv(row.integer_conditional_utilization_percent)
        }
        "F32 Utilization" => option_csv(row.f32_utilization_percent),
        "Kernel ALU Instructions" => option_csv(row.alu_instruction_count),
        "Kernel ALU Integer and Conditional Instructions" => {
            option_csv(row.branch_instruction_count)
        }
        _ => String::new(),
    }
}

const XCODE_COUNTER_NAMES: &[&str] = &[
    "1D Texture Array Sampler Calls",
    "1D Texture Sampler Calls",
    "2D MSAA Texture Sampler Calls",
    "2D Texture Array Sampler Calls",
    "2D Texture Sampler Calls",
    "2X MSAA Resolved Pixels Stored",
    "3D Texture Sampler Calls",
    "4X MSAA Resolved Pixels Stored",
    "ALU Utilization",
    "Anisotropic Sampler Calls",
    "Attachment Pixels Stored",
    "Average Anisotropic Level",
    "Average Pixel Overdraw",
    "Average Samples Per Pixel",
    "Average Sparse Texture Tile Size",
    "Back Face Clipped Primitives",
    "Block Compressed Texture Samples",
    "Buffer Device Memory Bytes Read",
    "Buffer Device Memory Bytes Written",
    "Buffer L1 Miss Rate",
    "Buffer L1 Read Accesses",
    "Buffer L1 Read Bandwidth",
    "Buffer L1 Write Accesses",
    "Buffer L1 Write Bandwidth",
    "Bytes Read From Device Memory",
    "Bytes Written To Device Memory",
    "Clip Unit Limiter",
    "Compression Ratio of Texture Memory Read",
    "Compression Ratio of Texture Memory Written",
    "Compute Shader Launch Limiter",
    "Compute Shader Launch Utilization",
    "Control Flow Limiter",
    "Control Flow Utilization",
    "Cube Array Texture Sampler Calls",
    "Cube Texture Sampler Calls",
    "Cull Unit Limiter",
    "Depth Load Utilization",
    "Depth Load Utilization",
    "Depth Store Utilization",
    "Depth Test Utilization",
    "Depth Test Utilization",
    "Depth Texture Bytes Loaded",
    "Depth Texture Bytes Stored",
    "Depth Texture Device Memory Bytes Read",
    "Depth Texture Device Memory Bytes Read",
    "Depth Texture Device Memory Bytes Written",
    "Depth Texture Device Memory Bytes Written",
    "Device Atomic Bytes Read",
    "Device Atomic Bytes Written",
    "Device Memory Bandwidth",
    "F16 Limiter",
    "F16 Utilization",
    "F32 Limiter",
    "F32 Utilization",
    "FS ALU Float Instructions",
    "FS ALU Half Instructions",
    "FS ALU Instructions",
    "FS ALU Integer and Complex Instructions",
    "FS ALU Integer and Conditional Instructions",
    "FS ALU Performance",
    "FS Buffer Device Memory Bytes Read",
    "FS Buffer Device Memory Bytes Written",
    "FS Bytes Read From Device Memory",
    "FS Bytes Written To Device Memory",
    "FS Device Atomic Bytes Read",
    "FS Device Atomic Bytes Written",
    "FS Device Memory Bandwidth",
    "FS Helper Invocations",
    "FS Helper Invocations Inefficiency",
    "FS Invocation Utilization",
    "FS Invocations",
    "FS Invocations per Primitive",
    "FS Last Level Cache Bytes Read",
    "FS Last Level Cache Bytes Written",
    "FS Occupancy",
    "FS Texture Cache Miss Rate",
    "FS Texture L1 Bytes Read",
    "FS Tiles Processed",
    "Fast Point Sampling Speedup",
    "Fragment Generator Pixel Processing",
    "Fragment Generator Primitive Processing",
    "Fragment Shader Launch Limiter",
    "Fragment Shader Launch Utilization",
    "Fragments Rasterized per Primitive",
    "GPU Read Bandwidth",
    "GPU Write Bandwidth",
    "ImageBlock L1 Read Accesses",
    "ImageBlock L1 Write Accesses",
    "Imageblock L1 Read Bandwidth",
    "Imageblock L1 Write Bandwidth",
    "Instruction Throughput Limiter",
    "Instruction Throughput Utilization",
    "Integer and Complex Limiter",
    "Integer and Complex Utilization",
    "Integer and Conditional Limiter",
    "Integer and Conditional Utilization",
    "Kernel ALU Float Instructions",
    "Kernel ALU Half Instructions",
    "Kernel ALU Instructions",
    "Kernel ALU Integer and Complex Instructions",
    "Kernel ALU Integer and Conditional Instructions",
    "Kernel ALU Performance",
    "Kernel Invocations",
    "Kernel Occupancy",
    "Kernel Texture Cache Miss Rate",
    "L1 Buffer Residency",
    "L1 Cache Limiter",
    "L1 Cache Utilization",
    "L1 Eviction Rate",
    "L1 Imageblock Residency",
    "L1 Other Residency",
    "L1 RT Scratch Residency",
    "L1 RT Scratch Residency",
    "L1 Read Bandwidth",
    "L1 Register Residency",
    "L1 Stack Residency",
    "L1 Threadgroup Residency",
    "L1 Total Residency",
    "L1 Write Bandwidth",
    "Last Level Cache Bandwidth",
    "Last Level Cache Bytes Read",
    "Last Level Cache Bytes Written",
    "Last Level Cache Limiter",
    "Last Level Cache Miss Rate",
    "Last Level Cache Utilization",
    "Lossless Compressed Pixels Stored",
    "Lossless Compressed Texture Bytes From Cache",
    "Lossless Compressed Texture Samples",
    "Lossy Compressed Pixels Stored",
    "Lossy Compressed Texture Bytes From Cache",
    "Lossy Compressed Texture Samples",
    "MMU Limiter",
    "MMU TLB Miss Rate",
    "MMU Utilization",
    "Mipmap Linear Sampler Calls",
    "Mipmap Nearest Sampler Calls",
    "New Triangles Generated",
    "New Vertices Generated",
    "Occupancy Manager Target",
    "Occupancy Manager Target",
    "Other L1 Read Accesses",
    "Other L1 Read Accesses",
    "Other L1 Write Accesses",
    "Other L1 Write Accesses",
    "Partial Render Count",
    "Pixels Rasterized",
    "Pixels Stored",
    "Pixels per Vertex",
    "Post Clip Cull Primitive Processing",
    "Post Clipped Primitives",
    "Pre Cull Primitive Processing",
    "PreZ Test Fails",
    "Predicated Texture Thread Reads",
    "Predicated Texture Thread Writes",
    "Primitive Block Tile Intersections",
    "Primitives",
    "Primitives Clipped",
    "Primitives Culled (Back-Face)",
    "Primitives Culled (Guard-Band)",
    "Primitives Culled (Off-Screen)",
    "Primitives Culled (Zero-Area)",
    "Primitives Per Tile",
    "Primitives Rendered",
    "RT Intersect Ray Threads",
    "RT Scratch L1 Read Accesses",
    "RT Scratch L1 Write Accesses",
    "RT Unit Active",
    "Rasterizer Sample Processing",
    "Ray Occupancy",
    "Register L1 Read Accesses",
    "Register L1 Read Accesses",
    "Register L1 Write Accesses",
    "Register L1 Write Accesses",
    "Sampler Calls/FS Invocation",
    "Sampler Calls/VS Invocation",
    "Samples Shaded Per Tile",
    "Shaded Vertex Read Limiter",
    "Small Triangles Clipped Pimitives",
    "Sparse Texture Translation Limiter",
    "Sparse Texture Translation Requests",
    "Stack L1 Read Accesses",
    "Stack L1 Read Bandwidth",
    "Stack L1 Write Accesses",
    "Stack L1 Write Bandwidth",
    "Texture Accesses",
    "Texture Cache Miss Rate",
    "Texture Cache Miss Rate",
    "Texture Device Memory Bytes Read",
    "Texture Device Memory Bytes Read",
    "Texture Device Memory Bytes Written",
    "Texture Device Memory Bytes Written",
    "Texture Filtering Limiter",
    "Texture Filtering Utilization",
    "Texture Gather Calls",
    "Texture L1 Bytes Read",
    "Texture L1 Bytes Read",
    "Texture Pixels Stored",
    "Texture Quads",
    "Texture Read Cache Limiter",
    "Texture Read Cache Miss Limiter",
    "Texture Read Cache Utilization",
    "Texture Read Limiter",
    "Texture Read Utilization",
    "Texture Sample Calls",
    "Texture Write Limiter",
    "Texture Write Utilization",
    "ThreadGroup L1 Read Accesses",
    "ThreadGroup L1 Write Accesses",
    "ThreadGroup L1 Write Accesses",
    "Threadgroup Memory L1 Read Bandwidth",
    "Threadgroup Memory L1 Write Bandwidth",
    "Threadgroup Memory L1 Write Bandwidth",
    "Tiled Vertex Buffer Bytes",
    "Tiled Vertex Buffer Primitive Blocks Bytes",
    "Tiling Block Utilization",
    "Total Resolved Pixels",
    "Uncompressed Texture Samples",
    "VS ALU Float Instructions",
    "VS ALU Half Instructions",
    "VS ALU Instructions",
    "VS ALU Integer and Complex Instructions",
    "VS ALU Integer and Conditional Instructions",
    "VS ALU Performance",
    "VS Buffer Device Memory Bytes Read",
    "VS Buffer Device Memory Bytes Written",
    "VS Bytes Read From Device Memory",
    "VS Bytes Written To Device Memory",
    "VS Device Atomic Bytes Read",
    "VS Device Atomic Bytes Written",
    "VS Device Memory Bandwidth",
    "VS Invocation Utilization",
    "VS Invocations",
    "VS Last Level Cache Bytes Read",
    "VS Last Level Cache Bytes Written",
    "VS Occupancy",
    "VS Texture Cache Miss Rate",
    "VS Texture L1 Bytes Read",
    "Vertex Shader Launch Limiter",
    "Vertex Shader Launch Utilization",
    "Vertices",
    "Vertices Reused",
];

fn truncate(value: &str, width: usize) -> String {
    if value.len() <= width {
        value.to_owned()
    } else {
        format!("{}...", &value[..width.saturating_sub(3)])
    }
}

fn csv_string(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn option_csv<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

#[cfg(test)]
fn normalize_for_matching(name: &str) -> String {
    name.chars()
        .filter_map(|ch| {
            if ch.is_ascii_alphanumeric() {
                Some(ch.to_ascii_lowercase())
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn formats_counter_export_report() {
        let report = CounterExportReport {
            trace_source: PathBuf::from("/tmp/example.gputrace"),
            source: "streamData".into(),
            total_rows: 1,
            rows: vec![CounterExportRow {
                row_index: 0,
                command_buffer_index: 0,
                encoder_index: 2,
                encoder_label: "main_encoder".into(),
                kernel_name: Some("blur".into()),
                pipeline_addr: Some(0x1234),
                start_time_ns: 100,
                end_time_ns: 200,
                duration_ns: Some(100),
                dispatch_count: 3,
                kernel_invocations: 3,
                metric_source: "execution-cost".into(),
                execution_cost_percent: Some(55.0),
                execution_cost_samples: 4,
                sample_count: 6,
                avg_sampling_density: Some(0.3),
                occupancy_percent: Some(37.5),
                alu_utilization_percent: Some(60.0),
                shader_launch_limiter_percent: Some(12.0),
                instruction_throughput_percent: Some(3.0),
                integer_complex_percent: Some(120.0),
                f32_limiter_percent: Some(650.0),
                l1_cache_percent: Some(8.0),
                last_level_cache_percent: Some(7.0),
                control_flow_percent: Some(9.0),
                device_memory_bandwidth_gbps: Some(3.2),
                gpu_read_bandwidth_gbps: Some(2.2),
                gpu_write_bandwidth_gbps: Some(1.0),
                buffer_device_memory_bytes_read: Some(4096.0),
                buffer_device_memory_bytes_written: Some(2048.0),
                bytes_read_from_device_memory: Some(8192.0),
                bytes_written_to_device_memory: Some(1024.0),
                buffer_l1_miss_rate_percent: Some(4.5),
                buffer_l1_read_accesses: Some(128.0),
                buffer_l1_read_bandwidth_gbps: Some(1.4),
                buffer_l1_write_accesses: Some(32.0),
                buffer_l1_write_bandwidth_gbps: Some(0.8),
                compute_shader_launch_utilization_percent: Some(70.0),
                control_flow_utilization_percent: Some(11.0),
                instruction_throughput_utilization_percent: Some(22.0),
                integer_complex_utilization_percent: Some(33.0),
                integer_conditional_utilization_percent: Some(44.0),
                f32_utilization_percent: Some(55.0),
                temporary_register_count: Some(24),
                uniform_register_count: Some(12),
                spilled_bytes: Some(64),
                threadgroup_memory: Some(128),
                instruction_count: Some(1024),
                alu_instruction_count: Some(700),
                branch_instruction_count: Some(20),
                compilation_time_ms: Some(1.2),
                metrics: BTreeMap::new(),
                metric_metadata: BTreeMap::new(),
            }],
        };

        let text = format_report(&report);
        assert!(text.contains("Counter export report"));
        assert!(text.contains("main_encoder"));
        assert!(text.contains("execution-cost"));

        let csv = format_csv(&report);
        assert!(csv.contains("encoder_label"));
        assert!(csv.contains("\"main_encoder\""));
        assert!(csv.contains("\"blur\""));
        assert!(csv.contains("55"));

        let xcode_csv = format_xcode_csv(&report);
        let header = xcode_csv.lines().next().unwrap();
        assert_eq!(header.split(',').count(), 247);
        assert!(header.starts_with(
            "Index,Encoder FunctionIndex,CommandBuffer Label,Debug Group,Encoder Label,,"
        ));
        assert!(header.contains("Kernel Invocations"));
        assert!(header.contains("GPU Read Bandwidth"));
        assert!(xcode_csv.contains("\"main_encoder\""));
        assert!(
            xcode_csv
                .lines()
                .nth(1)
                .unwrap()
                .split(',')
                .any(|column| column == "60")
        );
    }

    #[test]
    fn matches_xcode_encoder_by_index_then_label() {
        let data = xcode_counters::XcodeCounterData {
            source: PathBuf::from("/tmp/example.csv"),
            metrics: vec!["Kernel Occupancy".into()],
            encoders: vec![
                xcode_counters::XcodeEncoderCounters {
                    index: 7,
                    function_index: 0,
                    command_buffer_label: "cb0".into(),
                    encoder_label: "Compute Encoder 7 0x1234".into(),
                    counters: BTreeMap::new(),
                },
                xcode_counters::XcodeEncoderCounters {
                    index: 99,
                    function_index: 1,
                    command_buffer_label: "cb1".into(),
                    encoder_label: "main_encoder".into(),
                    counters: BTreeMap::new(),
                },
            ],
        };

        assert_eq!(
            match_xcode_encoder(7, "ignored", &data).map(|encoder| encoder.index),
            Some(7)
        );
        assert_eq!(
            match_xcode_encoder(1, "Main Encoder", &data).map(|encoder| encoder.index),
            Some(99)
        );
    }

    #[test]
    fn chooses_first_non_command_buffer_encoder_for_sample_rows() {
        let timeline = timeline::TimelineReport {
            synthetic: false,
            source: "fixture".to_owned(),
            command_buffers_profiler_backed: true,
            start_time_ns: 0,
            end_time_ns: 0,
            duration_ns: 0,
            command_buffer_count: 1,
            encoder_count: 2,
            dispatch_count: 0,
            counter_track_count: 0,
            command_buffers: Vec::new(),
            encoders: vec![
                timeline::TimelineEncoder {
                    index: 1,
                    command_buffer_index: 0,
                    label: "cb.command_buffer".to_owned(),
                    address: 0,
                    dispatch_count: 0,
                    start_time_ns: 0,
                    end_time_ns: 1,
                    duration_ns: Some(1),
                    synthetic: false,
                },
                timeline::TimelineEncoder {
                    index: 2,
                    command_buffer_index: 0,
                    label: "kernel_a".to_owned(),
                    address: 0,
                    dispatch_count: 0,
                    start_time_ns: 1,
                    end_time_ns: 2,
                    duration_ns: Some(1),
                    synthetic: false,
                },
            ],
            dispatches: Vec::new(),
            counter_tracks: Vec::new(),
            events: Vec::new(),
        };

        assert_eq!(
            preferred_timeline_encoder(&timeline, 0).map(|encoder| encoder.label.as_str()),
            Some("kernel_a")
        );
    }
}
