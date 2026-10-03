use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Cursor;
use std::mem;
use std::path::{Path, PathBuf};
use std::time::Instant;

use plist::{Dictionary, Uid, Value};
use rquickjs::{Context, Runtime};
use serde::Serialize;

use crate::profiler;
use crate::trace::TraceBundle;
use crate::xcode_counters;

type SampleBlob = (String, Vec<u8>);
type CounterSchemaByGroup = BTreeMap<usize, Vec<String>>;

#[derive(Debug, Clone, Default)]
struct StreamArchiveGroups {
    aps_data: Vec<Vec<u8>>,
    aps_counter_data: Vec<Vec<u8>>,
    aps_timeline_data: Vec<Vec<u8>>,
}

/// Per-encoder hardware limiters (percent, 0..100) and bandwidths (GB/s).
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct CounterLimiter {
    pub encoder_index: usize,
    /// `Compute Occupancy`.
    pub occupancy: Option<f64>,
    pub alu_utilization: Option<f64>,
    pub compute_shader_launch: Option<f64>,
    pub instruction_throughput: Option<f64>,
    pub integer_complex: Option<f64>,
    pub control_flow: Option<f64>,
    pub f32_limiter: Option<f64>,
    pub l1_cache: Option<f64>,
    pub last_level_cache: Option<f64>,
    pub device_memory_bandwidth_gbps: Option<f64>,
    pub buffer_l1_read_bandwidth_gbps: Option<f64>,
    pub buffer_l1_write_bandwidth_gbps: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterProbeReport {
    pub profiler_directory: PathBuf,
    pub csv_source: Option<PathBuf>,
    pub targets: Vec<RawCounterProbeTarget>,
    pub aggregate_metadata: Vec<RawCounterAggregateMetadata>,
    pub counter_info: Vec<RawCounterInfoEntry>,
    pub stream_archives: Vec<RawCounterStreamArchive>,
    pub structured_layouts: Vec<RawCounterStructuredLayout>,
    pub normalized_counters: Vec<RawCounterNormalizedMetric>,
    pub normalized_matches: Vec<RawCounterNormalizedMatch>,
    pub structured_samples: Vec<RawCounterStructuredSample>,
    pub files: Vec<RawCounterProbeFile>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCountersReport {
    pub trace_source: PathBuf,
    pub profiler_directory: PathBuf,
    pub timings: Vec<RawCounterDecodeTiming>,
    pub dispatch_alignment: Option<RawCounterDispatchAlignmentReport>,
    pub aggregate_metadata: Vec<RawCounterAggregateMetadata>,
    pub sample_trace_indices: Vec<RawCounterSampleTraceIndex>,
    pub trace_maps: Vec<RawCounterTraceMapEntry>,
    pub program_address_mappings: Vec<RawCounterProgramAddressMapping>,
    pub profiling_address_summary: Option<ProfilingAddressProbeReport>,
    pub counter_info: Vec<RawCounterInfoEntry>,
    pub schemas: Vec<RawCounterSchema>,
    pub streams: Vec<RawCounterDecodedStream>,
    pub metrics: Vec<RawCounterDecodedMetric>,
    pub derived_metrics: Vec<RawCounterJsDerivedMetric>,
    pub grouped_derived_metrics: Vec<RawCounterJsDerivedMetricGroup>,
    pub shifted_dispatch_metric_candidates: Vec<RawCounterShiftedDispatchMetricCandidate>,
    pub encoder_sample_metrics: Vec<RawCounterEncoderSampleMetric>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterDecodeTiming {
    pub stage: String,
    pub ms: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterDispatchAlignmentReport {
    pub timebase_numer: Option<u64>,
    pub timebase_denom: Option<u64>,
    pub dispatch_count: usize,
    pub dispatch_start_ticks: u64,
    pub dispatch_end_ticks: u64,
    pub dispatch_span_ticks: u64,
    pub dispatch_span_ns: Option<f64>,
    pub raw_record_count: usize,
    pub raw_start_ticks: u64,
    pub raw_end_ticks: u64,
    pub raw_span_ticks: u64,
    pub raw_span_ns: Option<f64>,
    pub direct_overlap_record_count: usize,
    pub raw_before_dispatch_record_count: usize,
    pub raw_after_dispatch_record_count: usize,
    pub raw_end_to_dispatch_start_gap_ticks: i64,
    pub raw_end_to_dispatch_start_gap_ns: Option<f64>,
    pub raw_start_to_dispatch_start_delta_ticks: i64,
    pub raw_start_to_dispatch_start_delta_ns: Option<f64>,
    pub candidate_shifts: Vec<RawCounterDispatchAlignmentCandidate>,
    pub source_spans: Vec<RawCounterTimestampSpan>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterDispatchAlignmentCandidate {
    pub name: String,
    pub shift_ticks: i64,
    pub shift_ns: Option<f64>,
    pub shifted_raw_start_ticks: u64,
    pub shifted_raw_end_ticks: u64,
    pub overlap_record_count: usize,
    pub dispatches_covered: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterTimestampSpan {
    pub sample_group: Option<usize>,
    pub source_index: Option<usize>,
    pub ring_indices: Vec<usize>,
    pub record_count: usize,
    pub start_ticks: u64,
    pub end_ticks: u64,
    pub span_ticks: u64,
    pub direct_overlap_record_count: usize,
    pub nearest_dispatch_gap_ticks: i64,
    pub nearest_dispatch_gap_ns: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterSchema {
    pub sample_group: usize,
    pub counter_count: usize,
    pub counter_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterDecodedStream {
    pub path: String,
    pub sample_group: Option<usize>,
    pub source_index: Option<usize>,
    pub ring_index: Option<usize>,
    pub byte_len: usize,
    pub record_size: usize,
    pub record_count: usize,
    pub counter_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterDecodedMetric {
    pub path: String,
    pub sample_group: Option<usize>,
    pub source_index: Option<usize>,
    pub ring_index: Option<usize>,
    pub counter_index: usize,
    pub raw_name: String,
    pub sample_count: usize,
    pub min_percent_of_gpu_cycles: f64,
    pub mean_percent_of_gpu_cycles: f64,
    pub max_percent_of_gpu_cycles: f64,
    pub encoder_ids: Vec<u64>,
    pub kick_trace_ids: Vec<u64>,
    pub source_ids: Vec<u64>,
    pub derived_counter_matches: Vec<RawCounterDerivedCounterMatch>,
    pub hardware_selectors: Vec<RawCounterHardwareSelector>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterEncoderSampleMetric {
    pub row_index: usize,
    pub sample_index: u32,
    pub path: String,
    pub sample_group: Option<usize>,
    pub source_index: Option<usize>,
    pub ring_index: Option<usize>,
    pub counter_index: usize,
    pub raw_name: String,
    pub sample_count: usize,
    pub raw_delta: u64,
    pub normalized_percent: Option<f64>,
    pub derived_counter_matches: Vec<RawCounterDerivedCounterMatch>,
    pub hardware_selectors: Vec<RawCounterHardwareSelector>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct RawCounterDerivedCounterMatch {
    pub key: String,
    pub name: String,
    pub counter_type: Option<String>,
    pub description: Option<String>,
    pub unit: Option<String>,
    pub groups: Vec<String>,
    pub timeline_groups: Vec<String>,
    pub visible: Option<bool>,
    pub batch_filtered: Option<bool>,
    pub sources: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct RawCounterHardwareSelector {
    pub partition: Option<u64>,
    pub select: Option<u64>,
    pub flag: Option<u64>,
    pub sources: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterJsDerivedMetric {
    pub key: String,
    pub name: String,
    pub counter_type: Option<String>,
    pub description: Option<String>,
    pub unit: Option<String>,
    pub groups: Vec<String>,
    pub timeline_groups: Vec<String>,
    pub visible: Option<bool>,
    pub batch_filtered: Option<bool>,
    pub value: f64,
    pub source_script: PathBuf,
    pub source_catalog: PathBuf,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterJsDerivedMetricGroup {
    pub group_kind: String,
    pub group_id: String,
    pub encoder_sample_row_index: Option<usize>,
    pub encoder_sample_index: Option<u32>,
    pub sample_group: Option<usize>,
    pub source_index: Option<usize>,
    pub ring_indices: Vec<usize>,
    pub start_ticks: Option<u64>,
    pub end_ticks: Option<u64>,
    pub record_count: usize,
    pub encoder_ids: Vec<u64>,
    pub kick_trace_ids: Vec<u64>,
    pub source_ids: Vec<u64>,
    pub profiler_dispatch_index: Option<usize>,
    pub profiler_encoder_index: Option<usize>,
    pub profiler_function_name: Option<String>,
    pub profiler_pipeline_id: Option<i64>,
    pub profiler_start_ticks: Option<u64>,
    pub profiler_end_ticks: Option<u64>,
    pub derived_metrics: Vec<RawCounterJsDerivedMetric>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterShiftedDispatchMetricCandidate {
    pub name: String,
    pub shift_ticks: i64,
    pub shift_ns: Option<f64>,
    pub overlap_record_count: usize,
    pub dispatches_covered: usize,
    pub groups: Vec<RawCounterJsDerivedMetricGroup>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterProbeTarget {
    pub metric: String,
    pub row_index: usize,
    pub encoder_label: String,
    pub value: f64,
    pub tolerance: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterAggregateMetadata {
    pub archive_index: usize,
    pub timebase_numer: Option<u64>,
    pub timebase_denom: Option<u64>,
    pub num_encoders: Option<u64>,
    pub perf_info: BTreeMap<String, u64>,
    pub encoder_sample_indices: Vec<RawCounterEncoderSampleIndex>,
    pub encoder_infos: Vec<RawCounterEncoderInfo>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterInfoEntry {
    pub archive_index: usize,
    pub raw_name: String,
    pub summary: String,
    pub fields: Vec<RawCounterFieldSummary>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterEncoderSampleIndex {
    pub row_index: usize,
    pub word0: u32,
    pub word1: u32,
    pub sample_index: u32,
    pub word3: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterEncoderInfo {
    pub row_index: usize,
    pub trace_ids: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterSampleTraceIndex {
    pub archive_index: usize,
    pub trace_id: u64,
    pub words: Vec<u64>,
    pub sample_index: Option<u32>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterTraceMapEntry {
    pub archive_index: usize,
    pub map_name: String,
    pub trace_id: u64,
    pub scalar_value: Option<u64>,
    pub words: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct RawCounterProgramAddressMapping {
    pub archive_index: usize,
    pub mapping_index: usize,
    pub mapping_type: String,
    pub binary_unique_id: Option<String>,
    pub draw_call_index: Option<u64>,
    pub draw_function_index: Option<u64>,
    pub encoder_trace_id: Option<u64>,
    pub encoder_index: Option<u64>,
    pub shader_index: Option<u64>,
    pub mapped_address: Option<u64>,
    pub mapped_size: Option<u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProfilingAddressProbeReport {
    pub trace_source: PathBuf,
    pub profiler_directory: PathBuf,
    pub mapping_count: usize,
    pub file_summaries: Vec<ProfilingAddressFileSummary>,
    pub top_full_address_hits: Vec<ProfilingAddressHit>,
    pub top_low32_address_hits: Vec<ProfilingAddressHit>,
    pub top_shader_low32_hits: Vec<ProfilingShaderAddressHit>,
    pub top_function_low32_hits: Vec<ProfilingFunctionAddressHit>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProfilingAddressFileSummary {
    pub file_index: usize,
    pub file_name: String,
    pub byte_len: usize,
    pub full_address_hits: usize,
    pub low32_address_hits: usize,
}

#[derive(Debug)]
struct ProfilingAddressFileProbe {
    file_summary: ProfilingAddressFileSummary,
    full_hits: Vec<usize>,
    low32_hits: Vec<usize>,
    first_full_offsets: Vec<Option<usize>>,
    first_low32_offsets: Vec<Option<usize>>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProfilingAddressHit {
    pub hit_count: usize,
    pub scan_kind: String,
    pub first_offset: usize,
    pub archive_index: usize,
    pub mapping_index: usize,
    pub mapping_type: String,
    pub binary_unique_id: Option<String>,
    pub draw_call_index: Option<u64>,
    pub draw_function_index: Option<u64>,
    pub encoder_trace_id: Option<u64>,
    pub encoder_index: Option<u64>,
    pub shader_index: Option<u64>,
    pub mapped_address: Option<u64>,
    pub mapped_size: Option<u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProfilingShaderAddressHit {
    pub hit_count: usize,
    pub dispatch_index: usize,
    pub function_name: Option<String>,
    pub encoder_index: Option<u64>,
    pub encoder_trace_id: Option<u64>,
    pub mapping_count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProfilingFunctionAddressHit {
    pub hit_count: usize,
    pub function_name: String,
    pub dispatch_count: usize,
    pub mapping_count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterProbeFile {
    pub file_index: usize,
    pub file_name: String,
    pub byte_len: usize,
    pub page_count_4k: Option<usize>,
    pub marker_count: usize,
    pub top_record_shapes: Vec<RawCounterRecordShape>,
    pub matches: Vec<RawCounterProbeMatch>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterRecordShape {
    pub tag: String,
    pub size: usize,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterStreamArchive {
    pub group: String,
    pub index: usize,
    pub byte_len: usize,
    pub source: Option<String>,
    pub serial: Option<u64>,
    pub source_index: Option<u64>,
    pub ring_buffer_index: Option<u64>,
    pub data_file: Option<String>,
    pub shader_profiler_data_len: Option<usize>,
    pub fields: Vec<RawCounterFieldSummary>,
    pub data_fields: Vec<RawCounterDataField>,
    pub keys: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterFieldSummary {
    pub key: String,
    pub kind: String,
    pub len: Option<usize>,
    pub keys: Vec<String>,
    pub children: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterDataField {
    pub key: String,
    pub byte_len: usize,
    pub prefix_hex: String,
    pub f32_preview: Vec<f64>,
    pub u32_preview: Vec<u32>,
    pub u64_preview: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterStructuredSample {
    pub path: String,
    pub byte_len: usize,
    pub gprw_record_size: Option<usize>,
    pub gprw_record_count: Option<usize>,
    pub matches: Vec<RawCounterProbeMatch>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterStructuredLayout {
    pub path: String,
    pub byte_len: usize,
    pub gprw_record_size: usize,
    pub gprw_record_count: usize,
    pub u64_columns: Vec<RawCounterColumnStat>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterColumnStat {
    pub index: usize,
    pub min: u64,
    pub max: u64,
    pub mean: f64,
    pub nonzero_count: usize,
    pub first_values: Vec<u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterNormalizedMetric {
    pub path: String,
    pub sample_group: Option<usize>,
    pub source_index: Option<usize>,
    pub ring_index: Option<usize>,
    pub encoder_ids: Vec<u64>,
    pub kick_trace_ids: Vec<u64>,
    pub source_ids: Vec<u64>,
    pub counter_index: usize,
    pub raw_name: String,
    pub sample_count: usize,
    pub min_percent: f64,
    pub mean_percent: f64,
    pub max_percent: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterNormalizedMatch {
    pub metric: String,
    pub row_index: usize,
    pub encoder_label: String,
    pub target: f64,
    pub delta: f64,
    pub tolerance: f64,
    pub confidence: f64,
    pub counter: RawCounterNormalizedMetric,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterProbeMatch {
    pub metric: String,
    pub row_index: usize,
    pub encoder_label: String,
    pub target: f64,
    pub tolerance: f64,
    pub encoding: String,
    pub count: usize,
    pub examples: Vec<RawCounterProbeExample>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RawCounterProbeExample {
    pub offset: usize,
    pub page_4k: usize,
    pub value: f64,
    pub record_tag: Option<String>,
    pub record_size: Option<usize>,
}

pub fn probe_raw_counters(
    trace: &TraceBundle,
    csv_path: Option<PathBuf>,
    metric_filter: Option<&str>,
    scan_files: bool,
) -> crate::Result<RawCounterProbeReport> {
    let profiler_directory = profiler::find_profiler_directory(&trace.path)
        .ok_or_else(|| crate::Error::NotFound(trace.path.clone()))?;
    let csv_data = xcode_counters::parse(trace, csv_path).ok();
    let mut targets = Vec::new();
    if let Some(csv_data) = &csv_data {
        for row in &csv_data.encoders {
            for (metric, value) in &row.counters {
                if let Some(filter) = metric_filter
                    && !metric.contains(filter)
                {
                    continue;
                }
                if is_probe_metric(metric) && value.is_finite() {
                    targets.push(RawCounterProbeTarget {
                        metric: metric.clone(),
                        row_index: row.index,
                        encoder_label: row.encoder_label.clone(),
                        value: *value,
                        tolerance: raw_probe_tolerance(*value),
                    });
                }
            }
        }
    }

    let mut file_reports = Vec::new();
    if scan_files {
        let mut files = fs::read_dir(&profiler_directory)?
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let name = entry.file_name().to_string_lossy().into_owned();
                let file_index = name
                    .strip_prefix("Counters_f_")
                    .and_then(|rest| rest.strip_suffix(".raw"))
                    .and_then(|rest| rest.parse::<usize>().ok())?;
                path.is_file().then_some((file_index, name, path))
            })
            .collect::<Vec<_>>();
        files.sort_by_key(|(file_index, _, _)| *file_index);

        for (file_index, file_name, path) in files {
            let data = fs::read(path)?;
            let markers = find_raw_counter_markers(&data);
            let top_record_shapes = summarize_raw_counter_shapes(&data, &markers);
            let matches = probe_counter_targets(&data, &markers, &targets);
            file_reports.push(RawCounterProbeFile {
                file_index,
                file_name,
                byte_len: data.len(),
                page_count_4k: (data.len() % 4096 == 0).then_some(data.len() / 4096),
                marker_count: markers.len(),
                top_record_shapes,
                matches,
            });
        }
    }

    let normalized_counters = probe_normalized_counter_metrics(&trace.path);
    let normalized_matches = match_normalized_counter_targets(&normalized_counters, &targets);
    let structured_samples = probe_structured_counter_samples(&trace.path, &targets);

    Ok(RawCounterProbeReport {
        profiler_directory,
        csv_source: csv_data.map(|data| data.source),
        targets,
        aggregate_metadata: probe_aggregate_counter_metadata(&trace.path),
        counter_info: probe_counter_info_entries(&trace.path),
        stream_archives: probe_stream_archives(&trace.path),
        structured_layouts: probe_structured_counter_layouts(&trace.path),
        normalized_counters,
        normalized_matches,
        structured_samples,
        files: file_reports,
    })
}

pub fn raw_counters_report(trace: &TraceBundle) -> crate::Result<RawCountersReport> {
    let mut timings = Vec::new();
    let profiler_directory_start = Instant::now();
    let profiler_directory = profiler::find_profiler_directory(&trace.path)
        .ok_or_else(|| crate::Error::NotFound(trace.path.clone()))?;
    timings.push(raw_counter_decode_timing(
        "find profiler directory",
        profiler_directory_start,
    ));
    let stream_groups =
        measure_raw_counter_stage(&mut timings, "load streamData APS archives", || {
            load_stream_archive_groups(&trace.path).unwrap_or_default()
        });
    let (schema_map, fallback_counter_names, sample_blobs) =
        measure_raw_counter_stage(&mut timings, "decode counter schemas/sample blobs", || {
            counter_schemas_and_sample_blobs_from_groups(&stream_groups)
        });
    let catalog = measure_raw_counter_stage(&mut timings, "load AGX counter catalog", || {
        load_agx_counter_catalog()
    });
    let profiler_summary =
        measure_raw_counter_stage(&mut timings, "profiler streamData summary", || {
            profiler::stream_data_summary(&trace.path).ok()
        });
    let aggregate_metadata =
        measure_raw_counter_stage(&mut timings, "decode APS aggregate metadata", || {
            probe_aggregate_counter_metadata_from_groups(&stream_groups)
        });
    let dispatch_alignment = measure_raw_counter_stage(
        &mut timings,
        "align raw counter/profiler timestamps",
        || {
            raw_counter_dispatch_alignment_report(
                &sample_blobs,
                profiler_summary
                    .as_ref()
                    .map(|summary| summary.dispatches.as_slice())
                    .unwrap_or(&[]),
                &aggregate_metadata,
            )
        },
    );
    let sample_trace_indices =
        measure_raw_counter_stage(&mut timings, "decode sample trace indices", || {
            probe_sample_trace_indices_from_groups(&stream_groups)
        });
    let trace_maps = measure_raw_counter_stage(&mut timings, "decode trace maps", || {
        probe_trace_maps_from_groups(&stream_groups)
    });
    let program_address_mappings =
        measure_raw_counter_stage(&mut timings, "decode program address mappings", || {
            probe_program_address_mappings_from_groups(&stream_groups)
        });
    let profiling_address_summary =
        measure_raw_counter_stage(&mut timings, "scan Profiling_f address hits", || {
            probe_profiling_addresses(trace).ok()
        });
    let counter_info =
        measure_raw_counter_stage(&mut timings, "decode counter info entries", || {
            probe_counter_info_entries_from_groups(&stream_groups)
        });
    let structured_layouts =
        measure_raw_counter_stage(&mut timings, "decode structured counter layouts", || {
            probe_structured_counter_layouts_from_groups(&stream_groups)
        });
    let normalized_counters =
        measure_raw_counter_stage(&mut timings, "normalize GPRW counter metrics", || {
            probe_normalized_counter_metrics_from_parts(
                &schema_map,
                &fallback_counter_names,
                &sample_blobs,
            )
        });
    let encoder_sample_metrics =
        measure_raw_counter_stage(&mut timings, "build encoder sample metrics", || {
            raw_counter_encoder_sample_metrics_from_parts(
                &schema_map,
                &fallback_counter_names,
                &sample_blobs,
                &aggregate_metadata,
                &catalog,
            )
        });
    let js_variables = measure_raw_counter_stage(&mut timings, "aggregate JS variables", || {
        raw_counter_js_variables_from_parts(&schema_map, &fallback_counter_names, &sample_blobs)
    });
    let device_identifier = measure_raw_counter_stage(&mut timings, "identify AGX device", || {
        trace_agx_device_identifier(&trace.path)
    });
    let js_variable_groups = measure_raw_counter_stage(
        &mut timings,
        "group JS variables by encoder/dispatch",
        || {
            raw_counter_js_variable_groups_from_parts(
                &schema_map,
                &fallback_counter_names,
                &sample_blobs,
                &aggregate_metadata,
                profiler_summary
                    .as_ref()
                    .map(|summary| summary.dispatches.as_slice())
                    .unwrap_or(&[]),
            )
        },
    );
    let derived_metrics =
        measure_raw_counter_stage(&mut timings, "evaluate AGX derived metrics", || {
            evaluate_agx_derived_metrics(&catalog, &js_variables, device_identifier.as_deref())
        });
    let grouped_derived_metrics =
        measure_raw_counter_stage(&mut timings, "evaluate grouped AGX derived metrics", || {
            evaluate_agx_derived_metric_groups(
                &catalog,
                js_variable_groups,
                device_identifier.as_deref(),
            )
        });
    let shifted_dispatch_metric_candidates = measure_raw_counter_stage(
        &mut timings,
        "evaluate shifted dispatch counter candidates",
        || {
            raw_counter_shifted_dispatch_metric_candidates(
                &schema_map,
                &fallback_counter_names,
                &sample_blobs,
                profiler_summary
                    .as_ref()
                    .map(|summary| summary.dispatches.as_slice())
                    .unwrap_or(&[]),
                dispatch_alignment.as_ref(),
                &catalog,
                device_identifier.as_deref(),
            )
        },
    );
    let schemas = measure_raw_counter_stage(&mut timings, "materialize schemas", || {
        schema_map
            .iter()
            .map(|(sample_group, counter_names)| RawCounterSchema {
                sample_group: *sample_group,
                counter_count: counter_names.len(),
                counter_names: counter_names.clone(),
            })
            .collect::<Vec<_>>()
    });
    let streams = measure_raw_counter_stage(&mut timings, "materialize streams", || {
        structured_layouts
            .iter()
            .map(|layout| {
                let path_ids = parse_derived_counter_sample_path(&layout.path);
                RawCounterDecodedStream {
                    path: layout.path.clone(),
                    sample_group: path_ids.sample_group,
                    source_index: path_ids.source_index,
                    ring_index: path_ids.ring_index,
                    byte_len: layout.byte_len,
                    record_size: layout.gprw_record_size,
                    record_count: layout.gprw_record_count,
                    counter_count: path_ids
                        .sample_group
                        .and_then(|group| schema_map.get(&group))
                        .map(Vec::len),
                }
            })
            .collect::<Vec<_>>()
    });
    let metrics = measure_raw_counter_stage(&mut timings, "materialize decoded metrics", || {
        normalized_counters
            .into_iter()
            .map(|metric| {
                let derived_counter_matches = catalog
                    .derived_by_hash
                    .get(&metric.raw_name)
                    .cloned()
                    .unwrap_or_default();
                let hardware_selectors = catalog
                    .hardware_by_hash
                    .get(&metric.raw_name)
                    .cloned()
                    .unwrap_or_default();
                RawCounterDecodedMetric {
                    path: metric.path,
                    sample_group: metric.sample_group,
                    source_index: metric.source_index,
                    ring_index: metric.ring_index,
                    counter_index: metric.counter_index,
                    raw_name: metric.raw_name,
                    sample_count: metric.sample_count,
                    min_percent_of_gpu_cycles: metric.min_percent,
                    mean_percent_of_gpu_cycles: metric.mean_percent,
                    max_percent_of_gpu_cycles: metric.max_percent,
                    encoder_ids: metric.encoder_ids,
                    kick_trace_ids: metric.kick_trace_ids,
                    source_ids: metric.source_ids,
                    derived_counter_matches,
                    hardware_selectors,
                }
            })
            .collect::<Vec<_>>()
    });
    let mut warnings = Vec::new();
    if schemas.is_empty() && !fallback_counter_names.is_empty() {
        warnings.push(
            "using fallback limiter sample counter names; per-pass schemas were not found"
                .to_owned(),
        );
    }
    if aggregate_metadata.is_empty() {
        warnings.push("no aggregate APSCounterData metadata was decoded".to_owned());
    }
    if metrics.is_empty() {
        warnings.push("no normalized GPRWCNTR counter metrics were decoded".to_owned());
    }
    if !js_variables.is_empty() && derived_metrics.is_empty() {
        warnings.push("no AGX JavaScript derived counters evaluated to finite values".to_owned());
    }
    if profiler_summary
        .as_ref()
        .is_some_and(|summary| !summary.dispatches.is_empty())
        && grouped_derived_metrics
            .iter()
            .all(|group| group.group_kind != "profiler_dispatch")
        && grouped_derived_metrics
            .iter()
            .any(|group| group.group_kind == "sample_group")
    {
        warnings.push(
            "raw counters are present as streamData GPRWCNTR sample-group/encoder aggregates, \
             but their timestamps did not overlap profiler dispatch tick windows; \
             dispatch-level derived counters are unavailable for this bundle"
                .to_owned(),
        );
    }

    Ok(RawCountersReport {
        trace_source: trace.path.clone(),
        profiler_directory,
        timings,
        dispatch_alignment,
        aggregate_metadata,
        sample_trace_indices,
        trace_maps,
        program_address_mappings,
        profiling_address_summary,
        counter_info,
        schemas,
        streams,
        metrics,
        derived_metrics,
        grouped_derived_metrics,
        shifted_dispatch_metric_candidates,
        encoder_sample_metrics,
        warnings,
    })
}

fn measure_raw_counter_stage<T>(
    timings: &mut Vec<RawCounterDecodeTiming>,
    stage: &str,
    build: impl FnOnce() -> T,
) -> T {
    let start = Instant::now();
    let value = build();
    timings.push(raw_counter_decode_timing(stage, start));
    value
}

fn raw_counter_decode_timing(stage: &str, start: Instant) -> RawCounterDecodeTiming {
    RawCounterDecodeTiming {
        stage: stage.to_owned(),
        ms: start.elapsed().as_secs_f64() * 1_000.0,
    }
}

pub fn probe_profiling_addresses(
    trace: &TraceBundle,
) -> crate::Result<ProfilingAddressProbeReport> {
    let profiler_directory = profiler::find_profiler_directory(&trace.path)
        .ok_or_else(|| crate::Error::NotFound(trace.path.clone()))?;
    let mappings = probe_program_address_mappings(&trace.path);
    let ranges = profiling_address_ranges(&mappings);
    let low32_ranges = profiling_low32_address_ranges(&mappings);
    let mut full_hits = vec![0usize; mappings.len()];
    let mut low32_hits = vec![0usize; mappings.len()];
    let mut first_full_offsets = vec![None; mappings.len()];
    let mut first_low32_offsets = vec![None; mappings.len()];
    let mut file_summaries = Vec::new();

    let mut files = fs::read_dir(&profiler_directory)?
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            let file_index = name
                .strip_prefix("Profiling_f_")
                .and_then(|rest| rest.strip_suffix(".raw"))
                .and_then(|rest| rest.parse::<usize>().ok())?;
            path.is_file().then_some((file_index, name, path))
        })
        .collect::<Vec<_>>();
    files.sort_by_key(|(file_index, _, _)| *file_index);

    for probe in probe_profiling_address_files(files, &ranges, &low32_ranges, mappings.len())? {
        for mapping_index in 0..mappings.len() {
            full_hits[mapping_index] += probe.full_hits[mapping_index];
            low32_hits[mapping_index] += probe.low32_hits[mapping_index];
            if first_full_offsets[mapping_index].is_none() {
                first_full_offsets[mapping_index] = probe.first_full_offsets[mapping_index];
            }
            if first_low32_offsets[mapping_index].is_none() {
                first_low32_offsets[mapping_index] = probe.first_low32_offsets[mapping_index];
            }
        }
        file_summaries.push(probe.file_summary);
    }

    Ok(ProfilingAddressProbeReport {
        trace_source: trace.path.clone(),
        profiler_directory,
        mapping_count: mappings.len(),
        file_summaries,
        top_full_address_hits: top_profiling_address_hits(
            &mappings,
            &full_hits,
            &first_full_offsets,
            "u64",
        ),
        top_low32_address_hits: top_profiling_address_hits(
            &mappings,
            &low32_hits,
            &first_low32_offsets,
            "low32",
        ),
        top_shader_low32_hits: top_shader_low32_hits(trace, &mappings, &low32_hits),
        top_function_low32_hits: top_function_low32_hits(trace, &mappings, &low32_hits),
    })
}

fn probe_profiling_address_files(
    files: Vec<(usize, String, PathBuf)>,
    ranges: &[ProfilingAddressRange],
    low32_ranges: &[ProfilingLow32AddressRange],
    mapping_count: usize,
) -> crate::Result<Vec<ProfilingAddressFileProbe>> {
    if files.is_empty() {
        return Ok(Vec::new());
    }
    let worker_count = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(files.len())
        .max(1);
    let chunk_size = files.len().div_ceil(worker_count);
    let mut probes = Vec::with_capacity(files.len());

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in files.chunks(chunk_size) {
            let jobs = chunk.to_vec();
            handles.push(scope.spawn(move || {
                jobs.into_iter()
                    .map(|(file_index, file_name, path)| {
                        probe_profiling_address_file(
                            file_index,
                            file_name,
                            &path,
                            ranges,
                            low32_ranges,
                            mapping_count,
                        )
                    })
                    .collect::<crate::Result<Vec<_>>>()
            }));
        }

        for handle in handles {
            let mut result = handle
                .join()
                .map_err(|_| crate::Error::InvalidTrace("profiling address worker panicked"))??;
            probes.append(&mut result);
        }
        Ok::<(), crate::Error>(())
    })?;

    probes.sort_by_key(|probe| probe.file_summary.file_index);
    Ok(probes)
}

fn probe_profiling_address_file(
    file_index: usize,
    file_name: String,
    path: &Path,
    ranges: &[ProfilingAddressRange],
    low32_ranges: &[ProfilingLow32AddressRange],
    mapping_count: usize,
) -> crate::Result<ProfilingAddressFileProbe> {
    let data = fs::read(path)?;
    let mut full_hits = vec![0usize; mapping_count];
    let mut low32_hits = vec![0usize; mapping_count];
    let mut first_full_offsets = vec![None; mapping_count];
    let mut first_low32_offsets = vec![None; mapping_count];
    let mut file_full_hits = 0usize;
    let mut file_low32_hits = 0usize;

    for offset in (0..data.len().saturating_sub(7)).step_by(4) {
        let value = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
        if let Some(mapping_index) = find_address_range(ranges, value) {
            full_hits[mapping_index] += 1;
            file_full_hits += 1;
            first_full_offsets[mapping_index].get_or_insert(offset);
        }
    }

    for offset in (0..data.len().saturating_sub(3)).step_by(4) {
        let value = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        if let Some(mapping_index) = find_low32_address_range(low32_ranges, value) {
            low32_hits[mapping_index] += 1;
            file_low32_hits += 1;
            first_low32_offsets[mapping_index].get_or_insert(offset);
        }
    }

    Ok(ProfilingAddressFileProbe {
        file_summary: ProfilingAddressFileSummary {
            file_index,
            file_name,
            byte_len: data.len(),
            full_address_hits: file_full_hits,
            low32_address_hits: file_low32_hits,
        },
        full_hits,
        low32_hits,
        first_full_offsets,
        first_low32_offsets,
    })
}

pub fn format_profiling_address_probe(report: &ProfilingAddressProbeReport) -> String {
    let mut out = String::new();
    out.push_str("Profiling address probe\n");
    out.push_str(&format!(
        "trace={} profiler_directory={} mappings={}\n\n",
        report.trace_source.display(),
        report.profiler_directory.display(),
        report.mapping_count
    ));
    out.push_str("files:\n");
    for file in &report.file_summaries {
        out.push_str(&format!(
            "  {:>2} {:<18} bytes={} u64_hits={} low32_hits={}\n",
            file.file_index,
            file.file_name,
            file.byte_len,
            file.full_address_hits,
            file.low32_address_hits
        ));
    }
    format_profiling_address_hits(
        &mut out,
        "top u64 address hits",
        &report.top_full_address_hits,
    );
    format_profiling_address_hits(
        &mut out,
        "top low32 address hits",
        &report.top_low32_address_hits,
    );
    out.push_str("\ntop shader low32 hits:\n");
    if report.top_shader_low32_hits.is_empty() {
        out.push_str("  none\n");
    }
    for hit in report.top_shader_low32_hits.iter().take(32) {
        out.push_str(&format!(
            "  hits={} dispatch={} encoder={} enc_trace_id={} mappings={} function={}\n",
            hit.hit_count,
            hit.dispatch_index,
            format_optional_u64(hit.encoder_index),
            format_optional_u64(hit.encoder_trace_id),
            hit.mapping_count,
            hit.function_name.as_deref().unwrap_or("-")
        ));
    }
    out.push_str("\ntop function low32 hits:\n");
    if report.top_function_low32_hits.is_empty() {
        out.push_str("  none\n");
    }
    for hit in report.top_function_low32_hits.iter().take(32) {
        out.push_str(&format!(
            "  hits={} dispatches={} mappings={} function={}\n",
            hit.hit_count, hit.dispatch_count, hit.mapping_count, hit.function_name
        ));
    }
    out
}

pub fn format_raw_counters_report(report: &RawCountersReport) -> String {
    let mut out = String::new();
    out.push_str("Raw counter report\n");
    out.push_str(&format!(
        "trace={} profiler_directory={}\n",
        report.trace_source.display(),
        report.profiler_directory.display()
    ));
    out.push_str(&format!(
        "metadata={} schemas={} streams={} metrics={} derived_metrics={} derived_groups={}\n\n",
        report.aggregate_metadata.len(),
        report.schemas.len(),
        report.streams.len(),
        report.metrics.len(),
        report.derived_metrics.len(),
        report.grouped_derived_metrics.len()
    ));
    if !report.timings.is_empty() {
        out.push_str("Decode timings:\n");
        for timing in &report.timings {
            out.push_str(&format!("  {}: {:.1} ms\n", timing.stage, timing.ms));
        }
        out.push('\n');
    }
    for warning in &report.warnings {
        out.push_str(&format!("warning: {warning}\n"));
    }
    if !report.warnings.is_empty() {
        out.push('\n');
    }
    if let Some(alignment) = &report.dispatch_alignment {
        out.push_str("Counter/dispatch timestamp alignment:\n");
        out.push_str(&format!(
            "  profiler dispatches={} ticks={}-{} span={} ({})\n",
            alignment.dispatch_count,
            alignment.dispatch_start_ticks,
            alignment.dispatch_end_ticks,
            alignment.dispatch_span_ticks,
            format_optional_duration_ns(alignment.dispatch_span_ns)
        ));
        out.push_str(&format!(
            "  raw counter records={} ticks={}-{} span={}\n",
            alignment.raw_record_count,
            alignment.raw_start_ticks,
            alignment.raw_end_ticks,
            format_ticks_with_duration(alignment.raw_span_ticks, alignment.raw_span_ns)
        ));
        out.push_str(&format!(
            "  direct_overlap_records={} before_dispatch={} after_dispatch={}\n",
            alignment.direct_overlap_record_count,
            alignment.raw_before_dispatch_record_count,
            alignment.raw_after_dispatch_record_count
        ));
        out.push_str(&format!(
            "  raw_end_to_dispatch_start_gap={} ({}) raw_start_to_dispatch_start_delta={} ({})\n",
            alignment.raw_end_to_dispatch_start_gap_ticks,
            format_optional_duration_ns(alignment.raw_end_to_dispatch_start_gap_ns),
            alignment.raw_start_to_dispatch_start_delta_ticks,
            format_optional_duration_ns(alignment.raw_start_to_dispatch_start_delta_ns)
        ));
        if !alignment.candidate_shifts.is_empty() {
            out.push_str("  diagnostic candidate shifts (not applied to counter joins):\n");
            for candidate in &alignment.candidate_shifts {
                out.push_str(&format!(
                    "    {} shift={} ({}) shifted_ticks={}-{} overlap_records={} dispatches_covered={}\n",
                    candidate.name,
                    candidate.shift_ticks,
                    format_optional_duration_ns(candidate.shift_ns),
                    candidate.shifted_raw_start_ticks,
                    candidate.shifted_raw_end_ticks,
                    candidate.overlap_record_count,
                    candidate.dispatches_covered
                ));
            }
        }
        if !alignment.source_spans.is_empty() {
            out.push_str("  nearest raw source spans:\n");
            for span in alignment.source_spans.iter().take(12) {
                out.push_str(&format!(
                    "    group={} source={} rings={:?} records={} ticks={}-{} gap={} ({}) overlap_records={}\n",
                    format_optional_usize(span.sample_group),
                    format_optional_usize(span.source_index),
                    span.ring_indices,
                    span.record_count,
                    span.start_ticks,
                    span.end_ticks,
                    span.nearest_dispatch_gap_ticks,
                    format_optional_duration_ns(span.nearest_dispatch_gap_ns),
                    span.direct_overlap_record_count
                ));
            }
        }
        if alignment.direct_overlap_record_count == 0 {
            out.push_str("  no raw counter records fall inside profiler dispatch windows; dispatch-level raw counters are not emitted.\n");
        }
        out.push('\n');
    }
    for metadata in &report.aggregate_metadata {
        out.push_str(&format!(
            "APSCounterData[{}]: timebase={}/{} num_encoders={} perf={:?}\n",
            metadata.archive_index,
            metadata
                .timebase_numer
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned()),
            metadata
                .timebase_denom
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned()),
            metadata
                .num_encoders
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned()),
            metadata.perf_info
        ));
    }
    if !report.sample_trace_indices.is_empty() {
        out.push_str("\nTraceId to SampleIndex:\n");
        for entry in report.sample_trace_indices.iter().take(16) {
            out.push_str(&format!(
                "  APSData[{}] trace_id={} sample_index={} words={:?}\n",
                entry.archive_index,
                entry.trace_id,
                entry
                    .sample_index
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                entry.words
            ));
        }
    }
    if !report.trace_maps.is_empty() {
        out.push_str("\nTrace maps:\n");
        for entry in report.trace_maps.iter().take(32) {
            out.push_str(&format!(
                "  APSData[{}] {} trace_id={} value={} words={:?}\n",
                entry.archive_index,
                entry.map_name,
                entry.trace_id,
                entry
                    .scalar_value
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                entry.words
            ));
        }
    }
    if !report.program_address_mappings.is_empty() {
        out.push_str(&format!(
            "\nProgram address mappings: entries={}\n",
            report.program_address_mappings.len()
        ));
        for mapping in report.program_address_mappings.iter().take(32) {
            out.push_str(&format!(
                "  APSData[{}] map={} enc_trace_id={} enc_index={} draw_function={} draw={} shader_index={} type={} binary={} addr={} size={}\n",
                mapping.archive_index,
                mapping.mapping_index,
                format_optional_u64(mapping.encoder_trace_id),
                format_optional_u64(mapping.encoder_index),
                format_optional_u64(mapping.draw_function_index),
                format_optional_u64(mapping.draw_call_index),
                format_optional_u64(mapping.shader_index),
                mapping.mapping_type,
                mapping.binary_unique_id.as_deref().unwrap_or("-"),
                format_optional_u64_hex(mapping.mapped_address),
                format_optional_u64(mapping.mapped_size)
            ));
        }
    }
    if let Some(summary) = &report.profiling_address_summary
        && !summary.top_function_low32_hits.is_empty()
    {
        out.push_str("\nProfiling_f address-derived shader hits (low32 range match):\n");
        for hit in summary.top_function_low32_hits.iter().take(16) {
            out.push_str(&format!(
                "  hits={} dispatches={} mappings={} function={}\n",
                hit.hit_count, hit.dispatch_count, hit.mapping_count, hit.function_name
            ));
        }
    }
    if !report.counter_info.is_empty() {
        out.push_str(&format!(
            "\nCounter Info: entries={} (from APSCounterData metadata)\n",
            report.counter_info.len()
        ));
        for entry in report.counter_info.iter().take(12) {
            let field_keys = entry
                .fields
                .iter()
                .map(|field| field.key.as_str())
                .collect::<Vec<_>>()
                .join("|");
            if field_keys.is_empty() {
                out.push_str(&format!(
                    "  APSCounterData[{}] {} {}\n",
                    entry.archive_index, entry.raw_name, entry.summary
                ));
            } else {
                out.push_str(&format!(
                    "  APSCounterData[{}] {} fields={} {}\n",
                    entry.archive_index, entry.raw_name, field_keys, entry.summary
                ));
            }
        }
    }
    if !report.schemas.is_empty() {
        out.push_str("\nSchemas:\n");
        for schema in &report.schemas {
            out.push_str(&format!(
                "  group {}: counters={}\n",
                schema.sample_group, schema.counter_count
            ));
        }
    }
    if !report.streams.is_empty() {
        out.push_str("\nStreams:\n");
        for stream in report.streams.iter().take(24) {
            out.push_str(&format!(
                "  group={} source={} ring={} records={} record_size={} counters={} {}\n",
                format_optional_usize(stream.sample_group),
                format_optional_usize(stream.source_index),
                format_optional_usize(stream.ring_index),
                stream.record_count,
                stream.record_size,
                stream
                    .counter_count
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                stream.path
            ));
        }
    }
    let plausible_metrics = report
        .metrics
        .iter()
        .filter(|metric| is_percent_like_decoded_metric(metric))
        .collect::<Vec<_>>();
    if !plausible_metrics.is_empty() {
        out.push_str("\nTop percent-like cycle-normalized counters (bounded estimate):\n");
        for metric in plausible_metrics.iter().take(32) {
            out.push_str(&format!(
                "  group={} source={} ring={} [{}] samples={} mean={:.2}% min={:.2}% max={:.2}% {}\n",
                format_optional_usize(metric.sample_group),
                format_optional_usize(metric.source_index),
                format_optional_usize(metric.ring_index),
                metric.counter_index,
                metric.sample_count,
                metric.mean_percent_of_gpu_cycles,
                metric.min_percent_of_gpu_cycles,
                metric.max_percent_of_gpu_cycles,
                format_raw_counter_metric_percent_label(metric)
            ));
        }
    }
    if !report.derived_metrics.is_empty() {
        let mut derived_metrics = report
            .derived_metrics
            .iter()
            .filter(|metric| metric.value.is_finite())
            .collect::<Vec<_>>();
        derived_metrics.sort_by(|left, right| {
            right
                .value
                .abs()
                .partial_cmp(&left.value.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.name.cmp(&right.name))
        });
        out.push_str("\nTop AGX JavaScript-derived counters:\n");
        for metric in derived_metrics.iter().take(32) {
            out.push_str(&format!(
                "  {:>12.4} {} ({}) type={} script={}\n",
                metric.value,
                metric.name,
                metric.key,
                metric.counter_type.as_deref().unwrap_or("-"),
                metric
                    .source_script
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("-")
            ));
        }
    }
    if !report.grouped_derived_metrics.is_empty() {
        let mut groups = report
            .grouped_derived_metrics
            .iter()
            .filter(|group| !group.derived_metrics.is_empty())
            .collect::<Vec<_>>();
        groups.sort_by(|left, right| {
            group_max_abs_derived_value(right)
                .partial_cmp(&group_max_abs_derived_value(left))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.record_count.cmp(&left.record_count))
                .then_with(|| left.group_kind.cmp(&right.group_kind))
                .then_with(|| left.group_id.cmp(&right.group_id))
        });
        out.push_str("\nTop grouped AGX JavaScript-derived counters:\n");
        for group in groups.iter().take(16) {
            out.push_str(&format!(
                "  {}={} records={} ticks={}-{} rings={:?}",
                group.group_kind,
                group.group_id,
                group.record_count,
                group
                    .start_ticks
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                group
                    .end_ticks
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                group.ring_indices
            ));
            if let Some(sample_index) = group.encoder_sample_index {
                out.push_str(&format!(
                    " sample_row={} sample_index={sample_index}",
                    group
                        .encoder_sample_row_index
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "-".to_owned())
                ));
            }
            if let Some(name) = group.profiler_function_name.as_deref() {
                out.push_str(&format!(
                    " dispatch={} function={}",
                    group
                        .profiler_dispatch_index
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "-".to_owned()),
                    name
                ));
            }
            out.push('\n');
            for metric in top_derived_metrics(&group.derived_metrics, 6) {
                out.push_str(&format!(
                    "    {:>12.4} {} ({}) type={}\n",
                    metric.value,
                    metric.name,
                    metric.key,
                    metric.counter_type.as_deref().unwrap_or("-")
                ));
            }
        }
    }
    if !report.shifted_dispatch_metric_candidates.is_empty() {
        out.push_str(
            "\nShifted dispatch AGX JavaScript-derived counter candidates (diagnostic only):\n",
        );
        for candidate in &report.shifted_dispatch_metric_candidates {
            out.push_str(&format!(
                "  candidate={} shift={} ({}) overlap_records={} dispatches_covered={}\n",
                candidate.name,
                candidate.shift_ticks,
                format_optional_duration_ns(candidate.shift_ns),
                candidate.overlap_record_count,
                candidate.dispatches_covered
            ));
            for group in candidate.groups.iter().take(12) {
                out.push_str(&format!(
                    "    dispatch={} function={} records={} ticks={}-{}\n",
                    group
                        .profiler_dispatch_index
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "-".to_owned()),
                    group.profiler_function_name.as_deref().unwrap_or("-"),
                    group.record_count,
                    group
                        .start_ticks
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "-".to_owned()),
                    group
                        .end_ticks
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "-".to_owned())
                ));
                let mut metrics =
                    top_derived_metrics_matching(&group.derived_metrics, 4, |metric| {
                        shifted_dispatch_metric_is_interesting(metric)
                    });
                if metrics.is_empty() {
                    metrics = top_derived_metrics(&group.derived_metrics, 4);
                }
                for metric in metrics {
                    out.push_str(&format!(
                        "      {:>12.4} {} ({}) type={}\n",
                        metric.value,
                        metric.name,
                        metric.key,
                        metric.counter_type.as_deref().unwrap_or("-")
                    ));
                }
            }
        }
    }
    let encoder_sample_groups = report
        .grouped_derived_metrics
        .iter()
        .filter(|group| group.group_kind == "encoder_sample" && !group.derived_metrics.is_empty())
        .collect::<Vec<_>>();
    if !encoder_sample_groups.is_empty() {
        out.push_str("\nEncoder sample AGX JavaScript-derived counters:\n");
        for group in encoder_sample_groups {
            out.push_str(&format!(
                "  row={} sample_index={} records={} ticks={}-{}\n",
                group
                    .encoder_sample_row_index
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                group
                    .encoder_sample_index
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                group.record_count,
                group
                    .start_ticks
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                group
                    .end_ticks
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned())
            ));
            let percentages = top_derived_metrics_matching(&group.derived_metrics, 8, |metric| {
                metric.counter_type.as_deref() == Some("Percentage")
            });
            if !percentages.is_empty() {
                out.push_str("    percentages:\n");
                for metric in percentages {
                    out.push_str(&format!(
                        "      {:>10.4}% {} ({})\n",
                        metric.value, metric.name, metric.key
                    ));
                }
            }
            let rates_and_counts =
                top_derived_metrics_matching(&group.derived_metrics, 6, |metric| {
                    metric.counter_type.as_deref() != Some("Percentage")
                });
            if !rates_and_counts.is_empty() {
                out.push_str("    rates/counts:\n");
                for metric in rates_and_counts {
                    out.push_str(&format!(
                        "      {:>12.4} {} ({}) type={}\n",
                        metric.value,
                        metric.name,
                        metric.key,
                        metric.counter_type.as_deref().unwrap_or("-")
                    ));
                }
            }
        }
    }
    if !report.encoder_sample_metrics.is_empty() {
        out.push_str("\nEncoder sample raw counters:\n");
        let rows = report
            .encoder_sample_metrics
            .iter()
            .map(|metric| metric.row_index)
            .collect::<BTreeSet<_>>();
        for row_index in rows.iter().take(8) {
            let row_metrics = report
                .encoder_sample_metrics
                .iter()
                .filter(|metric| metric.row_index == *row_index)
                .collect::<Vec<_>>();
            let sample_index = row_metrics
                .first()
                .map(|metric| metric.sample_index)
                .unwrap_or_default();
            out.push_str(&format!(
                "  row={} sample_index={} metrics={}\n",
                row_index,
                sample_index,
                row_metrics.len()
            ));
            let mut percent_like = row_metrics
                .iter()
                .copied()
                .filter(|metric| is_percent_like_encoder_sample_metric(metric))
                .collect::<Vec<_>>();
            percent_like.sort_by(|left, right| {
                right
                    .normalized_percent
                    .unwrap_or(0.0)
                    .abs()
                    .partial_cmp(&left.normalized_percent.unwrap_or(0.0).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| right.raw_delta.cmp(&left.raw_delta))
            });
            if !percent_like.is_empty() {
                out.push_str("    bounded percent-like counters:\n");
                for metric in percent_like.into_iter().take(8) {
                    out.push_str(&format!(
                        "      group={} source={} ring={} [{}] samples={} value={:.2}% raw={} {}\n",
                        format_optional_usize(metric.sample_group),
                        format_optional_usize(metric.source_index),
                        format_optional_usize(metric.ring_index),
                        metric.counter_index,
                        metric.sample_count,
                        metric.normalized_percent.unwrap_or(0.0),
                        metric.raw_delta,
                        format_raw_counter_encoder_sample_percent_label(metric)
                    ));
                }
            }
            let mut raw_or_rate = row_metrics
                .into_iter()
                .filter(|metric| !is_percent_like_encoder_sample_metric(metric))
                .collect::<Vec<_>>();
            raw_or_rate.sort_by(|left, right| {
                right.raw_delta.cmp(&left.raw_delta).then_with(|| {
                    right
                        .normalized_percent
                        .unwrap_or(0.0)
                        .abs()
                        .partial_cmp(&left.normalized_percent.unwrap_or(0.0).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            });
            if !raw_or_rate.is_empty() {
                out.push_str("    largest raw/rate/count counters:\n");
            }
            for metric in raw_or_rate.into_iter().take(6) {
                out.push_str(&format!(
                    "      group={} source={} ring={} [{}] samples={} raw={} norm={}\n        {}\n",
                    format_optional_usize(metric.sample_group),
                    format_optional_usize(metric.source_index),
                    format_optional_usize(metric.ring_index),
                    metric.counter_index,
                    metric.sample_count,
                    metric.raw_delta,
                    metric
                        .normalized_percent
                        .map(|value| format!("{value:.2}%"))
                        .unwrap_or_else(|| "-".to_owned()),
                    format_raw_counter_encoder_sample_label(metric)
                ));
            }
        }
    }
    let unbounded_count = report
        .metrics
        .iter()
        .filter(|metric| !is_percent_like_decoded_metric(metric))
        .count();
    if unbounded_count > 0 {
        out.push_str(&format!(
            "\n{unbounded_count} decoded counters are not shown in the percent-like summary; use --format json/csv for the full raw-id table.\n"
        ));
    }
    out
}

fn is_percent_like_decoded_metric(metric: &RawCounterDecodedMetric) -> bool {
    metric.mean_percent_of_gpu_cycles.is_finite()
        && metric.mean_percent_of_gpu_cycles >= 0.0
        && metric.mean_percent_of_gpu_cycles <= 500.0
        && metric.max_percent_of_gpu_cycles.is_finite()
        && metric.max_percent_of_gpu_cycles <= 500.0
        && !metric.derived_counter_matches.is_empty()
        && metric.derived_counter_matches.iter().any(|matched| {
            matched.counter_type.as_deref() == Some("Percentage")
                || percent_like_counter_name(&matched.name)
                || percent_like_counter_name(&matched.key)
        })
}

fn format_raw_counter_encoder_sample_label(metric: &RawCounterEncoderSampleMetric) -> String {
    format_counter_match_label(&metric.raw_name, &metric.derived_counter_matches, false)
}

fn format_raw_counter_encoder_sample_percent_label(
    metric: &RawCounterEncoderSampleMetric,
) -> String {
    format_counter_match_label(&metric.raw_name, &metric.derived_counter_matches, true)
}

fn is_percent_like_encoder_sample_metric(metric: &RawCounterEncoderSampleMetric) -> bool {
    let Some(value) = metric.normalized_percent else {
        return false;
    };
    if !value.is_finite() || !(0.0..=500.0).contains(&value) {
        return false;
    }
    if metric.derived_counter_matches.is_empty() {
        return false;
    }
    metric.derived_counter_matches.iter().any(|matched| {
        matched.counter_type.as_deref() == Some("Percentage")
            || percent_like_counter_name(&matched.name)
            || percent_like_counter_name(&matched.key)
    })
}

fn percent_like_counter_name(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    [
        "utilization",
        "limiter",
        "occupancy",
        "miss rate",
        "inefficiency",
        "residency",
        "compression ratio",
        "average overdraw",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

pub fn format_raw_counters_csv(report: &RawCountersReport) -> String {
    let mut out = String::new();
    out.push_str("path,sample_group,source_index,ring_index,counter_index,raw_name,derived_counter_names,derived_counter_keys,hardware_selector_count,sample_count,min_percent_of_gpu_cycles,mean_percent_of_gpu_cycles,max_percent_of_gpu_cycles,encoder_id_count,kick_trace_id_count,source_id_count\n");
    for metric in &report.metrics {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{},{},{}\n",
            csv_escape(&metric.path),
            optional_usize_csv(metric.sample_group),
            optional_usize_csv(metric.source_index),
            optional_usize_csv(metric.ring_index),
            metric.counter_index,
            csv_escape(&metric.raw_name),
            csv_escape(&derived_counter_names(metric)),
            csv_escape(&derived_counter_keys(metric)),
            metric.hardware_selectors.len(),
            metric.sample_count,
            metric.min_percent_of_gpu_cycles,
            metric.mean_percent_of_gpu_cycles,
            metric.max_percent_of_gpu_cycles,
            metric.encoder_ids.len(),
            metric.kick_trace_ids.len(),
            metric.source_ids.len()
        ));
    }
    out
}

fn format_raw_counter_metric_percent_label(metric: &RawCounterDecodedMetric) -> String {
    format_counter_match_label(&metric.raw_name, &metric.derived_counter_matches, true)
}

fn format_counter_match_label(
    raw_name: &str,
    matches: &[RawCounterDerivedCounterMatch],
    prefer_percent_like: bool,
) -> String {
    let filtered = matches
        .iter()
        .filter(|matched| {
            !prefer_percent_like
                || matched.counter_type.as_deref() == Some("Percentage")
                || percent_like_counter_name(&matched.name)
                || percent_like_counter_name(&matched.key)
        })
        .collect::<Vec<_>>();
    let selected = if filtered.is_empty() {
        matches.iter().collect::<Vec<_>>()
    } else {
        filtered
    };
    let Some(first) = selected.first() else {
        return raw_name.to_owned();
    };
    let extra = selected.len().saturating_sub(1);
    if extra == 0 {
        format!("{} ({})", first.name, raw_name)
    } else {
        format!("{} +{} ({})", first.name, extra, raw_name)
    }
}

fn derived_counter_names(metric: &RawCounterDecodedMetric) -> String {
    metric
        .derived_counter_matches
        .iter()
        .map(|matched| matched.name.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join("|")
}

fn derived_counter_keys(metric: &RawCounterDecodedMetric) -> String {
    metric
        .derived_counter_matches
        .iter()
        .map(|matched| matched.key.as_str())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join("|")
}

fn group_max_abs_derived_value(group: &RawCounterJsDerivedMetricGroup) -> f64 {
    group
        .derived_metrics
        .iter()
        .filter_map(|metric| metric.value.is_finite().then_some(metric.value.abs()))
        .fold(0.0, f64::max)
}

fn top_derived_metrics(
    metrics: &[RawCounterJsDerivedMetric],
    limit: usize,
) -> Vec<&RawCounterJsDerivedMetric> {
    top_derived_metrics_matching(metrics, limit, |_| true)
}

fn top_derived_metrics_matching(
    metrics: &[RawCounterJsDerivedMetric],
    limit: usize,
    predicate: impl Fn(&RawCounterJsDerivedMetric) -> bool,
) -> Vec<&RawCounterJsDerivedMetric> {
    let mut sorted = metrics
        .iter()
        .filter(|metric| metric.value.is_finite() && predicate(metric))
        .collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        right
            .value
            .abs()
            .partial_cmp(&left.value.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.name.cmp(&right.name))
            .then_with(|| left.key.cmp(&right.key))
    });
    sorted.truncate(limit);
    sorted
}

fn shifted_dispatch_metric_is_interesting(metric: &RawCounterJsDerivedMetric) -> bool {
    let key = metric.key.to_ascii_lowercase();
    let name = metric.name.to_ascii_lowercase();
    [
        "cs",
        "kernel",
        "alu",
        "invocation",
        "utilization",
        "performance",
    ]
    .iter()
    .any(|needle| key.contains(needle) || name.contains(needle))
}

fn optional_usize_csv(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

#[derive(Debug, Default)]
struct RawCounterCatalog {
    derived_by_hash: BTreeMap<String, Vec<RawCounterDerivedCounterMatch>>,
    hardware_by_hash: BTreeMap<String, Vec<RawCounterHardwareSelector>>,
    derived_definitions: Vec<RawCounterDerivedDefinition>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RawCounterDerivedCounterMatchKey {
    key: String,
    name: String,
    counter_type: Option<String>,
    description: Option<String>,
    unit: Option<String>,
    groups: Vec<String>,
    timeline_groups: Vec<String>,
    visible: Option<bool>,
    batch_filtered: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RawCounterHardwareSelectorKey {
    partition: Option<u64>,
    select: Option<u64>,
    flag: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RawCounterDerivedDefinition {
    key: String,
    name: String,
    counter_type: Option<String>,
    description: Option<String>,
    raw_counters: Vec<String>,
    unit: Option<String>,
    groups: Vec<String>,
    timeline_groups: Vec<String>,
    visible: Option<bool>,
    batch_filtered: Option<bool>,
    source_catalog: PathBuf,
    source_script: Option<PathBuf>,
}

#[derive(Debug, Default, Clone)]
struct RawCounterGraphCatalog {
    entries: BTreeMap<String, RawCounterGraphEntry>,
    aliases: BTreeMap<String, String>,
    groups_by_counter: BTreeMap<String, BTreeSet<String>>,
    timeline_groups_by_counter: BTreeMap<String, BTreeSet<String>>,
}

#[derive(Debug, Default, Clone)]
struct RawCounterGraphEntry {
    key: String,
    name: String,
    description: Option<String>,
    unit: Option<String>,
    vendor_counters: Vec<String>,
    visible: Option<bool>,
    batch_filtered: Option<bool>,
}

#[derive(Debug, Default, Clone)]
struct RawCounterGraphMetadata {
    description: Option<String>,
    unit: Option<String>,
    groups: Vec<String>,
    timeline_groups: Vec<String>,
    visible: Option<bool>,
    batch_filtered: Option<bool>,
}

#[derive(Debug)]
struct RawCounterLoadedDerivedScript {
    path: PathBuf,
    source: String,
    definitions: Vec<RawCounterDerivedDefinition>,
}

fn load_agx_counter_catalog() -> RawCounterCatalog {
    let mut statistics_files = Vec::new();
    let mut perf_files = Vec::new();
    let mut derived_script_files = Vec::new();
    let graph_catalog = load_gpu_counter_graph_catalog();
    collect_agx_counter_catalog_files(
        Path::new("/System/Library/Extensions"),
        &mut statistics_files,
        &mut perf_files,
        &mut derived_script_files,
    );
    let derived_script_by_stem = derived_script_files
        .into_iter()
        .filter_map(|path| agx_statistics_stem(&path).map(|stem| (stem, path)))
        .collect::<BTreeMap<_, _>>();

    let mut derived =
        BTreeMap::<String, BTreeMap<RawCounterDerivedCounterMatchKey, BTreeSet<PathBuf>>>::new();
    let mut derived_definitions = BTreeSet::<RawCounterDerivedDefinition>::new();
    for path in statistics_files {
        let script_path = agx_statistics_stem(&path)
            .and_then(|stem| derived_script_by_stem.get(&stem))
            .cloned();
        add_statistics_counter_catalog(
            &path,
            script_path.as_deref(),
            &graph_catalog,
            &mut derived,
            &mut derived_definitions,
        );
    }

    let mut hardware =
        BTreeMap::<String, BTreeMap<RawCounterHardwareSelectorKey, BTreeSet<PathBuf>>>::new();
    for path in perf_files {
        add_perf_counter_catalog(&path, &mut hardware);
    }

    RawCounterCatalog {
        derived_by_hash: derived
            .into_iter()
            .map(|(hash, matches)| {
                let matches = matches
                    .into_iter()
                    .map(|(key, sources)| RawCounterDerivedCounterMatch {
                        key: key.key,
                        name: key.name,
                        counter_type: key.counter_type,
                        description: key.description,
                        unit: key.unit,
                        groups: key.groups,
                        timeline_groups: key.timeline_groups,
                        visible: key.visible,
                        batch_filtered: key.batch_filtered,
                        sources: sources.into_iter().collect(),
                    })
                    .collect();
                (hash, matches)
            })
            .collect(),
        hardware_by_hash: hardware
            .into_iter()
            .map(|(hash, selectors)| {
                let selectors = selectors
                    .into_iter()
                    .map(|(key, sources)| RawCounterHardwareSelector {
                        partition: key.partition,
                        select: key.select,
                        flag: key.flag,
                        sources: sources.into_iter().collect(),
                    })
                    .collect();
                (hash, selectors)
            })
            .collect(),
        derived_definitions: derived_definitions.into_iter().collect(),
    }
}

fn collect_agx_counter_catalog_files(
    directory: &Path,
    statistics_files: &mut Vec<PathBuf>,
    perf_files: &mut Vec<PathBuf>,
    derived_script_files: &mut Vec<PathBuf>,
) {
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_agx_counter_catalog_files(
                &path,
                statistics_files,
                perf_files,
                derived_script_files,
            );
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|file_name| file_name.to_str()) else {
            continue;
        };
        if file_name.starts_with("AGXMetalStatisticsExternal")
            && file_name.ends_with("-counters.plist")
        {
            statistics_files.push(path);
        } else if file_name.starts_with("AGXMetalStatisticsExternal")
            && file_name.ends_with("-derived.js")
        {
            derived_script_files.push(path);
        } else if file_name == "AGXMetalPerfCountersExternal.plist" {
            perf_files.push(path);
        }
    }
}

fn load_gpu_counter_graph_catalog() -> RawCounterGraphCatalog {
    let mut catalog = RawCounterGraphCatalog::default();
    for path in gpu_counter_graph_paths() {
        add_gpu_counter_graph_catalog(&path, &mut catalog);
    }
    catalog
}

fn gpu_counter_graph_paths() -> Vec<PathBuf> {
    const REL: &str = "PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/GTShaderProfiler.framework/Versions/A/Resources/GPUCounterGraph.plist";
    let mut paths = Vec::new();
    // Prefer the active developer dir (`xcode-select -p` → `<app>/Contents/Developer`;
    // strip `/Developer`), so Xcode-beta / a `DEVELOPER_DIR` override is honored.
    if let Some(contents) = std::process::Command::new("xcode-select")
        .arg("-p")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| PathBuf::from(s.trim()))
        .and_then(|dev| dev.parent().map(|p| p.to_path_buf()))
    {
        paths.push(contents.join(REL));
    }
    // Stock fallback.
    paths.push(PathBuf::from(format!("/Applications/Xcode.app/Contents/{REL}")));
    paths
}

fn add_gpu_counter_graph_catalog(path: &Path, catalog: &mut RawCounterGraphCatalog) {
    let Ok(value) = Value::from_file(path) else {
        return;
    };
    let Some(root) = value.as_dictionary() else {
        return;
    };

    if let Some(groups) = root.get("groups").and_then(Value::as_array) {
        add_gpu_counter_graph_groups(groups, &mut catalog.groups_by_counter);
    }
    if let Some(groups) = root.get("timelineGroups").and_then(Value::as_array) {
        add_gpu_counter_graph_groups(groups, &mut catalog.timeline_groups_by_counter);
    }

    let Some(counters) = root.get("counters").and_then(Value::as_dictionary) else {
        return;
    };
    for (key, value) in counters {
        let Some(counter) = value.as_dictionary() else {
            continue;
        };
        let name = counter
            .get("name")
            .and_then(Value::as_string)
            .unwrap_or(key)
            .to_owned();
        let vendor_counters = counter
            .get("vendorCounters")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_string)
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let entry = RawCounterGraphEntry {
            key: key.clone(),
            name: name.clone(),
            description: counter
                .get("description")
                .and_then(Value::as_string)
                .map(str::to_owned),
            unit: counter
                .get("unit")
                .and_then(Value::as_string)
                .map(str::to_owned),
            vendor_counters,
            visible: counter.get("visible").and_then(plist_bool),
            batch_filtered: counter.get("batchfiltered").and_then(plist_bool),
        };
        catalog.aliases.insert(key.clone(), key.clone());
        catalog.aliases.insert(name.clone(), key.clone());
        for vendor_counter in &entry.vendor_counters {
            catalog.aliases.insert(vendor_counter.clone(), key.clone());
        }
        catalog.entries.insert(key.clone(), entry);
    }
}

fn add_gpu_counter_graph_groups(
    groups: &[Value],
    groups_by_counter: &mut BTreeMap<String, BTreeSet<String>>,
) {
    for group in groups {
        let Some(group) = group.as_dictionary() else {
            continue;
        };
        let Some(name) = group.get("name").and_then(Value::as_string) else {
            continue;
        };
        let Some(counters) = group.get("counters").and_then(Value::as_array) else {
            continue;
        };
        for counter in counters.iter().filter_map(Value::as_string) {
            groups_by_counter
                .entry(counter.to_owned())
                .or_default()
                .insert(name.to_owned());
        }
    }
}

impl RawCounterGraphCatalog {
    fn metadata_for(&self, key: &str, name: &str) -> RawCounterGraphMetadata {
        let entry_key = self
            .aliases
            .get(key)
            .or_else(|| self.aliases.get(name))
            .cloned();
        let entry = entry_key
            .as_deref()
            .and_then(|entry_key| self.entries.get(entry_key));

        let mut groups = BTreeSet::new();
        let mut timeline_groups = BTreeSet::new();
        for candidate in [key, name]
            .into_iter()
            .chain(entry.iter().flat_map(|entry| {
                std::iter::once(entry.key.as_str())
                    .chain(std::iter::once(entry.name.as_str()))
                    .chain(entry.vendor_counters.iter().map(String::as_str))
            }))
        {
            if let Some(values) = self.groups_by_counter.get(candidate) {
                groups.extend(values.iter().cloned());
            }
            if let Some(values) = self.timeline_groups_by_counter.get(candidate) {
                timeline_groups.extend(values.iter().cloned());
            }
        }

        RawCounterGraphMetadata {
            description: entry.and_then(|entry| entry.description.clone()),
            unit: entry.and_then(|entry| entry.unit.clone()),
            groups: groups.into_iter().collect(),
            timeline_groups: timeline_groups.into_iter().collect(),
            visible: entry.and_then(|entry| entry.visible),
            batch_filtered: entry.and_then(|entry| entry.batch_filtered),
        }
    }
}

fn add_statistics_counter_catalog(
    path: &Path,
    script_path: Option<&Path>,
    graph_catalog: &RawCounterGraphCatalog,
    derived: &mut BTreeMap<String, BTreeMap<RawCounterDerivedCounterMatchKey, BTreeSet<PathBuf>>>,
    derived_definitions: &mut BTreeSet<RawCounterDerivedDefinition>,
) {
    let Ok(value) = Value::from_file(path) else {
        return;
    };
    let Some(root) = value.as_dictionary() else {
        return;
    };
    let Some(counters) = root.get("DerivedCounters").and_then(Value::as_dictionary) else {
        return;
    };

    for (key, value) in counters {
        let Some(counter) = value.as_dictionary() else {
            continue;
        };
        let Some(raw_hashes) = counter.get("counters").and_then(Value::as_array) else {
            continue;
        };
        let raw_hashes = raw_hashes
            .iter()
            .filter_map(Value::as_string)
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let name = counter
            .get("name")
            .and_then(Value::as_string)
            .unwrap_or(key)
            .to_owned();
        let counter_type = counter
            .get("type")
            .and_then(Value::as_string)
            .map(str::to_owned);
        let description = counter
            .get("description")
            .and_then(Value::as_string)
            .map(str::to_owned);
        let graph = graph_catalog.metadata_for(key, &name);
        let description = description.or(graph.description);
        let match_key = RawCounterDerivedCounterMatchKey {
            key: key.clone(),
            name: name.clone(),
            counter_type: counter_type.clone(),
            description: description.clone(),
            unit: graph.unit.clone(),
            groups: graph.groups.clone(),
            timeline_groups: graph.timeline_groups.clone(),
            visible: graph.visible,
            batch_filtered: graph.batch_filtered,
        };
        derived_definitions.insert(RawCounterDerivedDefinition {
            key: key.clone(),
            name,
            counter_type,
            description,
            raw_counters: raw_hashes.clone(),
            unit: graph.unit,
            groups: graph.groups,
            timeline_groups: graph.timeline_groups,
            visible: graph.visible,
            batch_filtered: graph.batch_filtered,
            source_catalog: path.to_path_buf(),
            source_script: script_path.map(Path::to_path_buf),
        });
        for raw_hash in raw_hashes {
            derived
                .entry(raw_hash)
                .or_default()
                .entry(match_key.clone())
                .or_default()
                .insert(path.to_path_buf());
        }
    }
}

fn agx_statistics_stem(path: &Path) -> Option<String> {
    let file_name = path.file_name()?.to_str()?;
    file_name
        .strip_suffix("-counters.plist")
        .or_else(|| file_name.strip_suffix("-derived.js"))
        .map(str::to_owned)
}

fn evaluate_agx_derived_metrics(
    catalog: &RawCounterCatalog,
    variables: &BTreeMap<String, f64>,
    device_identifier: Option<&str>,
) -> Vec<RawCounterJsDerivedMetric> {
    if variables.is_empty() {
        return Vec::new();
    }

    let Some(script) = load_agx_derived_script(catalog, device_identifier, Some(variables)) else {
        return Vec::new();
    };
    evaluate_loaded_agx_derived_metrics(&script, variables)
}

fn load_agx_derived_script(
    catalog: &RawCounterCatalog,
    device_identifier: Option<&str>,
    variables: Option<&BTreeMap<String, f64>>,
) -> Option<RawCounterLoadedDerivedScript> {
    let definitions_by_script = catalog
        .derived_definitions
        .iter()
        .filter_map(|definition| {
            definition
                .source_script
                .as_ref()
                .map(|script| (script.clone(), definition.clone()))
        })
        .fold(
            BTreeMap::<PathBuf, Vec<RawCounterDerivedDefinition>>::new(),
            |mut grouped, (script, definition)| {
                grouped.entry(script).or_default().push(definition);
                grouped
            },
        );

    let (script_path, definitions) =
        choose_agx_derived_script(definitions_by_script, device_identifier, variables)?;
    let Ok(script_source) = fs::read_to_string(&script_path) else {
        return None;
    };
    Some(RawCounterLoadedDerivedScript {
        path: script_path,
        source: script_source,
        definitions,
    })
}

fn evaluate_loaded_agx_derived_metrics(
    script: &RawCounterLoadedDerivedScript,
    variables: &BTreeMap<String, f64>,
) -> Vec<RawCounterJsDerivedMetric> {
    if variables.is_empty() {
        return Vec::new();
    }

    let mut metrics =
        evaluate_agx_derived_script(&script.path, &script.source, &script.definitions, variables);
    metrics.sort_by(|left, right| {
        right
            .value
            .abs()
            .partial_cmp(&left.value.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.name.cmp(&right.name))
            .then_with(|| left.key.cmp(&right.key))
    });
    metrics
}

fn evaluate_agx_derived_metric_groups(
    catalog: &RawCounterCatalog,
    groups: Vec<RawCounterJsVariableGroup>,
    device_identifier: Option<&str>,
) -> Vec<RawCounterJsDerivedMetricGroup> {
    groups
        .into_iter()
        .filter_map(|group| {
            let script =
                load_agx_derived_script(catalog, device_identifier, Some(&group.variables))?;
            let metrics = evaluate_loaded_agx_derived_metrics(&script, &group.variables);
            (!metrics.is_empty()).then_some(RawCounterJsDerivedMetricGroup {
                group_kind: group.group_kind,
                group_id: group.group_id,
                encoder_sample_row_index: group.encoder_sample_row_index,
                encoder_sample_index: group.encoder_sample_index,
                sample_group: group.sample_group,
                source_index: group.source_index,
                ring_indices: group.ring_indices,
                start_ticks: group.start_ticks,
                end_ticks: group.end_ticks,
                record_count: group.record_count,
                encoder_ids: group.encoder_ids,
                kick_trace_ids: group.kick_trace_ids,
                source_ids: group.source_ids,
                profiler_dispatch_index: group.profiler_dispatch_index,
                profiler_encoder_index: group.profiler_encoder_index,
                profiler_function_name: group.profiler_function_name,
                profiler_pipeline_id: group.profiler_pipeline_id,
                profiler_start_ticks: group.profiler_start_ticks,
                profiler_end_ticks: group.profiler_end_ticks,
                derived_metrics: metrics,
            })
        })
        .collect()
}

/// The JavaScript derived-counter scripts shipped in the AGX bundles exist
/// only for G13/G14 (M1/M2). Their raw-counter hashes and formulas are
/// generation-specific, so a script is only used for its own generation;
/// newer GPUs go through agxps instead (`hw_counters`).
fn choose_agx_derived_script(
    definitions_by_script: BTreeMap<PathBuf, Vec<RawCounterDerivedDefinition>>,
    device_identifier: Option<&str>,
    variables: Option<&BTreeMap<String, f64>>,
) -> Option<(PathBuf, Vec<RawCounterDerivedDefinition>)> {
    let device_generation = device_identifier.and_then(agx_generation);
    definitions_by_script
        .into_iter()
        .filter(|(path, _)| {
            device_generation.is_none_or(|generation| {
                agx_statistics_stem(path).and_then(|stem| agx_generation(&stem))
                    == Some(generation)
            })
        })
        .max_by(
        |(left_path, left_definitions), (right_path, right_definitions)| {
            agx_derived_script_score(left_path, left_definitions, device_identifier, variables)
                .cmp(&agx_derived_script_score(
                    right_path,
                    right_definitions,
                    device_identifier,
                    variables,
                ))
                .then_with(|| right_path.cmp(left_path))
        },
    )
}

fn agx_derived_script_score(
    path: &Path,
    definitions: &[RawCounterDerivedDefinition],
    device_identifier: Option<&str>,
    variables: Option<&BTreeMap<String, f64>>,
) -> (usize, u8, u8, usize) {
    let stem = agx_statistics_stem(path).unwrap_or_default();
    let raw_hash_overlap = variables
        .map(|variables| {
            definitions
                .iter()
                .flat_map(|definition| definition.raw_counters.iter())
                .filter(|raw_counter| {
                    variables.contains_key(raw_counter.as_str())
                        || variables.contains_key(format!("{raw_counter}_norm").as_str())
                })
                .count()
        })
        .unwrap_or_default();
    let direct_match = device_identifier
        .filter(|identifier| stem.contains(identifier) || identifier.contains(&stem))
        .is_some() as u8;
    let compatibility = agx_derived_script_compatibility_rank(&stem);
    (
        raw_hash_overlap,
        direct_match,
        compatibility,
        definitions.len(),
    )
}

/// GPU generation named by a device identifier (`G16X` -> 16) or a derived
/// script stem (`AGXMetalStatisticsExternalG14S` -> 14; the `A14X` and
/// `13_3` scripts ship in the G13 bundles).
fn agx_generation(identifier: &str) -> Option<u32> {
    if identifier.contains("A14X") || identifier.contains("13_3") {
        return Some(13);
    }
    let bytes = identifier.as_bytes();
    (0..bytes.len().saturating_sub(2))
        .rev()
        .find(|&index| {
            bytes[index] == b'G'
                && bytes[index + 1].is_ascii_digit()
                && bytes[index + 2].is_ascii_digit()
        })
        .and_then(|index| identifier[index + 1..index + 3].parse().ok())
}

fn agx_derived_script_compatibility_rank(stem: &str) -> u8 {
    match () {
        _ if stem.contains("G14D") => 40,
        _ if stem.contains("G14C") => 35,
        _ if stem.contains("G14S") => 30,
        _ if stem.contains("G14G") => 25,
        _ if stem.contains("A14X") => 15,
        _ if stem.contains("13_3") => 10,
        _ => 0,
    }
}

fn trace_agx_device_identifier(trace_path: &Path) -> Option<String> {
    let profiler_directory = profiler::find_profiler_directory(trace_path)?;
    let stream_data = profiler_directory.join("streamData");
    let data = fs::read(stream_data).ok()?;
    let text = String::from_utf8_lossy(&data);
    // Driver bundle names such as `AGXMetalG16G_B0`: keep the `G16G` part.
    text.match_indices("AGXMetal").find_map(|(index, _)| {
        let identifier = text[index + "AGXMetal".len()..]
            .chars()
            .take_while(char::is_ascii_alphanumeric)
            .collect::<String>();
        (identifier.starts_with('G') && agx_generation(&identifier).is_some())
            .then_some(identifier)
    })
}

fn evaluate_agx_derived_script(
    script_path: &Path,
    script_source: &str,
    definitions: &[RawCounterDerivedDefinition],
    variables: &BTreeMap<String, f64>,
) -> Vec<RawCounterJsDerivedMetric> {
    let Ok(runtime) = Runtime::new() else {
        return Vec::new();
    };
    let Ok(context) = Context::full(&runtime) else {
        return Vec::new();
    };

    context.with(|ctx| {
        let globals = ctx.globals();
        for (name, value) in variables {
            let _ = globals.set(name.as_str(), *value);
        }
        if ctx.eval::<(), _>(script_source).is_err() {
            return Vec::new();
        }

        let mut metrics = Vec::new();
        for definition in definitions {
            if !is_javascript_identifier(&definition.key) {
                continue;
            }
            let source = format!(
                "(function() {{ try {{ if (typeof {key} !== 'function') return null; const v = {key}(); return Number.isFinite(v) ? v : null; }} catch (e) {{ return null; }} }})()",
                key = definition.key
            );
            let Ok(value) = ctx.eval::<Option<f64>, _>(source.as_str()) else {
                continue;
            };
            let Some(value) = value else {
                continue;
            };
            if !value.is_finite() {
                continue;
            }
            metrics.push(RawCounterJsDerivedMetric {
                key: definition.key.clone(),
                name: definition.name.clone(),
                counter_type: definition.counter_type.clone(),
                description: definition.description.clone(),
                unit: definition.unit.clone(),
                groups: definition.groups.clone(),
                timeline_groups: definition.timeline_groups.clone(),
                visible: definition.visible,
                batch_filtered: definition.batch_filtered,
                value,
                source_script: script_path.to_path_buf(),
                source_catalog: definition.source_catalog.clone(),
            });
        }
        metrics
    })
}

fn is_javascript_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first == '$' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|ch| ch == '_' || ch == '$' || ch.is_ascii_alphanumeric())
}

fn raw_counter_js_variables_from_parts(
    counter_schemas: &CounterSchemaByGroup,
    fallback_counter_names: &[String],
    sample_blobs: &[SampleBlob],
) -> BTreeMap<String, f64> {
    if counter_schemas.is_empty() && fallback_counter_names.is_empty() {
        return BTreeMap::new();
    }

    let mut accum = BTreeMap::<String, RawCounterJsVariableAccum>::new();
    for (path, data) in sample_blobs {
        if !path.contains("/Derived Counter Sample Data/") {
            continue;
        }
        let Some((record_size, _)) = gprw_record_info(&data) else {
            continue;
        };
        let records = gprw_u64_records(data, record_size);
        if records.is_empty() {
            continue;
        }
        let path_ids = parse_derived_counter_sample_path(&path);
        let Some(counter_names) = path_ids
            .sample_group
            .and_then(|group| counter_schemas.get(&group).map(Vec::as_slice))
            .or((!fallback_counter_names.is_empty()).then_some(fallback_counter_names))
        else {
            continue;
        };
        for (counter_index, raw_name) in counter_names.iter().enumerate() {
            let value_column = 8 + counter_index;
            for record in &records {
                let Some(value) = record.get(value_column).copied() else {
                    continue;
                };
                let normalized = record
                    .get(2)
                    .copied()
                    .filter(|denominator| *denominator != 0)
                    .map(|denominator| value as f64 / denominator as f64 * 100.0);
                accum
                    .entry(raw_name.clone())
                    .or_default()
                    .push(value as f64, normalized);
            }
        }
    }

    let mut variables = BTreeMap::new();
    for (raw_name, accum) in accum {
        let Some(raw_mean) = mean(&accum.raw_values) else {
            continue;
        };
        let normalized_mean = mean(&accum.normalized_values).unwrap_or(raw_mean);
        variables.insert(raw_name.clone(), raw_mean);
        variables.insert(format!("{raw_name}_norm"), normalized_mean);
    }
    variables
}

fn raw_counter_js_variable_groups_from_parts(
    counter_schemas: &CounterSchemaByGroup,
    fallback_counter_names: &[String],
    sample_blobs: &[SampleBlob],
    aggregate_metadata: &[RawCounterAggregateMetadata],
    profiler_dispatches: &[profiler::ProfilerDispatch],
) -> Vec<RawCounterJsVariableGroup> {
    if counter_schemas.is_empty() && fallback_counter_names.is_empty() {
        return Vec::new();
    }

    let mut sample_groups =
        BTreeMap::<(Option<usize>, Option<usize>), RawCounterJsGroupAccum>::new();
    let mut dispatch_groups = BTreeMap::<usize, RawCounterJsGroupAccum>::new();
    let encoder_sample_indices = aggregate_metadata
        .iter()
        .flat_map(|metadata| metadata.encoder_sample_indices.iter())
        .collect::<Vec<_>>();
    let encoder_sample_windows =
        raw_counter_encoder_sample_windows(&sample_blobs, &encoder_sample_indices);
    let mut encoder_sample_groups = BTreeMap::<usize, RawCounterJsGroupAccum>::new();
    for (path, data) in sample_blobs {
        if !path.contains("/Derived Counter Sample Data/") {
            continue;
        }
        let Some((record_size, _)) = gprw_record_info(&data) else {
            continue;
        };
        let records = gprw_u64_records(data, record_size);
        if records.is_empty() {
            continue;
        }
        let path_ids = parse_derived_counter_sample_path(&path);
        let Some(counter_names) = path_ids
            .sample_group
            .and_then(|group| counter_schemas.get(&group).map(Vec::as_slice))
            .or((!fallback_counter_names.is_empty()).then_some(fallback_counter_names))
        else {
            continue;
        };
        for record in &records {
            let sample_accum = sample_groups
                .entry((path_ids.sample_group, path_ids.source_index))
                .or_default();
            sample_accum.push_record_metadata(record, &path_ids);
            push_raw_counter_js_values(sample_accum, record, counter_names);

            let Some(dispatch) =
                profiler_dispatch_for_raw_counter_record(record, profiler_dispatches)
            else {
                continue;
            };
            let dispatch_accum = dispatch_groups.entry(dispatch.index).or_default();
            dispatch_accum.push_record_metadata(record, &path_ids);
            push_raw_counter_js_values(dispatch_accum, record, counter_names);
        }
        for sample_index in &encoder_sample_indices {
            let Some(window) =
                encoder_sample_windows.get(&(path_ids.sample_group, sample_index.row_index))
            else {
                continue;
            };
            let sample_accum = encoder_sample_groups
                .entry(sample_index.row_index)
                .or_default();
            sample_accum.encoder_sample_row_index = Some(sample_index.row_index);
            sample_accum.encoder_sample_index = Some(sample_index.sample_index);

            if path_ids.source_index == Some(4) {
                let Some(end_record) = records.get(window.end_record_index) else {
                    continue;
                };
                let Some(start_record) = records.get(window.start_record_index) else {
                    continue;
                };
                sample_accum.push_record_metadata(end_record, &path_ids);
                push_raw_counter_js_delta_values(
                    sample_accum,
                    start_record,
                    end_record,
                    counter_names,
                );
                continue;
            }

            for record in records_in_tick_window(&records, window.start_ticks, window.end_ticks) {
                sample_accum.push_record_metadata(record, &path_ids);
                push_raw_counter_js_values(sample_accum, record, counter_names);
            }
        }
    }

    let mut groups = Vec::new();
    groups.extend(
        sample_groups
            .into_iter()
            .filter_map(|((sample_group, source_index), accum)| {
                let group_id = match (sample_group, source_index) {
                    (Some(sample_group), Some(source_index)) => {
                        format!("{sample_group}/source{source_index}")
                    }
                    (Some(sample_group), None) => sample_group.to_string(),
                    (None, Some(source_index)) => format!("source{source_index}"),
                    (None, None) => "unknown".to_owned(),
                };
                RawCounterJsVariableGroup::from_accum(
                    "sample_group",
                    group_id,
                    sample_group,
                    source_index,
                    None,
                    accum,
                )
            }),
    );
    let dispatches_by_index = profiler_dispatches
        .iter()
        .map(|dispatch| (dispatch.index, dispatch))
        .collect::<BTreeMap<_, _>>();
    groups.extend(dispatch_groups.into_iter().filter_map(|(index, accum)| {
        let dispatch = dispatches_by_index.get(&index).copied();
        RawCounterJsVariableGroup::from_accum(
            "profiler_dispatch",
            index.to_string(),
            None,
            None,
            dispatch,
            accum,
        )
    }));
    groups.extend(
        encoder_sample_groups
            .into_iter()
            .filter_map(|(row_index, accum)| {
                let sample_index = accum.encoder_sample_index;
                RawCounterJsVariableGroup::from_accum(
                    "encoder_sample",
                    sample_index
                        .map(|sample_index| format!("row{row_index}/sample{sample_index}"))
                        .unwrap_or_else(|| format!("row{row_index}")),
                    None,
                    None,
                    None,
                    accum,
                )
            }),
    );
    groups.sort_by(|left, right| {
        left.group_kind
            .cmp(&right.group_kind)
            .then_with(|| left.group_id.cmp(&right.group_id))
    });
    groups
}

fn raw_counter_shifted_dispatch_metric_candidates(
    counter_schemas: &CounterSchemaByGroup,
    fallback_counter_names: &[String],
    sample_blobs: &[SampleBlob],
    profiler_dispatches: &[profiler::ProfilerDispatch],
    alignment: Option<&RawCounterDispatchAlignmentReport>,
    catalog: &RawCounterCatalog,
    device_identifier: Option<&str>,
) -> Vec<RawCounterShiftedDispatchMetricCandidate> {
    let Some(alignment) = alignment else {
        return Vec::new();
    };
    if alignment.direct_overlap_record_count != 0 {
        return Vec::new();
    }
    let Some(candidate) = alignment.candidate_shifts.iter().max_by(|left, right| {
        left.dispatches_covered
            .cmp(&right.dispatches_covered)
            .then_with(|| left.overlap_record_count.cmp(&right.overlap_record_count))
            .then_with(|| {
                right
                    .shift_ticks
                    .unsigned_abs()
                    .cmp(&left.shift_ticks.unsigned_abs())
            })
    }) else {
        return Vec::new();
    };
    if candidate.overlap_record_count == 0 || candidate.dispatches_covered == 0 {
        return Vec::new();
    }

    let variable_groups = raw_counter_shifted_dispatch_variable_groups_from_parts(
        counter_schemas,
        fallback_counter_names,
        sample_blobs,
        profiler_dispatches,
        candidate.shift_ticks,
    );
    let groups = evaluate_agx_derived_metric_groups(catalog, variable_groups, device_identifier);
    if groups.is_empty() {
        return Vec::new();
    }

    vec![RawCounterShiftedDispatchMetricCandidate {
        name: candidate.name.clone(),
        shift_ticks: candidate.shift_ticks,
        shift_ns: candidate.shift_ns,
        overlap_record_count: candidate.overlap_record_count,
        dispatches_covered: candidate.dispatches_covered,
        groups,
    }]
}

fn raw_counter_shifted_dispatch_variable_groups_from_parts(
    counter_schemas: &CounterSchemaByGroup,
    fallback_counter_names: &[String],
    sample_blobs: &[SampleBlob],
    profiler_dispatches: &[profiler::ProfilerDispatch],
    shift_ticks: i64,
) -> Vec<RawCounterJsVariableGroup> {
    if profiler_dispatches.is_empty()
        || (counter_schemas.is_empty() && fallback_counter_names.is_empty())
    {
        return Vec::new();
    }

    let mut dispatch_groups = BTreeMap::<usize, RawCounterJsGroupAccum>::new();
    for (path, data) in sample_blobs {
        if !path.contains("/Derived Counter Sample Data/") {
            continue;
        }
        let Some((record_size, _)) = gprw_record_info(data) else {
            continue;
        };
        let records = gprw_u64_records(data, record_size);
        if records.is_empty() {
            continue;
        }
        let path_ids = parse_derived_counter_sample_path(path);
        let Some(counter_names) = path_ids
            .sample_group
            .and_then(|group| counter_schemas.get(&group).map(Vec::as_slice))
            .or((!fallback_counter_names.is_empty()).then_some(fallback_counter_names))
        else {
            continue;
        };
        for record in &records {
            let Some(raw_ticks) = record.get(1).copied() else {
                continue;
            };
            let Some(shifted_ticks) = shifted_tick(raw_ticks, shift_ticks) else {
                continue;
            };
            let Some(dispatch) = profiler_dispatch_for_tick(shifted_ticks, profiler_dispatches)
            else {
                continue;
            };
            let dispatch_accum = dispatch_groups.entry(dispatch.index).or_default();
            dispatch_accum.push_record_metadata_with_ticks(record, &path_ids, Some(shifted_ticks));
            push_raw_counter_js_values(dispatch_accum, record, counter_names);
        }
    }

    let dispatches_by_index = profiler_dispatches
        .iter()
        .map(|dispatch| (dispatch.index, dispatch))
        .collect::<BTreeMap<_, _>>();
    dispatch_groups
        .into_iter()
        .filter_map(|(index, accum)| {
            let dispatch = dispatches_by_index.get(&index).copied();
            RawCounterJsVariableGroup::from_accum(
                "profiler_dispatch_shifted",
                index.to_string(),
                None,
                None,
                dispatch,
                accum,
            )
        })
        .collect()
}

fn raw_counter_dispatch_alignment_report(
    sample_blobs: &[SampleBlob],
    dispatches: &[profiler::ProfilerDispatch],
    aggregate_metadata: &[RawCounterAggregateMetadata],
) -> Option<RawCounterDispatchAlignmentReport> {
    if dispatches.is_empty() {
        return None;
    }

    let dispatch_start_ticks = dispatches
        .iter()
        .filter_map(|dispatch| (dispatch.start_ticks != 0).then_some(dispatch.start_ticks))
        .min()?;
    let dispatch_end_ticks = dispatches
        .iter()
        .filter_map(|dispatch| (dispatch.end_ticks != 0).then_some(dispatch.end_ticks))
        .max()?;
    if dispatch_end_ticks < dispatch_start_ticks {
        return None;
    }
    let dispatch_span_ticks = dispatch_end_ticks - dispatch_start_ticks;
    let (timebase_numer, timebase_denom) = raw_counter_timebase(aggregate_metadata);

    let mut timestamps = Vec::new();
    let mut raw_start_ticks = u64::MAX;
    let mut raw_end_ticks = 0u64;
    let mut direct_overlap_record_count = 0usize;
    let mut raw_before_dispatch_record_count = 0usize;
    let mut raw_after_dispatch_record_count = 0usize;
    let mut source_spans =
        BTreeMap::<(Option<usize>, Option<usize>), RawCounterTimestampSpanAccum>::new();

    for (path, data) in sample_blobs {
        if !path.contains("/Derived Counter Sample Data/") {
            continue;
        }
        let Some((record_size, _)) = gprw_record_info(data) else {
            continue;
        };
        let path_ids = parse_derived_counter_sample_path(path);
        for record in gprw_u64_records(data, record_size) {
            let Some(ticks) = record.get(1).copied() else {
                continue;
            };
            timestamps.push(ticks);
            raw_start_ticks = raw_start_ticks.min(ticks);
            raw_end_ticks = raw_end_ticks.max(ticks);
            let overlaps_dispatch = profiler_dispatch_for_tick(ticks, dispatches).is_some();
            if overlaps_dispatch {
                direct_overlap_record_count += 1;
            } else if ticks < dispatch_start_ticks {
                raw_before_dispatch_record_count += 1;
            } else if ticks > dispatch_end_ticks {
                raw_after_dispatch_record_count += 1;
            }
            source_spans
                .entry((path_ids.sample_group, path_ids.source_index))
                .or_default()
                .push(ticks, overlaps_dispatch, path_ids.ring_index);
        }
    }
    if timestamps.is_empty() || raw_start_ticks == u64::MAX {
        return None;
    }

    let raw_span_ticks = raw_end_ticks.saturating_sub(raw_start_ticks);
    let mut source_spans = source_spans
        .into_iter()
        .filter_map(|((sample_group, source_index), accum)| {
            accum.into_span(
                sample_group,
                source_index,
                dispatch_start_ticks,
                dispatch_end_ticks,
                timebase_numer,
                timebase_denom,
            )
        })
        .collect::<Vec<_>>();
    source_spans.sort_by(|left, right| {
        left.nearest_dispatch_gap_ticks
            .unsigned_abs()
            .cmp(&right.nearest_dispatch_gap_ticks.unsigned_abs())
            .then_with(|| left.sample_group.cmp(&right.sample_group))
            .then_with(|| left.source_index.cmp(&right.source_index))
    });

    let candidate_shifts = raw_counter_dispatch_alignment_candidates(
        &timestamps,
        raw_start_ticks,
        raw_end_ticks,
        dispatch_start_ticks,
        dispatch_end_ticks,
        dispatches,
        timebase_numer,
        timebase_denom,
    );

    Some(RawCounterDispatchAlignmentReport {
        timebase_numer,
        timebase_denom,
        dispatch_count: dispatches.len(),
        dispatch_start_ticks,
        dispatch_end_ticks,
        dispatch_span_ticks,
        dispatch_span_ns: tick_delta_to_ns(
            dispatch_span_ticks as i64,
            timebase_numer,
            timebase_denom,
        ),
        raw_record_count: timestamps.len(),
        raw_start_ticks,
        raw_end_ticks,
        raw_span_ticks,
        raw_span_ns: tick_delta_to_ns(raw_span_ticks as i64, timebase_numer, timebase_denom),
        direct_overlap_record_count,
        raw_before_dispatch_record_count,
        raw_after_dispatch_record_count,
        raw_end_to_dispatch_start_gap_ticks: signed_tick_delta(dispatch_start_ticks, raw_end_ticks),
        raw_end_to_dispatch_start_gap_ns: tick_delta_to_ns(
            signed_tick_delta(dispatch_start_ticks, raw_end_ticks),
            timebase_numer,
            timebase_denom,
        ),
        raw_start_to_dispatch_start_delta_ticks: signed_tick_delta(
            dispatch_start_ticks,
            raw_start_ticks,
        ),
        raw_start_to_dispatch_start_delta_ns: tick_delta_to_ns(
            signed_tick_delta(dispatch_start_ticks, raw_start_ticks),
            timebase_numer,
            timebase_denom,
        ),
        candidate_shifts,
        source_spans,
    })
}

#[derive(Debug, Default)]
struct RawCounterTimestampSpanAccum {
    record_count: usize,
    start_ticks: Option<u64>,
    end_ticks: Option<u64>,
    direct_overlap_record_count: usize,
    ring_indices: BTreeSet<usize>,
}

impl RawCounterTimestampSpanAccum {
    fn push(&mut self, ticks: u64, overlaps_dispatch: bool, ring_index: Option<usize>) {
        self.record_count += 1;
        self.start_ticks = Some(
            self.start_ticks
                .map(|current| current.min(ticks))
                .unwrap_or(ticks),
        );
        self.end_ticks = Some(
            self.end_ticks
                .map(|current| current.max(ticks))
                .unwrap_or(ticks),
        );
        if overlaps_dispatch {
            self.direct_overlap_record_count += 1;
        }
        if let Some(ring_index) = ring_index {
            self.ring_indices.insert(ring_index);
        }
    }

    fn into_span(
        self,
        sample_group: Option<usize>,
        source_index: Option<usize>,
        dispatch_start_ticks: u64,
        dispatch_end_ticks: u64,
        timebase_numer: Option<u64>,
        timebase_denom: Option<u64>,
    ) -> Option<RawCounterTimestampSpan> {
        let start_ticks = self.start_ticks?;
        let end_ticks = self.end_ticks?;
        let span_ticks = end_ticks.saturating_sub(start_ticks);
        let nearest_dispatch_gap_ticks = tick_span_gap_to_dispatch(
            start_ticks,
            end_ticks,
            dispatch_start_ticks,
            dispatch_end_ticks,
        );
        Some(RawCounterTimestampSpan {
            sample_group,
            source_index,
            ring_indices: self.ring_indices.into_iter().collect(),
            record_count: self.record_count,
            start_ticks,
            end_ticks,
            span_ticks,
            direct_overlap_record_count: self.direct_overlap_record_count,
            nearest_dispatch_gap_ticks,
            nearest_dispatch_gap_ns: tick_delta_to_ns(
                nearest_dispatch_gap_ticks,
                timebase_numer,
                timebase_denom,
            ),
        })
    }
}

fn raw_counter_timebase(
    aggregate_metadata: &[RawCounterAggregateMetadata],
) -> (Option<u64>, Option<u64>) {
    aggregate_metadata
        .iter()
        .find_map(|metadata| Some((metadata.timebase_numer?, metadata.timebase_denom?)))
        .map(|(numer, denom)| (Some(numer), Some(denom)))
        .unwrap_or((None, None))
}

fn raw_counter_dispatch_alignment_candidates(
    timestamps: &[u64],
    raw_start_ticks: u64,
    raw_end_ticks: u64,
    dispatch_start_ticks: u64,
    dispatch_end_ticks: u64,
    dispatches: &[profiler::ProfilerDispatch],
    timebase_numer: Option<u64>,
    timebase_denom: Option<u64>,
) -> Vec<RawCounterDispatchAlignmentCandidate> {
    [
        (
            "raw_start_to_dispatch_start",
            signed_tick_delta(dispatch_start_ticks, raw_start_ticks),
        ),
        (
            "raw_end_to_dispatch_end",
            signed_tick_delta(dispatch_end_ticks, raw_end_ticks),
        ),
        (
            "raw_end_to_dispatch_start",
            signed_tick_delta(dispatch_start_ticks, raw_end_ticks),
        ),
    ]
    .into_iter()
    .filter_map(|(name, shift_ticks)| {
        let shifted_raw_start_ticks = shifted_tick(raw_start_ticks, shift_ticks)?;
        let shifted_raw_end_ticks = shifted_tick(raw_end_ticks, shift_ticks)?;
        let mut dispatches_covered = BTreeSet::new();
        let overlap_record_count = timestamps
            .iter()
            .filter_map(|ticks| shifted_tick(*ticks, shift_ticks))
            .filter(|ticks| {
                let Some(dispatch) = profiler_dispatch_for_tick(*ticks, dispatches) else {
                    return false;
                };
                dispatches_covered.insert(dispatch.index);
                true
            })
            .count();
        Some(RawCounterDispatchAlignmentCandidate {
            name: name.to_owned(),
            shift_ticks,
            shift_ns: tick_delta_to_ns(shift_ticks, timebase_numer, timebase_denom),
            shifted_raw_start_ticks,
            shifted_raw_end_ticks,
            overlap_record_count,
            dispatches_covered: dispatches_covered.len(),
        })
    })
    .collect()
}

fn profiler_dispatch_for_tick(
    ticks: u64,
    dispatches: &[profiler::ProfilerDispatch],
) -> Option<&profiler::ProfilerDispatch> {
    let index = dispatches
        .partition_point(|dispatch| dispatch.end_ticks != 0 && dispatch.end_ticks < ticks);
    dispatches
        .get(index)
        .filter(|dispatch| dispatch.start_ticks <= ticks && ticks <= dispatch.end_ticks)
}

fn tick_span_gap_to_dispatch(
    start_ticks: u64,
    end_ticks: u64,
    dispatch_start_ticks: u64,
    dispatch_end_ticks: u64,
) -> i64 {
    if end_ticks < dispatch_start_ticks {
        signed_tick_delta(dispatch_start_ticks, end_ticks)
    } else if start_ticks > dispatch_end_ticks {
        signed_tick_delta(dispatch_end_ticks, start_ticks)
    } else {
        0
    }
}

fn signed_tick_delta(target_ticks: u64, source_ticks: u64) -> i64 {
    let delta = target_ticks as i128 - source_ticks as i128;
    delta.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

fn shifted_tick(ticks: u64, shift_ticks: i64) -> Option<u64> {
    let shifted = ticks as i128 + shift_ticks as i128;
    (0..=u64::MAX as i128)
        .contains(&shifted)
        .then_some(shifted as u64)
}

fn tick_delta_to_ns(
    ticks: i64,
    timebase_numer: Option<u64>,
    timebase_denom: Option<u64>,
) -> Option<f64> {
    let numer = timebase_numer?;
    let denom = timebase_denom?;
    (denom != 0).then(|| ticks as f64 * numer as f64 / denom as f64)
}

#[derive(Debug, Clone, Copy)]
struct RawCounterEncoderSampleWindow {
    start_record_index: usize,
    end_record_index: usize,
    start_ticks: u64,
    end_ticks: u64,
}

fn raw_counter_encoder_sample_windows(
    sample_blobs: &[SampleBlob],
    encoder_sample_indices: &[&RawCounterEncoderSampleIndex],
) -> BTreeMap<(Option<usize>, usize), RawCounterEncoderSampleWindow> {
    let mut windows = BTreeMap::new();
    if encoder_sample_indices.is_empty() {
        return windows;
    }

    for (path, data) in sample_blobs {
        let path_ids = parse_derived_counter_sample_path(path);
        if path_ids.source_index != Some(4) {
            continue;
        }
        let Some((record_size, _)) = gprw_record_info(data) else {
            continue;
        };
        let records = gprw_u64_records(data, record_size);
        if records.is_empty() {
            continue;
        }

        for (position, sample_index) in encoder_sample_indices.iter().enumerate() {
            let end_record_index = sample_index.sample_index as usize;
            let start_record_index = position
                .checked_sub(1)
                .and_then(|position| encoder_sample_indices.get(position))
                .map(|sample_index| sample_index.sample_index as usize)
                .unwrap_or(0);
            let Some(start_record) = records.get(start_record_index) else {
                continue;
            };
            let Some(end_record) = records.get(end_record_index) else {
                continue;
            };
            let Some(start_ticks) = start_record.get(1).copied() else {
                continue;
            };
            let Some(end_ticks) = end_record.get(1).copied() else {
                continue;
            };
            if end_ticks < start_ticks {
                continue;
            }
            windows.insert(
                (path_ids.sample_group, sample_index.row_index),
                RawCounterEncoderSampleWindow {
                    start_record_index,
                    end_record_index,
                    start_ticks,
                    end_ticks,
                },
            );
        }
    }

    windows
}

fn records_in_tick_window(
    records: &[Vec<u64>],
    start_ticks: u64,
    end_ticks: u64,
) -> impl Iterator<Item = &[u64]> {
    records.iter().filter_map(move |record| {
        let ticks = record.get(1).copied()?;
        (ticks > start_ticks && ticks <= end_ticks).then_some(record.as_slice())
    })
}

fn raw_counter_encoder_sample_metrics_from_parts(
    counter_schemas: &CounterSchemaByGroup,
    fallback_counter_names: &[String],
    sample_blobs: &[SampleBlob],
    aggregate_metadata: &[RawCounterAggregateMetadata],
    catalog: &RawCounterCatalog,
) -> Vec<RawCounterEncoderSampleMetric> {
    if counter_schemas.is_empty() && fallback_counter_names.is_empty() {
        return Vec::new();
    }

    let encoder_sample_indices = aggregate_metadata
        .iter()
        .flat_map(|metadata| metadata.encoder_sample_indices.iter())
        .collect::<Vec<_>>();
    if encoder_sample_indices.is_empty() {
        return Vec::new();
    }
    let encoder_sample_windows =
        raw_counter_encoder_sample_windows(&sample_blobs, &encoder_sample_indices);

    let mut metrics = Vec::new();
    for (path, data) in sample_blobs {
        if !path.contains("/Derived Counter Sample Data/") {
            continue;
        }
        let Some((record_size, _)) = gprw_record_info(&data) else {
            continue;
        };
        let records = gprw_u64_records(data, record_size);
        if records.is_empty() {
            continue;
        }
        let path_ids = parse_derived_counter_sample_path(&path);
        let Some(counter_names) = path_ids
            .sample_group
            .and_then(|group| counter_schemas.get(&group).map(Vec::as_slice))
            .or((!fallback_counter_names.is_empty()).then_some(fallback_counter_names))
        else {
            continue;
        };

        for sample_index in &encoder_sample_indices {
            let Some(window) =
                encoder_sample_windows.get(&(path_ids.sample_group, sample_index.row_index))
            else {
                continue;
            };

            if path_ids.source_index == Some(4) {
                let Some(end_record) = records.get(window.end_record_index) else {
                    continue;
                };
                let Some(start_record) = records.get(window.start_record_index) else {
                    continue;
                };
                push_encoder_sample_metric_deltas(
                    &mut metrics,
                    sample_index,
                    &path,
                    &path_ids,
                    counter_names,
                    start_record,
                    end_record,
                    catalog,
                );
            } else {
                push_encoder_sample_metric_window_totals(
                    &mut metrics,
                    sample_index,
                    &path,
                    &path_ids,
                    counter_names,
                    records_in_tick_window(&records, window.start_ticks, window.end_ticks),
                    catalog,
                );
            }
        }
    }

    metrics.sort_by(|left, right| {
        left.row_index
            .cmp(&right.row_index)
            .then_with(|| {
                right
                    .normalized_percent
                    .unwrap_or(0.0)
                    .abs()
                    .partial_cmp(&left.normalized_percent.unwrap_or(0.0).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| right.raw_delta.cmp(&left.raw_delta))
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.counter_index.cmp(&right.counter_index))
    });
    metrics
}

fn push_encoder_sample_metric_deltas(
    metrics: &mut Vec<RawCounterEncoderSampleMetric>,
    sample_index: &RawCounterEncoderSampleIndex,
    path: &str,
    path_ids: &DerivedCounterSamplePath,
    counter_names: &[String],
    start_record: &[u64],
    end_record: &[u64],
    catalog: &RawCounterCatalog,
) {
    let denominator_delta = match (start_record.get(2).copied(), end_record.get(2).copied()) {
        (Some(start), Some(end)) if end > start => Some(end - start),
        _ => None,
    };
    for (counter_index, raw_name) in counter_names.iter().enumerate() {
        let value_column = 8 + counter_index;
        let Some(start) = start_record.get(value_column).copied() else {
            continue;
        };
        let Some(end) = end_record.get(value_column).copied() else {
            continue;
        };
        let raw_delta = end.saturating_sub(start);
        let normalized_percent = denominator_delta
            .filter(|denominator| *denominator != 0)
            .map(|denominator| raw_delta as f64 / denominator as f64 * 100.0);
        metrics.push(RawCounterEncoderSampleMetric {
            row_index: sample_index.row_index,
            sample_index: sample_index.sample_index,
            path: path.to_owned(),
            sample_group: path_ids.sample_group,
            source_index: path_ids.source_index,
            ring_index: path_ids.ring_index,
            counter_index,
            raw_name: raw_name.clone(),
            sample_count: 1,
            raw_delta,
            normalized_percent,
            derived_counter_matches: catalog
                .derived_by_hash
                .get(raw_name)
                .cloned()
                .unwrap_or_default(),
            hardware_selectors: catalog
                .hardware_by_hash
                .get(raw_name)
                .cloned()
                .unwrap_or_default(),
        });
    }
}

fn push_encoder_sample_metric_window_totals<'a>(
    metrics: &mut Vec<RawCounterEncoderSampleMetric>,
    sample_index: &RawCounterEncoderSampleIndex,
    path: &str,
    path_ids: &DerivedCounterSamplePath,
    counter_names: &[String],
    records: impl Iterator<Item = &'a [u64]>,
    catalog: &RawCounterCatalog,
) {
    let mut totals = vec![0u64; counter_names.len()];
    let mut denominator_total = 0u64;
    let mut sample_count = 0usize;
    for record in records {
        sample_count += 1;
        denominator_total = denominator_total.saturating_add(record.get(2).copied().unwrap_or(0));
        for (counter_index, total) in totals.iter_mut().enumerate() {
            let value_column = 8 + counter_index;
            *total = total.saturating_add(record.get(value_column).copied().unwrap_or(0));
        }
    }
    if sample_count == 0 {
        return;
    }
    for (counter_index, raw_name) in counter_names.iter().enumerate() {
        let raw_delta = totals[counter_index];
        let normalized_percent =
            (denominator_total != 0).then(|| raw_delta as f64 / denominator_total as f64 * 100.0);
        metrics.push(RawCounterEncoderSampleMetric {
            row_index: sample_index.row_index,
            sample_index: sample_index.sample_index,
            path: path.to_owned(),
            sample_group: path_ids.sample_group,
            source_index: path_ids.source_index,
            ring_index: path_ids.ring_index,
            counter_index,
            raw_name: raw_name.clone(),
            sample_count,
            raw_delta,
            normalized_percent,
            derived_counter_matches: catalog
                .derived_by_hash
                .get(raw_name)
                .cloned()
                .unwrap_or_default(),
            hardware_selectors: catalog
                .hardware_by_hash
                .get(raw_name)
                .cloned()
                .unwrap_or_default(),
        });
    }
}

fn profiler_dispatch_for_raw_counter_record<'a>(
    record: &[u64],
    dispatches: &'a [profiler::ProfilerDispatch],
) -> Option<&'a profiler::ProfilerDispatch> {
    let timestamp = record.get(1).copied()?;
    profiler_dispatch_for_tick(timestamp, dispatches)
}

#[derive(Debug)]
struct RawCounterJsVariableGroup {
    group_kind: String,
    group_id: String,
    encoder_sample_row_index: Option<usize>,
    encoder_sample_index: Option<u32>,
    sample_group: Option<usize>,
    source_index: Option<usize>,
    ring_indices: Vec<usize>,
    start_ticks: Option<u64>,
    end_ticks: Option<u64>,
    record_count: usize,
    encoder_ids: Vec<u64>,
    kick_trace_ids: Vec<u64>,
    source_ids: Vec<u64>,
    profiler_dispatch_index: Option<usize>,
    profiler_encoder_index: Option<usize>,
    profiler_function_name: Option<String>,
    profiler_pipeline_id: Option<i64>,
    profiler_start_ticks: Option<u64>,
    profiler_end_ticks: Option<u64>,
    variables: BTreeMap<String, f64>,
}

impl RawCounterJsVariableGroup {
    fn from_accum(
        group_kind: &str,
        group_id: String,
        sample_group: Option<usize>,
        source_index: Option<usize>,
        dispatch: Option<&profiler::ProfilerDispatch>,
        accum: RawCounterJsGroupAccum,
    ) -> Option<Self> {
        let variables = raw_counter_js_variables_from_accum(accum.counters);
        (!variables.is_empty()).then(|| Self {
            group_kind: group_kind.to_owned(),
            group_id,
            encoder_sample_row_index: accum.encoder_sample_row_index,
            encoder_sample_index: accum.encoder_sample_index,
            sample_group,
            source_index,
            ring_indices: accum.ring_indices.into_iter().collect(),
            start_ticks: accum.start_ticks,
            end_ticks: accum.end_ticks,
            record_count: accum.record_count,
            encoder_ids: accum.encoder_ids.into_iter().collect(),
            kick_trace_ids: accum.kick_trace_ids.into_iter().collect(),
            source_ids: accum.source_ids.into_iter().collect(),
            profiler_dispatch_index: dispatch.map(|dispatch| dispatch.index),
            profiler_encoder_index: dispatch.map(|dispatch| dispatch.encoder_index),
            profiler_function_name: dispatch.and_then(|dispatch| dispatch.function_name.clone()),
            profiler_pipeline_id: dispatch.and_then(|dispatch| dispatch.pipeline_id),
            profiler_start_ticks: dispatch.map(|dispatch| dispatch.start_ticks),
            profiler_end_ticks: dispatch.map(|dispatch| dispatch.end_ticks),
            variables,
        })
    }
}

#[derive(Debug, Default)]
struct RawCounterJsGroupAccum {
    counters: BTreeMap<String, RawCounterJsVariableAccum>,
    record_count: usize,
    ring_indices: BTreeSet<usize>,
    start_ticks: Option<u64>,
    end_ticks: Option<u64>,
    encoder_ids: BTreeSet<u64>,
    kick_trace_ids: BTreeSet<u64>,
    source_ids: BTreeSet<u64>,
    encoder_sample_row_index: Option<usize>,
    encoder_sample_index: Option<u32>,
}

impl RawCounterJsGroupAccum {
    fn push_record_metadata(&mut self, record: &[u64], path_ids: &DerivedCounterSamplePath) {
        self.push_record_metadata_with_ticks(record, path_ids, record.get(1).copied());
    }

    fn push_record_metadata_with_ticks(
        &mut self,
        record: &[u64],
        path_ids: &DerivedCounterSamplePath,
        ticks: impl Into<Option<u64>>,
    ) {
        self.record_count += 1;
        if let Some(ring_index) = path_ids.ring_index {
            self.ring_indices.insert(ring_index);
        }
        if let Some(timestamp) = ticks.into() {
            self.start_ticks = Some(
                self.start_ticks
                    .map(|current| current.min(timestamp))
                    .unwrap_or(timestamp),
            );
            self.end_ticks = Some(
                self.end_ticks
                    .map(|current| current.max(timestamp))
                    .unwrap_or(timestamp),
            );
        }
        if let Some(encoder_id) = record.get(4).copied() {
            self.encoder_ids.insert(encoder_id);
        }
        if let Some(kick_trace_id) = record.get(5).copied() {
            self.kick_trace_ids.insert(kick_trace_id);
        }
        if let Some(source_id) = record.get(7).copied() {
            self.source_ids.insert(source_id);
        }
    }
}

fn push_raw_counter_js_values(
    accum: &mut RawCounterJsGroupAccum,
    record: &[u64],
    counter_names: &[String],
) {
    for (counter_index, raw_name) in counter_names.iter().enumerate() {
        let value_column = 8 + counter_index;
        let Some(value) = record.get(value_column).copied() else {
            continue;
        };
        let normalized = record
            .get(2)
            .copied()
            .filter(|denominator| *denominator != 0)
            .map(|denominator| value as f64 / denominator as f64 * 100.0);
        accum
            .counters
            .entry(raw_name.clone())
            .or_default()
            .push(value as f64, normalized);
    }
}

fn push_raw_counter_js_delta_values(
    accum: &mut RawCounterJsGroupAccum,
    start_record: &[u64],
    end_record: &[u64],
    counter_names: &[String],
) {
    let denominator_delta = match (start_record.get(2).copied(), end_record.get(2).copied()) {
        (Some(start), Some(end)) if end > start => Some(end - start),
        _ => None,
    };
    for (counter_index, raw_name) in counter_names.iter().enumerate() {
        let value_column = 8 + counter_index;
        let Some(start) = start_record.get(value_column).copied() else {
            continue;
        };
        let Some(end) = end_record.get(value_column).copied() else {
            continue;
        };
        let value = end.saturating_sub(start);
        let normalized = denominator_delta
            .filter(|denominator| *denominator != 0)
            .map(|denominator| value as f64 / denominator as f64 * 100.0);
        accum
            .counters
            .entry(raw_name.clone())
            .or_default()
            .push(value as f64, normalized);
    }
}

fn raw_counter_js_variables_from_accum(
    accum: BTreeMap<String, RawCounterJsVariableAccum>,
) -> BTreeMap<String, f64> {
    let mut variables = BTreeMap::new();
    for (raw_name, accum) in accum {
        let Some(raw_mean) = mean(&accum.raw_values) else {
            continue;
        };
        let normalized_mean = mean(&accum.normalized_values).unwrap_or(raw_mean);
        variables.insert(raw_name.clone(), raw_mean);
        variables.insert(format!("{raw_name}_norm"), normalized_mean);
    }
    variables
}

#[derive(Debug, Default)]
struct RawCounterJsVariableAccum {
    raw_values: Vec<f64>,
    normalized_values: Vec<f64>,
}

impl RawCounterJsVariableAccum {
    fn push(&mut self, raw: f64, normalized: Option<f64>) {
        self.raw_values.push(raw);
        if let Some(normalized) = normalized {
            self.normalized_values.push(normalized);
        }
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn add_perf_counter_catalog(
    path: &Path,
    hardware: &mut BTreeMap<String, BTreeMap<RawCounterHardwareSelectorKey, BTreeSet<PathBuf>>>,
) {
    let Ok(value) = Value::from_file(path) else {
        return;
    };
    let Some(root) = value.as_dictionary() else {
        return;
    };
    for (hash, value) in root {
        if !hash.starts_with('_') {
            continue;
        }
        let Some(selector) = value.as_dictionary() else {
            continue;
        };
        let key = RawCounterHardwareSelectorKey {
            partition: selector.get("Partition").and_then(plist_u64),
            select: selector.get("Select").and_then(plist_u64),
            flag: selector.get("Flag").and_then(plist_u64),
        };
        hardware
            .entry(hash.clone())
            .or_default()
            .entry(key)
            .or_default()
            .insert(path.to_path_buf());
    }
}

fn plist_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Integer(value) => value.as_unsigned(),
        _ => None,
    }
}

fn plist_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Boolean(value) => Some(*value),
        _ => None,
    }
}

pub fn format_raw_counter_probe(report: &RawCounterProbeReport) -> String {
    let mut out = String::new();
    out.push_str("Raw Counter Probe\n");
    out.push_str("=================\n");
    out.push_str(&format!(
        "profiler_directory={}\n",
        report.profiler_directory.display()
    ));
    if let Some(csv_source) = &report.csv_source {
        out.push_str(&format!("csv_source={}\n", csv_source.display()));
    }
    out.push_str(&format!("targets={}\n\n", report.targets.len()));

    if !report.aggregate_metadata.is_empty() {
        out.push_str("aggregate counter metadata\n");
        for metadata in &report.aggregate_metadata {
            out.push_str(&format!(
                "  APSCounterData[{}]: timebase={}/{} num_encoders={} perf={:?}\n",
                metadata.archive_index,
                metadata
                    .timebase_numer
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                metadata
                    .timebase_denom
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                metadata
                    .num_encoders
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                metadata.perf_info
            ));
            if !metadata.encoder_sample_indices.is_empty() {
                out.push_str("    encoder sample indices:");
                for row in metadata.encoder_sample_indices.iter().take(8) {
                    out.push_str(&format!(
                        " #{}=({}, {}, sample={}, {})",
                        row.row_index, row.word0, row.word1, row.sample_index, row.word3
                    ));
                }
                out.push('\n');
            }
            if !metadata.encoder_infos.is_empty() {
                out.push_str("    encoder infos:");
                for row in metadata.encoder_infos.iter().take(8) {
                    out.push_str(&format!(" #{}={:?}", row.row_index, row.trace_ids));
                }
                out.push('\n');
            }
        }
        out.push('\n');
    }

    if !report.counter_info.is_empty() {
        out.push_str("counter info entries\n");
        for entry in report.counter_info.iter().take(35) {
            let fields = entry
                .fields
                .iter()
                .map(|field| {
                    let len = field
                        .len
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "-".to_owned());
                    if field.keys.is_empty() && field.children.is_empty() {
                        format!("{}:{}:{}", field.key, field.kind, len)
                    } else {
                        format!(
                            "{}:{}:{}:{}:{}",
                            field.key,
                            field.kind,
                            len,
                            field.keys.join("|"),
                            field.children.join("|")
                        )
                    }
                })
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!(
                "  APSCounterData[{}] {} {} fields={}\n",
                entry.archive_index, entry.raw_name, entry.summary, fields
            ));
        }
        out.push('\n');
    }

    if !report.stream_archives.is_empty() {
        out.push_str("streamData archives\n");
        for archive in &report.stream_archives {
            out.push_str(&format!(
                "  {}[{}]: bytes={} source={} serial={} source_index={} ring_buffer={} file={} shader_data={} keys={}\n",
                archive.group,
                archive.index,
                archive.byte_len,
                archive.source.as_deref().unwrap_or("-"),
                archive.serial.map(|value| value.to_string()).unwrap_or_else(|| "-".to_owned()),
                archive
                    .source_index
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                archive
                    .ring_buffer_index
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                archive.data_file.as_deref().unwrap_or("-"),
                archive
                    .shader_profiler_data_len
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                archive.keys.join(",")
            ));
            for field in archive.data_fields.iter().take(8) {
                out.push_str(&format!(
                    "    data {}: bytes={} prefix={} u32={:?} u64={:?} f32={:?}\n",
                    field.key,
                    field.byte_len,
                    field.prefix_hex,
                    field.u32_preview,
                    field.u64_preview,
                    field.f32_preview
                ));
            }
            for field in archive
                .fields
                .iter()
                .filter(|field| field.kind != "data")
                .take(8)
            {
                out.push_str(&format!(
                    "    field {}: kind={} len={} keys={} children={}\n",
                    field.key,
                    field.kind,
                    field
                        .len
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "-".to_owned()),
                    field.keys.join(","),
                    field.children.join(",")
                ));
            }
        }
        out.push('\n');
    }

    if !report.structured_samples.is_empty() {
        out.push_str("structured derived samples\n");
        for sample in report.structured_samples.iter().take(32) {
            out.push_str(&format!(
                "  {}: bytes={} record_size={} records={}\n",
                sample.path,
                sample.byte_len,
                sample
                    .gprw_record_size
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned()),
                sample
                    .gprw_record_count
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_owned())
            ));
            for matched in sample.matches.iter().take(6) {
                out.push_str(&format!(
                    "    {} row={} target={:.4} {} hits={}",
                    matched.metric,
                    matched.row_index,
                    matched.target,
                    matched.encoding,
                    matched.count
                ));
                for example in matched.examples.iter().take(3) {
                    out.push_str(&format!(
                        " @{}(p{} {:.4})",
                        example.offset, example.page_4k, example.value
                    ));
                }
                out.push('\n');
            }
        }
        out.push('\n');
    }

    if !report.structured_layouts.is_empty() {
        out.push_str("structured derived layouts\n");
        for layout in report.structured_layouts.iter().take(32) {
            out.push_str(&format!(
                "  {}: bytes={} record_size={} records={} columns={}\n",
                layout.path,
                layout.byte_len,
                layout.gprw_record_size,
                layout.gprw_record_count,
                layout.u64_columns.len()
            ));
            for column in layout.u64_columns.iter().take(10) {
                out.push_str(&format!(
                    "    u64[{}]: min={} max={} mean={:.2} nonzero={} first={:?}\n",
                    column.index,
                    column.min,
                    column.max,
                    column.mean,
                    column.nonzero_count,
                    column.first_values
                ));
            }
        }
        out.push('\n');
    }

    if !report.normalized_counters.is_empty() {
        if !report.normalized_matches.is_empty() {
            out.push_str("normalized counter target matches\n");
            for matched in report.normalized_matches.iter().take(32) {
                out.push_str(&format!(
                    "  {} row={} target={:.2}% delta={:.2}% confidence={:.2} -> {} group={} ring={} [{}] {}\n",
                    matched.metric,
                    matched.row_index,
                    matched.target,
                    matched.delta,
                    matched.confidence,
                    matched.counter.path,
                    format_optional_usize(matched.counter.sample_group),
                    format_optional_usize(matched.counter.ring_index),
                    matched.counter.counter_index,
                    matched.counter.raw_name
                ));
            }
            out.push('\n');
        }

        out.push_str("normalized derived counters\n");
        for metric in report.normalized_counters.iter().take(32) {
            out.push_str(&format!(
                "  {} group={} source={} ring={} encoders={} kicks={} sources={} [{}] {}: mean={:.2}% max={:.2}% samples={}\n",
                metric.path,
                format_optional_usize(metric.sample_group),
                format_optional_usize(metric.source_index),
                format_optional_usize(metric.ring_index),
                format_u64_hex_list(&metric.encoder_ids, 4),
                format_u64_hex_list(&metric.kick_trace_ids, 4),
                format_u64_hex_list(&metric.source_ids, 4),
                metric.counter_index,
                metric.raw_name,
                metric.mean_percent,
                metric.max_percent,
                metric.sample_count
            ));
        }
        out.push('\n');
    }

    for file in &report.files {
        out.push_str(&format!(
            "Counters_f_{}: bytes={} pages4k={} markers={}\n",
            file.file_index,
            file.byte_len,
            file.page_count_4k
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned()),
            file.marker_count,
        ));
        if !file.top_record_shapes.is_empty() {
            out.push_str("  top shapes:");
            for shape in file.top_record_shapes.iter().take(8) {
                out.push_str(&format!(" {}:{}x{}", shape.tag, shape.size, shape.count));
            }
            out.push('\n');
        }
        for matched in file.matches.iter().take(12) {
            out.push_str(&format!(
                "  {} row={} target={:.4} +/- {:.4} {} hits={}",
                matched.metric,
                matched.row_index,
                matched.target,
                matched.tolerance,
                matched.encoding,
                matched.count
            ));
            for example in matched.examples.iter().take(3) {
                out.push_str(&format!(
                    " @{}(p{} {} {} {:.4})",
                    example.offset,
                    example.page_4k,
                    example.record_tag.as_deref().unwrap_or("-"),
                    example
                        .record_size
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "-".to_owned()),
                    example.value
                ));
            }
            out.push('\n');
        }
        out.push('\n');
    }
    out
}

fn format_optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_owned())
}

fn format_optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_owned())
}

fn format_optional_u64_hex(value: Option<u64>) -> String {
    value
        .map(|value| format!("0x{value:x}"))
        .unwrap_or_else(|| "-".to_owned())
}

fn format_ticks_with_duration(ticks: u64, ns: Option<f64>) -> String {
    format!("{ticks} ({})", format_optional_duration_ns(ns))
}

fn format_optional_duration_ns(ns: Option<f64>) -> String {
    let Some(ns) = ns else {
        return "-".to_owned();
    };
    let abs = ns.abs();
    if abs >= 1_000_000.0 {
        format!("{:.3} ms", ns / 1_000_000.0)
    } else if abs >= 1_000.0 {
        format!("{:.3} us", ns / 1_000.0)
    } else {
        format!("{ns:.1} ns")
    }
}

fn format_u64_hex_list(values: &[u64], limit: usize) -> String {
    if values.is_empty() {
        return "-".to_owned();
    }
    let mut formatted = values
        .iter()
        .take(limit)
        .map(|value| format!("0x{value:x}"))
        .collect::<Vec<_>>();
    if values.len() > limit {
        formatted.push(format!("+{}", values.len() - limit));
    }
    formatted.join("|")
}

fn probe_aggregate_counter_metadata(trace_path: &Path) -> Vec<RawCounterAggregateMetadata> {
    load_stream_archive_groups(trace_path)
        .as_ref()
        .map(probe_aggregate_counter_metadata_from_groups)
        .unwrap_or_default()
}

fn probe_aggregate_counter_metadata_from_groups(
    groups: &StreamArchiveGroups,
) -> Vec<RawCounterAggregateMetadata> {
    groups
        .aps_counter_data
        .iter()
        .enumerate()
        .filter_map(|(archive_index, bytes)| {
            let keyed = parse_keyed_archive_dictionary(bytes)?;
            if !keyed.contains_key("Derived Counter Sample Data") {
                return None;
            }
            Some(RawCounterAggregateMetadata {
                archive_index,
                timebase_numer: array_integer(&keyed, "Timebase", 0),
                timebase_denom: array_integer(&keyed, "Timebase", 1),
                num_encoders: keyed
                    .get("Num Encoders")
                    .and_then(StreamArchiveValue::as_u64),
                perf_info: keyed
                    .get("Perf Info")
                    .and_then(StreamArchiveValue::as_dictionary)
                    .map(dictionary_u64_values)
                    .unwrap_or_default(),
                encoder_sample_indices: keyed
                    .get("Encoder Sample Index Data")
                    .and_then(StreamArchiveValue::as_data)
                    .map(parse_encoder_sample_indices)
                    .unwrap_or_default(),
                encoder_infos: keyed
                    .get("Encoder Infos")
                    .map(parse_encoder_infos)
                    .unwrap_or_default(),
            })
        })
        .collect()
}

fn probe_counter_info_entries(trace_path: &Path) -> Vec<RawCounterInfoEntry> {
    load_stream_archive_groups(trace_path)
        .as_ref()
        .map(probe_counter_info_entries_from_groups)
        .unwrap_or_default()
}

fn probe_counter_info_entries_from_groups(
    groups: &StreamArchiveGroups,
) -> Vec<RawCounterInfoEntry> {
    let mut entries = Vec::new();
    for (archive_index, bytes) in groups.aps_counter_data.iter().enumerate() {
        let Some(keyed) = parse_keyed_archive_dictionary(bytes) else {
            continue;
        };
        let Some(counter_info) = keyed
            .get("Counter Info")
            .and_then(StreamArchiveValue::as_dictionary)
        else {
            continue;
        };
        for (raw_name, value) in counter_info {
            let fields = value
                .as_dictionary()
                .map(|fields| {
                    fields
                        .iter()
                        .map(|(key, value)| summarize_field(key, value))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            entries.push(RawCounterInfoEntry {
                archive_index,
                raw_name: raw_name.clone(),
                summary: value.short_summary(),
                fields,
            });
        }
    }
    entries.sort_by(|left, right| {
        left.archive_index
            .cmp(&right.archive_index)
            .then_with(|| left.raw_name.cmp(&right.raw_name))
    });
    entries
}

fn probe_sample_trace_indices_from_groups(
    groups: &StreamArchiveGroups,
) -> Vec<RawCounterSampleTraceIndex> {
    let mut entries = Vec::new();
    for (archive_index, bytes) in groups.aps_data.iter().enumerate() {
        let Some(keyed) = parse_keyed_archive_dictionary(bytes) else {
            continue;
        };
        let Some(data) = keyed
            .get("TraceId to SampleIndex")
            .and_then(StreamArchiveValue::as_data)
        else {
            continue;
        };
        entries.extend(parse_trace_id_sample_index_map(archive_index, data));
    }
    entries.sort_by(|left, right| {
        left.sample_index
            .cmp(&right.sample_index)
            .then_with(|| left.trace_id.cmp(&right.trace_id))
    });
    entries
}

fn probe_trace_maps_from_groups(groups: &StreamArchiveGroups) -> Vec<RawCounterTraceMapEntry> {
    let map_names = [
        "TraceId to BatchId",
        "TraceId to Coalesced BatchId",
        "TraceId to SampleIndex",
        "TraceId to Tile Info",
        "Blit TraceId to Sample Index",
        "Blit Split TraceId to Sample Index",
        "MTLFX TraceIds",
    ];
    let mut entries = Vec::new();
    for (archive_index, bytes) in groups.aps_data.iter().enumerate() {
        let Some(keyed) = parse_keyed_archive_dictionary(bytes) else {
            continue;
        };
        for map_name in map_names {
            let Some(data) = keyed.get(map_name).and_then(StreamArchiveValue::as_data) else {
                continue;
            };
            entries.extend(parse_trace_map_entries(archive_index, map_name, data));
        }
    }
    entries.sort_by(|left, right| {
        left.map_name
            .cmp(&right.map_name)
            .then_with(|| left.trace_id.cmp(&right.trace_id))
    });
    entries
}

#[derive(Debug, Clone)]
struct ProfilingAddressRange {
    start: u64,
    end: u64,
    mapping_index: usize,
}

#[derive(Debug, Clone)]
struct ProfilingLow32AddressRange {
    start: u32,
    end: u32,
    mapping_index: usize,
}

fn profiling_address_ranges(
    mappings: &[RawCounterProgramAddressMapping],
) -> Vec<ProfilingAddressRange> {
    let mut ranges = mappings
        .iter()
        .enumerate()
        .filter_map(|(mapping_index, mapping)| {
            let start = mapping.mapped_address?;
            let size = mapping.mapped_size?;
            (size > 0).then_some(ProfilingAddressRange {
                start,
                end: start.saturating_add(size),
                mapping_index,
            })
        })
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| (range.start, range.end, range.mapping_index));
    ranges
}

fn profiling_low32_address_ranges(
    mappings: &[RawCounterProgramAddressMapping],
) -> Vec<ProfilingLow32AddressRange> {
    let mut ranges = mappings
        .iter()
        .enumerate()
        .filter_map(|(mapping_index, mapping)| {
            let start = mapping.mapped_address?;
            let size = mapping.mapped_size?;
            let end = start.saturating_add(size);
            let low_start = start as u32;
            let low_end = end as u32;
            (size > 0 && low_start < low_end).then_some(ProfilingLow32AddressRange {
                start: low_start,
                end: low_end,
                mapping_index,
            })
        })
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| (range.start, range.end, range.mapping_index));
    ranges
}

fn find_address_range(ranges: &[ProfilingAddressRange], value: u64) -> Option<usize> {
    let mut index = ranges.partition_point(|range| range.start <= value);
    while index > 0 {
        index -= 1;
        let range = &ranges[index];
        if value >= range.end {
            break;
        }
        if value >= range.start {
            return Some(range.mapping_index);
        }
    }
    None
}

fn find_low32_address_range(ranges: &[ProfilingLow32AddressRange], value: u32) -> Option<usize> {
    let mut index = ranges.partition_point(|range| range.start <= value);
    while index > 0 {
        index -= 1;
        let range = &ranges[index];
        if value >= range.end {
            break;
        }
        if value >= range.start {
            return Some(range.mapping_index);
        }
    }
    None
}

fn top_profiling_address_hits(
    mappings: &[RawCounterProgramAddressMapping],
    hits: &[usize],
    first_offsets: &[Option<usize>],
    scan_kind: &str,
) -> Vec<ProfilingAddressHit> {
    let mut rows = hits
        .iter()
        .enumerate()
        .filter_map(|(mapping_index, hit_count)| {
            let hit_count = *hit_count;
            (hit_count > 0).then(|| {
                let mapping = &mappings[mapping_index];
                ProfilingAddressHit {
                    hit_count,
                    scan_kind: scan_kind.to_owned(),
                    first_offset: first_offsets[mapping_index].unwrap_or_default(),
                    archive_index: mapping.archive_index,
                    mapping_index: mapping.mapping_index,
                    mapping_type: mapping.mapping_type.clone(),
                    binary_unique_id: mapping.binary_unique_id.clone(),
                    draw_call_index: mapping.draw_call_index,
                    draw_function_index: mapping.draw_function_index,
                    encoder_trace_id: mapping.encoder_trace_id,
                    encoder_index: mapping.encoder_index,
                    shader_index: mapping.shader_index,
                    mapped_address: mapping.mapped_address,
                    mapped_size: mapping.mapped_size,
                }
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .hit_count
            .cmp(&left.hit_count)
            .then_with(|| left.mapping_index.cmp(&right.mapping_index))
    });
    rows.truncate(64);
    rows
}

fn format_profiling_address_hits(out: &mut String, title: &str, hits: &[ProfilingAddressHit]) {
    out.push_str(&format!("\n{title}:\n"));
    if hits.is_empty() {
        out.push_str("  none\n");
        return;
    }
    for hit in hits.iter().take(32) {
        out.push_str(&format!(
            "  hits={} first_offset={} map={} enc_trace_id={} enc_index={} draw_function={} draw={} shader_index={} type={} binary={} addr={} size={}\n",
            hit.hit_count,
            hit.first_offset,
            hit.mapping_index,
            format_optional_u64(hit.encoder_trace_id),
            format_optional_u64(hit.encoder_index),
            format_optional_u64(hit.draw_function_index),
            format_optional_u64(hit.draw_call_index),
            format_optional_u64(hit.shader_index),
            hit.mapping_type,
            hit.binary_unique_id.as_deref().unwrap_or("-"),
            format_optional_u64_hex(hit.mapped_address),
            format_optional_u64(hit.mapped_size)
        ));
    }
}

fn top_shader_low32_hits(
    trace: &TraceBundle,
    mappings: &[RawCounterProgramAddressMapping],
    low32_hits: &[usize],
) -> Vec<ProfilingShaderAddressHit> {
    #[derive(Default)]
    struct Accum {
        hit_count: usize,
        encoder_index: Option<u64>,
        encoder_trace_id: Option<u64>,
        mapping_count: usize,
    }

    let profiler_summary = profiler::stream_data_summary(&trace.path).ok();
    let mut by_dispatch = BTreeMap::<usize, Accum>::new();
    for (mapping, hit_count) in mappings.iter().zip(low32_hits.iter().copied()) {
        if hit_count == 0 || !mapping.mapping_type.starts_with("compute") {
            continue;
        }
        let Some(dispatch_index) = mapping
            .draw_call_index
            .and_then(|value| usize::try_from(value).ok())
        else {
            continue;
        };
        let entry = by_dispatch.entry(dispatch_index).or_default();
        entry.hit_count += hit_count;
        entry.encoder_index = entry.encoder_index.or(mapping.encoder_index);
        entry.encoder_trace_id = entry.encoder_trace_id.or(mapping.encoder_trace_id);
        entry.mapping_count += 1;
    }

    let mut rows = by_dispatch
        .into_iter()
        .map(|(dispatch_index, accum)| {
            let function_name = profiler_summary
                .as_ref()
                .and_then(|summary| summary.dispatches.get(dispatch_index))
                .and_then(|dispatch| dispatch.function_name.clone());
            ProfilingShaderAddressHit {
                hit_count: accum.hit_count,
                dispatch_index,
                function_name,
                encoder_index: accum.encoder_index,
                encoder_trace_id: accum.encoder_trace_id,
                mapping_count: accum.mapping_count,
            }
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .hit_count
            .cmp(&left.hit_count)
            .then_with(|| left.dispatch_index.cmp(&right.dispatch_index))
    });
    rows.truncate(64);
    rows
}

fn top_function_low32_hits(
    trace: &TraceBundle,
    mappings: &[RawCounterProgramAddressMapping],
    low32_hits: &[usize],
) -> Vec<ProfilingFunctionAddressHit> {
    #[derive(Default)]
    struct Accum {
        hit_count: usize,
        dispatches: BTreeSet<usize>,
        mapping_count: usize,
    }

    let profiler_summary = profiler::stream_data_summary(&trace.path).ok();
    let mut by_function = BTreeMap::<String, Accum>::new();
    for (mapping, hit_count) in mappings.iter().zip(low32_hits.iter().copied()) {
        if hit_count == 0 || !mapping.mapping_type.starts_with("compute") {
            continue;
        }
        let Some(dispatch_index) = mapping
            .draw_call_index
            .and_then(|value| usize::try_from(value).ok())
        else {
            continue;
        };
        let function_name = profiler_summary
            .as_ref()
            .and_then(|summary| summary.dispatches.get(dispatch_index))
            .and_then(|dispatch| dispatch.function_name.clone())
            .unwrap_or_else(|| format!("dispatch_{dispatch_index}"));
        let entry = by_function.entry(function_name).or_default();
        entry.hit_count += hit_count;
        entry.dispatches.insert(dispatch_index);
        entry.mapping_count += 1;
    }

    let mut rows = by_function
        .into_iter()
        .map(|(function_name, accum)| ProfilingFunctionAddressHit {
            hit_count: accum.hit_count,
            function_name,
            dispatch_count: accum.dispatches.len(),
            mapping_count: accum.mapping_count,
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .hit_count
            .cmp(&left.hit_count)
            .then_with(|| left.function_name.cmp(&right.function_name))
    });
    rows.truncate(64);
    rows
}

fn probe_program_address_mappings(trace_path: &Path) -> Vec<RawCounterProgramAddressMapping> {
    load_stream_archive_groups(trace_path)
        .as_ref()
        .map(probe_program_address_mappings_from_groups)
        .unwrap_or_default()
}

fn probe_program_address_mappings_from_groups(
    groups: &StreamArchiveGroups,
) -> Vec<RawCounterProgramAddressMapping> {
    let mut mappings = Vec::new();
    for (archive_index, bytes) in groups.aps_data.iter().enumerate() {
        let Some(keyed) = parse_keyed_archive_dictionary(bytes) else {
            continue;
        };
        let Some(StreamArchiveValue::Array(entries)) = keyed.get("Program Address Mappings") else {
            continue;
        };
        for (mapping_index, entry) in entries.iter().enumerate() {
            let Some(entry) = entry.as_dictionary() else {
                continue;
            };
            mappings.push(RawCounterProgramAddressMapping {
                archive_index,
                mapping_index,
                mapping_type: stream_dict_string(entry, "type")
                    .unwrap_or("unknown")
                    .to_owned(),
                binary_unique_id: stream_dict_string(entry, "binaryUniqueId")
                    .map(ToOwned::to_owned),
                draw_call_index: stream_dict_u64(entry, "drawCallIndex"),
                draw_function_index: stream_dict_u64(entry, "drawFunctionIndex"),
                encoder_trace_id: stream_dict_u64(entry, "encID"),
                encoder_index: stream_dict_u64(entry, "encIndex"),
                shader_index: stream_dict_u64(entry, "index"),
                mapped_address: stream_dict_u64(entry, "mappedAddress"),
                mapped_size: stream_dict_u64(entry, "mappedSize"),
            });
        }
    }
    mappings.sort_by(|left, right| {
        left.archive_index
            .cmp(&right.archive_index)
            .then_with(|| left.encoder_trace_id.cmp(&right.encoder_trace_id))
            .then_with(|| left.draw_function_index.cmp(&right.draw_function_index))
            .then_with(|| left.mapping_index.cmp(&right.mapping_index))
    });
    mappings
}

fn stream_dict_u64(values: &BTreeMap<String, StreamArchiveValue>, key: &str) -> Option<u64> {
    values.get(key).and_then(StreamArchiveValue::as_u64)
}

fn stream_dict_string<'a>(
    values: &'a BTreeMap<String, StreamArchiveValue>,
    key: &str,
) -> Option<&'a str> {
    values.get(key).and_then(StreamArchiveValue::as_string)
}

fn parse_trace_id_sample_index_map(
    archive_index: usize,
    data: &[u8],
) -> Vec<RawCounterSampleTraceIndex> {
    let Ok(plist) = Value::from_reader(Cursor::new(data)) else {
        return Vec::new();
    };
    let Some(archive) = plist.as_dictionary() else {
        return Vec::new();
    };
    let Some(objects) = archive.get("$objects").and_then(Value::as_array) else {
        return Vec::new();
    };
    let Some(top) = archive.get("$top").and_then(Value::as_dictionary) else {
        return Vec::new();
    };
    let Some(root_uid) = top.get("root").and_then(as_uid) else {
        return Vec::new();
    };
    let Some(root) = object_dictionary(objects, root_uid) else {
        return Vec::new();
    };
    let Some(keys) = root.get("NS.keys").and_then(Value::as_array) else {
        return Vec::new();
    };
    let Some(values) = root.get("NS.objects").and_then(Value::as_array) else {
        return Vec::new();
    };
    if keys.len() != values.len() {
        return Vec::new();
    }

    let mut entries = Vec::new();
    for (key, value) in keys.iter().zip(values.iter()) {
        let Some(trace_id) = resolve_value(objects, key).and_then(as_u64) else {
            continue;
        };
        let Some(value) = resolve_value(objects, value) else {
            continue;
        };
        let Some(words) = ns_number_array(objects, value) else {
            continue;
        };
        entries.push(RawCounterSampleTraceIndex {
            archive_index,
            trace_id,
            sample_index: words.get(3).and_then(|value| (*value).try_into().ok()),
            words,
        });
    }
    entries
}

fn parse_trace_map_entries(
    archive_index: usize,
    map_name: &str,
    data: &[u8],
) -> Vec<RawCounterTraceMapEntry> {
    let Ok(plist) = Value::from_reader(Cursor::new(data)) else {
        return Vec::new();
    };
    let Some(archive) = plist.as_dictionary() else {
        return Vec::new();
    };
    let Some(objects) = archive.get("$objects").and_then(Value::as_array) else {
        return Vec::new();
    };
    let Some(top) = archive.get("$top").and_then(Value::as_dictionary) else {
        return Vec::new();
    };
    let Some(root_uid) = top.get("root").and_then(as_uid) else {
        return Vec::new();
    };
    let Some(root) = object_dictionary(objects, root_uid) else {
        return Vec::new();
    };
    let Some(keys) = root.get("NS.keys").and_then(Value::as_array) else {
        return Vec::new();
    };
    let Some(values) = root.get("NS.objects").and_then(Value::as_array) else {
        return Vec::new();
    };
    if keys.len() != values.len() {
        return Vec::new();
    }

    let mut entries = Vec::new();
    for (key, value) in keys.iter().zip(values.iter()) {
        let Some(trace_id) = resolve_value(objects, key).and_then(as_u64) else {
            continue;
        };
        let Some(value) = resolve_value(objects, value) else {
            continue;
        };
        let scalar_value = as_u64(value);
        let words = ns_number_array(objects, value).unwrap_or_default();
        entries.push(RawCounterTraceMapEntry {
            archive_index,
            map_name: map_name.to_owned(),
            trace_id,
            scalar_value,
            words,
        });
    }
    entries
}

fn ns_number_array(objects: &[Value], value: &Value) -> Option<Vec<u64>> {
    let values = value
        .as_dictionary()
        .and_then(|dict| dict.get("NS.objects"))
        .and_then(Value::as_array)
        .or_else(|| value.as_array())?;
    Some(
        values
            .iter()
            .filter_map(|value| resolve_value(objects, value))
            .filter_map(as_u64)
            .collect(),
    )
}

fn array_integer(
    keyed: &BTreeMap<String, StreamArchiveValue>,
    key: &str,
    index: usize,
) -> Option<u64> {
    let StreamArchiveValue::Array(values) = keyed.get(key)? else {
        return None;
    };
    values.get(index).and_then(StreamArchiveValue::as_u64)
}

fn dictionary_u64_values(values: &BTreeMap<String, StreamArchiveValue>) -> BTreeMap<String, u64> {
    values
        .iter()
        .filter_map(|(key, value)| value.as_u64().map(|value| (key.clone(), value)))
        .collect()
}

fn parse_encoder_sample_indices(data: &[u8]) -> Vec<RawCounterEncoderSampleIndex> {
    data.chunks_exact(16)
        .enumerate()
        .map(|(row_index, chunk)| RawCounterEncoderSampleIndex {
            row_index,
            word0: u32::from_le_bytes(chunk[0..4].try_into().unwrap()),
            word1: u32::from_le_bytes(chunk[4..8].try_into().unwrap()),
            sample_index: u32::from_le_bytes(chunk[8..12].try_into().unwrap()),
            word3: u32::from_le_bytes(chunk[12..16].try_into().unwrap()),
        })
        .collect()
}

fn parse_encoder_infos(value: &StreamArchiveValue) -> Vec<RawCounterEncoderInfo> {
    let StreamArchiveValue::Array(values) = value else {
        return Vec::new();
    };
    values
        .iter()
        .enumerate()
        .filter_map(|(row_index, value)| {
            let data = value.as_data()?;
            let trace_ids = data
                .chunks_exact(mem::size_of::<u32>())
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            Some(RawCounterEncoderInfo {
                row_index,
                trace_ids,
            })
        })
        .collect()
}

fn probe_stream_archives(trace_path: &Path) -> Vec<RawCounterStreamArchive> {
    load_stream_archive_groups(trace_path)
        .as_ref()
        .map(probe_stream_archives_from_groups)
        .unwrap_or_default()
}

fn probe_stream_archives_from_groups(groups: &StreamArchiveGroups) -> Vec<RawCounterStreamArchive> {
    [
        ("APSData", &groups.aps_data),
        ("APSCounterData", &groups.aps_counter_data),
        ("APSTimelineData", &groups.aps_timeline_data),
    ]
    .into_iter()
    .flat_map(|(group, archives)| {
        archives
            .iter()
            .enumerate()
            .filter_map(move |(index, bytes)| summarize_stream_archive(group, index, bytes))
    })
    .collect()
}

fn load_stream_archive_groups(trace_path: &Path) -> Option<StreamArchiveGroups> {
    let Some(profiler_dir) = profiler::find_profiler_directory(trace_path) else {
        return None;
    };
    let stream_data_path = profiler_dir.join("streamData");
    let Ok(plist) = Value::from_file(stream_data_path) else {
        return None;
    };
    let archive = plist.as_dictionary()?;
    let objects = archive.get("$objects").and_then(Value::as_array)?;
    let root = objects.get(1).and_then(Value::as_dictionary)?;

    Some(StreamArchiveGroups {
        aps_data: ns_data_array_from_root_key(objects, root, "APSData"),
        aps_counter_data: ns_data_array_from_root_key(objects, root, "APSCounterData"),
        aps_timeline_data: ns_data_array_from_root_key(objects, root, "APSTimelineData"),
    })
}

fn probe_structured_counter_samples(
    trace_path: &Path,
    targets: &[RawCounterProbeTarget],
) -> Vec<RawCounterStructuredSample> {
    if targets.is_empty() {
        return Vec::new();
    }
    let Some(profiler_dir) = profiler::find_profiler_directory(trace_path) else {
        return Vec::new();
    };
    let stream_data_path = profiler_dir.join("streamData");
    let Ok(plist) = Value::from_file(stream_data_path) else {
        return Vec::new();
    };
    let Some(archive) = plist.as_dictionary() else {
        return Vec::new();
    };
    let Some(objects) = archive.get("$objects").and_then(Value::as_array) else {
        return Vec::new();
    };
    let Some(root) = objects.get(1).and_then(Value::as_dictionary) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for (archive_index, bytes) in ns_data_array_from_root_key(objects, root, "APSCounterData")
        .into_iter()
        .enumerate()
    {
        let Some(keyed) = parse_keyed_archive_dictionary(&bytes) else {
            continue;
        };
        let Some(samples) = keyed.get("Derived Counter Sample Data") else {
            continue;
        };
        let mut blobs = Vec::new();
        collect_data_blobs(
            &format!("APSCounterData[{archive_index}]/Derived Counter Sample Data"),
            samples,
            &mut blobs,
        );
        for (path, data) in blobs {
            if !data.starts_with(b"GPRWCNTR") {
                continue;
            }
            let matches = probe_counter_targets(&data, &[], targets);
            if matches.is_empty() {
                continue;
            }
            let (gprw_record_size, gprw_record_count) = gprw_record_info(&data).unwrap_or((0, 0));
            out.push(RawCounterStructuredSample {
                path,
                byte_len: data.len(),
                gprw_record_size: (gprw_record_size > 0).then_some(gprw_record_size),
                gprw_record_count: (gprw_record_count > 0).then_some(gprw_record_count),
                matches,
            });
        }
    }
    out.sort_by(|left, right| {
        let left_hits: usize = left.matches.iter().map(|matched| matched.count).sum();
        let right_hits: usize = right.matches.iter().map(|matched| matched.count).sum();
        right_hits
            .cmp(&left_hits)
            .then_with(|| left.path.cmp(&right.path))
    });
    out
}

fn probe_structured_counter_layouts(trace_path: &Path) -> Vec<RawCounterStructuredLayout> {
    load_stream_archive_groups(trace_path)
        .as_ref()
        .map(probe_structured_counter_layouts_from_groups)
        .unwrap_or_default()
}

fn probe_structured_counter_layouts_from_groups(
    groups: &StreamArchiveGroups,
) -> Vec<RawCounterStructuredLayout> {
    let mut layouts = Vec::new();
    for (archive_index, bytes) in groups.aps_counter_data.iter().enumerate() {
        let Some(keyed) = parse_keyed_archive_dictionary(bytes) else {
            continue;
        };
        let Some(samples) = keyed.get("Derived Counter Sample Data") else {
            continue;
        };
        let mut blobs = Vec::new();
        collect_data_blobs(
            &format!("APSCounterData[{archive_index}]/Derived Counter Sample Data"),
            samples,
            &mut blobs,
        );
        for (path, data) in blobs {
            let Some((record_size, record_count)) = gprw_record_info(&data) else {
                continue;
            };
            if record_count < 2 {
                continue;
            }
            layouts.push(RawCounterStructuredLayout {
                path,
                byte_len: data.len(),
                gprw_record_size: record_size,
                gprw_record_count: record_count,
                u64_columns: summarize_gprw_u64_columns(&data, record_size),
            });
        }
    }
    layouts.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then_with(|| right.gprw_record_count.cmp(&left.gprw_record_count))
    });
    layouts
}

fn probe_normalized_counter_metrics(trace_path: &Path) -> Vec<RawCounterNormalizedMetric> {
    let Some((counter_schemas, fallback_counter_names, sample_blobs)) =
        counter_schemas_and_sample_blobs(trace_path)
    else {
        return Vec::new();
    };
    probe_normalized_counter_metrics_from_parts(
        &counter_schemas,
        &fallback_counter_names,
        &sample_blobs,
    )
}

fn probe_normalized_counter_metrics_from_parts(
    counter_schemas: &CounterSchemaByGroup,
    fallback_counter_names: &[String],
    sample_blobs: &[SampleBlob],
) -> Vec<RawCounterNormalizedMetric> {
    if counter_schemas.is_empty() && fallback_counter_names.is_empty() {
        return Vec::new();
    }

    let mut accum = BTreeMap::<RawCounterMetricKey, RawCounterMetricAccum>::new();
    for (path, data) in sample_blobs {
        if !path.contains("/Derived Counter Sample Data/") {
            continue;
        }
        let Some((record_size, _)) = gprw_record_info(&data) else {
            continue;
        };
        let records = gprw_u64_records(data, record_size);
        if records.is_empty() {
            continue;
        }
        let path_ids = parse_derived_counter_sample_path(&path);
        let Some(counter_names) = path_ids
            .sample_group
            .and_then(|group| counter_schemas.get(&group).map(Vec::as_slice))
            .or((!fallback_counter_names.is_empty()).then_some(fallback_counter_names))
        else {
            continue;
        };
        for (counter_index, raw_name) in counter_names.iter().enumerate() {
            let value_column = 8 + counter_index;
            for record in &records {
                let Some(denominator) = record.get(2).copied() else {
                    continue;
                };
                let Some(value) = record.get(value_column).copied() else {
                    continue;
                };
                if denominator == 0 {
                    continue;
                }
                accum
                    .entry(RawCounterMetricKey {
                        path: path.clone(),
                        sample_group: path_ids.sample_group,
                        source_index: path_ids.source_index,
                        ring_index: path_ids.ring_index,
                        counter_index,
                        raw_name: raw_name.clone(),
                    })
                    .or_default()
                    .push(
                        value as f64 / denominator as f64 * 100.0,
                        record.get(4).copied(),
                        record.get(5).copied(),
                        record.get(7).copied(),
                    );
            }
        }
    }
    let mut metrics = accum
        .into_iter()
        .map(|(key, accum)| {
            let max_percent = accum.values.iter().copied().fold(0.0, f64::max);
            let min_percent = accum.values.iter().copied().fold(f64::INFINITY, f64::min);
            let mean_percent = accum.values.iter().sum::<f64>() / accum.values.len() as f64;
            RawCounterNormalizedMetric {
                path: key.path,
                sample_group: key.sample_group,
                source_index: key.source_index,
                ring_index: key.ring_index,
                encoder_ids: accum.encoder_ids.into_iter().collect(),
                kick_trace_ids: accum.kick_trace_ids.into_iter().collect(),
                source_ids: accum.source_ids.into_iter().collect(),
                counter_index: key.counter_index,
                raw_name: key.raw_name,
                sample_count: accum.values.len(),
                min_percent,
                mean_percent,
                max_percent,
            }
        })
        .collect::<Vec<_>>();
    metrics.sort_by(|left, right| {
        right
            .mean_percent
            .partial_cmp(&left.mean_percent)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.path.cmp(&right.path))
            .then_with(|| left.counter_index.cmp(&right.counter_index))
    });
    metrics
}

fn match_normalized_counter_targets(
    counters: &[RawCounterNormalizedMetric],
    targets: &[RawCounterProbeTarget],
) -> Vec<RawCounterNormalizedMatch> {
    let mut matches = Vec::new();
    for target in targets {
        if target.value.abs() < 5.0 {
            continue;
        }
        let tolerance = normalized_match_tolerance(target.value);
        for counter in counters {
            if counter.sample_count < 20 {
                continue;
            }
            let delta = (counter.mean_percent - target.value).abs();
            if delta > tolerance {
                continue;
            }
            matches.push(RawCounterNormalizedMatch {
                metric: target.metric.clone(),
                row_index: target.row_index,
                encoder_label: target.encoder_label.clone(),
                target: target.value,
                delta,
                tolerance,
                confidence: (1.0 - delta / tolerance).clamp(0.0, 1.0),
                counter: counter.clone(),
            });
        }
    }
    matches.sort_by(|left, right| {
        right
            .confidence
            .partial_cmp(&left.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left.delta
                    .partial_cmp(&right.delta)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left.metric.cmp(&right.metric))
            .then_with(|| left.row_index.cmp(&right.row_index))
    });
    matches
}

fn normalized_match_tolerance(target: f64) -> f64 {
    (target.abs() * 0.10).max(2.0)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RawCounterMetricKey {
    path: String,
    sample_group: Option<usize>,
    source_index: Option<usize>,
    ring_index: Option<usize>,
    counter_index: usize,
    raw_name: String,
}

#[derive(Debug, Default)]
struct RawCounterMetricAccum {
    values: Vec<f64>,
    encoder_ids: BTreeSet<u64>,
    kick_trace_ids: BTreeSet<u64>,
    source_ids: BTreeSet<u64>,
}

impl RawCounterMetricAccum {
    fn push(
        &mut self,
        value: f64,
        encoder_id: Option<u64>,
        kick_trace_id: Option<u64>,
        source_id: Option<u64>,
    ) {
        self.values.push(value);
        if let Some(encoder_id) = encoder_id {
            self.encoder_ids.insert(encoder_id);
        }
        if let Some(kick_trace_id) = kick_trace_id {
            self.kick_trace_ids.insert(kick_trace_id);
        }
        if let Some(source_id) = source_id {
            self.source_ids.insert(source_id);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct DerivedCounterSamplePath {
    sample_group: Option<usize>,
    source_index: Option<usize>,
    ring_index: Option<usize>,
}

fn parse_derived_counter_sample_path(path: &str) -> DerivedCounterSamplePath {
    let mut parts = path.split('/');
    while let Some(part) = parts.next() {
        if part == "Derived Counter Sample Data" {
            return DerivedCounterSamplePath {
                sample_group: parts.next().and_then(|part| part.parse().ok()),
                source_index: parts.next().and_then(|part| part.parse().ok()),
                ring_index: parts.next().and_then(|part| part.parse().ok()),
            };
        }
    }
    DerivedCounterSamplePath {
        sample_group: None,
        source_index: None,
        ring_index: None,
    }
}

fn counter_schemas_and_sample_blobs(
    trace_path: &Path,
) -> Option<(CounterSchemaByGroup, Vec<String>, Vec<SampleBlob>)> {
    let groups = load_stream_archive_groups(trace_path)?;
    Some(counter_schemas_and_sample_blobs_from_groups(&groups))
}

fn counter_schemas_and_sample_blobs_from_groups(
    groups: &StreamArchiveGroups,
) -> (CounterSchemaByGroup, Vec<String>, Vec<SampleBlob>) {
    let mut counter_schemas = BTreeMap::new();
    let mut fallback_counter_names = Vec::new();
    let mut sample_blobs = Vec::new();
    for (archive_index, bytes) in groups.aps_counter_data.iter().enumerate() {
        let Some(keyed) = parse_keyed_archive_dictionary(bytes) else {
            continue;
        };
        if fallback_counter_names.is_empty()
            && let Some(names) = keyed.get("limiter sample counters")
        {
            fallback_counter_names = string_array_values(names);
        }
        if counter_schemas.is_empty()
            && let Some(subdivided) = keyed.get("Subdivided Dictionary")
        {
            counter_schemas = counter_schemas_from_subdivided_dictionary(subdivided);
        }
        if let Some(samples) = keyed.get("Derived Counter Sample Data") {
            collect_data_blobs(
                &format!("APSCounterData[{archive_index}]/Derived Counter Sample Data"),
                samples,
                &mut sample_blobs,
            );
        }
    }
    (counter_schemas, fallback_counter_names, sample_blobs)
}

fn counter_schemas_from_subdivided_dictionary(value: &StreamArchiveValue) -> CounterSchemaByGroup {
    let mut schemas = BTreeMap::new();
    let Some(pass_list) = value
        .as_dictionary()
        .and_then(|values| values.get("passList"))
    else {
        return schemas;
    };
    let StreamArchiveValue::Array(passes) = pass_list else {
        return schemas;
    };
    for (group_index, pass) in passes.iter().enumerate() {
        let Some(names) = first_counter_schema(pass) else {
            continue;
        };
        let raw_names = names.into_iter().skip(7).collect::<Vec<_>>();
        if !raw_names.is_empty() {
            schemas.insert(group_index, raw_names);
        }
    }
    schemas
}

fn first_counter_schema(value: &StreamArchiveValue) -> Option<Vec<String>> {
    match value {
        StreamArchiveValue::Array(children) => {
            let direct = string_array_values(value);
            if direct.first().is_some_and(|name| name == "GRC_TIMESTAMP") && direct.len() > 7 {
                return Some(direct);
            }
            children.iter().find_map(first_counter_schema)
        }
        StreamArchiveValue::Dictionary(values) => values.values().find_map(first_counter_schema),
        _ => None,
    }
}

fn string_array_values(value: &StreamArchiveValue) -> Vec<String> {
    let StreamArchiveValue::Array(children) = value else {
        return Vec::new();
    };
    children
        .iter()
        .filter_map(|child| match child {
            StreamArchiveValue::String(value) => Some(value.clone()),
            _ => None,
        })
        .collect()
}

fn gprw_u64_records(data: &[u8], record_size: usize) -> Vec<Vec<u64>> {
    let mut records = Vec::new();
    for offset in gprw_magic_offsets(data) {
        let Some(record) = data.get(offset..offset + record_size) else {
            continue;
        };
        records.push(
            record
                .chunks_exact(mem::size_of::<u64>())
                .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        );
    }
    records
}

fn summarize_gprw_u64_columns(data: &[u8], record_size: usize) -> Vec<RawCounterColumnStat> {
    if record_size < mem::size_of::<u64>() {
        return Vec::new();
    }
    let columns = record_size / mem::size_of::<u64>();
    let mut values_by_column = vec![Vec::<u64>::new(); columns];
    for offset in gprw_magic_offsets(data) {
        let Some(record) = data.get(offset..offset + record_size) else {
            continue;
        };
        for (index, chunk) in record.chunks_exact(mem::size_of::<u64>()).enumerate() {
            values_by_column[index].push(u64::from_le_bytes(chunk.try_into().unwrap()));
        }
    }
    values_by_column
        .into_iter()
        .enumerate()
        .filter_map(|(index, values)| summarize_u64_column(index, values))
        .collect()
}

fn summarize_u64_column(index: usize, values: Vec<u64>) -> Option<RawCounterColumnStat> {
    if values.is_empty() {
        return None;
    }
    let min = values.iter().min().copied()?;
    let max = values.iter().max().copied()?;
    let total = values
        .iter()
        .fold(0.0, |total, value| total + *value as f64);
    let nonzero_count = values.iter().filter(|value| **value != 0).count();
    Some(RawCounterColumnStat {
        index,
        min,
        max,
        mean: total / values.len() as f64,
        nonzero_count,
        first_values: values.into_iter().take(6).collect(),
    })
}

fn collect_data_blobs(path: &str, value: &StreamArchiveValue, out: &mut Vec<(String, Vec<u8>)>) {
    match value {
        StreamArchiveValue::Data(data) => out.push((path.to_owned(), data.clone())),
        StreamArchiveValue::Array(children) => {
            for (index, child) in children.iter().enumerate() {
                collect_data_blobs(&format!("{path}/{index}"), child, out);
            }
        }
        _ => {}
    }
}

fn summarize_stream_archive(
    group: &str,
    index: usize,
    bytes: &[u8],
) -> Option<RawCounterStreamArchive> {
    let keyed = parse_keyed_archive_dictionary(bytes)?;
    let keys = keyed.keys().cloned().collect::<Vec<_>>();
    let data_file = keyed
        .get("APSCounterDataFile")
        .or_else(|| keyed.get("APSTraceDataFile"))
        .or_else(|| keyed.get("File"))
        .and_then(StreamArchiveValue::as_string)
        .map(ToOwned::to_owned);
    Some(RawCounterStreamArchive {
        group: group.to_owned(),
        index,
        byte_len: bytes.len(),
        source: keyed
            .get("Source")
            .and_then(StreamArchiveValue::as_string)
            .map(ToOwned::to_owned),
        serial: keyed.get("Serial").and_then(StreamArchiveValue::as_u64),
        source_index: keyed
            .get("SourceIndex")
            .and_then(StreamArchiveValue::as_u64),
        ring_buffer_index: keyed
            .get("RingBufferIndex")
            .and_then(StreamArchiveValue::as_u64),
        data_file,
        shader_profiler_data_len: keyed
            .get("ShaderProfilerData")
            .and_then(StreamArchiveValue::as_data_len),
        fields: keyed
            .iter()
            .map(|(key, value)| summarize_field(key, value))
            .collect(),
        data_fields: keyed
            .iter()
            .filter_map(|(key, value)| {
                let StreamArchiveValue::Data(data) = value else {
                    return None;
                };
                Some(summarize_data_field(key, data))
            })
            .collect(),
        keys,
    })
}

fn summarize_field(key: &str, value: &StreamArchiveValue) -> RawCounterFieldSummary {
    let (kind, len, keys) = match value {
        StreamArchiveValue::String(_) => ("string", None, Vec::new()),
        StreamArchiveValue::Integer(_) => ("integer", None, Vec::new()),
        StreamArchiveValue::Data(data) => ("data", Some(data.len()), Vec::new()),
        StreamArchiveValue::Array(children) => ("array", Some(children.len()), Vec::new()),
        StreamArchiveValue::Dictionary(values) => (
            "dictionary",
            Some(values.len()),
            values.keys().cloned().collect(),
        ),
        StreamArchiveValue::Other => ("other", None, Vec::new()),
    };
    let children = match value {
        StreamArchiveValue::Array(children) => children
            .iter()
            .take(32)
            .enumerate()
            .map(|(index, child)| format!("{index}:{}", child.short_summary()))
            .collect(),
        _ => Vec::new(),
    };
    RawCounterFieldSummary {
        key: key.to_owned(),
        kind: kind.to_owned(),
        len,
        keys,
        children,
    }
}

#[derive(Debug, Clone, PartialEq)]
enum StreamArchiveValue {
    String(String),
    Integer(u64),
    Data(Vec<u8>),
    Array(Vec<StreamArchiveValue>),
    Dictionary(BTreeMap<String, StreamArchiveValue>),
    Other,
}

impl StreamArchiveValue {
    fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Integer(value) => Some(*value),
            _ => None,
        }
    }

    fn as_data_len(&self) -> Option<usize> {
        match self {
            Self::Data(value) => Some(value.len()),
            _ => None,
        }
    }

    fn as_data(&self) -> Option<&[u8]> {
        match self {
            Self::Data(value) => Some(value),
            _ => None,
        }
    }

    fn short_summary(&self) -> String {
        match self {
            Self::String(value) => format!("string:{value}"),
            Self::Integer(value) => format!("integer:{value}"),
            Self::Data(data) => match summarize_gprw_data(data) {
                Some(summary) => summary,
                None => format!("data:{}:{}", data.len(), format_hex_prefix(data, 16)),
            },
            Self::Array(children) => {
                let nested = children
                    .iter()
                    .take(8)
                    .map(StreamArchiveValue::short_summary)
                    .collect::<Vec<_>>()
                    .join("|");
                format!("array:{}:[{}]", children.len(), nested)
            }
            Self::Dictionary(values) => format!(
                "dictionary:{}:{}",
                values.len(),
                values.keys().cloned().collect::<Vec<_>>().join("|")
            ),
            Self::Other => "other".to_owned(),
        }
    }

    fn as_dictionary(&self) -> Option<&BTreeMap<String, StreamArchiveValue>> {
        match self {
            Self::Dictionary(values) => Some(values),
            _ => None,
        }
    }
}

fn summarize_gprw_data(data: &[u8]) -> Option<String> {
    let (record_size, record_count) = gprw_record_info(data)?;
    let magic_offsets = gprw_magic_offsets(data);
    let repeated_magic = magic_offsets.len() > 1;
    let first = if repeated_magic {
        data.get(..record_size)?
    } else {
        data.get(8..8 + record_size)?
    };
    let fields = preview_u64_values(first, 8)
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join("|");
    let magic_deltas = summarize_magic_deltas(&magic_offsets);
    Some(format!(
        "gprw:{}:record_size={record_size}:records={record_count}:magics={}:deltas={}:u64={fields}",
        data.len(),
        magic_offsets.len(),
        magic_deltas
    ))
}

fn gprw_record_info(data: &[u8]) -> Option<(usize, usize)> {
    if data.get(..8)? != b"GPRWCNTR" {
        return None;
    }
    let magic_offsets = gprw_magic_offsets(data);
    if let Some(record_size) = dominant_magic_delta(&magic_offsets)
        && record_size > 0
    {
        return Some((record_size, magic_offsets.len()));
    }
    let record_size = 168;
    let record_count = data[8..].len() / record_size;
    Some((record_size, record_count))
}

fn gprw_magic_offsets(data: &[u8]) -> Vec<usize> {
    data.windows(8)
        .enumerate()
        .filter_map(|(offset, window)| (window == b"GPRWCNTR").then_some(offset))
        .collect()
}

fn summarize_magic_deltas(offsets: &[usize]) -> String {
    let mut counts = BTreeMap::<usize, usize>::new();
    for pair in offsets.windows(2) {
        *counts.entry(pair[1] - pair[0]).or_default() += 1;
    }
    let mut counts = counts.into_iter().collect::<Vec<_>>();
    counts.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    counts
        .into_iter()
        .take(4)
        .map(|(delta, count)| format!("{delta}x{count}"))
        .collect::<Vec<_>>()
        .join("|")
}

fn dominant_magic_delta(offsets: &[usize]) -> Option<usize> {
    let mut counts = BTreeMap::<usize, usize>::new();
    for pair in offsets.windows(2) {
        *counts.entry(pair[1] - pair[0]).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(&left.0)))
        .map(|(delta, _)| delta)
}

fn parse_keyed_archive_dictionary(bytes: &[u8]) -> Option<BTreeMap<String, StreamArchiveValue>> {
    let plist = Value::from_reader(Cursor::new(bytes)).ok()?;
    let archive = plist.as_dictionary()?;
    let objects = archive.get("$objects").and_then(Value::as_array)?;
    let top = archive.get("$top").and_then(Value::as_dictionary)?;
    let root_uid = top.get("root").and_then(as_uid)?;
    let root = object_dictionary(objects, root_uid)?;
    keyed_dictionary_values(objects, root)
}

fn keyed_dictionary_values(
    objects: &[Value],
    root: &Dictionary,
) -> Option<BTreeMap<String, StreamArchiveValue>> {
    let keys = root.get("NS.keys").and_then(Value::as_array)?;
    let values = root.get("NS.objects").and_then(Value::as_array)?;
    if keys.len() != values.len() {
        return None;
    }

    let mut out = BTreeMap::new();
    for (key, value) in keys.iter().zip(values.iter()) {
        let key_uid = as_uid(key)?;
        let key_name = object(objects, key_uid).and_then(Value::as_string)?;
        let resolved = resolve_value(objects, value)?;
        out.insert(
            key_name.to_owned(),
            summarize_stream_archive_value(objects, resolved),
        );
    }
    Some(out)
}

fn summarize_stream_archive_value(objects: &[Value], value: &Value) -> StreamArchiveValue {
    if let Some(value) = value.as_string() {
        return StreamArchiveValue::String(value.to_owned());
    }
    if let Some(value) = as_u64(value) {
        return StreamArchiveValue::Integer(value);
    }
    if let Some(data) = ns_data_from_value(value) {
        return StreamArchiveValue::Data(data.to_vec());
    }
    if let Some(dict) = value.as_dictionary()
        && let Some(values) = keyed_dictionary_values(objects, dict)
    {
        return StreamArchiveValue::Dictionary(values);
    }
    if let Some(array) = value
        .as_dictionary()
        .and_then(|dict| dict.get("NS.objects"))
        .and_then(Value::as_array)
        .or_else(|| value.as_array())
    {
        return StreamArchiveValue::Array(
            array
                .iter()
                .filter_map(|value| resolve_value(objects, value))
                .map(|value| summarize_stream_archive_value(objects, value))
                .collect(),
        );
    }
    StreamArchiveValue::Other
}

fn summarize_data_field(key: &str, data: &[u8]) -> RawCounterDataField {
    RawCounterDataField {
        key: key.to_owned(),
        byte_len: data.len(),
        prefix_hex: format_hex_prefix(data, 32),
        f32_preview: preview_f32_values(data, 8),
        u32_preview: preview_u32_values(data, 8),
        u64_preview: preview_u64_values(data, 4),
    }
}

fn format_hex_prefix(data: &[u8], max_len: usize) -> String {
    data.iter()
        .take(max_len)
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join("")
}

fn preview_f32_values(data: &[u8], max_count: usize) -> Vec<f64> {
    data.chunks_exact(mem::size_of::<f32>())
        .take(max_count)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()) as f64)
        .collect()
}

fn preview_u32_values(data: &[u8], max_count: usize) -> Vec<u32> {
    data.chunks_exact(mem::size_of::<u32>())
        .take(max_count)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn preview_u64_values(data: &[u8], max_count: usize) -> Vec<u64> {
    data.chunks_exact(mem::size_of::<u64>())
        .take(max_count)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn ns_data_array_from_root_key(objects: &[Value], root: &Dictionary, key: &str) -> Vec<Vec<u8>> {
    let Some(uid) = root.get(key).and_then(as_uid) else {
        return Vec::new();
    };
    let Some(array_dict) = object_dictionary(objects, uid) else {
        return Vec::new();
    };
    let Some(values) = array_dict.get("NS.objects").and_then(Value::as_array) else {
        return Vec::new();
    };

    values
        .iter()
        .filter_map(|value| resolve_value(objects, value))
        .filter_map(ns_data_from_value)
        .map(ToOwned::to_owned)
        .collect()
}

fn ns_data_from_value(value: &Value) -> Option<&[u8]> {
    value
        .as_dictionary()
        .and_then(|dict| dict.get("NS.data"))
        .and_then(Value::as_data)
        .or_else(|| value.as_data())
}

fn object(objects: &[Value], uid: Uid) -> Option<&Value> {
    objects.get(uid.get() as usize)
}

fn object_dictionary(objects: &[Value], uid: Uid) -> Option<&Dictionary> {
    object(objects, uid).and_then(Value::as_dictionary)
}

fn resolve_value<'a>(objects: &'a [Value], value: &'a Value) -> Option<&'a Value> {
    match value {
        Value::Uid(uid) => object(objects, *uid),
        value => Some(value),
    }
}

fn as_uid(value: &Value) -> Option<Uid> {
    match value {
        Value::Uid(uid) => Some(*uid),
        _ => None,
    }
}

fn as_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Integer(value) => value.as_unsigned().or_else(|| {
            value
                .as_signed()
                .and_then(|value| u64::try_from(value).ok())
        }),
        _ => None,
    }
}

/// Per-encoder limiters and utilizations of the limiter pass (see
/// [`crate::hw_counters`]): percentages in 0..100, bandwidths in GB/s.
pub fn extract_limiters(profiler_dir: &Path) -> Vec<CounterLimiter> {
    crate::hw_counters::cached_report(profiler_dir)
        .map(|report| {
            report
                .encoders
                .iter()
                .filter_map(|row| {
                    Some(CounterLimiter {
                        encoder_index: row.encoder_index?,
                        occupancy: row.percent("Compute Occupancy"),
                        alu_utilization: row.percent("ALU Utilization"),
                        compute_shader_launch: row.percent("Compute Shader Launch Limiter"),
                        instruction_throughput: row.percent("Instruction Issue Limiter"),
                        integer_complex: row.percent("IC Limiter"),
                        control_flow: row.percent("Control Flow Limiter"),
                        f32_limiter: row.percent("F32 Limiter"),
                        l1_cache: row.percent("L1 Cache Limiter"),
                        last_level_cache: row.percent("L2 Cache Limiter"),
                        device_memory_bandwidth_gbps: row.dram_gbps(),
                        buffer_l1_read_bandwidth_gbps: row.gbps("Buffer L1 Load Bandwidth"),
                        buffer_l1_write_bandwidth_gbps: row.gbps("Buffer L1 Store Bandwidth"),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

pub fn extract_limiters_for_trace(path: &Path) -> Vec<CounterLimiter> {
    profiler::find_profiler_directory(path)
        .map(|dir| extract_limiters(&dir))
        .unwrap_or_default()
}

fn is_probe_metric(metric: &str) -> bool {
    matches!(
        metric,
        "ALU Utilization"
            | "Kernel Occupancy"
            | "Occupancy Manager Target"
            | "Compute Shader Launch Limiter"
            | "Instruction Throughput Limiter"
            | "Integer & Complex Limiter"
            | "Control Flow Limiter"
            | "F32 Limiter"
            | "L1 Cache Limiter"
            | "Last Level Cache Limiter"
            | "Device Memory Bandwidth"
            | "GPU Read Bandwidth"
            | "GPU Write Bandwidth"
            | "Buffer L1 Miss Rate"
            | "Buffer L1 Read Accesses"
            | "Buffer L1 Read Bandwidth"
            | "Buffer L1 Write Accesses"
            | "Buffer L1 Write Bandwidth"
            | "Kernel Invocations"
    )
}

fn raw_probe_tolerance(value: f64) -> f64 {
    if value.abs() >= 1000.0 {
        (value.abs() * 0.001).max(1.0)
    } else {
        (value.abs() * 0.005).max(0.02)
    }
}

fn find_raw_counter_markers(data: &[u8]) -> Vec<usize> {
    const TAGS: &[u8] = &[0x0e, 0x2e, 0x4e, 0x6e, 0x8e, 0xae, 0xce, 0xee];
    let mut offsets = Vec::new();
    for offset in 0..data.len().saturating_sub(3) {
        if TAGS.contains(&data[offset])
            && data[offset + 1] == 0
            && data[offset + 2] == 0
            && data[offset + 3] == 0
        {
            offsets.push(offset);
        }
    }
    offsets
}

fn summarize_raw_counter_shapes(data: &[u8], markers: &[usize]) -> Vec<RawCounterRecordShape> {
    let mut counts = BTreeMap::<(u8, usize), usize>::new();
    for (index, offset) in markers.iter().enumerate() {
        let next = markers.get(index + 1).copied().unwrap_or(data.len());
        if next <= *offset {
            continue;
        }
        let tag = data[*offset];
        let size = next - *offset;
        *counts.entry((tag, size)).or_default() += 1;
    }
    let mut shapes = counts
        .into_iter()
        .map(|((tag, size), count)| RawCounterRecordShape {
            tag: format!("0x{tag:02x}"),
            size,
            count,
        })
        .collect::<Vec<_>>();
    shapes.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.size.cmp(&right.size))
            .then_with(|| left.tag.cmp(&right.tag))
    });
    shapes
}

fn probe_counter_targets(
    data: &[u8],
    markers: &[usize],
    targets: &[RawCounterProbeTarget],
) -> Vec<RawCounterProbeMatch> {
    let mut accum = BTreeMap::<(usize, &'static str), RawProbeAccum>::new();
    for offset in (0..data.len().saturating_sub(mem::size_of::<u32>())).step_by(4) {
        let float_value = f32::from_bits(u32::from_le_bytes(
            data[offset..offset + 4].try_into().unwrap(),
        )) as f64;
        if float_value.is_finite() {
            for (target_index, target) in targets.iter().enumerate() {
                if (float_value - target.value).abs() <= target.tolerance {
                    push_probe_match(
                        &mut accum,
                        target_index,
                        "f32",
                        data,
                        markers,
                        offset,
                        float_value,
                    );
                }
            }
        }

        if offset + mem::size_of::<u64>() <= data.len() {
            let uint_value = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            for (target_index, target) in targets.iter().enumerate() {
                if !target_allows_integer_encoding(&target.metric) {
                    continue;
                }
                let value = uint_value as f64;
                if (value - target.value).abs() <= target.tolerance {
                    push_probe_match(
                        &mut accum,
                        target_index,
                        "u64",
                        data,
                        markers,
                        offset,
                        value,
                    );
                }
            }
        }
    }

    let mut matches = accum
        .into_iter()
        .filter_map(|((target_index, encoding), accum)| {
            let target = targets.get(target_index)?;
            Some(RawCounterProbeMatch {
                metric: target.metric.clone(),
                row_index: target.row_index,
                encoder_label: target.encoder_label.clone(),
                target: target.value,
                tolerance: target.tolerance,
                encoding: encoding.to_owned(),
                count: accum.count,
                examples: accum.examples,
            })
        })
        .collect::<Vec<_>>();
    matches.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.metric.cmp(&right.metric))
            .then_with(|| left.row_index.cmp(&right.row_index))
            .then_with(|| left.encoding.cmp(&right.encoding))
    });
    matches
}

fn target_allows_integer_encoding(metric: &str) -> bool {
    metric.contains("Invocations") || metric.contains("Bytes")
}

#[derive(Default)]
struct RawProbeAccum {
    count: usize,
    examples: Vec<RawCounterProbeExample>,
}

fn push_probe_match(
    accum: &mut BTreeMap<(usize, &'static str), RawProbeAccum>,
    target_index: usize,
    encoding: &'static str,
    data: &[u8],
    markers: &[usize],
    offset: usize,
    value: f64,
) {
    let entry = accum.entry((target_index, encoding)).or_default();
    entry.count += 1;
    if entry.examples.len() >= 5 {
        return;
    }
    let (record_tag, record_size) = marker_for_offset(data, markers, offset)
        .map(|(tag, size)| (Some(format!("0x{tag:02x}")), Some(size)))
        .unwrap_or((None, None));
    entry.examples.push(RawCounterProbeExample {
        offset,
        page_4k: offset / 4096,
        value,
        record_tag,
        record_size,
    });
}

fn marker_for_offset(data: &[u8], markers: &[usize], offset: usize) -> Option<(u8, usize)> {
    let index = match markers.binary_search(&offset) {
        Ok(index) => index,
        Err(0) => return None,
        Err(index) => index - 1,
    };
    let start = markers[index];
    let next = markers.get(index + 1).copied().unwrap_or(data.len());
    (next > start).then_some((data[start], next - start))
}

#[cfg(test)]
mod tests {
    use super::*;
    use plist::Integer;
    use tempfile::tempdir;

    fn test_integer(value: u64) -> Value {
        Value::Integer(Integer::from(value))
    }

    fn gprw_blob(timestamps: &[u64]) -> Vec<u8> {
        let mut data = Vec::new();
        for timestamp in timestamps {
            let mut record = vec![0u8; 64];
            record[0..8].copy_from_slice(b"GPRWCNTR");
            record[8..16].copy_from_slice(&timestamp.to_le_bytes());
            record[16..24].copy_from_slice(&1_u64.to_le_bytes());
            data.extend_from_slice(&record);
        }
        data
    }

    #[test]
    fn parses_derived_counter_sample_group_source_and_ring() {
        let path = "APSCounterData[44]/Derived Counter Sample Data/10/1/0";

        let parsed = parse_derived_counter_sample_path(path);

        assert_eq!(parsed.sample_group, Some(10));
        assert_eq!(parsed.source_index, Some(1));
        assert_eq!(parsed.ring_index, Some(0));
    }

    #[test]
    fn matches_raw_counter_record_to_profiler_dispatch_window() {
        let dispatches = vec![
            profiler::ProfilerDispatch {
                index: 7,
                pipeline_index: 0,
                pipeline_id: Some(42),
                function_name: Some("kernel_a".to_owned()),
                encoder_index: 2,
                cumulative_us: 10,
                duration_us: 10,
                sample_count: 0,
                sampling_density: 0.0,
                start_ticks: 100,
                end_ticks: 200,
            },
            profiler::ProfilerDispatch {
                index: 8,
                pipeline_index: 0,
                pipeline_id: Some(43),
                function_name: Some("kernel_b".to_owned()),
                encoder_index: 2,
                cumulative_us: 20,
                duration_us: 10,
                sample_count: 0,
                sampling_density: 0.0,
                start_ticks: 201,
                end_ticks: 300,
            },
        ];
        let record = vec![0, 250];

        let matched = profiler_dispatch_for_raw_counter_record(&record, &dispatches).unwrap();

        assert_eq!(matched.index, 8);
        assert_eq!(matched.function_name.as_deref(), Some("kernel_b"));
    }

    #[test]
    fn reports_raw_counter_dispatch_timestamp_alignment() {
        let sample_blobs = vec![(
            "APSCounterData[0]/Derived Counter Sample Data/2/1/0".to_owned(),
            gprw_blob(&[10, 20, 30]),
        )];
        let dispatches = vec![profiler::ProfilerDispatch {
            index: 0,
            pipeline_index: 0,
            pipeline_id: Some(42),
            function_name: Some("kernel".to_owned()),
            encoder_index: 0,
            cumulative_us: 0,
            duration_us: 1,
            sample_count: 0,
            sampling_density: 0.0,
            start_ticks: 100,
            end_ticks: 120,
        }];
        let metadata = vec![RawCounterAggregateMetadata {
            archive_index: 0,
            timebase_numer: Some(2),
            timebase_denom: Some(1),
            num_encoders: Some(1),
            perf_info: BTreeMap::new(),
            encoder_sample_indices: Vec::new(),
            encoder_infos: Vec::new(),
        }];

        let report =
            raw_counter_dispatch_alignment_report(&sample_blobs, &dispatches, &metadata).unwrap();

        assert_eq!(report.direct_overlap_record_count, 0);
        assert_eq!(report.raw_end_to_dispatch_start_gap_ticks, 70);
        assert_eq!(report.raw_end_to_dispatch_start_gap_ns, Some(140.0));
        assert_eq!(report.source_spans[0].nearest_dispatch_gap_ticks, 70);
        let candidate = report
            .candidate_shifts
            .iter()
            .find(|candidate| candidate.name == "raw_end_to_dispatch_start")
            .unwrap();
        assert_eq!(candidate.overlap_record_count, 1);
        assert_eq!(candidate.dispatches_covered, 1);
    }

    #[test]
    fn parses_trace_map_entries_from_keyed_dictionary() {
        let mut root = Dictionary::new();
        root.insert(
            "NS.keys".to_owned(),
            Value::Array(vec![test_integer(42), test_integer(43)]),
        );
        root.insert(
            "NS.objects".to_owned(),
            Value::Array(vec![
                test_integer(7),
                Value::Array(vec![test_integer(1), test_integer(2), test_integer(3)]),
            ]),
        );
        let mut top = Dictionary::new();
        top.insert("root".to_owned(), Value::Uid(Uid::new(1)));
        let mut archive = Dictionary::new();
        archive.insert("$top".to_owned(), Value::Dictionary(top));
        archive.insert(
            "$objects".to_owned(),
            Value::Array(vec![
                Value::String("$null".to_owned()),
                Value::Dictionary(root),
            ]),
        );
        let mut bytes = Vec::new();
        Value::Dictionary(archive)
            .to_writer_binary(&mut bytes)
            .unwrap();

        let entries = parse_trace_map_entries(26, "TraceId to BatchId", &bytes);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].trace_id, 42);
        assert_eq!(entries[0].scalar_value, Some(7));
        assert_eq!(entries[1].trace_id, 43);
        assert_eq!(entries[1].words, vec![1, 2, 3]);
    }

    #[test]
    fn formats_raw_counter_report_without_unbounded_rows_in_text_summary() {
        let report = RawCountersReport {
            trace_source: PathBuf::from("trace.gputrace"),
            profiler_directory: PathBuf::from("trace.gputrace/raw"),
            timings: Vec::new(),
            dispatch_alignment: None,
            aggregate_metadata: Vec::new(),
            sample_trace_indices: Vec::new(),
            trace_maps: Vec::new(),
            program_address_mappings: Vec::new(),
            profiling_address_summary: None,
            counter_info: Vec::new(),
            schemas: vec![RawCounterSchema {
                sample_group: 0,
                counter_count: 2,
                counter_names: vec!["_hot".to_owned(), "_huge".to_owned()],
            }],
            streams: Vec::new(),
            metrics: vec![
                RawCounterDecodedMetric {
                    path: "APSCounterData[0]/Derived Counter Sample Data/0/1/0".to_owned(),
                    sample_group: Some(0),
                    source_index: Some(1),
                    ring_index: Some(0),
                    counter_index: 0,
                    raw_name: "_hot".to_owned(),
                    sample_count: 2,
                    min_percent_of_gpu_cycles: 1.0,
                    mean_percent_of_gpu_cycles: 12.0,
                    max_percent_of_gpu_cycles: 20.0,
                    encoder_ids: Vec::new(),
                    kick_trace_ids: Vec::new(),
                    source_ids: Vec::new(),
                    derived_counter_matches: vec![RawCounterDerivedCounterMatch {
                        key: "TextureCacheLimiter".to_owned(),
                        name: "Texture Cache Limiter".to_owned(),
                        counter_type: Some("Percentage".to_owned()),
                        description: None,
                        unit: Some("%".to_owned()),
                        groups: Vec::new(),
                        timeline_groups: Vec::new(),
                        visible: Some(true),
                        batch_filtered: None,
                        sources: Vec::new(),
                    }],
                    hardware_selectors: Vec::new(),
                },
                RawCounterDecodedMetric {
                    path: "APSCounterData[0]/Derived Counter Sample Data/0/1/0".to_owned(),
                    sample_group: Some(0),
                    source_index: Some(1),
                    ring_index: Some(0),
                    counter_index: 1,
                    raw_name: "_huge".to_owned(),
                    sample_count: 2,
                    min_percent_of_gpu_cycles: 1.0,
                    mean_percent_of_gpu_cycles: 12.0,
                    max_percent_of_gpu_cycles: 900.0,
                    encoder_ids: Vec::new(),
                    kick_trace_ids: Vec::new(),
                    source_ids: Vec::new(),
                    derived_counter_matches: Vec::new(),
                    hardware_selectors: Vec::new(),
                },
            ],
            derived_metrics: Vec::new(),
            grouped_derived_metrics: Vec::new(),
            shifted_dispatch_metric_candidates: Vec::new(),
            encoder_sample_metrics: Vec::new(),
            warnings: Vec::new(),
        };

        let formatted = format_raw_counters_report(&report);

        assert!(formatted.contains("_hot"));
        assert!(!formatted.contains("_huge"));
        assert!(formatted.contains("1 decoded counters are not shown"));
    }

    #[test]
    fn formats_raw_counter_catalog_names_when_present() {
        let metric = RawCounterDecodedMetric {
            path: "APSCounterData[0]/Derived Counter Sample Data/0/1/0".to_owned(),
            sample_group: Some(0),
            source_index: Some(1),
            ring_index: Some(0),
            counter_index: 0,
            raw_name: "_hash".to_owned(),
            sample_count: 2,
            min_percent_of_gpu_cycles: 1.0,
            mean_percent_of_gpu_cycles: 12.0,
            max_percent_of_gpu_cycles: 20.0,
            encoder_ids: Vec::new(),
            kick_trace_ids: Vec::new(),
            source_ids: Vec::new(),
            derived_counter_matches: vec![RawCounterDerivedCounterMatch {
                key: "ALUUtilization".to_owned(),
                name: "ALU Utilization".to_owned(),
                counter_type: Some("Percentage".to_owned()),
                description: None,
                unit: Some("%".to_owned()),
                groups: vec!["Performance Limiters".to_owned()],
                timeline_groups: Vec::new(),
                visible: Some(true),
                batch_filtered: None,
                sources: vec![PathBuf::from("AGXMetalStatisticsExternal-counters.plist")],
            }],
            hardware_selectors: vec![RawCounterHardwareSelector {
                partition: Some(1),
                select: Some(2),
                flag: None,
                sources: vec![PathBuf::from("AGXMetalPerfCountersExternal.plist")],
            }],
        };
        let report = RawCountersReport {
            trace_source: PathBuf::from("trace.gputrace"),
            profiler_directory: PathBuf::from("trace.gputrace/raw"),
            timings: Vec::new(),
            dispatch_alignment: None,
            aggregate_metadata: Vec::new(),
            sample_trace_indices: Vec::new(),
            trace_maps: Vec::new(),
            program_address_mappings: Vec::new(),
            profiling_address_summary: None,
            counter_info: Vec::new(),
            schemas: Vec::new(),
            streams: Vec::new(),
            metrics: vec![metric],
            derived_metrics: Vec::new(),
            grouped_derived_metrics: Vec::new(),
            shifted_dispatch_metric_candidates: Vec::new(),
            encoder_sample_metrics: Vec::new(),
            warnings: Vec::new(),
        };

        let text = format_raw_counters_report(&report);
        let csv = format_raw_counters_csv(&report);

        assert!(text.contains("ALU Utilization (_hash)"));
        assert!(csv.contains("ALU Utilization"));
        assert!(csv.contains("ALUUtilization"));
    }

    #[test]
    fn formats_program_address_mappings_when_present() {
        let report = RawCountersReport {
            trace_source: PathBuf::from("trace.gputrace"),
            profiler_directory: PathBuf::from("trace.gputrace/raw"),
            timings: Vec::new(),
            dispatch_alignment: None,
            aggregate_metadata: Vec::new(),
            sample_trace_indices: Vec::new(),
            trace_maps: Vec::new(),
            program_address_mappings: vec![RawCounterProgramAddressMapping {
                archive_index: 26,
                mapping_index: 1,
                mapping_type: "compute".to_owned(),
                binary_unique_id: Some("000000000000000e".to_owned()),
                draw_call_index: Some(0),
                draw_function_index: Some(1564),
                encoder_trace_id: Some(2680109419),
                encoder_index: Some(0),
                shader_index: Some(0),
                mapped_address: Some(0x10000005840),
                mapped_size: Some(1590),
            }],
            profiling_address_summary: Some(ProfilingAddressProbeReport {
                trace_source: PathBuf::from("trace.gputrace"),
                profiler_directory: PathBuf::from("trace.gputrace/raw"),
                mapping_count: 1,
                file_summaries: Vec::new(),
                top_full_address_hits: Vec::new(),
                top_low32_address_hits: Vec::new(),
                top_shader_low32_hits: Vec::new(),
                top_function_low32_hits: vec![ProfilingFunctionAddressHit {
                    hit_count: 42,
                    function_name: "kernel".to_owned(),
                    dispatch_count: 2,
                    mapping_count: 3,
                }],
            }),
            counter_info: Vec::new(),
            schemas: Vec::new(),
            streams: Vec::new(),
            metrics: Vec::new(),
            derived_metrics: Vec::new(),
            grouped_derived_metrics: Vec::new(),
            shifted_dispatch_metric_candidates: Vec::new(),
            encoder_sample_metrics: Vec::new(),
            warnings: Vec::new(),
        };

        let text = format_raw_counters_report(&report);

        assert!(text.contains("Program address mappings: entries=1"));
        assert!(text.contains("enc_trace_id=2680109419"));
        assert!(text.contains("draw_function=1564"));
        assert!(text.contains("addr=0x10000005840"));
        assert!(text.contains("Profiling_f address-derived shader hits"));
        assert!(text.contains("function=kernel"));
    }

    #[test]
    fn evaluates_agx_derived_javascript_formula() {
        let definitions = vec![RawCounterDerivedDefinition {
            key: "ALUUtilization".to_owned(),
            name: "ALU Utilization".to_owned(),
            counter_type: Some("Percentage".to_owned()),
            description: None,
            raw_counters: vec!["_hash".to_owned()],
            unit: Some("%".to_owned()),
            groups: Vec::new(),
            timeline_groups: Vec::new(),
            visible: Some(true),
            batch_filtered: None,
            source_catalog: PathBuf::from("AGXMetalStatisticsExternalTest-counters.plist"),
            source_script: Some(PathBuf::from("AGXMetalStatisticsExternalTest-derived.js")),
        }];
        let variables = BTreeMap::from([("_hash_norm".to_owned(), 50.0)]);
        let metrics = evaluate_agx_derived_script(
            Path::new("AGXMetalStatisticsExternalTest-derived.js"),
            "var num_cores = 4; function ALUUtilization() { return _hash_norm / (2.0 * num_cores); }",
            &definitions,
            &variables,
        );

        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].key, "ALUUtilization");
        assert!((metrics[0].value - 6.25).abs() < 0.001);
    }

    #[test]
    fn derived_scripts_only_apply_to_their_gpu_generation() {
        let definitions_by_script = BTreeMap::from([
            (
                PathBuf::from("AGXMetalStatisticsExternalG14S-derived.js"),
                vec![RawCounterDerivedDefinition {
                    key: "A".to_owned(),
                    name: "A".to_owned(),
                    counter_type: None,
                    description: None,
                    raw_counters: Vec::new(),
                    unit: None,
                    groups: Vec::new(),
                    timeline_groups: Vec::new(),
                    visible: None,
                    batch_filtered: None,
                    source_catalog: PathBuf::from("AGXMetalStatisticsExternalG14S-counters.plist"),
                    source_script: None,
                }],
            ),
            (
                PathBuf::from("AGXMetalStatisticsExternalG14D-derived.js"),
                vec![RawCounterDerivedDefinition {
                    key: "B".to_owned(),
                    name: "B".to_owned(),
                    counter_type: None,
                    description: None,
                    raw_counters: Vec::new(),
                    unit: None,
                    groups: Vec::new(),
                    timeline_groups: Vec::new(),
                    visible: None,
                    batch_filtered: None,
                    source_catalog: PathBuf::from("AGXMetalStatisticsExternalG14D-counters.plist"),
                    source_script: None,
                }],
            ),
        ]);

        assert!(
            choose_agx_derived_script(definitions_by_script.clone(), Some("G16X"), None).is_none()
        );
        let (path, _) =
            choose_agx_derived_script(definitions_by_script, Some("G14X"), None).unwrap();
        assert_eq!(
            path,
            PathBuf::from("AGXMetalStatisticsExternalG14D-derived.js")
        );
    }

    #[test]
    fn chooses_agx_derived_script_by_raw_hash_overlap() {
        let definitions_by_script = BTreeMap::from([
            (
                PathBuf::from("AGXMetalStatisticsExternalG14D-derived.js"),
                vec![RawCounterDerivedDefinition {
                    key: "A".to_owned(),
                    name: "A".to_owned(),
                    counter_type: None,
                    description: None,
                    raw_counters: vec!["_missing".to_owned()],
                    unit: None,
                    groups: Vec::new(),
                    timeline_groups: Vec::new(),
                    visible: None,
                    batch_filtered: None,
                    source_catalog: PathBuf::from("AGXMetalStatisticsExternalG14D-counters.plist"),
                    source_script: None,
                }],
            ),
            (
                PathBuf::from("AGXMetalStatisticsExternalG14G-derived.js"),
                vec![RawCounterDerivedDefinition {
                    key: "B".to_owned(),
                    name: "B".to_owned(),
                    counter_type: None,
                    description: None,
                    raw_counters: vec!["_present".to_owned()],
                    unit: None,
                    groups: Vec::new(),
                    timeline_groups: Vec::new(),
                    visible: None,
                    batch_filtered: None,
                    source_catalog: PathBuf::from("AGXMetalStatisticsExternalG14G-counters.plist"),
                    source_script: None,
                }],
            ),
        ]);
        let variables = BTreeMap::from([("_present_norm".to_owned(), 1.0)]);

        let (path, _) =
            choose_agx_derived_script(definitions_by_script, Some("G14X"), Some(&variables))
                .unwrap();

        assert_eq!(
            path,
            PathBuf::from("AGXMetalStatisticsExternalG14G-derived.js")
        );
    }

    #[test]
    fn detects_gprw_record_size_from_magic_stride() {
        let mut data = Vec::new();
        for index in 0..3u64 {
            let mut record = vec![0u8; 64];
            record[0..8].copy_from_slice(b"GPRWCNTR");
            record[8..16].copy_from_slice(&(100 + index).to_le_bytes());
            record[16..24].copy_from_slice(&(200 + index).to_le_bytes());
            data.extend_from_slice(&record);
        }

        assert_eq!(gprw_record_info(&data), Some((64, 3)));
        let stats = summarize_gprw_u64_columns(&data, 64);
        assert_eq!(stats[0].min, u64::from_le_bytes(*b"GPRWCNTR"));
        assert_eq!(stats[1].first_values, vec![100, 101, 102]);
        assert_eq!(stats[2].first_values, vec![200, 201, 202]);
    }
}
