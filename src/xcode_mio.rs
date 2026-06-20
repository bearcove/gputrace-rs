use std::path::PathBuf;
use std::time::Instant;

use serde::Serialize;

use crate::error::{Error, Result};
use crate::profiler;
use crate::trace::TraceBundle;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioReport {
    pub trace_source: PathBuf,
    pub profiler_directory: PathBuf,
    pub stream_data_path: PathBuf,
    pub framework_path: PathBuf,
    pub timings: XcodeMioTimings,
    pub gpu_command_count: usize,
    pub encoder_count: usize,
    pub pipeline_state_count: usize,
    pub draw_count: usize,
    pub cost_record_count: usize,
    pub gpu_time_ns: u64,
    pub cost_timeline: Option<XcodeMioCostTimeline>,
    pub timeline_candidates: Vec<XcodeMioTimelineCandidate>,
    pub timeline_binary_count: usize,
    pub timeline_binaries: Vec<XcodeMioTimelineBinary>,
    pub timeline_pipeline_state_ids: Vec<u64>,
    pub shader_binary_info: Vec<XcodeMioShaderBinaryInfo>,
    pub decoded_cost_records: Vec<XcodeMioDecodedCostRecord>,
    pub draw_timeline_records: Vec<XcodeMioDrawTimelineRecord>,
    pub draw_metadata_records: Vec<XcodeMioDrawMetadataRecord>,
    pub pipelines: Vec<XcodeMioPipeline>,
    pub encoders: Vec<XcodeMioEncoder>,
    pub gpu_commands: Vec<XcodeMioGpuCommand>,
    pub gpu_command_function_times: Vec<XcodeMioGpuCommandFunctionTime>,
    pub gpu_command_function_time_probes: Vec<XcodeMioFunctionTimeProbe>,
    pub draw_array_probes: Vec<XcodeMioDrawArrayProbe>,
    pub usc_clique_summaries: Vec<XcodeMioUSCCliqueSummary>,
    pub usc_clique_probes: Vec<XcodeMioUSCCliqueProbe>,
    pub encoder_quad_probes: Vec<XcodeMioEncoderQuadProbe>,
    pub draw_execution_history_probes: Vec<XcodeMioDrawExecutionHistoryProbe>,
    pub top_draw_tracks: Vec<XcodeMioTopDrawTrack>,
    pub gpu_command_direct_costs: Vec<XcodeMioGpuCommandDirectCost>,
    pub gpu_command_shader_profiler_costs: Vec<XcodeMioGpuCommandShaderProfilerCost>,
    pub gpu_command_counter_rows: Vec<XcodeMioGpuCommandCounterRow>,
    pub shader_profiler_numeric_arrays: Vec<XcodeMioPrivateNumericArray>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct XcodeMioTimings {
    pub total_ms: f64,
    pub locate_profiler_ms: f64,
    pub stream_data_summary_ms: f64,
    pub framework_load_ms: f64,
    pub stream_data_load_ms: f64,
    pub processor_init_ms: f64,
    pub process_stream_ms: f64,
    pub extract_result_ms: f64,
    pub decode_pipeline_commands_ms: f64,
    pub shader_profiler_probe_ms: f64,
    pub cost_timeline_request_ms: f64,
    pub cost_timeline_decode_ms: f64,
    pub final_metadata_ms: f64,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct XcodeMioDecodeOptions {
    pub decode_cost_details: bool,
}

impl Default for XcodeMioDecodeOptions {
    fn default() -> Self {
        Self {
            decode_cost_details: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioPipeline {
    pub index: usize,
    pub object_id: u64,
    pub pointer_id: u64,
    pub function_index: u64,
    pub gpu_command_count: usize,
    pub timeline_draw_count: usize,
    pub timeline_duration_ns: u64,
    pub timeline_total_cost: f64,
    pub pipeline_address: Option<u64>,
    pub function_name: Option<String>,
    pub binary_keys: Vec<u64>,
    pub all_binary_keys: Vec<u64>,
    pub shader_stats: Vec<XcodeMioPipelineShaderStat>,
    pub profiler_timings: Vec<XcodeMioPipelineProfilerTiming>,
    pub scope_costs: Vec<XcodeMioPipelineScopeCost>,
    pub shader_tracks: Vec<XcodeMioPipelineShaderTrack>,
    pub shader_binaries: Vec<XcodeMioPipelineShaderBinary>,
    pub shader_binary_costs: Vec<XcodeMioPipelineShaderBinaryCost>,
    pub agxps_trace_costs: Vec<XcodeMioPipelineAgxpsTraceCost>,
    pub shader_profiler_costs: Vec<XcodeMioPipelineShaderProfilerCost>,
    pub execution_history: Vec<XcodeMioPipelineExecutionHistory>,
    pub shader_binary_references: Vec<XcodeMioPipelineShaderBinaryReference>,
    pub pipeline_counters: Vec<XcodeMioPipelineCounter>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioPipelineProfilerTiming {
    pub source: &'static str,
    pub cycle_average: f64,
    pub cycle_min: f64,
    pub cycle_max: f64,
    pub time_average: f64,
    pub time_min: f64,
    pub time_max: f64,
    pub percentage_average: f64,
    pub percentage_min: f64,
    pub percentage_max: f64,
    pub surplus_cycles: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioPipelineShaderProfilerCost {
    pub binary_key: String,
    pub full_path: Option<String>,
    pub type_name: Option<String>,
    pub shader_type: u32,
    pub addr_start: u32,
    pub addr_end: u32,
    pub total_binary_cost: f64,
    pub total_binary_samples: u64,
    pub pipeline_cost: f64,
    pub pipeline_cost_percent_sum: f64,
    pub nonzero_draw_count: usize,
    pub first_draw_index: usize,
    pub last_draw_index: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioPipelineAgxpsTraceCost {
    pub source: &'static str,
    pub shader_address: u64,
    pub work_shader_address: u64,
    pub command_count: usize,
    pub record_cliques: u64,
    pub analyzer_weighted_duration: u64,
    pub analyzer_avg_duration_sum: u64,
    pub matched_work_cliques: usize,
    pub duration_ns: u64,
    pub execution_events: u64,
    pub stats_word0: u64,
    pub stats_word1: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioPipelineCounter {
    pub name: String,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioTimelineBinary {
    pub index: u64,
    pub address: u64,
    pub program_type: u16,
    pub instruction_info_count: u64,
    pub instruction_executed: u64,
    pub duration_ns: u64,
    pub trace_count: u64,
    pub cost_count: u64,
    pub total_cost: f64,
    pub total_instruction_count: u64,
    pub instruction_cost_record_count: u64,
    pub instruction_nonzero_record_count: u64,
    pub instruction_total_cost: f64,
    pub instruction_total_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioTimelineCandidate {
    pub source: &'static str,
    pub draw_count: usize,
    pub pipeline_state_count: usize,
    pub cost_record_count: usize,
    pub gpu_time_ns: u64,
    pub global_gpu_time_ns: u64,
    pub timeline_duration_ns: u64,
    pub total_clique_cost: u64,
    pub gpu_cost: f64,
    pub gpu_cost_instruction_count: u64,
    pub timeline_binary_count: usize,
    pub shader_binary_info_count: usize,
    pub nonzero_cost_records: usize,
    pub decoded_total_cost: f64,
    pub decoded_instruction_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioPipelineShaderBinaryReference {
    pub raw5: u16,
    pub raw6: u16,
    pub raw1: u32,
    pub address: u64,
    pub timeline_binary_index: Option<u64>,
    pub record_count: usize,
    pub first_command_index: usize,
    pub last_command_index: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioPipelineShaderBinary {
    pub pipeline_id_kind: &'static str,
    pub pipeline_id: u64,
    pub binary_index: u64,
    pub address: u64,
    pub program_type: u16,
    pub instruction_info_count: u64,
    pub instruction_executed: u64,
    pub duration_ns: u64,
    pub trace_count: u64,
    pub cost_count: u64,
    pub total_cost: f64,
    pub total_instruction_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioPipelineShaderBinaryCost {
    pub source: &'static str,
    pub pipeline_id_kind: &'static str,
    pub pipeline_id: u64,
    pub binary_index: u64,
    pub address: u64,
    pub program_type: u16,
    pub record_count: u64,
    pub nonzero_record_count: u64,
    pub total_cost: f64,
    pub total_instruction_count: u64,
    pub alu_cost: f64,
    pub non_alu_cost: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioPipelineExecutionHistory {
    pub style: u32,
    pub options: u32,
    pub program_type: u16,
    pub pipeline_id_kind: &'static str,
    pub pipeline_id: u64,
    pub top_cost_percentage: f64,
    pub duration_percentage: f64,
    pub total_duration_ns: u64,
    pub total_cost: f64,
    pub instruction_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioShaderBinaryInfo {
    pub index: usize,
    pub raw0: u32,
    pub raw1: u32,
    pub raw2: u64,
    pub raw3: u64,
    pub raw4: u16,
    pub raw5: u16,
    pub raw6: u16,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioPipelineShaderTrack {
    pub source: &'static str,
    pub pipeline_id_kind: &'static str,
    pub pipeline_id: u64,
    pub program_type: u16,
    pub track_id: i32,
    pub first_index: u64,
    pub start_timestamp_ns: u64,
    pub end_timestamp_ns: u64,
    pub duration_ns: u64,
    pub trace_count: u64,
    pub traces: Vec<XcodeMioBinaryTrace>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioBinaryTrace {
    pub index: usize,
    pub start_timestamp_ns: u64,
    pub end_timestamp_ns: u64,
    pub raw_identifier: u64,
    pub raw_index: u32,
    pub raw_count: u32,
    pub raw_program_type: u16,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioPipelineShaderStat {
    pub shader_id_kind: &'static str,
    pub shader_id: u64,
    pub program_type: u16,
    pub number_of_cliques: u64,
    pub total_gpu_cycles: u64,
    pub total_latency: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioPipelineScopeCost {
    pub scope: u16,
    pub identifier_kind: &'static str,
    pub identifier: u64,
    pub level: u16,
    pub level_identifier: u32,
    pub alu_cost: f64,
    pub non_alu_cost: f64,
    pub total_cost: f64,
    pub instruction_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioDecodedCostRecord {
    pub index: usize,
    pub level: u16,
    pub scope: u16,
    pub level_identifier: u32,
    pub scope_identifier: u64,
    pub alu_cost: f64,
    pub non_alu_cost: f64,
    pub total_cost: f64,
    pub instruction_count: u64,
    pub threads_executing_instruction: u64,
    pub cpi_weighted_instruction_count: u64,
    pub active_thread_instruction_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioDrawTimelineRecord {
    pub index: usize,
    pub raw0: u64,
    pub raw1: u64,
    pub raw2: u32,
    pub raw3: u16,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioDrawMetadataRecord {
    pub index: usize,
    pub raw0: u32,
    pub raw1: u32,
    pub raw2: u32,
    pub raw3: u32,
    pub raw4: i32,
    pub raw5: u32,
    pub raw6: u64,
    pub raw7: u32,
    pub raw8: u32,
    pub raw9: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioGpuCommand {
    pub index: usize,
    pub function_index: u64,
    pub sub_command_index: i32,
    pub encoder_index: usize,
    pub pipeline_index: usize,
    pub pipeline_object_id: u64,
    pub command_buffer_index: usize,
    pub function_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioEncoder {
    pub index: usize,
    pub function_index: u64,
    pub gpu_command_start_index: usize,
    pub gpu_command_count: usize,
    pub load_time: u64,
    pub store_time: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioGpuCommandFunctionTime {
    pub source: &'static str,
    pub command_index: usize,
    pub function_index: u64,
    pub sub_command_index: i32,
    pub encoder_index: usize,
    pub pipeline_index: usize,
    pub function_name: Option<String>,
    pub draw_index: u32,
    pub data_master: u16,
    pub duration_ns: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioFunctionTimeProbe {
    pub source: &'static str,
    pub target_kind: &'static str,
    pub target_id_kind: &'static str,
    pub target_id: u64,
    pub pipeline_index: Option<usize>,
    pub encoder_index: Option<usize>,
    pub function_name: Option<String>,
    pub reported_draw_count: u64,
    pub enumerated_draw_count: usize,
    pub sampled_draws: Vec<u32>,
    pub best_draw_index: Option<u32>,
    pub best_data_master: Option<u16>,
    pub best_duration_ns: u64,
    pub kick_duration_ns: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioDrawArrayProbe {
    pub source: &'static str,
    pub array_index_kind: &'static str,
    pub array_index: usize,
    pub command_index: usize,
    pub function_index: u64,
    pub sub_command_index: i32,
    pub encoder_index: usize,
    pub pipeline_index: usize,
    pub function_name: Option<String>,
    pub trace_raw0: u64,
    pub trace_raw1: u64,
    pub trace_raw2: u32,
    pub trace_raw3: u16,
    pub trace_duration_ns: u64,
    pub metadata_raw0: u32,
    pub metadata_raw1: u32,
    pub metadata_raw2: u32,
    pub metadata_raw3: u32,
    pub metadata_raw4: i32,
    pub metadata_raw5: u32,
    pub metadata_raw6: u64,
    pub metadata_raw7: u32,
    pub metadata_raw8: u32,
    pub metadata_raw9: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioTopDrawTrack {
    pub source: &'static str,
    pub track_index: usize,
    pub trace_index: usize,
    pub track_id: i32,
    pub first_index: u64,
    pub start_timestamp_ns: u64,
    pub end_timestamp_ns: u64,
    pub duration_ns: u64,
    pub trace_count: u64,
    pub trace_raw0: u64,
    pub trace_raw1: u64,
    pub trace_raw2: u32,
    pub trace_raw3: u16,
    pub trace_duration_ns: u64,
    pub command_index: Option<usize>,
    pub function_index: Option<u64>,
    pub sub_command_index: Option<i32>,
    pub encoder_index: Option<usize>,
    pub pipeline_index: Option<usize>,
    pub function_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioUSCCliqueProbe {
    pub source: &'static str,
    pub usc_index: usize,
    pub field_index: usize,
    pub data_master: Option<u16>,
    pub match_kind: &'static str,
    pub matched_value: u32,
    pub command_index: usize,
    pub function_index: u64,
    pub sub_command_index: i32,
    pub encoder_index: usize,
    pub pipeline_index: usize,
    pub function_name: Option<String>,
    pub clique_count: usize,
    pub first_clique_index: usize,
    pub last_clique_index: usize,
    pub duration_sum_ns: u64,
    pub span_duration_ns: u64,
    pub min_duration_ns: u64,
    pub max_duration_ns: u64,
    pub min_timestamp_ns: u64,
    pub max_timestamp_ns: u64,
    pub sample_raw0: u64,
    pub sample_raw1: u64,
    pub sample_u32_fields: Vec<u32>,
    pub sample_u16_fields: Vec<u16>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioUSCCliqueSummary {
    pub source: &'static str,
    pub usc_index: Option<usize>,
    pub usc_count: u64,
    pub clique_count: u64,
    pub has_cliques: bool,
    pub has_enumerate_kick_cliques_by_function: bool,
    pub sample_raw0: u64,
    pub sample_raw1: u64,
    pub sample_duration_ns: u64,
    pub sample_u32_fields: Vec<u32>,
    pub sample_u16_fields: Vec<u16>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioEncoderQuadProbe {
    pub source: &'static str,
    pub mode: &'static str,
    pub encoder_index: Option<usize>,
    pub encoder_function_index: u32,
    pub pipeline_index: Option<usize>,
    pub pipeline_id_kind: Option<&'static str>,
    pub pipeline_id: Option<u64>,
    pub draw_id_kind: Option<&'static str>,
    pub draw_index: Option<u32>,
    pub function_name: Option<String>,
    pub program_type: u16,
    pub options: u64,
    pub draw_count: u64,
    pub quad_count: u64,
    pub min_timestamp_ns: u64,
    pub max_timestamp_ns: u64,
    pub duration_ns: u64,
    pub max_cost: f64,
    pub min_cost: f64,
    pub sampled_draws: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioDrawExecutionHistoryProbe {
    pub source: &'static str,
    pub mode: &'static str,
    pub node_source: &'static str,
    pub command_index: usize,
    pub draw_index: u32,
    pub pipeline_index: usize,
    pub function_name: Option<String>,
    pub style: u32,
    pub options: u32,
    pub program_type: u16,
    pub generated: bool,
    pub top_cost_percentage: f64,
    pub duration_percentage: f64,
    pub total_duration_ns: u64,
    pub total_cost: f64,
    pub instruction_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioGpuCommandCounterRow {
    pub command_index: usize,
    pub function_index: u64,
    pub sub_command_index: i32,
    pub encoder_index: usize,
    pub pipeline_index: usize,
    pub function_name: Option<String>,
    pub counters: Vec<XcodeMioPipelineCounter>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioGpuCommandDirectCost {
    pub source: &'static str,
    pub command_index: usize,
    pub function_index: u64,
    pub sub_command_index: i32,
    pub encoder_index: usize,
    pub pipeline_index: usize,
    pub function_name: Option<String>,
    pub cost: f64,
    pub cost_percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioGpuCommandShaderProfilerCost {
    pub source: &'static str,
    pub draw_index: usize,
    pub command_index: usize,
    pub function_index: u64,
    pub sub_command_index: i32,
    pub encoder_index: usize,
    pub pipeline_index: usize,
    pub function_name: Option<String>,
    pub binary_key: String,
    pub full_path: Option<String>,
    pub type_name: Option<String>,
    pub shader_type: u32,
    pub addr_start: u32,
    pub addr_end: u32,
    pub total_binary_cost: f64,
    pub total_binary_samples: u64,
    pub cost: f64,
    pub cost_percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioPrivateNumericArray {
    pub source: &'static str,
    pub rows: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioCostTimeline {
    pub draw_count: usize,
    pub pipeline_state_count: usize,
    pub cost_record_count: usize,
    pub gpu_time_ns: u64,
    pub global_gpu_time_ns: u64,
    pub timeline_duration_ns: u64,
    pub total_clique_cost: u64,
    pub gpu_cost: f64,
    pub gpu_cost_instruction_count: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioAnalysisReport {
    pub backend: &'static str,
    pub trace_source: PathBuf,
    pub timings: XcodeMioTimings,
    pub gpu_time_ns: u64,
    pub gpu_command_count: usize,
    pub pipeline_state_count: usize,
    pub cost_record_count: usize,
    pub top_pipelines: Vec<XcodeMioPipelineAnalysis>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct XcodeMioPipelineAnalysis {
    pub pipeline_index: usize,
    pub object_id: u64,
    pub pipeline_address: Option<u64>,
    pub function_name: Option<String>,
    pub command_count: usize,
    pub command_percent: f64,
    pub shader_binary_reference_count: usize,
    pub executable_shader_binary_reference_count: usize,
    pub unique_timeline_binary_count: usize,
    pub referenced_instruction_info_count: u64,
    pub xcode_time_percent: Option<f64>,
    pub xcode_time_average: Option<f64>,
    pub xcode_cycle_average: Option<f64>,
    pub timeline_duration_ns: u64,
    pub timeline_duration_percent: Option<f64>,
    pub timeline_total_cost: f64,
    pub timeline_cost_percent: Option<f64>,
    pub shader_profiler_cost: f64,
    pub shader_profiler_cost_percent: Option<f64>,
    pub shader_binary_cost: f64,
    pub shader_binary_cost_percent: Option<f64>,
    pub agxps_trace_cost: u64,
    pub agxps_trace_cost_percent: Option<f64>,
    pub agxps_trace_events: u64,
    pub agxps_trace_matched_work_cliques: usize,
    pub agxps_trace_duration_ns: u64,
    pub agxps_trace_duration_percent: Option<f64>,
    pub agxps_analyzer_cost: u64,
    pub agxps_analyzer_cost_percent: Option<f64>,
    pub agxps_analyzer_avg_duration_sum: u64,
    pub agxps_analyzer_record_cliques: u64,
    pub execution_top_cost_percent: Option<f64>,
    pub execution_duration_percent: Option<f64>,
    pub execution_total_cost: Option<f64>,
    pub execution_instruction_count: Option<u64>,
    pub counters: Vec<XcodeMioPipelineCounter>,
    pub metric_sources: Vec<String>,
}

pub fn report(trace: &TraceBundle) -> Result<XcodeMioReport> {
    report_with_profiler_summary(trace, None)
}

pub fn report_with_profiler_summary(
    trace: &TraceBundle,
    precomputed_profiler_summary: Option<&profiler::ProfilerStreamDataSummary>,
) -> Result<XcodeMioReport> {
    report_with_options(
        trace,
        precomputed_profiler_summary,
        XcodeMioDecodeOptions::default(),
    )
}

pub fn report_with_options(
    trace: &TraceBundle,
    precomputed_profiler_summary: Option<&profiler::ProfilerStreamDataSummary>,
    options: XcodeMioDecodeOptions,
) -> Result<XcodeMioReport> {
    let total_start = Instant::now();
    let locate_start = Instant::now();
    let profiler_directory = profiler::find_profiler_directory(&trace.path)
        .ok_or_else(|| Error::NotFound(trace.path.clone()))?;
    let locate_profiler_ms = elapsed_ms(locate_start);
    let stream_data_path = profiler_directory.join("streamData");
    if !stream_data_path.is_file() {
        return Err(Error::NotFound(stream_data_path));
    }

    let summary_start = Instant::now();
    let profiler_summary_owned;
    let profiler_summary = if let Some(summary) = precomputed_profiler_summary {
        Some(summary)
    } else {
        profiler_summary_owned = profiler::stream_data_summary(&trace.path).ok();
        profiler_summary_owned.as_ref()
    };
    let mut timings = XcodeMioTimings {
        locate_profiler_ms,
        stream_data_summary_ms: elapsed_ms(summary_start),
        ..XcodeMioTimings::default()
    };
    let mut report = platform::decode(
        trace.path.clone(),
        profiler_directory,
        stream_data_path,
        profiler_summary,
        timings.clone(),
        options,
    )?;
    timings = report.timings.clone();
    timings.total_ms = elapsed_ms(total_start);
    report.timings = timings;
    Ok(report)
}

pub fn analysis_report(trace: &TraceBundle) -> Result<XcodeMioAnalysisReport> {
    Ok(summarize_report(&report(trace)?))
}

pub fn agxps_analysis_report(
    trace: &TraceBundle,
    precomputed_profiler_summary: Option<&profiler::ProfilerStreamDataSummary>,
) -> Result<XcodeMioAnalysisReport> {
    let report = report_with_options(
        trace,
        precomputed_profiler_summary,
        XcodeMioDecodeOptions {
            decode_cost_details: false,
        },
    )?;
    Ok(summarize_report(&report))
}

pub fn summarize_report(report: &XcodeMioReport) -> XcodeMioAnalysisReport {
    let command_denominator = report.gpu_command_count.max(1) as f64;
    let timeline_duration_denominator = report
        .cost_timeline
        .as_ref()
        .map(|timeline| timeline.global_gpu_time_ns)
        .filter(|value| *value > 0)
        .unwrap_or(report.gpu_time_ns);
    let timeline_cost_denominator = report
        .pipelines
        .iter()
        .map(|pipeline| pipeline.timeline_total_cost)
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum::<f64>();
    let shader_profiler_cost_denominator = report
        .pipelines
        .iter()
        .flat_map(|pipeline| pipeline.shader_profiler_costs.iter())
        .map(|cost| cost.pipeline_cost)
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum::<f64>();
    let shader_binary_cost_denominator = report
        .pipelines
        .iter()
        .flat_map(|pipeline| pipeline.shader_binary_costs.iter())
        .map(|cost| cost.total_cost)
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum::<f64>();
    let agxps_trace_cost_denominator = report
        .pipelines
        .iter()
        .flat_map(|pipeline| pipeline.agxps_trace_costs.iter())
        .map(|cost| cost.stats_word1)
        .sum::<u64>();
    let agxps_trace_duration_denominator = report
        .pipelines
        .iter()
        .flat_map(|pipeline| pipeline.agxps_trace_costs.iter())
        .map(|cost| cost.duration_ns)
        .sum::<u64>();
    let agxps_analyzer_cost_denominator = report
        .pipelines
        .iter()
        .flat_map(|pipeline| pipeline.agxps_trace_costs.iter())
        .map(|cost| cost.analyzer_weighted_duration)
        .sum::<u64>();
    let timeline_binaries_by_index = report
        .timeline_binaries
        .iter()
        .map(|binary| (binary.index, binary))
        .collect::<std::collections::BTreeMap<_, _>>();

    let mut top_pipelines = report
        .pipelines
        .iter()
        .map(|pipeline| {
            let best_timing = pipeline.profiler_timings.iter().max_by(|left, right| {
                left.percentage_average
                    .partial_cmp(&right.percentage_average)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let shader_profiler_cost = pipeline
                .shader_profiler_costs
                .iter()
                .map(|cost| cost.pipeline_cost)
                .filter(|value| value.is_finite())
                .sum::<f64>();
            let shader_binary_cost = pipeline
                .shader_binary_costs
                .iter()
                .map(|cost| cost.total_cost)
                .filter(|value| value.is_finite())
                .sum::<f64>();
            let agxps_trace_cost = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.stats_word1)
                .sum::<u64>();
            let agxps_trace_duration_ns = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.duration_ns)
                .sum::<u64>();
            let agxps_analyzer_cost = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.analyzer_weighted_duration)
                .sum::<u64>();
            let agxps_analyzer_avg_duration_sum = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.analyzer_avg_duration_sum)
                .sum::<u64>();
            let agxps_analyzer_record_cliques = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.record_cliques)
                .sum::<u64>();
            let agxps_trace_events = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.execution_events)
                .sum::<u64>();
            let agxps_trace_matched_work_cliques = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.matched_work_cliques)
                .sum::<usize>();
            let best_execution = pipeline.execution_history.iter().max_by(|left, right| {
                left.top_cost_percentage
                    .partial_cmp(&right.top_cost_percentage)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let shader_binary_reference_count = pipeline
                .shader_binary_references
                .iter()
                .map(|reference| reference.record_count)
                .sum::<usize>();
            let executable_shader_binary_reference_count = pipeline
                .shader_binary_references
                .iter()
                .filter(|reference| reference.raw6 == 28)
                .map(|reference| reference.record_count)
                .sum::<usize>();
            let unique_timeline_binary_count = pipeline
                .shader_binary_references
                .iter()
                .map(|reference| reference.timeline_binary_index.unwrap_or(reference.address))
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            let referenced_instruction_info_count = pipeline
                .shader_binary_references
                .iter()
                .filter_map(|reference| {
                    let binary =
                        timeline_binaries_by_index.get(&reference.timeline_binary_index?)?;
                    Some(
                        binary
                            .instruction_info_count
                            .saturating_mul(reference.record_count as u64),
                    )
                })
                .fold(0u64, |sum, count| sum.saturating_add(count));
            let mut counters = pipeline.pipeline_counters.clone();
            counters.sort_by(|left, right| {
                right
                    .value
                    .abs()
                    .partial_cmp(&left.value.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| left.name.cmp(&right.name))
            });
            counters.truncate(8);

            let mut metric_sources = Vec::new();
            if best_timing
                .is_some_and(|timing| timing.percentage_average > 0.0 || timing.time_average > 0.0)
            {
                metric_sources.push("xcode-profiler-timing".to_owned());
            }
            if pipeline.timeline_duration_ns > 0 || pipeline.timeline_total_cost > 0.0 {
                metric_sources.push("xcode-cost-timeline".to_owned());
            }
            if shader_profiler_cost > 0.0 {
                metric_sources.push("xcode-shader-profiler-cost".to_owned());
            }
            if shader_binary_cost > 0.0 {
                metric_sources.push("xcode-shader-binary-cost".to_owned());
            }
            if agxps_trace_cost > 0 || agxps_analyzer_cost > 0 {
                metric_sources.push("agxps-timing-trace".to_owned());
            }
            if best_execution.is_some_and(|history| {
                history.top_cost_percentage > 0.0
                    || history.duration_percentage > 0.0
                    || history.total_cost > 0.0
            }) {
                metric_sources.push("xcode-execution-history".to_owned());
            }
            if !counters.is_empty() {
                metric_sources.push("xcode-pipeline-counters".to_owned());
            }
            if pipeline.gpu_command_count > 0 {
                metric_sources.push("xcode-gpu-command-topology".to_owned());
            }
            if shader_binary_reference_count > 0 {
                metric_sources.push("xcode-shader-binary-references".to_owned());
            }

            XcodeMioPipelineAnalysis {
                pipeline_index: pipeline.index,
                object_id: pipeline.object_id,
                pipeline_address: pipeline.pipeline_address,
                function_name: pipeline.function_name.clone(),
                command_count: pipeline.gpu_command_count,
                command_percent: pipeline.gpu_command_count as f64 * 100.0 / command_denominator,
                shader_binary_reference_count,
                executable_shader_binary_reference_count,
                unique_timeline_binary_count,
                referenced_instruction_info_count,
                xcode_time_percent: best_timing.map(|timing| timing.percentage_average),
                xcode_time_average: best_timing.map(|timing| timing.time_average),
                xcode_cycle_average: best_timing.map(|timing| timing.cycle_average),
                timeline_duration_ns: pipeline.timeline_duration_ns,
                timeline_duration_percent: (timeline_duration_denominator > 0
                    && pipeline.timeline_duration_ns > 0)
                    .then(|| {
                        pipeline.timeline_duration_ns as f64 * 100.0
                            / timeline_duration_denominator as f64
                    }),
                timeline_total_cost: pipeline.timeline_total_cost,
                timeline_cost_percent: (timeline_cost_denominator > 0.0
                    && pipeline.timeline_total_cost > 0.0)
                    .then(|| pipeline.timeline_total_cost * 100.0 / timeline_cost_denominator),
                shader_profiler_cost,
                shader_profiler_cost_percent: (shader_profiler_cost_denominator > 0.0
                    && shader_profiler_cost > 0.0)
                    .then(|| shader_profiler_cost * 100.0 / shader_profiler_cost_denominator),
                shader_binary_cost,
                shader_binary_cost_percent: (shader_binary_cost_denominator > 0.0
                    && shader_binary_cost > 0.0)
                    .then(|| shader_binary_cost * 100.0 / shader_binary_cost_denominator),
                agxps_trace_cost,
                agxps_trace_cost_percent: (agxps_trace_cost_denominator > 0
                    && agxps_trace_cost > 0)
                    .then(|| agxps_trace_cost as f64 * 100.0 / agxps_trace_cost_denominator as f64),
                agxps_trace_events,
                agxps_trace_matched_work_cliques,
                agxps_trace_duration_ns,
                agxps_trace_duration_percent: (agxps_trace_duration_denominator > 0
                    && agxps_trace_duration_ns > 0)
                    .then(|| {
                        agxps_trace_duration_ns as f64 * 100.0
                            / agxps_trace_duration_denominator as f64
                    }),
                agxps_analyzer_cost,
                agxps_analyzer_cost_percent: (agxps_analyzer_cost_denominator > 0
                    && agxps_analyzer_cost > 0)
                    .then(|| {
                        agxps_analyzer_cost as f64 * 100.0 / agxps_analyzer_cost_denominator as f64
                    }),
                agxps_analyzer_avg_duration_sum,
                agxps_analyzer_record_cliques,
                execution_top_cost_percent: best_execution
                    .map(|history| history.top_cost_percentage),
                execution_duration_percent: best_execution
                    .map(|history| history.duration_percentage),
                execution_total_cost: best_execution.map(|history| history.total_cost),
                execution_instruction_count: best_execution
                    .map(|history| history.instruction_count),
                counters,
                metric_sources,
            }
        })
        .collect::<Vec<_>>();

    top_pipelines.sort_by(|left, right| {
        pipeline_rank_score(right)
            .partial_cmp(&pipeline_rank_score(left))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.command_count.cmp(&left.command_count))
            .then_with(|| left.pipeline_index.cmp(&right.pipeline_index))
    });

    XcodeMioAnalysisReport {
        backend: "xcode-mio",
        trace_source: report.trace_source.clone(),
        timings: report.timings.clone(),
        gpu_time_ns: report.gpu_time_ns,
        gpu_command_count: report.gpu_command_count,
        pipeline_state_count: report.pipeline_state_count,
        cost_record_count: report.cost_record_count,
        top_pipelines,
        warnings: report.warnings.clone(),
    }
}

pub fn format_analysis_report(report: &XcodeMioAnalysisReport) -> String {
    let mut out = String::new();
    out.push_str("Xcode MIO analysis\n");
    out.push_str(&format!(
        "backend={} gpu_time={:.3} ms commands={} pipelines={} cost_records={} wall={:.1} ms\n\n",
        report.backend,
        report.gpu_time_ns as f64 / 1_000_000.0,
        report.gpu_command_count,
        report.pipeline_state_count,
        report.cost_record_count,
        report.timings.total_ms,
    ));
    out.push_str(&format!(
        "timings: locate={:.1} ms stream_summary={:.1} ms framework={:.1} ms stream_load={:.1} ms process={:.1} ms extract={:.1} ms topology={:.1} ms probes={:.1} ms cost_request={:.1} ms cost_decode={:.1} ms metadata={:.1} ms\n\n",
        report.timings.locate_profiler_ms,
        report.timings.stream_data_summary_ms,
        report.timings.framework_load_ms,
        report.timings.stream_data_load_ms,
        report.timings.process_stream_ms,
        report.timings.extract_result_ms,
        report.timings.decode_pipeline_commands_ms,
        report.timings.shader_profiler_probe_ms,
        report.timings.cost_timeline_request_ms,
        report.timings.cost_timeline_decode_ms,
        report.timings.final_metadata_ms,
    ));
    if report
        .top_pipelines
        .iter()
        .any(|pipeline| pipeline.agxps_trace_duration_percent.is_some())
    {
        out.push_str(
            "Cost % is per-pipeline AGXPS clique duration_ns normalized to total. On the synthetic single-dispatch-per-pipeline trace it matches Xcode Shader Cost within +/-1pp; on realistic traces it deviates up to +/-4pp on heavily aggregated pipelines.\n\n",
        );
    }
    out.push_str(&format!(
        "{:<42} {:>5} {:>7} {:>8} {:>7} {:>7} {:>8} {:>8} {:>8}\n",
        "Function", "Cmds", "Cmd %", "Cost %", "Bins", "ExecBin", "AGX Ana", "AGX W1", "Time %",
    ));
    for pipeline in report.top_pipelines.iter().take(25) {
        out.push_str(&format!(
            "{:<42} {:>5} {:>6.2}% {:>8} {:>7} {:>7} {:>7} {:>7} {:>7}\n",
            truncate(
                pipeline
                    .function_name
                    .as_deref()
                    .unwrap_or("<unknown function>"),
                42,
            ),
            pipeline.command_count,
            pipeline.command_percent,
            format_optional_percent(pipeline.agxps_trace_duration_percent),
            pipeline.unique_timeline_binary_count,
            pipeline.executable_shader_binary_reference_count,
            format_optional_percent(pipeline.agxps_analyzer_cost_percent),
            format_optional_percent(pipeline.agxps_trace_cost_percent),
            format_optional_percent(pipeline.xcode_time_percent),
        ));
    }
    if !report.warnings.is_empty() {
        out.push_str("\nWarnings:\n");
        for warning in &report.warnings {
            out.push_str(&format!("  - {warning}\n"));
        }
    }
    out
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn pipeline_rank_score(pipeline: &XcodeMioPipelineAnalysis) -> f64 {
    if let Some(value) = pipeline
        .agxps_trace_duration_percent
        .filter(|value| value.is_finite())
    {
        return value;
    }
    [
        pipeline.execution_top_cost_percent,
        pipeline.agxps_analyzer_cost_percent,
        pipeline.agxps_trace_cost_percent,
        pipeline.shader_profiler_cost_percent,
        pipeline.timeline_cost_percent,
        pipeline.timeline_duration_percent,
        pipeline.xcode_time_percent,
        Some(pipeline.command_percent),
    ]
    .into_iter()
    .flatten()
    .filter(|value| value.is_finite())
    .fold(0.0, f64::max)
}

fn format_optional_percent(value: Option<f64>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.2}%"))
        .unwrap_or_else(|| "-".to_owned())
}

fn truncate(value: &str, width: usize) -> String {
    if value.len() <= width {
        return value.to_owned();
    }
    let keep = width.saturating_sub(3);
    format!("{}...", &value[..keep])
}

pub fn format_report(report: &XcodeMioReport) -> String {
    let mut out = String::new();
    out.push_str("Xcode private MIO decode\n");
    out.push_str(&format!(
        "streamData: {}\nframework: {}\n\n",
        report.stream_data_path.display(),
        report.framework_path.display()
    ));
    out.push_str(&format!(
        "gpu_commands={} encoders={} pipelines={} draws={} cost_records={} gpu_time={:.3} ms\n\n",
        report.gpu_command_count,
        report.encoder_count,
        report.pipeline_state_count,
        report.draw_count,
        report.cost_record_count,
        report.gpu_time_ns as f64 / 1_000_000.0
    ));
    if let Some(timeline) = &report.cost_timeline {
        out.push_str(&format!(
            "cost_timeline: draws={} pipelines={} cost_records={} global_gpu_time={:.3} ms total_clique_cost={} gpu_cost={:.3} instr={}\n",
            timeline.draw_count,
            timeline.pipeline_state_count,
            timeline.cost_record_count,
            timeline.global_gpu_time_ns as f64 / 1_000_000.0,
            timeline.total_clique_cost,
            timeline.gpu_cost,
            timeline.gpu_cost_instruction_count,
        ));
        out.push('\n');
    } else {
        out.push('\n');
    }
    if !report.timeline_candidates.is_empty() {
        out.push_str("timeline candidates:\n");
        for candidate in &report.timeline_candidates {
            out.push_str(&format!(
                "  {:<28} draws={:<4} pipelines={:<3} costs={:<4} gpu={:>7.3} ms global={:>7.3} ms clique_cost={} gpu_cost={:.3} instr={} binaries={} shader_info={} nonzero_costs={} decoded_cost={:.3}\n",
                candidate.source,
                candidate.draw_count,
                candidate.pipeline_state_count,
                candidate.cost_record_count,
                candidate.gpu_time_ns as f64 / 1_000_000.0,
                candidate.global_gpu_time_ns as f64 / 1_000_000.0,
                candidate.total_clique_cost,
                candidate.gpu_cost,
                candidate.gpu_cost_instruction_count,
                candidate.timeline_binary_count,
                candidate.shader_binary_info_count,
                candidate.nonzero_cost_records,
                candidate.decoded_total_cost,
            ));
        }
        out.push('\n');
    }
    if report.timeline_binary_count > 0 || !report.shader_binary_info.is_empty() {
        out.push_str(&format!(
            "timeline_binaries={} timeline_pipeline_states={} shader_binary_info={}\n\n",
            report
                .timeline_binaries
                .len()
                .max(report.timeline_binary_count),
            report.timeline_pipeline_state_ids.len(),
            report.shader_binary_info.len()
        ));
    }
    if !report.encoders.is_empty() {
        out.push_str("MIO encoders:\n");
        out.push_str("  idx function_index start_cmd commands load_time store_time\n");
        for encoder in report.encoders.iter().take(16) {
            out.push_str(&format!(
                "  {idx:>3} {function_index:>14} {start:>9} {commands:>8} {load:>9} {store:>10}\n",
                idx = encoder.index,
                function_index = encoder.function_index,
                start = encoder.gpu_command_start_index,
                commands = encoder.gpu_command_count,
                load = encoder.load_time,
                store = encoder.store_time,
            ));
        }
        out.push('\n');
    }
    let mut binary_references = report
        .pipelines
        .iter()
        .flat_map(|pipeline| {
            let mut groups = std::collections::BTreeMap::<(u16, u16, u32), (usize, usize)>::new();
            for reference in pipeline
                .shader_binary_references
                .iter()
                .filter(|reference| reference.raw5 == 6 && reference.raw6 == 28)
            {
                let entry = groups
                    .entry((reference.raw5, reference.raw6, reference.raw1))
                    .or_insert((0, 0));
                entry.0 += reference.record_count;
                entry.1 += 1;
            }
            groups
                .into_iter()
                .map(move |((raw5, raw6, raw1), (record_count, binary_count))| {
                    (pipeline, raw5, raw6, raw1, record_count, binary_count)
                })
        })
        .collect::<Vec<_>>();
    binary_references.sort_by(
        |(left_pipeline, _, _, left_raw1, left_count, _),
         (right_pipeline, _, _, right_raw1, right_count, _)| {
            right_count
                .cmp(left_count)
                .then_with(|| left_pipeline.index.cmp(&right_pipeline.index))
                .then_with(|| left_raw1.cmp(right_raw1))
        },
    );
    if !binary_references.is_empty() {
        out.push_str("Pipelines by shader-binary references:\n");
        for (pipeline, raw5, raw6, raw1, record_count, binary_count) in
            binary_references.iter().take(20)
        {
            let name = pipeline
                .function_name
                .as_deref()
                .unwrap_or("<unknown function>");
            out.push_str(&format!(
                "  {:>4} refs  {:>4} binaries  raw1={:<5} kind={}/{}  {:<56}\n",
                record_count, binary_count, raw1, raw5, raw6, name,
            ));
        }
        out.push('\n');
    }
    let mut pipeline_counters = report
        .pipelines
        .iter()
        .filter(|pipeline| !pipeline.pipeline_counters.is_empty())
        .collect::<Vec<_>>();
    pipeline_counters.sort_by(|left, right| {
        let left_max = left
            .pipeline_counters
            .iter()
            .map(|counter| counter.value.abs())
            .fold(0.0, f64::max);
        let right_max = right
            .pipeline_counters
            .iter()
            .map(|counter| counter.value.abs())
            .fold(0.0, f64::max);
        right_max
            .partial_cmp(&left_max)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.index.cmp(&right.index))
    });
    if !pipeline_counters.is_empty() {
        out.push_str("Pipelines by private pipeline counters:\n");
        for pipeline in pipeline_counters.iter().take(12) {
            let name = pipeline
                .function_name
                .as_deref()
                .unwrap_or("<unknown function>");
            let counters = pipeline
                .pipeline_counters
                .iter()
                .take(4)
                .map(|counter| format!("{}={:.3}", counter.name, counter.value))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!("  {:<56} {}\n", name, counters));
        }
        out.push('\n');
    }
    if !report.gpu_command_counter_rows.is_empty() {
        out.push_str("GPU command counters from Xcode non-overlapping counters:\n");
        out.push_str(
            "  cmd sub  execution_cost  invocations  alu_instructions  alu_float% function                 nonzero_internal_counters\n",
        );
        for row in report.gpu_command_counter_rows.iter().take(80) {
            let name = row.function_name.as_deref().unwrap_or("<unknown>");
            let execution_cost = counter_value(row, "Execution Cost");
            let invocations = counter_value(row, "Kernel Invocations");
            let alu_instructions = counter_value(row, "Kernel ALU Instructions");
            let alu_float = counter_value(row, "Kernel ALU Float Instructions");
            let internal_counters = row
                .counters
                .iter()
                .take(4)
                .map(|counter| format!("{}={:.3}", counter.name, counter.value))
                .collect::<Vec<_>>()
                .join(", ");
            out.push_str(&format!(
                "  {cmd:>3} {sub:>3} {cost:>14} {invocations:>12} {alu:>17} {float_pct:>10} {name:<24} {internal_counters}\n",
                cmd = row.command_index,
                sub = row.sub_command_index,
                cost = format_optional_counter(execution_cost, 3),
                invocations = format_optional_counter(invocations, 0),
                alu = format_optional_counter(alu_instructions, 0),
                float_pct = format_optional_counter(alu_float, 2),
            ));
        }
        out.push_str("  note: these names are Xcode's internal draw counters. The public Compute Kernel display columns above are not populated through this object on raw-directory profiles.\n");
        out.push('\n');
    }
    if !report.top_draw_tracks.is_empty() {
        let mut rows = report.top_draw_tracks.iter().collect::<Vec<_>>();
        rows.sort_by(|left, right| {
            right
                .trace_duration_ns
                .max(right.duration_ns)
                .cmp(&left.trace_duration_ns.max(left.duration_ns))
                .then_with(|| left.source.cmp(right.source))
                .then_with(|| left.track_index.cmp(&right.track_index))
                .then_with(|| left.trace_index.cmp(&right.trace_index))
        });
        let total_ns = best_top_draw_track_rows(rows.iter().copied())
            .iter()
            .map(|row| row.trace_duration_ns.max(row.duration_ns))
            .sum::<u64>();
        out.push_str("Top draw tracks from GTMioTraceDataHelper.generateTopDrawTracks:\n");
        out.push_str(
            "  src                    track trace cmd   normalized% duration trace_duration raw0 raw1 raw2 raw3 function\n",
        );
        for row in rows.into_iter().take(80) {
            let duration_ns = row.trace_duration_ns.max(row.duration_ns);
            let pct = percent_u64(duration_ns, total_ns);
            let cmd = row
                .command_index
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned());
            let name = row.function_name.as_deref().unwrap_or("<unknown>");
            out.push_str(&format!(
                "  {source:<22} {track:>5} {trace:>5} {cmd:>4} {pct:>10.4}% {duration:>10} {trace_duration:>14} {raw0:>10} {raw1:>10} {raw2:>5} {raw3:>4} {name}\n",
                source = row.source,
                track = row.track_index,
                trace = row.trace_index,
                duration = format_duration_ns(duration_ns),
                trace_duration = format_duration_ns(row.trace_duration_ns),
                raw0 = row.trace_raw0,
                raw1 = row.trace_raw1,
                raw2 = row.trace_raw2,
                raw3 = row.trace_raw3,
            ));
        }
        out.push('\n');
    }
    if !report.gpu_command_function_times.is_empty() {
        out.push_str("GPU command function times from Xcode durationForDraw:dataMaster:\n");
        let mut by_source =
            std::collections::BTreeMap::<&'static str, Vec<&XcodeMioGpuCommandFunctionTime>>::new();
        for row in &report.gpu_command_function_times {
            by_source.entry(row.source).or_default().push(row);
        }
        let mut source_totals = std::collections::BTreeMap::new();
        for (source, rows) in &by_source {
            let best_rows = best_function_time_rows(rows.iter().copied());
            let total_ns = best_rows.iter().map(|row| row.duration_ns).sum::<u64>();
            source_totals.insert(*source, total_ns);
            out.push_str(&format!(
                "  source={source:<23} rows={:<4} commands={:<4} total={}\n",
                rows.len(),
                best_rows.len(),
                format_duration_ns(total_ns),
            ));

            let mut pipeline_totals =
                std::collections::BTreeMap::<usize, (u64, usize, Option<String>)>::new();
            for row in best_rows {
                let entry = pipeline_totals
                    .entry(row.pipeline_index)
                    .or_insert_with(|| (0, 0, row.function_name.clone()));
                entry.0 = entry.0.saturating_add(row.duration_ns);
                entry.1 += 1;
            }
            let mut pipeline_totals = pipeline_totals.into_iter().collect::<Vec<_>>();
            pipeline_totals
                .sort_by(|left, right| right.1.0.cmp(&left.1.0).then_with(|| left.0.cmp(&right.0)));
            for (_, (duration_ns, command_count, function_name)) in pipeline_totals.iter().take(8) {
                let pct = percent_u64(*duration_ns, total_ns);
                let name = function_name.as_deref().unwrap_or("<unknown function>");
                out.push_str(&format!(
                    "    {:>7.3}% {:>10} cmds={:<4} {}\n",
                    pct,
                    format_duration_ns(*duration_ns),
                    command_count,
                    truncate(name, 56),
                ));
            }
        }
        out.push_str("  rows: source cmd draw dm duration normalized% function\n");
        for row in report.gpu_command_function_times.iter().take(80) {
            let total_ns = source_totals.get(row.source).copied().unwrap_or(0);
            let pct = percent_u64(row.duration_ns, total_ns);
            let name = row.function_name.as_deref().unwrap_or("<unknown>");
            out.push_str(&format!(
                "  {source:<23} {cmd:>4} {draw:>5} {dm:>2} {duration:>10} {pct:>10.4}% {name}\n",
                source = row.source,
                cmd = row.command_index,
                draw = row.draw_index,
                dm = row.data_master,
                duration = format_duration_ns(row.duration_ns),
            ));
        }
        out.push('\n');
    }
    if !report.gpu_command_function_time_probes.is_empty() {
        out.push_str("GPU command function-time probes:\n");
        out.push_str(
            "  src                       kind     id_kind                    id pipe enc reported enum samples                       best_draw dm best_duration kick_duration function\n",
        );
        let mut probes = report
            .gpu_command_function_time_probes
            .iter()
            .collect::<Vec<_>>();
        probes.sort_by(|left, right| {
            right
                .best_duration_ns
                .cmp(&left.best_duration_ns)
                .then_with(|| right.kick_duration_ns.cmp(&left.kick_duration_ns))
                .then_with(|| right.reported_draw_count.cmp(&left.reported_draw_count))
                .then_with(|| right.enumerated_draw_count.cmp(&left.enumerated_draw_count))
                .then_with(|| left.source.cmp(right.source))
                .then_with(|| left.target_kind.cmp(right.target_kind))
                .then_with(|| left.target_id.cmp(&right.target_id))
        });
        for probe in probes.iter().take(120) {
            let name = probe.function_name.as_deref().unwrap_or("-");
            let pipeline = probe
                .pipeline_index
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned());
            let encoder = probe
                .encoder_index
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned());
            let best_draw = probe
                .best_draw_index
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned());
            let best_data_master = probe
                .best_data_master
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned());
            let samples = format_u32_samples(&probe.sampled_draws, 8);
            out.push_str(&format!(
                "  {source:<25} {kind:<8} {id_kind:<24} {id:>6} {pipe:>4} {enc:>3} {reported:>8} {enumerated:>4} {samples:<29} {best_draw:>9} {dm:>2} {best:>13} {kick:>13} {name}\n",
                source = probe.source,
                kind = probe.target_kind,
                id_kind = probe.target_id_kind,
                id = probe.target_id,
                pipe = pipeline,
                enc = encoder,
                reported = probe.reported_draw_count,
                enumerated = probe.enumerated_draw_count,
                samples = samples,
                dm = best_data_master,
                best = format_duration_ns(probe.best_duration_ns),
                kick = format_duration_ns(probe.kick_duration_ns),
            ));
        }
        out.push('\n');
    }
    if !report.draw_array_probes.is_empty() {
        out.push_str("MIO raw draw-array probes:\n");
        out.push_str(
            "  src                       idx_kind       arr_idx  cmd  fn_idx sub enc pipe trace_duration trace0        trace1        t2  dm meta0 meta7 function\n",
        );
        let mut probes = report.draw_array_probes.iter().collect::<Vec<_>>();
        probes.sort_by(|left, right| {
            left.command_index
                .cmp(&right.command_index)
                .then_with(|| left.source.cmp(right.source))
                .then_with(|| left.array_index_kind.cmp(right.array_index_kind))
                .then_with(|| left.array_index.cmp(&right.array_index))
        });
        for probe in probes.iter().take(500) {
            let name = probe.function_name.as_deref().unwrap_or("-");
            out.push_str(&format!(
                "  {source:<25} {idx_kind:<14} {arr_idx:>7} {cmd:>4} {fn_idx:>7} {sub:>3} {enc:>3} {pipe:>4} {duration:>14} {trace0:>12} {trace1:>12} {t2:>3} {dm:>3} {meta0:>5} {meta7:>5} {name}\n",
                source = probe.source,
                idx_kind = probe.array_index_kind,
                arr_idx = probe.array_index,
                cmd = probe.command_index,
                fn_idx = probe.function_index,
                sub = probe.sub_command_index,
                enc = probe.encoder_index,
                pipe = probe.pipeline_index,
                duration = format_duration_ns(probe.trace_duration_ns),
                trace0 = probe.trace_raw0,
                trace1 = probe.trace_raw1,
                t2 = probe.trace_raw2,
                dm = probe.trace_raw3,
                meta0 = probe.metadata_raw0,
                meta7 = probe.metadata_raw7,
            ));
        }
        out.push('\n');
    }
    if !report.usc_clique_summaries.is_empty() {
        out.push_str("MIO USC clique inventory:\n");
        out.push_str(
            "  src                       usc usc_count clique_count cliques enum_by_fn sample_duration sample0      sample1      u32_fields                              u16_fields\n",
        );
        for summary in report.usc_clique_summaries.iter().take(80) {
            let usc = summary
                .usc_index
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned());
            let u32_fields = format_u32_samples(&summary.sample_u32_fields, 14);
            let u16_fields = format_u16_samples(&summary.sample_u16_fields, 3);
            out.push_str(&format!(
                "  {source:<25} {usc:>3} {usc_count:>9} {clique_count:>12} {has_cliques:>7} {has_enum:>10} {duration:>15} {raw0:>12} {raw1:>12} {u32_fields:<39} {u16_fields}\n",
                source = summary.source,
                usc_count = summary.usc_count,
                clique_count = summary.clique_count,
                has_cliques = summary.has_cliques,
                has_enum = summary.has_enumerate_kick_cliques_by_function,
                duration = format_duration_ns(summary.sample_duration_ns),
                raw0 = summary.sample_raw0,
                raw1 = summary.sample_raw1,
            ));
        }
        out.push('\n');
    }
    if !report.usc_clique_probes.is_empty() {
        out.push_str("MIO USC clique candidate timings:\n");
        out.push_str(
            "  src                       usc field  dm match          value  cmd  fn_idx sub enc pipe cliques sum_duration span_duration min       max       first  last   sample0      sample1      u32_fields                              u16_fields function\n",
        );
        let mut probes = report.usc_clique_probes.iter().collect::<Vec<_>>();
        probes.sort_by(|left, right| {
            left.command_index
                .cmp(&right.command_index)
                .then_with(|| right.duration_sum_ns.cmp(&left.duration_sum_ns))
                .then_with(|| left.source.cmp(right.source))
                .then_with(|| left.usc_index.cmp(&right.usc_index))
                .then_with(|| left.field_index.cmp(&right.field_index))
                .then_with(|| left.match_kind.cmp(right.match_kind))
        });
        for probe in probes.iter().take(500) {
            let name = probe.function_name.as_deref().unwrap_or("-");
            let u32_fields = format_u32_samples(&probe.sample_u32_fields, 14);
            let u16_fields = format_u16_samples(&probe.sample_u16_fields, 3);
            let data_master = probe
                .data_master
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned());
            out.push_str(&format!(
                "  {source:<25} {usc:>3} r{field:<3} {dm:>2} {kind:<14} {value:>6} {cmd:>4} {fn_idx:>7} {sub:>3} {enc:>3} {pipe:>4} {cliques:>7} {sum:>12} {span:>13} {min:>9} {max:>9} {first:>6} {last:>6} {raw0:>12} {raw1:>12} {u32_fields:<39} {u16_fields:<10} {name}\n",
                source = probe.source,
                usc = probe.usc_index,
                field = probe.field_index,
                dm = data_master,
                kind = probe.match_kind,
                value = probe.matched_value,
                cmd = probe.command_index,
                fn_idx = probe.function_index,
                sub = probe.sub_command_index,
                enc = probe.encoder_index,
                pipe = probe.pipeline_index,
                cliques = probe.clique_count,
                sum = format_duration_ns(probe.duration_sum_ns),
                span = format_duration_ns(probe.span_duration_ns),
                min = format_duration_ns(probe.min_duration_ns),
                max = format_duration_ns(probe.max_duration_ns),
                first = probe.first_clique_index,
                last = probe.last_clique_index,
                raw0 = probe.sample_raw0,
                raw1 = probe.sample_raw1,
            ));
        }
        out.push('\n');
    }
    if !report.encoder_quad_probes.is_empty() {
        out.push_str("GTMioEncoderQuadData probes:\n");
        out.push_str(
            "  src                  mode              enc enc_fn pipe id_kind      id draw_kind draw     pt opt draws quads duration     min_ts     max_ts max_cost samples                       function\n",
        );
        let mut probes = report.encoder_quad_probes.iter().collect::<Vec<_>>();
        probes.sort_by(|left, right| {
            right
                .duration_ns
                .cmp(&left.duration_ns)
                .then_with(|| right.draw_count.cmp(&left.draw_count))
                .then_with(|| right.quad_count.cmp(&left.quad_count))
                .then_with(|| left.source.cmp(right.source))
                .then_with(|| left.mode.cmp(right.mode))
                .then_with(|| {
                    left.encoder_function_index
                        .cmp(&right.encoder_function_index)
                })
        });
        for probe in probes.iter().take(160) {
            let encoder = probe
                .encoder_index
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned());
            let pipeline = probe
                .pipeline_index
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned());
            let pipeline_id_kind = probe.pipeline_id_kind.unwrap_or("-");
            let pipeline_id = probe
                .pipeline_id
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned());
            let draw_id_kind = probe.draw_id_kind.unwrap_or("-");
            let draw_index = probe
                .draw_index
                .map(|value| value.to_string())
                .unwrap_or_else(|| "-".to_owned());
            let samples = format_u32_samples(&probe.sampled_draws, 8);
            let name = probe.function_name.as_deref().unwrap_or("-");
            out.push_str(&format!(
                "  {source:<20} {mode:<17} {enc:>3} {enc_fn:>6} {pipe:>4} {id_kind:<12} {id:>6} {draw_kind:<9} {draw:>5} {pt:>6} {opt:>3} {draws:>5} {quads:>5} {duration:>10} {min_ts:>10} {max_ts:>10} {max_cost:>8.3} {samples:<29} {name}\n",
                source = probe.source,
                mode = probe.mode,
                enc = encoder,
                enc_fn = probe.encoder_function_index,
                pipe = pipeline,
                id_kind = pipeline_id_kind,
                id = pipeline_id,
                draw_kind = draw_id_kind,
                draw = draw_index,
                pt = probe.program_type,
                opt = probe.options,
                draws = probe.draw_count,
                quads = probe.quad_count,
                duration = format_duration_ns(probe.duration_ns),
                min_ts = probe.min_timestamp_ns,
                max_ts = probe.max_timestamp_ns,
                max_cost = probe.max_cost,
                samples = samples,
            ));
        }
        out.push('\n');
    }
    if !report.draw_execution_history_probes.is_empty() {
        out.push_str("GTMioShaderExecutionHistory draw probes:\n");
        out.push_str(
            "  src                  mode             node             cmd  draw pipe style opt     pt generated duration   dur%   top% total_cost instr function\n",
        );
        let mut probes = report
            .draw_execution_history_probes
            .iter()
            .collect::<Vec<_>>();
        probes.sort_by(|left, right| {
            right
                .total_duration_ns
                .cmp(&left.total_duration_ns)
                .then_with(|| {
                    right
                        .duration_percentage
                        .partial_cmp(&left.duration_percentage)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| left.command_index.cmp(&right.command_index))
                .then_with(|| left.source.cmp(right.source))
                .then_with(|| left.mode.cmp(right.mode))
        });
        for probe in probes.iter().take(160) {
            let name = probe.function_name.as_deref().unwrap_or("-");
            out.push_str(&format!(
                "  {source:<20} {mode:<16} {node:<16} {cmd:>4} {draw:>5} {pipe:>4} {style:>5} {opt:>3} {pt:>6} {generated:>9} {duration:>10} {dur_pct:>6.2} {top_pct:>6.2} {cost:>10.3} {instr:>5} {name}\n",
                source = probe.source,
                mode = probe.mode,
                node = probe.node_source,
                cmd = probe.command_index,
                draw = probe.draw_index,
                pipe = probe.pipeline_index,
                style = probe.style,
                opt = probe.options,
                pt = probe.program_type,
                generated = probe.generated,
                duration = format_duration_ns(probe.total_duration_ns),
                dur_pct = probe.duration_percentage,
                top_pct = probe.top_cost_percentage,
                cost = probe.total_cost,
                instr = probe.instruction_count,
            ));
        }
        out.push('\n');
    }
    if !report.gpu_command_shader_profiler_costs.is_empty() {
        let denominator = report
            .gpu_command_shader_profiler_costs
            .iter()
            .map(|row| row.cost)
            .filter(|value| value.is_finite() && *value > 0.0)
            .sum::<f64>();
        out.push_str("GPU commands by Xcode shader-profiler per-draw cost:\n");
        out.push_str(
            "  src       cmd draw sub   normalized%    private%       cost function                 key\n",
        );
        for row in report.gpu_command_shader_profiler_costs.iter().take(80) {
            let normalized_percent = if denominator > 0.0 {
                Some(row.cost * 100.0 / denominator)
            } else {
                None
            };
            let name = row.function_name.as_deref().unwrap_or("<unknown>");
            out.push_str(&format!(
                "  {source:<8} {cmd:>3} {draw:>4} {sub:>3} {norm:>12} {private:>10} {cost:>10.3} {name:<24} {key}\n",
                source = row.source,
                cmd = row.command_index,
                draw = row.draw_index,
                sub = row.sub_command_index,
                norm = format_optional_percent(normalized_percent),
                private = format_optional_percent(row.cost_percent),
                cost = row.cost,
                key = row.binary_key,
            ));
        }
        out.push('\n');
    }
    if !report.gpu_command_direct_costs.is_empty() {
        let denominator = report
            .gpu_command_direct_costs
            .iter()
            .map(|row| row.cost)
            .filter(|value| value.is_finite() && *value > 0.0)
            .sum::<f64>();
        out.push_str("GPU commands by direct trace-data per-draw cost:\n");
        out.push_str(
            "  src                    cmd sub   normalized%    private%       cost function\n",
        );
        let mut rows = report.gpu_command_direct_costs.iter().collect::<Vec<_>>();
        rows.sort_by(|left, right| {
            right
                .cost
                .partial_cmp(&left.cost)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.source.cmp(right.source))
                .then_with(|| left.command_index.cmp(&right.command_index))
        });
        for row in rows.into_iter().take(80) {
            let normalized_percent = if denominator > 0.0 {
                Some(row.cost * 100.0 / denominator)
            } else {
                None
            };
            let name = row.function_name.as_deref().unwrap_or("<unknown>");
            out.push_str(&format!(
                "  {source:<22} {cmd:>3} {sub:>3} {norm:>12} {private:>10} {cost:>10.3} {name}\n",
                source = row.source,
                cmd = row.command_index,
                sub = row.sub_command_index,
                norm = format_optional_percent(normalized_percent),
                private = format_optional_percent(row.cost_percent),
                cost = row.cost,
            ));
        }
        out.push('\n');
    }
    if !report.shader_profiler_numeric_arrays.is_empty() {
        out.push_str("Private shader-profiler numeric arrays:\n");
        for probe in &report.shader_profiler_numeric_arrays {
            out.push_str(&format!("  {} rows={}\n", probe.source, probe.rows.len()));
            for (row_index, row) in probe.rows.iter().enumerate().take(32) {
                let values = row
                    .iter()
                    .take(24)
                    .map(|value| format_private_numeric_value(*value))
                    .collect::<Vec<_>>()
                    .join(", ");
                let suffix = if row.len() > 24 { ", ..." } else { "" };
                out.push_str(&format!(
                    "    row {row_index:>3} len={len:>4}: [{values}{suffix}]\n",
                    len = row.len(),
                ));
            }
        }
        out.push('\n');
    }
    let mut agxps_trace_costs = report
        .pipelines
        .iter()
        .filter(|pipeline| !pipeline.agxps_trace_costs.is_empty())
        .collect::<Vec<_>>();
    agxps_trace_costs.sort_by(|left, right| {
        let left_cost = left
            .agxps_trace_costs
            .iter()
            .map(|cost| cost.stats_word1)
            .sum::<u64>();
        let right_cost = right
            .agxps_trace_costs
            .iter()
            .map(|cost| cost.stats_word1)
            .sum::<u64>();
        right_cost
            .cmp(&left_cost)
            .then_with(|| left.index.cmp(&right.index))
    });
    let agxps_trace_denominator = report
        .pipelines
        .iter()
        .flat_map(|pipeline| pipeline.agxps_trace_costs.iter())
        .map(|cost| cost.stats_word1)
        .sum::<u64>();
    if agxps_trace_denominator > 0 {
        out.push_str("Pipelines by AGXPS timing-trace instruction stats:\n");
        for pipeline in agxps_trace_costs.iter().take(20) {
            let name = pipeline
                .function_name
                .as_deref()
                .unwrap_or("<unknown function>");
            let total_cost = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.stats_word1)
                .sum::<u64>();
            let analyzer_weighted = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.analyzer_weighted_duration)
                .sum::<u64>();
            let total_events = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.execution_events)
                .sum::<u64>();
            let matched_cliques = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.matched_work_cliques)
                .sum::<usize>();
            let command_count = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.command_count)
                .sum::<usize>();
            let pct = 100.0 * total_cost as f64 / agxps_trace_denominator as f64;
            let top = pipeline
                .agxps_trace_costs
                .iter()
                .max_by_key(|cost| cost.stats_word1);
            let top_address = top
                .map(|cost| format!("0x{:x}", cost.shader_address))
                .unwrap_or_else(|| "-".to_owned());
            out.push_str(&format!(
                "  {:>6.2}% w1={:>12} analyzer_weighted={:>12} events={:>9} cliques={:>7} cmds={:>4} {:<56} top_esl={}\n",
                pct,
                total_cost,
                analyzer_weighted,
                total_events,
                matched_cliques,
                command_count,
                name,
                top_address,
            ));
        }
        out.push('\n');
    }
    let agxps_analyzer_denominator = report
        .pipelines
        .iter()
        .flat_map(|pipeline| pipeline.agxps_trace_costs.iter())
        .map(|cost| cost.analyzer_weighted_duration)
        .sum::<u64>();
    if agxps_analyzer_denominator > 0 {
        let mut pipelines = report
            .pipelines
            .iter()
            .filter(|pipeline| {
                pipeline
                    .agxps_trace_costs
                    .iter()
                    .any(|cost| cost.analyzer_weighted_duration > 0)
            })
            .collect::<Vec<_>>();
        pipelines.sort_by(|left, right| {
            let left_cost = left
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.analyzer_weighted_duration)
                .sum::<u64>();
            let right_cost = right
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.analyzer_weighted_duration)
                .sum::<u64>();
            right_cost
                .cmp(&left_cost)
                .then_with(|| left.index.cmp(&right.index))
        });
        out.push_str("Pipelines by AGXPS timing-analyzer clique duration:\n");
        for pipeline in pipelines.iter().take(20) {
            let name = pipeline
                .function_name
                .as_deref()
                .unwrap_or("<unknown function>");
            let weighted = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.analyzer_weighted_duration)
                .sum::<u64>();
            let avg_sum = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.analyzer_avg_duration_sum)
                .sum::<u64>();
            let record_cliques = pipeline
                .agxps_trace_costs
                .iter()
                .map(|cost| cost.record_cliques)
                .sum::<u64>();
            let pct = 100.0 * weighted as f64 / agxps_analyzer_denominator as f64;
            out.push_str(&format!(
                "  {:>6.3}% weighted={:>12} avg_sum={:>10} rec_cliques={:>7} {:<56}\n",
                pct, weighted, avg_sum, record_cliques, name,
            ));
        }
        out.push('\n');
    }
    let agxps_event_denominator = report
        .pipelines
        .iter()
        .flat_map(|pipeline| pipeline.agxps_trace_costs.iter())
        .map(|cost| cost.execution_events)
        .sum::<u64>();
    let mut agxps_rows = report
        .pipelines
        .iter()
        .flat_map(|pipeline| {
            pipeline
                .agxps_trace_costs
                .iter()
                .map(move |cost| (pipeline, cost))
        })
        .collect::<Vec<_>>();
    agxps_rows.sort_by(|(left_pipeline, left_cost), (right_pipeline, right_cost)| {
        left_cost
            .shader_address
            .cmp(&right_cost.shader_address)
            .then_with(|| left_pipeline.index.cmp(&right_pipeline.index))
    });
    if !agxps_rows.is_empty() {
        out.push_str("AGXPS timing rows by ESL shader address:\n");
        out.push_str(
            "  row analyzer%      w1%   events% analyzer_weighted           w1     events cmds rec_cliques matched function      esl_shader\n",
        );
        for (row_index, (pipeline, cost)) in agxps_rows.iter().enumerate().take(80) {
            let name = pipeline.function_name.as_deref().unwrap_or("<unknown>");
            let analyzer_pct =
                percent_u64(cost.analyzer_weighted_duration, agxps_analyzer_denominator);
            let w1_pct = percent_u64(cost.stats_word1, agxps_trace_denominator);
            let events_pct = percent_u64(cost.execution_events, agxps_event_denominator);
            out.push_str(&format!(
                "  {row:>3} {analyzer_pct:>8.4}% {w1_pct:>8.4}% {events_pct:>8.4}% {weighted:>17} {w1:>12} {events:>10} {cmds:>4} {cliques:>11} {matched:>7} {name:<13} 0x{address:x}\n",
                row = row_index,
                weighted = cost.analyzer_weighted_duration,
                w1 = cost.stats_word1,
                events = cost.execution_events,
                cmds = cost.command_count,
                cliques = cost.record_cliques,
                matched = cost.matched_work_cliques,
                address = cost.shader_address,
            ));
        }
        out.push('\n');
    }
    let mut shader_binary_costs = report
        .pipelines
        .iter()
        .flat_map(|pipeline| {
            pipeline
                .shader_binary_costs
                .iter()
                .map(move |cost| (pipeline, cost))
        })
        .filter(|(_, cost)| cost.total_cost > 0.0 || cost.total_instruction_count > 0)
        .collect::<Vec<_>>();
    shader_binary_costs.sort_by(|(left_pipeline, left), (right_pipeline, right)| {
        right
            .total_cost
            .partial_cmp(&left.total_cost)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .total_instruction_count
                    .cmp(&left.total_instruction_count)
            })
            .then_with(|| left_pipeline.index.cmp(&right_pipeline.index))
    });
    if !shader_binary_costs.is_empty() {
        let denominator = shader_binary_costs
            .iter()
            .map(|(_, cost)| cost.total_cost)
            .filter(|value| value.is_finite() && *value > 0.0)
            .sum::<f64>();
        out.push_str("Pipelines by shader instruction-cost arrays:\n");
        for (pipeline, cost) in shader_binary_costs.iter().take(20) {
            let name = pipeline
                .function_name
                .as_deref()
                .unwrap_or("<unknown function>");
            let pct = if denominator > 0.0 {
                100.0 * cost.total_cost / denominator
            } else {
                0.0
            };
            out.push_str(&format!(
                "  {:>6.2}% cost={:>10.3} instr={:>10} nonzero={:>4}/{:<4} ptype={:<2} {:<56} {}={} binary={} addr=0x{:x}\n",
                pct,
                cost.total_cost,
                cost.total_instruction_count,
                cost.nonzero_record_count,
                cost.record_count,
                cost.program_type,
                name,
                cost.pipeline_id_kind,
                cost.pipeline_id,
                cost.binary_index,
                cost.address,
            ));
        }
        out.push('\n');
    }
    let mut profiler_timings = report
        .pipelines
        .iter()
        .filter_map(|pipeline| {
            let best = pipeline.profiler_timings.iter().max_by(|left, right| {
                left.percentage_average
                    .partial_cmp(&right.percentage_average)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?;
            (best.percentage_average > 0.0 || best.time_average > 0.0).then_some((pipeline, best))
        })
        .collect::<Vec<_>>();
    profiler_timings.sort_by(|(left_pipeline, left), (right_pipeline, right)| {
        right
            .percentage_average
            .partial_cmp(&left.percentage_average)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .time_average
                    .partial_cmp(&left.time_average)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left_pipeline.index.cmp(&right_pipeline.index))
    });
    if !profiler_timings.is_empty() {
        out.push_str("Pipelines by Xcode profiler timing:\n");
        for (pipeline, timing) in profiler_timings.iter().take(20) {
            let name = pipeline
                .function_name
                .as_deref()
                .unwrap_or("<unknown function>");
            out.push_str(&format!(
                "  {:>6.2}% time={:>10.3} cycles={:>10.3} source={:<8} {:<56}\n",
                timing.percentage_average,
                timing.time_average,
                timing.cycle_average,
                timing.source,
                name,
            ));
        }
        out.push('\n');
    }
    let mut shader_profiler_costs = report
        .pipelines
        .iter()
        .filter(|pipeline| !pipeline.shader_profiler_costs.is_empty())
        .collect::<Vec<_>>();
    shader_profiler_costs.sort_by(|left, right| {
        let left_cost = left
            .shader_profiler_costs
            .iter()
            .map(|cost| cost.pipeline_cost)
            .sum::<f64>();
        let right_cost = right
            .shader_profiler_costs
            .iter()
            .map(|cost| cost.pipeline_cost)
            .sum::<f64>();
        right_cost
            .partial_cmp(&left_cost)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.index.cmp(&right.index))
    });
    let shader_profiler_denominator = report
        .pipelines
        .iter()
        .flat_map(|pipeline| pipeline.shader_profiler_costs.iter())
        .map(|cost| cost.pipeline_cost)
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum::<f64>();
    if shader_profiler_denominator > 0.0 {
        out.push_str("Pipelines by Xcode shader-profiler per-draw cost:\n");
        for pipeline in shader_profiler_costs.iter().take(20) {
            let name = pipeline
                .function_name
                .as_deref()
                .unwrap_or("<unknown function>");
            let pipeline_cost = pipeline
                .shader_profiler_costs
                .iter()
                .map(|cost| cost.pipeline_cost)
                .sum::<f64>();
            let draw_count = pipeline
                .shader_profiler_costs
                .iter()
                .map(|cost| cost.nonzero_draw_count)
                .sum::<usize>();
            let pct = 100.0 * pipeline_cost / shader_profiler_denominator;
            let top_binary = pipeline.shader_profiler_costs.iter().max_by(|left, right| {
                left.pipeline_cost
                    .partial_cmp(&right.pipeline_cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let key = top_binary
                .map(|cost| cost.binary_key.as_str())
                .unwrap_or("-");
            out.push_str(&format!(
                "  {:>6.2}% cost={:>10.3} draws={:>4} {:<56} key={}\n",
                pct, pipeline_cost, draw_count, name, key,
            ));
        }
        out.push('\n');
    }
    let mut execution_history = report
        .pipelines
        .iter()
        .flat_map(|pipeline| {
            pipeline
                .execution_history
                .iter()
                .map(move |history| (pipeline, history))
        })
        .filter(|(_, history)| {
            history.top_cost_percentage > 0.0
                || history.duration_percentage > 0.0
                || history.total_cost > 0.0
        })
        .collect::<Vec<_>>();
    execution_history.sort_by(|(left_pipeline, left), (right_pipeline, right)| {
        right
            .top_cost_percentage
            .partial_cmp(&left.top_cost_percentage)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                right
                    .total_cost
                    .partial_cmp(&left.total_cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left_pipeline.index.cmp(&right_pipeline.index))
    });
    if !execution_history.is_empty() {
        out.push_str("Pipelines by Xcode execution-history cost:\n");
        for (pipeline, history) in execution_history.iter().take(20) {
            let name = pipeline
                .function_name
                .as_deref()
                .unwrap_or("<unknown function>");
            out.push_str(&format!(
                "  {:>6.2}% dur={:>6.2}% cost={:>10.3} instr={:>10} ptype={:<2} style={}/{} {:<56} {}={}\n",
                history.top_cost_percentage,
                history.duration_percentage,
                history.total_cost,
                history.instruction_count,
                history.program_type,
                history.style,
                history.options,
                name,
                history.pipeline_id_kind,
                history.pipeline_id,
            ));
        }
        out.push('\n');
    }
    let mut shader_tracks = report
        .pipelines
        .iter()
        .flat_map(|pipeline| {
            pipeline
                .shader_tracks
                .iter()
                .map(move |track| (pipeline, track))
        })
        .filter(|(_, track)| track.duration_ns > 0 || track.trace_count > 0)
        .collect::<Vec<_>>();
    shader_tracks.sort_by(|(left_pipeline, left), (right_pipeline, right)| {
        right
            .duration_ns
            .cmp(&left.duration_ns)
            .then_with(|| right.trace_count.cmp(&left.trace_count))
            .then_with(|| left_pipeline.index.cmp(&right_pipeline.index))
            .then_with(|| left.program_type.cmp(&right.program_type))
    });
    if !shader_tracks.is_empty() {
        out.push_str("Pipelines by Xcode shader track duration:\n");
        let denominator = report
            .cost_timeline
            .as_ref()
            .map(|timeline| timeline.global_gpu_time_ns)
            .filter(|value| *value > 0)
            .unwrap_or(report.gpu_time_ns);
        for (pipeline, track) in shader_tracks.iter().take(20) {
            let name = pipeline
                .function_name
                .as_deref()
                .unwrap_or("<unknown function>");
            let pct = if denominator > 0 {
                100.0 * track.duration_ns as f64 / denominator as f64
            } else {
                0.0
            };
            out.push_str(&format!(
                "  {:>6.2}% {:>9.3} ms  {:>4} traces  ptype={:<2} {:<56} id={}:{}\n",
                pct,
                track.duration_ns as f64 / 1_000_000.0,
                track.trace_count,
                track.program_type,
                name,
                track.pipeline_id_kind,
                track.pipeline_id,
            ));
        }
        out.push('\n');
    }
    let mut shader_binaries = report
        .pipelines
        .iter()
        .flat_map(|pipeline| {
            pipeline
                .shader_binaries
                .iter()
                .map(move |binary| (pipeline, binary))
        })
        .filter(|(_, binary)| {
            binary.duration_ns > 0 || binary.total_cost > 0.0 || binary.trace_count > 0
        })
        .collect::<Vec<_>>();
    shader_binaries.sort_by(|(left_pipeline, left), (right_pipeline, right)| {
        right
            .duration_ns
            .cmp(&left.duration_ns)
            .then_with(|| {
                right
                    .total_cost
                    .partial_cmp(&left.total_cost)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| left_pipeline.index.cmp(&right_pipeline.index))
    });
    if !shader_binaries.is_empty() {
        out.push_str("Pipelines by Xcode shader binary duration:\n");
        let denominator = report
            .cost_timeline
            .as_ref()
            .map(|timeline| timeline.global_gpu_time_ns)
            .filter(|value| *value > 0)
            .unwrap_or(report.gpu_time_ns);
        for (pipeline, binary) in shader_binaries.iter().take(20) {
            let name = pipeline
                .function_name
                .as_deref()
                .unwrap_or("<unknown function>");
            let pct = if denominator > 0 {
                100.0 * binary.duration_ns as f64 / denominator as f64
            } else {
                0.0
            };
            out.push_str(&format!(
                "  {:>6.2}% {:>9.3} ms  cost={:>10.3} traces={:>4} ptype={:<2} {:<56} binary={} addr=0x{:x}\n",
                pct,
                binary.duration_ns as f64 / 1_000_000.0,
                binary.total_cost,
                binary.trace_count,
                binary.program_type,
                name,
                binary.binary_index,
                binary.address,
            ));
        }
        out.push('\n');
    }
    let mut draw_timeline = report
        .pipelines
        .iter()
        .filter(|pipeline| pipeline.timeline_duration_ns > 0 || pipeline.timeline_draw_count > 0)
        .collect::<Vec<_>>();
    draw_timeline.sort_by(|left, right| {
        right
            .timeline_duration_ns
            .cmp(&left.timeline_duration_ns)
            .then_with(|| left.index.cmp(&right.index))
    });
    let draw_duration_denominator = report
        .cost_timeline
        .as_ref()
        .map(|timeline| timeline.global_gpu_time_ns)
        .filter(|value| *value > 0)
        .unwrap_or(report.gpu_time_ns);
    if !draw_timeline.is_empty()
        && draw_timeline
            .iter()
            .any(|pipeline| pipeline.timeline_duration_ns > 0)
    {
        out.push_str("Pipelines by Xcode draw duration:\n");
        for pipeline in draw_timeline.iter().take(20) {
            let name = pipeline
                .function_name
                .as_deref()
                .unwrap_or("<unknown function>");
            let pct = if draw_duration_denominator > 0 {
                100.0 * pipeline.timeline_duration_ns as f64 / draw_duration_denominator as f64
            } else {
                0.0
            };
            out.push_str(&format!(
                "  {:>6.2}% {:>9.3} ms  {:>4} draws  {:<56}\n",
                pct,
                pipeline.timeline_duration_ns as f64 / 1_000_000.0,
                pipeline.timeline_draw_count,
                name,
            ));
        }
        out.push('\n');
    }
    draw_timeline.sort_by(|left, right| {
        right
            .timeline_total_cost
            .partial_cmp(&left.timeline_total_cost)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.index.cmp(&right.index))
    });
    let draw_cost_denominator = report
        .pipelines
        .iter()
        .map(|pipeline| pipeline.timeline_total_cost)
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum::<f64>();
    if !draw_timeline.is_empty() && draw_cost_denominator > 0.0 {
        out.push_str("Pipelines by Xcode draw-scope cost:\n");
        for pipeline in draw_timeline.iter().take(20) {
            let name = pipeline
                .function_name
                .as_deref()
                .unwrap_or("<unknown function>");
            let pct = 100.0 * pipeline.timeline_total_cost / draw_cost_denominator;
            out.push_str(&format!(
                "  {:>6.2}% cost={:>10.3} {:>4} draws  {:<56}\n",
                pct, pipeline.timeline_total_cost, pipeline.timeline_draw_count, name,
            ));
        }
        out.push('\n');
    }
    out.push_str("Pipelines by command count:\n");
    let mut pipelines = report.pipelines.clone();
    pipelines.sort_by(|left, right| {
        right
            .gpu_command_count
            .cmp(&left.gpu_command_count)
            .then_with(|| left.index.cmp(&right.index))
    });
    for pipeline in pipelines {
        let name = pipeline
            .function_name
            .as_deref()
            .unwrap_or("<unknown function>");
        let address = pipeline
            .pipeline_address
            .map(|value| format!("0x{value:x}"))
            .unwrap_or_else(|| "-".to_owned());
        out.push_str(&format!(
            "  {:>2} {:>4} commands  {:<56} addr={} object_id={}\n",
            pipeline.index, pipeline.gpu_command_count, name, address, pipeline.object_id
        ));
    }

    if !report.warnings.is_empty() {
        out.push_str("\nWarnings:\n");
        for warning in &report.warnings {
            out.push_str(&format!("  - {warning}\n"));
        }
    }

    out
}

fn percent_u64(value: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        100.0 * value as f64 / denominator as f64
    }
}

fn counter_value(row: &XcodeMioGpuCommandCounterRow, name: &str) -> Option<f64> {
    row.counters
        .iter()
        .find(|counter| counter.name == name)
        .map(|counter| counter.value)
}

fn format_optional_counter(value: Option<f64>, precision: usize) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:.precision$}"),
        _ => "-".to_owned(),
    }
}

fn best_function_time_rows<'a, I>(rows: I) -> Vec<&'a XcodeMioGpuCommandFunctionTime>
where
    I: IntoIterator<Item = &'a XcodeMioGpuCommandFunctionTime>,
{
    let mut by_command =
        std::collections::BTreeMap::<usize, &XcodeMioGpuCommandFunctionTime>::new();
    for row in rows {
        by_command
            .entry(row.command_index)
            .and_modify(|existing| {
                if row.duration_ns > existing.duration_ns {
                    *existing = row;
                }
            })
            .or_insert(row);
    }
    by_command.into_values().collect()
}

fn best_top_draw_track_rows<'a, I>(rows: I) -> Vec<&'a XcodeMioTopDrawTrack>
where
    I: IntoIterator<Item = &'a XcodeMioTopDrawTrack>,
{
    let mut by_command = std::collections::BTreeMap::<usize, &XcodeMioTopDrawTrack>::new();
    for row in rows {
        let Some(command_index) = row.command_index else {
            continue;
        };
        by_command
            .entry(command_index)
            .and_modify(|existing| {
                let duration = row.trace_duration_ns.max(row.duration_ns);
                let existing_duration = existing.trace_duration_ns.max(existing.duration_ns);
                if duration > existing_duration {
                    *existing = row;
                }
            })
            .or_insert(row);
    }
    by_command.into_values().collect()
}

fn format_duration_ns(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.3} ms", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.3} us", value as f64 / 1_000.0)
    } else {
        format!("{value} ns")
    }
}

fn format_u32_samples(values: &[u32], limit: usize) -> String {
    if values.is_empty() {
        return "-".to_owned();
    }
    let mut out = values
        .iter()
        .take(limit)
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    if values.len() > limit {
        out.push_str(",...");
    }
    out
}

fn format_u16_samples(values: &[u16], limit: usize) -> String {
    if values.is_empty() {
        return "-".to_owned();
    }
    let mut out = values
        .iter()
        .take(limit)
        .map(u16::to_string)
        .collect::<Vec<_>>()
        .join(",");
    if values.len() > limit {
        out.push_str(",...");
    }
    out
}

fn format_private_numeric_value(value: f64) -> String {
    if !value.is_finite() {
        value.to_string()
    } else if value.abs() >= 1000.0 || value.fract().abs() < 0.000_001 {
        format!("{value:.0}")
    } else {
        format!("{value:.6}")
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use std::collections::{BTreeMap, BTreeSet};
    use std::ffi::{CStr, CString, c_char, c_int, c_long, c_void};
    use std::fs;
    use std::mem;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use super::{
        XcodeMioBinaryTrace, XcodeMioCostTimeline, XcodeMioDecodeOptions,
        XcodeMioDecodedCostRecord, XcodeMioDrawArrayProbe, XcodeMioDrawExecutionHistoryProbe,
        XcodeMioDrawMetadataRecord, XcodeMioDrawTimelineRecord, XcodeMioEncoder,
        XcodeMioEncoderQuadProbe, XcodeMioFunctionTimeProbe, XcodeMioGpuCommand,
        XcodeMioGpuCommandCounterRow, XcodeMioGpuCommandDirectCost, XcodeMioGpuCommandFunctionTime,
        XcodeMioGpuCommandShaderProfilerCost, XcodeMioPipeline, XcodeMioPipelineAgxpsTraceCost,
        XcodeMioPipelineCounter, XcodeMioPipelineExecutionHistory, XcodeMioPipelineProfilerTiming,
        XcodeMioPipelineScopeCost, XcodeMioPipelineShaderBinary, XcodeMioPipelineShaderBinaryCost,
        XcodeMioPipelineShaderBinaryReference, XcodeMioPipelineShaderProfilerCost,
        XcodeMioPipelineShaderStat, XcodeMioPipelineShaderTrack, XcodeMioPrivateNumericArray,
        XcodeMioReport, XcodeMioShaderBinaryInfo, XcodeMioTimelineBinary,
        XcodeMioTimelineCandidate, XcodeMioTimings, XcodeMioTopDrawTrack, XcodeMioUSCCliqueProbe,
        XcodeMioUSCCliqueSummary, elapsed_ms,
    };
    use crate::error::{Error, Result};
    use crate::profiler;
    use block2::RcBlock;

    type Id = *mut c_void;
    type Class = *mut c_void;
    type Sel = *mut c_void;
    type Ivar = *mut c_void;
    type CfTypeRef = *const c_void;
    type CfStringRef = *const c_void;

    const RTLD_NOW: c_int = 0x2;
    const RTLD_GLOBAL: c_int = 0x8;
    const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
    /// `<Xcode.app>/Contents`, resolved from the active developer dir via
    /// `xcode-select -p` (which honors `DEVELOPER_DIR` and points at Xcode-beta
    /// when that's selected). Falls back to the stock `/Applications/Xcode.app`.
    fn xcode_contents_dir() -> std::path::PathBuf {
        std::process::Command::new("xcode-select")
            .arg("-p")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| std::path::PathBuf::from(s.trim()))
            // `xcode-select -p` → `<app>/Contents/Developer`; strip `/Developer`.
            .and_then(|dev| dev.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| std::path::PathBuf::from("/Applications/Xcode.app/Contents"))
    }

    fn gt_shader_profiler_framework() -> String {
        xcode_contents_dir()
            .join("PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/GTShaderProfiler.framework/Versions/A/GTShaderProfiler")
            .to_string_lossy()
            .into_owned()
    }

    fn mtl_tools_shader_profiler_framework() -> String {
        xcode_contents_dir()
            .join("SharedFrameworks/MTLToolsShaderProfiler.framework/Versions/A/MTLToolsShaderProfiler")
            .to_string_lossy()
            .into_owned()
    }

    unsafe extern "C" {
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn open(path: *const c_char, oflag: c_int, ...) -> c_int;
        fn dup(fd: c_int) -> c_int;
        fn dup2(src: c_int, dst: c_int) -> c_int;
        fn close(fd: c_int) -> c_int;
        fn objc_lookUpClass(name: *const c_char) -> Class;
        fn object_getClass(obj: Id) -> Class;
        fn class_getName(cls: Class) -> *const c_char;
        fn class_getInstanceVariable(cls: Class, name: *const c_char) -> Ivar;
        fn ivar_getTypeEncoding(ivar: Ivar) -> *const c_char;
        fn object_getIvar(obj: Id, ivar: Ivar) -> Id;
        fn sel_registerName(name: *const c_char) -> Sel;
        fn objc_msgSend();
        fn CFStringCreateWithCString(
            alloc: CfTypeRef,
            c_str: *const c_char,
            encoding: u32,
        ) -> CfStringRef;
        fn CFRelease(cf: CfTypeRef);
    }

    #[link(name = "Foundation", kind = "framework")]
    unsafe extern "C" {}

    #[link(name = "objc")]
    unsafe extern "C" {}

    pub fn decode(
        trace_source: PathBuf,
        profiler_directory: PathBuf,
        stream_data_path: PathBuf,
        profiler_summary: Option<&profiler::ProfilerStreamDataSummary>,
        mut timings: XcodeMioTimings,
        options: XcodeMioDecodeOptions,
    ) -> Result<XcodeMioReport> {
        let framework_path = PathBuf::from(gt_shader_profiler_framework());
        let silence = FdSilencer::new();
        let framework_start = Instant::now();
        let mut runtime = unsafe { Runtime::load()? };
        timings.framework_load_ms = elapsed_ms(framework_start);
        let stream_start = Instant::now();
        let stream = runtime.stream_data(&stream_data_path)?;
        timings.stream_data_load_ms = elapsed_ms(stream_start);
        let processor_start = Instant::now();
        let processor = unsafe { runtime.processor(stream)? };
        timings.processor_init_ms = elapsed_ms(processor_start);
        let process_start = Instant::now();
        unsafe {
            runtime.send_void(processor, "processStreamData")?;
            runtime.send_void(processor, "waitUntilFinished")?;
            if responds_to_selector(processor, "processShaderProfilerStreamData") {
                runtime.send_void(processor, "processShaderProfilerStreamData")?;
            }
            if responds_to_selector(processor, "waitUntilShaderProfilerFinished") {
                runtime.send_void(processor, "waitUntilShaderProfilerFinished")?;
            }
        }
        timings.process_stream_ms = elapsed_ms(process_start);
        let extract_start = Instant::now();
        let mio = unsafe { runtime.send_id(processor, "mioData")? };
        let result = unsafe { runtime.send_id(processor, "result")? };
        let shader_result = unsafe { runtime.send_id(result, "shaderProfilerResult")? };
        let gpu_commands = unsafe { runtime.send_id(shader_result, "gpuCommands")? };
        let pipelines = unsafe { runtime.send_id(shader_result, "pipelineStates")? };
        let encoders = unsafe { runtime.send_id(shader_result, "encoders")? };

        let pipeline_count = unsafe { runtime.array_count(pipelines)? };
        let command_count = unsafe { runtime.array_count(gpu_commands)? };
        let encoder_count = unsafe { runtime.array_count(encoders)? };
        timings.extract_result_ms = elapsed_ms(extract_start);
        let topology_start = Instant::now();
        let mut decoded_pipelines = Vec::with_capacity(pipeline_count);
        for index in 0..pipeline_count {
            let pipeline = unsafe { runtime.array_object(pipelines, index)? };
            let summary_pipeline =
                profiler_summary.and_then(|summary| summary.pipelines.get(index));
            let binary_keys = unsafe {
                runtime
                    .send_id_allow_nil(pipeline, "binaryKeys")
                    .ok()
                    .filter(|array| !array.is_null())
                    .map(|array| runtime.u64_array(array))
                    .transpose()?
                    .unwrap_or_default()
            };
            let all_binary_keys = unsafe {
                runtime
                    .send_id_allow_nil(pipeline, "allBinaryKeys")
                    .ok()
                    .filter(|array| !array.is_null())
                    .map(|array| runtime.u64_array(array))
                    .transpose()?
                    .unwrap_or_default()
            };
            decoded_pipelines.push(XcodeMioPipeline {
                index: unsafe { runtime.send_u32(pipeline, "index")? as usize },
                object_id: unsafe { runtime.send_u64(pipeline, "objectId")? },
                pointer_id: unsafe { runtime.send_u64(pipeline, "pointerId")? },
                function_index: unsafe { runtime.send_u64(pipeline, "functionIndex")? },
                gpu_command_count: unsafe {
                    runtime.send_u32(pipeline, "numGPUCommands")? as usize
                },
                timeline_draw_count: 0,
                timeline_duration_ns: 0,
                timeline_total_cost: 0.0,
                pipeline_address: summary_pipeline.map(|pipeline| pipeline.pipeline_address),
                function_name: summary_pipeline.and_then(|pipeline| pipeline.function_name.clone()),
                binary_keys,
                all_binary_keys,
                shader_stats: Vec::new(),
                profiler_timings: unsafe { decode_pipeline_profiler_timings(pipeline) },
                scope_costs: Vec::new(),
                shader_tracks: Vec::new(),
                shader_binaries: Vec::new(),
                shader_binary_costs: Vec::new(),
                agxps_trace_costs: Vec::new(),
                shader_profiler_costs: Vec::new(),
                execution_history: Vec::new(),
                shader_binary_references: Vec::new(),
                pipeline_counters: Vec::new(),
            });
        }

        let mut decoded_commands = Vec::with_capacity(command_count);
        for index in 0..command_count {
            let command = unsafe { runtime.array_object(gpu_commands, index)? };
            let pipeline_index =
                unsafe { runtime.send_u32(command, "pipelineInfoIndex")? as usize };
            let function_name = profiler_summary
                .and_then(|summary| summary.pipelines.get(pipeline_index))
                .and_then(|pipeline| pipeline.function_name.clone());
            decoded_commands.push(XcodeMioGpuCommand {
                index: unsafe { runtime.send_u32(command, "index")? as usize },
                function_index: unsafe { runtime.send_u64(command, "functionIndex")? },
                sub_command_index: unsafe { runtime.send_i32(command, "subCommandIndex")? },
                encoder_index: unsafe { runtime.send_u32(command, "encoderInfoIndex")? as usize },
                pipeline_index,
                pipeline_object_id: unsafe { runtime.send_u64(command, "pipelineStateObjectId")? },
                command_buffer_index: unsafe {
                    runtime.send_u32(command, "commandBufferIndex")? as usize
                },
                function_name,
            });
        }

        let mut decoded_encoders = Vec::with_capacity(encoder_count);
        for index in 0..encoder_count {
            let encoder = unsafe { runtime.array_object(encoders, index)? };
            decoded_encoders.push(XcodeMioEncoder {
                index,
                function_index: unsafe { runtime.send_u64(encoder, "functionIndex")? },
                gpu_command_start_index: unsafe {
                    runtime.send_u32(encoder, "gpuCommandStartIndex")? as usize
                },
                gpu_command_count: unsafe { runtime.send_u32(encoder, "numGPUCommands")? as usize },
                load_time: unsafe { runtime.send_u64(encoder, "loadTime")? },
                store_time: unsafe { runtime.send_u64(encoder, "storeTime")? },
            });
        }
        timings.decode_pipeline_commands_ms = elapsed_ms(topology_start);

        let mut warnings = Vec::new();
        let mut gpu_command_shader_profiler_costs = Vec::new();
        let shader_probe_start = Instant::now();
        if let Err(error) = unsafe {
            runtime.decode_shader_profiler_costs(
                "processor",
                shader_result,
                &decoded_commands,
                &mut decoded_pipelines,
                &mut gpu_command_shader_profiler_costs,
            )
        } {
            warnings.push(format!(
                "private shader-profiler cost probe failed: {error}"
            ));
        }
        if let Err(error) = unsafe {
            runtime.decode_shader_profiler_timing_info(shader_result, &mut decoded_pipelines)
        } {
            warnings.push(format!(
                "private shader-profiler timingInfo probe failed: {error}"
            ));
        }
        let mut shader_profiler_numeric_arrays = Vec::new();
        unsafe {
            push_parent_processor_numeric_probes(&mut shader_profiler_numeric_arrays, processor);
        }
        let mut shader_profiler_results = vec![("processor.shader_result", shader_result)];
        match unsafe { runtime.direct_shader_profiler_result(stream) } {
            Ok(direct) => {
                let direct_shader_result = direct.shader_result;
                shader_profiler_results.push(("direct.shader_result", direct_shader_result));
                shader_profiler_numeric_arrays.extend(direct.numeric_arrays);
                if let Err(error) = unsafe {
                    runtime.merge_shader_profiler_pipeline_state_data(
                        direct_shader_result,
                        &mut decoded_pipelines,
                    )
                } {
                    warnings.push(format!(
                        "direct shader-profiler pipeline probe failed: {error}"
                    ));
                }
                if let Err(error) = unsafe {
                    runtime.decode_shader_profiler_timing_info(
                        direct_shader_result,
                        &mut decoded_pipelines,
                    )
                } {
                    warnings.push(format!(
                        "direct shader-profiler timingInfo probe failed: {error}"
                    ));
                }
                if let Err(error) = unsafe {
                    runtime.decode_shader_profiler_costs(
                        "direct",
                        direct_shader_result,
                        &decoded_commands,
                        &mut decoded_pipelines,
                        &mut gpu_command_shader_profiler_costs,
                    )
                } {
                    warnings.push(format!("direct shader-profiler cost probe failed: {error}"));
                }
            }
            Err(error) => {
                warnings.push(format!("direct shader-profiler processor failed: {error}"));
            }
        }
        gpu_command_shader_profiler_costs.sort_by(|left, right| {
            left.source
                .cmp(right.source)
                .then_with(|| left.draw_index.cmp(&right.draw_index))
                .then_with(|| left.command_index.cmp(&right.command_index))
                .then_with(|| left.binary_key.cmp(&right.binary_key))
        });
        for (source, shader_result) in &shader_profiler_results {
            let class_name = unsafe { object_class_name(*shader_result) }
                .unwrap_or_else(|| "<unknown class>".to_owned());
            let has_gpu_command = unsafe {
                responds_to_selector(
                    *shader_result,
                    "gpuCommandForFunctionIndex:subCommandIndex:",
                )
            };
            warnings.push(format!(
                "{source}: class={class_name} gpuCommandForFunctionIndex={has_gpu_command}"
            ));
        }
        timings.shader_profiler_probe_ms = elapsed_ms(shader_probe_start);
        let cost_request_start = Instant::now();
        let (cost_timeline, cost_timeline_object) =
            match unsafe { runtime.request_cost_timeline(mio) } {
                Ok(Some(timeline)) => {
                    let object = timeline.object;
                    (Some(timeline.summary), Some(object))
                }
                Ok(None) => (None, None),
                Err(error) => {
                    warnings.push(format!("private cost timeline callback failed: {error}"));
                    (None, None)
                }
            };
        let timeline_candidates = unsafe { runtime.timeline_candidates(mio, cost_timeline_object) };
        timings.cost_timeline_request_ms = elapsed_ms(cost_request_start);
        let mut decoded_cost_records = Vec::new();
        let mut draw_timeline_records = Vec::new();
        let mut draw_metadata_records = Vec::new();
        let mut timeline_binary_count = 0;
        let mut timeline_binaries = Vec::new();
        let mut timeline_pipeline_state_ids = Vec::new();
        let mut shader_binary_info = Vec::new();
        let mut gpu_command_counter_rows = Vec::new();
        if let Some(timeline) = cost_timeline_object {
            let cost_decode_start = Instant::now();
            timeline_binary_count = unsafe { runtime.timeline_binary_count(timeline) };
            timeline_binaries = unsafe { runtime.decode_timeline_binaries(timeline) };
            timeline_pipeline_state_ids =
                unsafe { runtime.decode_timeline_pipeline_state_ids(timeline) };
            shader_binary_info = unsafe { runtime.decode_shader_binary_info(timeline) };
            attach_shader_binary_references(
                &shader_binary_info,
                &timeline_binaries,
                &decoded_commands,
                &mut decoded_pipelines,
            );
            if let Err(error) =
                decode_agxps_timing_trace_costs(&profiler_directory, &mut decoded_pipelines)
            {
                warnings.push(format!("AGXPS timing-trace probe failed: {error}"));
            }
            if options.decode_cost_details {
                decoded_cost_records = unsafe { runtime.decode_timeline_cost_records(timeline) };
                draw_timeline_records = unsafe { runtime.decode_draw_timeline_records(timeline) };
                draw_metadata_records = unsafe { runtime.decode_draw_metadata_records(timeline) };
                unsafe {
                    runtime.decode_pipeline_draw_timeline(
                        mio,
                        timeline,
                        &draw_timeline_records,
                        &draw_metadata_records,
                        &decoded_commands,
                        &mut decoded_pipelines,
                    )
                };
                if let Err(error) = unsafe {
                    runtime.decode_pipeline_private_costs(
                        mio,
                        timeline,
                        &decoded_cost_records,
                        &mut decoded_pipelines,
                        &timeline_pipeline_state_ids,
                    )
                } {
                    warnings.push(format!("private per-pipeline cost probe failed: {error}"));
                }
                if let Err(error) =
                    unsafe { runtime.decode_pipeline_counters(mio, &mut decoded_pipelines) }
                {
                    warnings.push(format!("private pipeline counter probe failed: {error}"));
                }
            }
            timings.cost_timeline_decode_ms = elapsed_ms(cost_decode_start);
        }
        let (mut gpu_command_function_times, gpu_command_function_time_probes) = unsafe {
            runtime.decode_gpu_command_function_times(
                mio,
                cost_timeline_object,
                &decoded_pipelines,
                &decoded_encoders,
                &decoded_commands,
                &timeline_pipeline_state_ids,
            )
        };
        for (source, shader_result) in shader_profiler_results {
            let (rows, warning) = unsafe {
                runtime.decode_shader_profiler_gpu_command_times(
                    source,
                    shader_result,
                    &decoded_commands,
                )
            };
            gpu_command_function_times.extend(rows);
            warnings.push(warning);
        }
        gpu_command_function_times.sort_by(|left, right| {
            left.source
                .cmp(right.source)
                .then_with(|| left.command_index.cmp(&right.command_index))
                .then_with(|| left.draw_index.cmp(&right.draw_index))
                .then_with(|| left.data_master.cmp(&right.data_master))
        });
        let gpu_command_direct_costs = unsafe {
            runtime.decode_gpu_command_direct_costs(mio, cost_timeline_object, &decoded_commands)
        };
        let draw_array_probes = unsafe {
            runtime.decode_draw_array_probes(
                mio,
                cost_timeline_object,
                &decoded_pipelines,
                &decoded_commands,
            )
        };
        let usc_clique_summaries =
            unsafe { runtime.decode_usc_clique_summaries(mio, cost_timeline_object) };
        let usc_clique_probes = unsafe {
            runtime.decode_usc_clique_probes(
                mio,
                cost_timeline_object,
                &decoded_pipelines,
                &decoded_commands,
            )
        };
        let encoder_quad_probes = unsafe {
            runtime.decode_encoder_quad_probes(
                mio,
                cost_timeline_object,
                &decoded_pipelines,
                &decoded_encoders,
                &decoded_commands,
                &timeline_pipeline_state_ids,
            )
        };
        let draw_execution_history_probes = unsafe {
            runtime.decode_draw_execution_history_probes(
                mio,
                cost_timeline_object,
                &decoded_pipelines,
                &decoded_commands,
            )
        };
        let top_draw_tracks =
            unsafe { runtime.decode_top_draw_tracks(mio, cost_timeline_object, &decoded_commands) };
        if options.decode_cost_details {
            match unsafe { runtime.decode_gpu_command_counters(mio, &decoded_commands) } {
                Ok(rows) => gpu_command_counter_rows = rows,
                Err(error) => {
                    warnings.push(format!("private GPU command counter probe failed: {error}"))
                }
            }
        }
        if profiler_summary.is_some_and(|summary| summary.num_gpu_commands != command_count) {
            warnings.push("private MIO command count differs from streamData summary".to_owned());
        }
        if decoded_pipelines
            .iter()
            .any(|pipeline| pipeline.function_name.is_none())
        {
            warnings.push("one or more private MIO pipelines could not be named".to_owned());
        }

        let final_metadata_start = Instant::now();
        let draw_count = unsafe { runtime.send_u64(mio, "drawCount")? as usize };
        let cost_record_count = unsafe { runtime.send_u64(mio, "costCount")? as usize };
        let gpu_time_ns = unsafe { runtime.send_u64(mio, "gpuTime")? };
        timings.final_metadata_ms = elapsed_ms(final_metadata_start);

        let report = XcodeMioReport {
            trace_source,
            profiler_directory,
            stream_data_path,
            framework_path,
            timings,
            gpu_command_count: command_count,
            encoder_count,
            pipeline_state_count: pipeline_count,
            draw_count,
            cost_record_count,
            gpu_time_ns,
            cost_timeline,
            timeline_candidates,
            timeline_binary_count,
            timeline_binaries,
            timeline_pipeline_state_ids,
            shader_binary_info,
            decoded_cost_records,
            draw_timeline_records,
            draw_metadata_records,
            pipelines: decoded_pipelines,
            encoders: decoded_encoders,
            gpu_commands: decoded_commands,
            gpu_command_function_times,
            gpu_command_function_time_probes,
            draw_array_probes,
            usc_clique_summaries,
            usc_clique_probes,
            encoder_quad_probes,
            draw_execution_history_probes,
            top_draw_tracks,
            gpu_command_direct_costs,
            gpu_command_shader_profiler_costs,
            gpu_command_counter_rows,
            shader_profiler_numeric_arrays,
            warnings,
        };
        drop(runtime);
        std::thread::sleep(std::time::Duration::from_millis(100));
        drop(silence);
        Ok(report)
    }

    struct Runtime {
        pool: Id,
    }

    struct DirectShaderProfilerResult {
        shader_result: Id,
        numeric_arrays: Vec<XcodeMioPrivateNumericArray>,
    }

    #[derive(Clone, Copy)]
    struct EnumFunctionTimeSources {
        object_id: &'static str,
        pointer_id: &'static str,
        function_index: &'static str,
        pipeline_index: &'static str,
    }

    #[derive(Clone, Copy)]
    struct FunctionTimeTraceDataSource {
        object: Id,
        source: &'static str,
        function_index_source: &'static str,
        command_index_source: &'static str,
        pipeline_sources: EnumFunctionTimeSources,
    }

    impl Runtime {
        unsafe fn load() -> Result<Self> {
            unsafe {
                let _ = load_framework(&mtl_tools_shader_profiler_framework());
                load_framework(&gt_shader_profiler_framework())?;
                let pool_class = lookup_class("NSAutoreleasePool")?;
                let pool = send_id(send_id(pool_class, "alloc")?, "init")?;
                Ok(Self { pool })
            }
        }

        fn stream_data(&mut self, stream_data_path: &std::path::Path) -> Result<Id> {
            let path = stream_data_path.to_string_lossy();
            let path = CString::new(path.as_bytes())
                .map_err(|_| Error::InvalidInput("streamData path contains NUL".to_owned()))?;
            unsafe {
                let path = CfString::new(path.as_ptr())?;
                let url_class = lookup_class("NSURL")?;
                let url = send_id_id(url_class, "fileURLWithPath:", path.ptr.cast_mut())?;
                let stream_class = lookup_class("GTShaderProfilerStreamData")?;
                send_id_id(stream_class, "dataFromArchivedDataURL:", url)
            }
        }

        unsafe fn processor(&mut self, stream: Id) -> Result<Id> {
            unsafe {
                let processor_class = lookup_class("GTShaderProfilerStreamDataProcessor")?;
                let processor = send_id(processor_class, "alloc")?;
                send_id_id_id(
                    processor,
                    "initWithStreamData:llvmHelperPath:",
                    stream,
                    std::ptr::null_mut(),
                )
            }
        }

        unsafe fn direct_shader_profiler_result(
            &mut self,
            stream: Id,
        ) -> Result<DirectShaderProfilerResult> {
            unsafe {
                let processor_class = lookup_class("GTAGX2StreamDataShaderProfilerProcessor")?;
                let processor = send_id(processor_class, "alloc")?;
                let processor = send_id_id(processor, "initWithStreamData:", stream)?;
                send_void(processor, "processStreamData")?;
                if responds_to_selector(processor, "waitUntilStreamDataFinished") {
                    send_void(processor, "waitUntilStreamDataFinished")?;
                }
                if responds_to_selector(processor, "waitUntilBatchIDCounterFinished") {
                    let _ = send_void(processor, "waitUntilBatchIDCounterFinished");
                }
                if responds_to_selector(processor, "processBatchIDFilteringData") {
                    let _ = send_void(processor, "processBatchIDFilteringData");
                }
                if responds_to_selector(processor, "waitUntilBatchIDCounterFinished") {
                    let _ = send_void(processor, "waitUntilBatchIDCounterFinished");
                }
                let mut numeric_arrays = Vec::new();
                push_numeric_ivar_probe(
                    &mut numeric_arrays,
                    processor,
                    "_effectivePerEncoderDrawKickTimes",
                    "processor._effectivePerEncoderDrawKickTimes",
                );
                push_numeric_ivar_probe(
                    &mut numeric_arrays,
                    processor,
                    "_shaderProfilerFrameTimes",
                    "processor._shaderProfilerFrameTimes",
                );
                if let Some(shader_profiler) =
                    object_ivar_assume_object(processor, "_shaderProfiler")
                {
                    if responds_to_selector(shader_profiler, "updatePerDrawCounters") {
                        let _ = send_void(shader_profiler, "updatePerDrawCounters");
                    }
                    push_numeric_array_probe(
                        &mut numeric_arrays,
                        shader_profiler,
                        "effectiveKickTimes",
                        "shader_profiler.effectiveKickTimes",
                    );
                    push_numeric_array_probe(
                        &mut numeric_arrays,
                        shader_profiler,
                        "averagePerDrawKickDurations",
                        "shader_profiler.averagePerDrawKickDurations",
                    );
                }
                Ok(DirectShaderProfilerResult {
                    shader_result: send_id(processor, "shaderProfilerResult")?,
                    numeric_arrays,
                })
            }
        }

        unsafe fn send_id(&mut self, receiver: Id, selector: &str) -> Result<Id> {
            unsafe { send_id(receiver, selector) }
        }

        unsafe fn send_id_allow_nil(&mut self, receiver: Id, selector: &str) -> Result<Id> {
            unsafe { send_id_allow_nil(receiver, selector) }
        }

        unsafe fn send_void(&mut self, receiver: Id, selector: &str) -> Result<()> {
            unsafe { send_void(receiver, selector) }
        }

        unsafe fn send_u64(&mut self, receiver: Id, selector: &str) -> Result<u64> {
            unsafe { send_u64(receiver, selector) }
        }

        unsafe fn send_u32(&mut self, receiver: Id, selector: &str) -> Result<u32> {
            unsafe { send_u32(receiver, selector) }
        }

        unsafe fn send_i32(&mut self, receiver: Id, selector: &str) -> Result<i32> {
            unsafe { send_i32(receiver, selector) }
        }

        unsafe fn array_count(&mut self, array: Id) -> Result<usize> {
            unsafe { Ok(send_u64(array, "count")? as usize) }
        }

        unsafe fn array_object(&mut self, array: Id, index: usize) -> Result<Id> {
            unsafe { send_id_usize(array, "objectAtIndex:", index) }
        }

        unsafe fn u64_array(&mut self, array: Id) -> Result<Vec<u64>> {
            let count = unsafe { self.array_count(array)? };
            let mut values = Vec::with_capacity(count);
            for index in 0..count {
                let value = unsafe { self.array_object(array, index)? };
                values.push(unsafe { send_u64(value, "unsignedLongLongValue")? });
            }
            Ok(values)
        }

        unsafe fn request_cost_timeline(&mut self, mio: Id) -> Result<Option<DecodedTimeline>> {
            let slot = Arc::new(Mutex::new(None::<usize>));
            let callback_slot = Arc::clone(&slot);
            let block = RcBlock::new(move |timeline: Id| {
                if !timeline.is_null()
                    && let Ok(mut slot) = callback_slot.lock()
                {
                    *slot = Some(timeline as usize);
                }
            });
            let block_ptr = RcBlock::as_ptr(&block).cast::<c_void>();
            unsafe {
                send_id_id_allow_nil(mio, "requestCostTimeline:", block_ptr.cast())?;
            }

            let deadline = Instant::now() + Duration::from_secs(20);
            let mut timeline = None;
            while Instant::now() < deadline {
                timeline = *slot
                    .lock()
                    .map_err(|_| Error::InvalidInput("cost timeline lock poisoned".to_owned()))?;
                if timeline.is_some() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }

            let Some(timeline) = timeline.map(|value| value as Id) else {
                return Ok(None);
            };

            let (gpu_cost, gpu_cost_instruction_count) = unsafe { decode_gpu_cost(timeline) };
            Ok(Some(DecodedTimeline {
                object: timeline,
                summary: XcodeMioCostTimeline {
                    draw_count: unsafe { send_u64(timeline, "drawCount")? as usize },
                    pipeline_state_count: unsafe {
                        send_u64(timeline, "pipelineStateCount")? as usize
                    },
                    cost_record_count: unsafe { send_u64(timeline, "costCount")? as usize },
                    gpu_time_ns: unsafe { send_u64(timeline, "gpuTime")? },
                    global_gpu_time_ns: unsafe { send_u64(timeline, "globalGPUTime")? },
                    timeline_duration_ns: unsafe { send_u64(timeline, "timelineDuration")? },
                    total_clique_cost: unsafe { send_u64(timeline, "totalCliqueCost")? },
                    gpu_cost,
                    gpu_cost_instruction_count,
                },
            }))
        }

        unsafe fn timeline_candidates(
            &mut self,
            mio: Id,
            requested_timeline: Option<Id>,
        ) -> Vec<XcodeMioTimelineCandidate> {
            let mut seen = BTreeSet::<usize>::new();
            let mut candidates = Vec::new();

            if let Some(timeline) = requested_timeline
                && seen.insert(timeline as usize)
                && let Some(candidate) =
                    unsafe { self.describe_timeline("requestCostTimeline", timeline) }
            {
                candidates.push(candidate);
            }

            for (source, selector) in [
                ("mio.costTimeline", "costTimeline"),
                ("mio.overlappingTimeline", "overlappingTimeline"),
                ("mio.nonOverlappingTimeline", "nonOverlappingTimeline"),
            ] {
                if !unsafe { responds_to_selector(mio, selector) } {
                    continue;
                }
                let Ok(timeline) = (unsafe { send_id_allow_nil(mio, selector) }) else {
                    continue;
                };
                if timeline.is_null() || !seen.insert(timeline as usize) {
                    continue;
                }
                if let Some(candidate) = unsafe { self.describe_timeline(source, timeline) } {
                    candidates.push(candidate);
                }
            }

            candidates
        }

        unsafe fn describe_timeline(
            &mut self,
            source: &'static str,
            timeline: Id,
        ) -> Option<XcodeMioTimelineCandidate> {
            if timeline.is_null() {
                return None;
            }
            let cost_records = unsafe { self.decode_timeline_cost_records(timeline) };
            let nonzero_cost_records = cost_records
                .iter()
                .filter(|record| {
                    record.total_cost.is_finite() && record.total_cost > 0.0
                        || record.instruction_count > 0
                })
                .count();
            let decoded_total_cost = cost_records
                .iter()
                .map(|record| record.total_cost)
                .filter(|value| value.is_finite())
                .sum::<f64>();
            let decoded_instruction_count = cost_records
                .iter()
                .map(|record| record.instruction_count)
                .sum::<u64>();
            let (gpu_cost, gpu_cost_instruction_count) = unsafe { decode_gpu_cost(timeline) };

            Some(XcodeMioTimelineCandidate {
                source,
                draw_count: unsafe { send_u64_if_supported(timeline, "drawCount") as usize },
                pipeline_state_count: unsafe {
                    send_u64_if_supported(timeline, "pipelineStateCount") as usize
                },
                cost_record_count: unsafe { send_u64_if_supported(timeline, "costCount") as usize },
                gpu_time_ns: unsafe { send_u64_if_supported(timeline, "gpuTime") },
                global_gpu_time_ns: unsafe { send_u64_if_supported(timeline, "globalGPUTime") },
                timeline_duration_ns: unsafe {
                    send_u64_if_supported(timeline, "timelineDuration")
                },
                total_clique_cost: unsafe { send_u64_if_supported(timeline, "totalCliqueCost") },
                gpu_cost,
                gpu_cost_instruction_count,
                timeline_binary_count: unsafe { self.timeline_binary_count(timeline) },
                shader_binary_info_count: unsafe {
                    send_u64_if_supported(timeline, "shaderBinaryInfoCount") as usize
                },
                nonzero_cost_records,
                decoded_total_cost,
                decoded_instruction_count,
            })
        }

        unsafe fn decode_pipeline_private_costs(
            &mut self,
            mio: Id,
            timeline: Id,
            cost_records: &[XcodeMioDecodedCostRecord],
            pipelines: &mut [XcodeMioPipeline],
            timeline_pipeline_state_ids: &[u64],
        ) -> Result<()> {
            unsafe {
                self.decode_pipeline_shader_stats(timeline, pipelines)?;
                self.decode_pipeline_scope_costs(cost_records, pipelines);
                self.decode_pipeline_shader_tracks(
                    timeline,
                    pipelines,
                    "timeline_pipeline_state_program_type",
                )?;
                if !pipelines
                    .iter()
                    .any(|pipeline| !pipeline.shader_tracks.is_empty())
                {
                    self.decode_pipeline_shader_tracks(
                        mio,
                        pipelines,
                        "mio_pipeline_state_program_type",
                    )?;
                }
                self.decode_pipeline_shader_binaries(timeline, pipelines)?;
                self.decode_pipeline_execution_history(
                    timeline,
                    pipelines,
                    timeline_pipeline_state_ids,
                )?;
                self.decode_pipeline_execution_history(
                    mio,
                    pipelines,
                    timeline_pipeline_state_ids,
                )?;
            }
            Ok(())
        }

        unsafe fn decode_shader_profiler_costs(
            &mut self,
            source: &'static str,
            shader_result: Id,
            gpu_commands: &[XcodeMioGpuCommand],
            pipelines: &mut [XcodeMioPipeline],
            gpu_command_costs: &mut Vec<XcodeMioGpuCommandShaderProfilerCost>,
        ) -> Result<()> {
            unsafe {
                let binaries = self.send_id_allow_nil(shader_result, "shaderBinaries")?;
                if binaries.is_null() {
                    return Ok(());
                }
                let keys = self.send_id_allow_nil(binaries, "allKeys")?;
                let values = self.send_id_allow_nil(binaries, "allValues")?;
                if keys.is_null() || values.is_null() {
                    return Ok(());
                }
                let count = self.array_count(keys)?.min(self.array_count(values)?);
                for index in 0..count {
                    let key_object = self.array_object(keys, index)?;
                    let binary = self.array_object(values, index)?;
                    if !responds_to_selector(binary, "costForDrawAtIndex:")
                        || !responds_to_selector(binary, "totalCost")
                    {
                        continue;
                    }
                    let key = if responds_to_selector(key_object, "UTF8String") {
                        nsstring_to_string(key_object)
                    } else {
                        None
                    }
                    .unwrap_or_else(|| format!("shader_binary_{index}"));
                    let full_path = if responds_to_selector(binary, "fullPath") {
                        self.send_id_allow_nil(binary, "fullPath")
                            .ok()
                            .filter(|value| !value.is_null())
                            .filter(|value| responds_to_selector(*value, "UTF8String"))
                            .and_then(|value| nsstring_to_string(value))
                    } else {
                        None
                    };
                    let type_name = if responds_to_selector(binary, "typeName") {
                        self.send_id_allow_nil(binary, "typeName")
                            .ok()
                            .filter(|value| !value.is_null())
                            .filter(|value| responds_to_selector(*value, "UTF8String"))
                            .and_then(|value| nsstring_to_string(value))
                    } else {
                        None
                    };
                    let shader_type = if responds_to_selector(binary, "type") {
                        send_u32(binary, "type").unwrap_or(0)
                    } else {
                        0
                    };
                    let addr_start = if responds_to_selector(binary, "addrStart") {
                        send_u32(binary, "addrStart").unwrap_or(0)
                    } else {
                        0
                    };
                    let addr_end = if responds_to_selector(binary, "addrEnd") {
                        send_u32(binary, "addrEnd").unwrap_or(0)
                    } else {
                        0
                    };
                    let total_binary_cost = send_f64(binary, "totalCost").unwrap_or(0.0);
                    let total_binary_samples = if responds_to_selector(binary, "numSamples") {
                        send_u64(binary, "numSamples").unwrap_or(0)
                    } else {
                        0
                    };
                    if total_binary_cost <= 0.0 && total_binary_samples == 0 {
                        continue;
                    }
                    let has_cost_percentage =
                        responds_to_selector(binary, "costPercentageForDrawAtIndex:");
                    let mut by_pipeline =
                        BTreeMap::<usize, XcodeMioPipelineShaderProfilerCost>::new();
                    for command in gpu_commands {
                        let mut candidates = Vec::with_capacity(2);
                        if let Ok(command_index) = u32::try_from(command.index) {
                            candidates.push(command_index);
                        }
                        if let Ok(function_index) = u32::try_from(command.function_index)
                            && !candidates.contains(&function_index)
                        {
                            candidates.push(function_index);
                        }
                        let mut hit = None;
                        for draw_index in candidates {
                            let cost = send_f64_u32(binary, "costForDrawAtIndex:", draw_index)
                                .unwrap_or(0.0);
                            if !cost.is_finite() || cost <= 0.0 {
                                continue;
                            }
                            let percent = if has_cost_percentage {
                                send_f64_u32(binary, "costPercentageForDrawAtIndex:", draw_index)
                                    .ok()
                                    .filter(|value| value.is_finite() && *value > 0.0)
                            } else {
                                None
                            };
                            hit = Some((draw_index as usize, cost, percent));
                            break;
                        }
                        let Some((draw_index, cost, percent)) = hit else {
                            continue;
                        };
                        gpu_command_costs.push(XcodeMioGpuCommandShaderProfilerCost {
                            source,
                            draw_index,
                            command_index: command.index,
                            function_index: command.function_index,
                            sub_command_index: command.sub_command_index,
                            encoder_index: command.encoder_index,
                            pipeline_index: command.pipeline_index,
                            function_name: command.function_name.clone(),
                            binary_key: key.clone(),
                            full_path: full_path.clone(),
                            type_name: type_name.clone(),
                            shader_type,
                            addr_start,
                            addr_end,
                            total_binary_cost,
                            total_binary_samples,
                            cost,
                            cost_percent: percent,
                        });
                        by_pipeline
                            .entry(command.pipeline_index)
                            .and_modify(|entry| {
                                entry.pipeline_cost += cost;
                                if let Some(percent) = percent {
                                    entry.pipeline_cost_percent_sum += percent;
                                }
                                entry.nonzero_draw_count += 1;
                                entry.first_draw_index = entry.first_draw_index.min(draw_index);
                                entry.last_draw_index = entry.last_draw_index.max(draw_index);
                            })
                            .or_insert_with(|| XcodeMioPipelineShaderProfilerCost {
                                binary_key: key.clone(),
                                full_path: full_path.clone(),
                                type_name: type_name.clone(),
                                shader_type,
                                addr_start,
                                addr_end,
                                total_binary_cost,
                                total_binary_samples,
                                pipeline_cost: cost,
                                pipeline_cost_percent_sum: percent.unwrap_or(0.0),
                                nonzero_draw_count: 1,
                                first_draw_index: draw_index,
                                last_draw_index: draw_index,
                            });
                    }
                    for (pipeline_index, cost) in by_pipeline {
                        if let Some(pipeline) = pipelines.get_mut(pipeline_index) {
                            pipeline.shader_profiler_costs.push(cost);
                        }
                    }
                }
                for pipeline in pipelines {
                    pipeline.shader_profiler_costs.sort_by(|left, right| {
                        right
                            .pipeline_cost
                            .partial_cmp(&left.pipeline_cost)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| left.binary_key.cmp(&right.binary_key))
                    });
                }
            }
            Ok(())
        }

        unsafe fn merge_shader_profiler_pipeline_state_data(
            &mut self,
            shader_result: Id,
            pipelines: &mut [XcodeMioPipeline],
        ) -> Result<()> {
            unsafe {
                let pipeline_states = self.send_id_allow_nil(shader_result, "pipelineStates")?;
                if pipeline_states.is_null() {
                    return Ok(());
                }
                let count = self.array_count(pipeline_states)?;
                for index in 0..count {
                    let pipeline_state = self.array_object(pipeline_states, index)?;
                    let object_id = if responds_to_selector(pipeline_state, "objectId") {
                        send_u64(pipeline_state, "objectId").unwrap_or(0)
                    } else {
                        0
                    };
                    let pipeline_index = if responds_to_selector(pipeline_state, "index") {
                        send_u32(pipeline_state, "index").unwrap_or(index as u32) as usize
                    } else {
                        index
                    };
                    let timings = decode_pipeline_profiler_timings(pipeline_state);
                    if timings.is_empty() {
                        continue;
                    }
                    let target_index = pipelines
                        .iter()
                        .position(|pipeline| pipeline.object_id == object_id && object_id != 0)
                        .or_else(|| (pipeline_index < pipelines.len()).then_some(pipeline_index));
                    let Some(target_index) = target_index else {
                        continue;
                    };
                    let target = &mut pipelines[target_index];
                    target.profiler_timings.extend(timings);
                    dedup_profiler_timings(&mut target.profiler_timings);
                }
            }
            Ok(())
        }

        unsafe fn decode_shader_profiler_timing_info(
            &mut self,
            shader_result: Id,
            pipelines: &mut [XcodeMioPipeline],
        ) -> Result<()> {
            unsafe {
                let result_timing = decode_timing_info(shader_result).unwrap_or_default();
                let denominator = result_timing
                    .compute_time
                    .max(result_timing.time)
                    .max(result_timing.vertex_time)
                    .max(result_timing.fragment_time);
                let pipeline_states = self.send_id_allow_nil(shader_result, "pipelineStates")?;
                if pipeline_states.is_null() {
                    return Ok(());
                }
                let count = self.array_count(pipeline_states)?;
                for index in 0..count {
                    let pipeline_state = self.array_object(pipeline_states, index)?;
                    let Some(timing) = decode_timing_info(pipeline_state) else {
                        continue;
                    };
                    let pipeline_time = timing
                        .compute_time
                        .max(timing.time)
                        .max(timing.vertex_time)
                        .max(timing.fragment_time);
                    if pipeline_time == 0 {
                        continue;
                    }
                    let object_id = if responds_to_selector(pipeline_state, "objectId") {
                        send_u64(pipeline_state, "objectId").unwrap_or(0)
                    } else {
                        0
                    };
                    let pipeline_index = if responds_to_selector(pipeline_state, "index") {
                        send_u32(pipeline_state, "index").unwrap_or(index as u32) as usize
                    } else {
                        index
                    };
                    let target_index = pipelines
                        .iter()
                        .position(|pipeline| pipeline.object_id == object_id && object_id != 0)
                        .or_else(|| (pipeline_index < pipelines.len()).then_some(pipeline_index));
                    let Some(target_index) = target_index else {
                        continue;
                    };
                    let percentage = if denominator > 0 {
                        100.0 * pipeline_time as f64 / denominator as f64
                    } else {
                        0.0
                    };
                    let target = &mut pipelines[target_index];
                    target
                        .profiler_timings
                        .push(XcodeMioPipelineProfilerTiming {
                            source: "timingInfo",
                            cycle_average: 0.0,
                            cycle_min: 0.0,
                            cycle_max: 0.0,
                            time_average: pipeline_time as f64 / 1_000_000.0,
                            time_min: 0.0,
                            time_max: 0.0,
                            percentage_average: percentage,
                            percentage_min: 0.0,
                            percentage_max: 0.0,
                            surplus_cycles: 0.0,
                        });
                    dedup_profiler_timings(&mut target.profiler_timings);
                }
            }
            Ok(())
        }

        unsafe fn decode_pipeline_counters(
            &mut self,
            mio: Id,
            pipelines: &mut [XcodeMioPipeline],
        ) -> Result<()> {
            unsafe {
                let counters = self.send_id_allow_nil(mio, "nonOverlappingCounters")?;
                if counters.is_null() {
                    return Ok(());
                }
                let names = self.send_id_allow_nil(counters, "pipelineStateCounterNames")?;
                if names.is_null() {
                    return Ok(());
                }
                let name_count = self.array_count(names)?;
                let value_count = send_u64(counters, "numPipelineStateCounters")? as usize;
                let count = name_count.min(value_count);
                let mut counter_names = Vec::with_capacity(count);
                for index in 0..count {
                    let name = self.array_object(names, index)?;
                    counter_names.push(
                        nsstring_to_string(name)
                            .unwrap_or_else(|| format!("pipeline_counter_{index}")),
                    );
                }
                for pipeline in pipelines {
                    let values = send_ptr_u64_u32(
                        counters,
                        "counterValuesForPipelineStateId:encoderFunctionIndex:",
                        pipeline.object_id,
                        pipeline.function_index as u32,
                    )?;
                    if values.is_null() {
                        continue;
                    }
                    let values = values.cast::<f64>();
                    for (index, name) in counter_names.iter().enumerate() {
                        let value = *values.add(index);
                        if !value.is_finite() || value == 0.0 {
                            continue;
                        }
                        pipeline.pipeline_counters.push(XcodeMioPipelineCounter {
                            name: name.clone(),
                            value,
                        });
                    }
                    pipeline.pipeline_counters.sort_by(|left, right| {
                        right
                            .value
                            .abs()
                            .partial_cmp(&left.value.abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
            Ok(())
        }

        unsafe fn decode_gpu_command_counters(
            &mut self,
            mio: Id,
            gpu_commands: &[XcodeMioGpuCommand],
        ) -> Result<Vec<XcodeMioGpuCommandCounterRow>> {
            unsafe {
                let counters = self.send_id_allow_nil(mio, "nonOverlappingCounters")?;
                if counters.is_null() {
                    return Ok(Vec::new());
                }
                let names = self.send_id_allow_nil(counters, "drawCounterNames")?;
                if names.is_null() {
                    return Ok(Vec::new());
                }
                let name_count = self.array_count(names)?;
                let value_count = send_u64(counters, "numDrawCounters")? as usize;
                let count = name_count.min(value_count);
                let mut counter_names = Vec::with_capacity(count);
                for index in 0..count {
                    let name = self.array_object(names, index)?;
                    counter_names.push(
                        nsstring_to_string(name).unwrap_or_else(|| format!("draw_counter_{index}")),
                    );
                }

                let mut rows = Vec::new();
                for command in gpu_commands {
                    let values = send_ptr_u64_i32(
                        counters,
                        "counterValuesForGPUCommandAtFunctionIndex:subCommandIndex:",
                        command.function_index,
                        command.sub_command_index,
                    )?;
                    if values.is_null() {
                        continue;
                    }
                    let values = values.cast::<f64>();
                    let mut row_counters = Vec::new();
                    for (index, name) in counter_names.iter().enumerate() {
                        let value = *values.add(index);
                        if !value.is_finite() || value == 0.0 {
                            continue;
                        }
                        row_counters.push(XcodeMioPipelineCounter {
                            name: name.clone(),
                            value,
                        });
                    }
                    if row_counters.is_empty() {
                        continue;
                    }
                    rows.push(XcodeMioGpuCommandCounterRow {
                        command_index: command.index,
                        function_index: command.function_index,
                        sub_command_index: command.sub_command_index,
                        encoder_index: command.encoder_index,
                        pipeline_index: command.pipeline_index,
                        function_name: command.function_name.clone(),
                        counters: row_counters,
                    });
                }
                rows.sort_by(|left, right| {
                    left.command_index
                        .cmp(&right.command_index)
                        .then_with(|| left.sub_command_index.cmp(&right.sub_command_index))
                });
                Ok(rows)
            }
        }

        unsafe fn decode_gpu_command_function_times(
            &mut self,
            trace_data: Id,
            timeline: Option<Id>,
            pipelines: &[XcodeMioPipeline],
            encoders: &[XcodeMioEncoder],
            gpu_commands: &[XcodeMioGpuCommand],
            timeline_pipeline_state_ids: &[u64],
        ) -> (
            Vec<XcodeMioGpuCommandFunctionTime>,
            Vec<XcodeMioFunctionTimeProbe>,
        ) {
            let mut rows = Vec::new();
            let mut probes = Vec::new();
            unsafe {
                for source in function_time_trace_data_sources(trace_data, timeline) {
                    if responds_to_selector(source.object, "durationForDraw:dataMaster:") {
                        push_gpu_command_function_time_source(
                            &mut rows,
                            source.object,
                            source.function_index_source,
                            source.command_index_source,
                            gpu_commands,
                        );
                    }
                    if responds_to_selector(
                        source.object,
                        "enumerateDrawsForPipelineState:enumerator:",
                    ) || responds_to_selector(source.object, "numDrawsForPipelineState:")
                    {
                        push_enumerated_gpu_command_function_time_source(
                            &mut rows,
                            &mut probes,
                            source,
                            pipelines,
                            gpu_commands,
                            timeline_pipeline_state_ids,
                        );
                    }
                    if responds_to_selector(source.object, "enumerateDrawsForEncoder:enumerator:")
                        || responds_to_selector(source.object, "numDrawsForEncoder:")
                        || responds_to_selector(source.object, "kickDurationForEncoder:dataMaster:")
                        || responds_to_selector(source.object, "kickDurationForEncoder:")
                    {
                        push_enumerated_encoder_function_time_source(
                            &mut rows,
                            &mut probes,
                            source,
                            encoders,
                            gpu_commands,
                        );
                    }
                }
            }
            rows.sort_by(|left, right| {
                left.source
                    .cmp(right.source)
                    .then_with(|| left.command_index.cmp(&right.command_index))
                    .then_with(|| left.draw_index.cmp(&right.draw_index))
                    .then_with(|| left.data_master.cmp(&right.data_master))
            });
            probes.sort_by(|left, right| {
                left.source
                    .cmp(right.source)
                    .then_with(|| left.target_kind.cmp(right.target_kind))
                    .then_with(|| left.target_id_kind.cmp(right.target_id_kind))
                    .then_with(|| left.target_id.cmp(&right.target_id))
            });
            (rows, probes)
        }

        unsafe fn decode_shader_profiler_gpu_command_times(
            &mut self,
            source: &'static str,
            shader_result: Id,
            gpu_commands: &[XcodeMioGpuCommand],
        ) -> (Vec<XcodeMioGpuCommandFunctionTime>, String) {
            let mut rows = Vec::new();
            let mut attempts = 0_usize;
            let mut non_nil = 0_usize;
            let mut timing_info = 0_usize;
            let mut nonzero = 0_usize;
            let mut first_object_class = None;
            unsafe {
                if shader_result.is_null()
                    || !responds_to_selector(
                        shader_result,
                        "gpuCommandForFunctionIndex:subCommandIndex:",
                    )
                {
                    return (
                        rows,
                        format!("{source}: gpuCommand scan skipped; selector unavailable"),
                    );
                }
                for command in gpu_commands {
                    let mut function_indices = vec![
                        command.function_index,
                        command.index as u64,
                        (command.index * 2) as u64,
                        (command.index * 2 + 1) as u64,
                    ];
                    if let Some(display_index) = command.function_index.checked_sub(1) {
                        function_indices.push(display_index);
                    }
                    if let Some(previous_index) = command.function_index.checked_sub(2) {
                        function_indices.push(previous_index);
                    }
                    function_indices.push(command.function_index.saturating_add(1));
                    function_indices.push(command.index as u64);
                    function_indices.sort_unstable();
                    function_indices.dedup();

                    let mut sub_command_indices = vec![command.sub_command_index, -1, 0];
                    sub_command_indices.sort_unstable();
                    sub_command_indices.dedup();

                    for function_index in function_indices {
                        let Ok(draw_index) = u32::try_from(function_index) else {
                            continue;
                        };
                        for sub_command_index in &sub_command_indices {
                            attempts += 1;
                            let gpu_command = send_id_u32_i32_allow_nil(
                                shader_result,
                                "gpuCommandForFunctionIndex:subCommandIndex:",
                                draw_index,
                                *sub_command_index,
                            )
                            .unwrap_or(std::ptr::null_mut());
                            if gpu_command.is_null() {
                                continue;
                            }
                            non_nil += 1;
                            if first_object_class.is_none() {
                                first_object_class = object_class_name(gpu_command);
                            }
                            let Some(timing) = decode_timing_info(gpu_command) else {
                                continue;
                            };
                            timing_info += 1;
                            let (duration_ns, data_master) =
                                shader_profiler_command_duration(timing);
                            if duration_ns == 0 {
                                continue;
                            }
                            nonzero += 1;
                            rows.push(XcodeMioGpuCommandFunctionTime {
                                source,
                                command_index: command.index,
                                function_index,
                                sub_command_index: *sub_command_index,
                                encoder_index: command.encoder_index,
                                pipeline_index: command.pipeline_index,
                                function_name: command.function_name.clone(),
                                draw_index,
                                data_master,
                                duration_ns,
                            });
                        }
                    }
                }
            }
            let class_name = first_object_class.unwrap_or_else(|| "-".to_owned());
            (
                rows,
                format!(
                    "{source}: gpuCommand scan attempts={attempts} non_nil={non_nil} timing_info={timing_info} nonzero={nonzero} first_class={class_name}"
                ),
            )
        }

        unsafe fn decode_gpu_command_direct_costs(
            &mut self,
            trace_data: Id,
            timeline: Option<Id>,
            gpu_commands: &[XcodeMioGpuCommand],
        ) -> Vec<XcodeMioGpuCommandDirectCost> {
            let mut rows = Vec::new();
            unsafe {
                for source in function_time_trace_data_sources(trace_data, timeline) {
                    if !responds_to_selector(source.object, "costForDrawAtIndex:") {
                        continue;
                    }
                    let has_cost_percent =
                        responds_to_selector(source.object, "costPercentageForDrawAtIndex:");
                    for command in gpu_commands {
                        let Ok(draw_index) = u32::try_from(command.index) else {
                            continue;
                        };
                        let Ok(cost) =
                            send_f64_u32(source.object, "costForDrawAtIndex:", draw_index)
                        else {
                            continue;
                        };
                        if !cost.is_finite() || cost <= 0.0 {
                            continue;
                        }
                        let cost_percent = if has_cost_percent {
                            send_f64_u32(source.object, "costPercentageForDrawAtIndex:", draw_index)
                                .ok()
                                .filter(|value| value.is_finite() && *value > 0.0)
                        } else {
                            None
                        };
                        rows.push(XcodeMioGpuCommandDirectCost {
                            source: source.source,
                            command_index: command.index,
                            function_index: command.function_index,
                            sub_command_index: command.sub_command_index,
                            encoder_index: command.encoder_index,
                            pipeline_index: command.pipeline_index,
                            function_name: command.function_name.clone(),
                            cost,
                            cost_percent,
                        });
                    }
                }
            }
            rows.sort_by(|left, right| {
                left.source
                    .cmp(right.source)
                    .then_with(|| left.command_index.cmp(&right.command_index))
            });
            rows
        }

        unsafe fn decode_top_draw_tracks(
            &mut self,
            trace_data: Id,
            timeline: Option<Id>,
            gpu_commands: &[XcodeMioGpuCommand],
        ) -> Vec<XcodeMioTopDrawTrack> {
            let mut rows = Vec::new();
            unsafe {
                let Ok(helper_class) = lookup_class("GTMioTraceDataHelper") else {
                    return rows;
                };
                let mut seen_sources = BTreeSet::new();
                for source in function_time_trace_data_sources(trace_data, timeline) {
                    if source.object.is_null() || !seen_sources.insert(source.object as usize) {
                        continue;
                    }
                    let Ok(helper) = send_id(helper_class, "alloc") else {
                        continue;
                    };
                    let Ok(helper) = send_id_id(helper, "initWithTraceData:", source.object) else {
                        continue;
                    };
                    let _ = send_void_i8(helper, "setShowDriverInternalShaders:", 1);
                    let _ = send_void_i8(helper, "setShowDriverIntersectionShaders:", 1);
                    let _ = send_void_i8(helper, "setShowESLShaders:", 1);
                    let Ok(tracks) = send_id_allow_nil(helper, "generateTopDrawTracks") else {
                        continue;
                    };
                    if tracks.is_null() {
                        continue;
                    }
                    let Ok(track_count) = self.array_count(tracks) else {
                        continue;
                    };
                    for track_index in 0..track_count {
                        let Ok(track) = self.array_object(tracks, track_index) else {
                            continue;
                        };
                        let track_id = send_i32(track, "trackId").unwrap_or(0);
                        let first_index = send_u64(track, "firstIndex").unwrap_or(0);
                        let start_timestamp_ns = send_u64(track, "startTimestamp").unwrap_or(0);
                        let end_timestamp_ns = send_u64(track, "endTimestamp").unwrap_or(0);
                        let duration_ns = send_u64(track, "duration").unwrap_or(0);
                        let trace_count = send_u64(track, "traceCount").unwrap_or(0);
                        let traces = send_ptr(track, "traces")
                            .ok()
                            .filter(|ptr| !ptr.is_null())
                            .map(|ptr| ptr.cast::<RawGtmioDrawTrace>());
                        if trace_count == 0 || traces.is_none() {
                            let matched =
                                command_for_top_draw_track(first_index, None, gpu_commands);
                            rows.push(top_draw_track_row(
                                source.source,
                                track_index,
                                0,
                                track_id,
                                first_index,
                                start_timestamp_ns,
                                end_timestamp_ns,
                                duration_ns,
                                trace_count,
                                RawGtmioDrawTrace::default(),
                                matched,
                            ));
                            continue;
                        }
                        let traces = traces.expect("checked above");
                        for trace_index in 0..trace_count as usize {
                            let trace = *traces.add(trace_index);
                            let matched =
                                command_for_top_draw_track(first_index, Some(trace), gpu_commands);
                            rows.push(top_draw_track_row(
                                source.source,
                                track_index,
                                trace_index,
                                track_id,
                                first_index,
                                start_timestamp_ns,
                                end_timestamp_ns,
                                duration_ns,
                                trace_count,
                                trace,
                                matched,
                            ));
                        }
                    }
                }
            }
            rows.sort_by(|left, right| {
                left.source
                    .cmp(right.source)
                    .then_with(|| left.track_index.cmp(&right.track_index))
                    .then_with(|| left.trace_index.cmp(&right.trace_index))
            });
            rows
        }

        unsafe fn decode_draw_array_probes(
            &mut self,
            trace_data: Id,
            timeline: Option<Id>,
            pipelines: &[XcodeMioPipeline],
            gpu_commands: &[XcodeMioGpuCommand],
        ) -> Vec<XcodeMioDrawArrayProbe> {
            let command_indices = draw_array_probe_command_indices(pipelines, gpu_commands);
            let mut probes = Vec::new();
            let mut seen = BTreeSet::new();
            unsafe {
                for source in function_time_trace_data_sources(trace_data, timeline) {
                    let Ok(draw_count) = send_u64(source.object, "drawCount") else {
                        continue;
                    };
                    if draw_count == 0 {
                        continue;
                    }
                    let draw_traces = send_ptr(source.object, "drawTraces")
                        .ok()
                        .filter(|ptr| !ptr.is_null())
                        .map(|ptr| ptr.cast::<RawGtmioDrawTrace>());
                    let draws = send_ptr(source.object, "draws")
                        .ok()
                        .filter(|ptr| !ptr.is_null())
                        .map(|ptr| ptr.cast::<u8>());
                    if draw_traces.is_none() && draws.is_none() {
                        continue;
                    }

                    for command_index in &command_indices {
                        let Some(command) = gpu_commands.get(*command_index) else {
                            continue;
                        };
                        let mut array_indices = Vec::with_capacity(2);
                        array_indices.push(("command_index", command.index));
                        if let Ok(function_index) = usize::try_from(command.function_index)
                            && function_index != command.index
                            && function_index < draw_count as usize
                        {
                            array_indices.push(("function_index", function_index));
                        }
                        for (array_index_kind, array_index) in array_indices {
                            if array_index >= draw_count as usize
                                || !seen.insert((source.source, array_index_kind, array_index))
                            {
                                continue;
                            }
                            let trace = draw_traces
                                .map(|draw_traces| *draw_traces.add(array_index))
                                .unwrap_or_default();
                            let metadata =
                                draws.map(|draws| decode_draw_metadata_record(draws, array_index));
                            let metadata = metadata.unwrap_or(XcodeMioDrawMetadataRecord {
                                index: array_index,
                                raw0: 0,
                                raw1: 0,
                                raw2: 0,
                                raw3: 0,
                                raw4: 0,
                                raw5: 0,
                                raw6: 0,
                                raw7: 0,
                                raw8: 0,
                                raw9: 0,
                            });
                            probes.push(XcodeMioDrawArrayProbe {
                                source: source.source,
                                array_index_kind,
                                array_index,
                                command_index: command.index,
                                function_index: command.function_index,
                                sub_command_index: command.sub_command_index,
                                encoder_index: command.encoder_index,
                                pipeline_index: command.pipeline_index,
                                function_name: command.function_name.clone(),
                                trace_raw0: trace.raw0,
                                trace_raw1: trace.raw1,
                                trace_raw2: trace.raw2,
                                trace_raw3: trace.raw3,
                                trace_duration_ns: trace.raw1.saturating_sub(trace.raw0),
                                metadata_raw0: metadata.raw0,
                                metadata_raw1: metadata.raw1,
                                metadata_raw2: metadata.raw2,
                                metadata_raw3: metadata.raw3,
                                metadata_raw4: metadata.raw4,
                                metadata_raw5: metadata.raw5,
                                metadata_raw6: metadata.raw6,
                                metadata_raw7: metadata.raw7,
                                metadata_raw8: metadata.raw8,
                                metadata_raw9: metadata.raw9,
                            });
                        }
                    }
                }
            }
            probes.sort_by(|left, right| {
                left.command_index
                    .cmp(&right.command_index)
                    .then_with(|| left.source.cmp(right.source))
                    .then_with(|| left.array_index_kind.cmp(right.array_index_kind))
                    .then_with(|| left.array_index.cmp(&right.array_index))
            });
            probes
        }

        unsafe fn decode_usc_clique_summaries(
            &mut self,
            trace_data: Id,
            timeline: Option<Id>,
        ) -> Vec<XcodeMioUSCCliqueSummary> {
            let mut summaries = Vec::new();
            unsafe {
                for source in function_time_trace_data_sources(trace_data, timeline) {
                    if !responds_to_selector(source.object, "uscs") {
                        summaries.push(XcodeMioUSCCliqueSummary {
                            source: source.source,
                            usc_index: None,
                            usc_count: 0,
                            clique_count: 0,
                            has_cliques: false,
                            has_enumerate_kick_cliques_by_function: false,
                            sample_raw0: 0,
                            sample_raw1: 0,
                            sample_duration_ns: 0,
                            sample_u32_fields: Vec::new(),
                            sample_u16_fields: Vec::new(),
                        });
                        continue;
                    }
                    let Ok(uscs) = send_id_allow_nil(source.object, "uscs") else {
                        continue;
                    };
                    if uscs.is_null() || !responds_to_selector(uscs, "count") {
                        summaries.push(XcodeMioUSCCliqueSummary {
                            source: source.source,
                            usc_index: None,
                            usc_count: 0,
                            clique_count: 0,
                            has_cliques: false,
                            has_enumerate_kick_cliques_by_function: false,
                            sample_raw0: 0,
                            sample_raw1: 0,
                            sample_duration_ns: 0,
                            sample_u32_fields: Vec::new(),
                            sample_u16_fields: Vec::new(),
                        });
                        continue;
                    }
                    let usc_count = send_u64(uscs, "count").unwrap_or(0);
                    if usc_count == 0 {
                        summaries.push(XcodeMioUSCCliqueSummary {
                            source: source.source,
                            usc_index: None,
                            usc_count,
                            clique_count: 0,
                            has_cliques: false,
                            has_enumerate_kick_cliques_by_function: false,
                            sample_raw0: 0,
                            sample_raw1: 0,
                            sample_duration_ns: 0,
                            sample_u32_fields: Vec::new(),
                            sample_u16_fields: Vec::new(),
                        });
                        continue;
                    }
                    for usc_index in 0..(usc_count as usize).min(256) {
                        let Ok(usc) = send_id_usize(uscs, "objectAtIndex:", usc_index) else {
                            continue;
                        };
                        let has_cliques = !usc.is_null()
                            && responds_to_selector(usc, "cliquesCount")
                            && responds_to_selector(usc, "cliques");
                        let has_enumerate = !usc.is_null()
                            && responds_to_selector(
                                usc,
                                "enumerateKickCliquesAtFunctionIndex:dataMaster:enumerator:",
                            );
                        let mut sample = RawGtmioUSCCliqueMetadata::default();
                        let clique_count = if has_cliques {
                            send_u64(usc, "cliquesCount").unwrap_or(0)
                        } else {
                            0
                        };
                        if clique_count > 0
                            && clique_count <= 5_000_000
                            && let Ok(cliques) = send_ptr(usc, "cliques")
                            && !cliques.is_null()
                        {
                            sample = *cliques.cast::<RawGtmioUSCCliqueMetadata>();
                        }
                        summaries.push(XcodeMioUSCCliqueSummary {
                            source: source.source,
                            usc_index: Some(usc_index),
                            usc_count,
                            clique_count,
                            has_cliques,
                            has_enumerate_kick_cliques_by_function: has_enumerate,
                            sample_raw0: sample.raw0,
                            sample_raw1: sample.raw1,
                            sample_duration_ns: sample.duration_ns(),
                            sample_u32_fields: sample.u32_fields().to_vec(),
                            sample_u16_fields: sample.u16_fields().to_vec(),
                        });
                    }
                }
            }
            summaries
        }

        unsafe fn decode_usc_clique_probes(
            &mut self,
            trace_data: Id,
            timeline: Option<Id>,
            pipelines: &[XcodeMioPipeline],
            gpu_commands: &[XcodeMioGpuCommand],
        ) -> Vec<XcodeMioUSCCliqueProbe> {
            #[derive(Clone, Copy)]
            struct CliqueAggregate {
                clique_count: usize,
                first_clique_index: usize,
                last_clique_index: usize,
                duration_sum_ns: u64,
                min_duration_ns: u64,
                max_duration_ns: u64,
                min_timestamp_ns: u64,
                max_timestamp_ns: u64,
                sample: RawGtmioUSCCliqueMetadata,
            }

            impl CliqueAggregate {
                fn new(index: usize, clique: RawGtmioUSCCliqueMetadata) -> Self {
                    let duration = clique.duration_ns();
                    Self {
                        clique_count: 1,
                        first_clique_index: index,
                        last_clique_index: index,
                        duration_sum_ns: duration,
                        min_duration_ns: duration,
                        max_duration_ns: duration,
                        min_timestamp_ns: clique.raw0,
                        max_timestamp_ns: clique.raw1,
                        sample: clique,
                    }
                }

                fn push(&mut self, index: usize, clique: RawGtmioUSCCliqueMetadata) {
                    let duration = clique.duration_ns();
                    self.clique_count += 1;
                    self.last_clique_index = index;
                    self.duration_sum_ns = self.duration_sum_ns.saturating_add(duration);
                    self.min_duration_ns = self.min_duration_ns.min(duration);
                    self.max_duration_ns = self.max_duration_ns.max(duration);
                    self.min_timestamp_ns = self.min_timestamp_ns.min(clique.raw0);
                    self.max_timestamp_ns = self.max_timestamp_ns.max(clique.raw1);
                }
            }

            let command_indices = draw_array_probe_command_indices(pipelines, gpu_commands);
            let mut target_values = BTreeMap::<u32, Vec<(&'static str, usize)>>::new();
            for command_index in &command_indices {
                let Some(command) = gpu_commands.get(*command_index) else {
                    continue;
                };
                if let Ok(index) = u32::try_from(command.index)
                    && index != 0
                {
                    target_values
                        .entry(index)
                        .or_default()
                        .push(("command_index", command.index));
                }
                if let Ok(function_index) = u32::try_from(command.function_index) {
                    target_values
                        .entry(function_index)
                        .or_default()
                        .push(("function_index", command.index));
                }
            }
            if target_values.is_empty() {
                return Vec::new();
            }

            let mut probes = Vec::new();
            unsafe {
                for source in function_time_trace_data_sources(trace_data, timeline) {
                    if !responds_to_selector(source.object, "uscs") {
                        continue;
                    }
                    let Ok(uscs) = send_id_allow_nil(source.object, "uscs") else {
                        continue;
                    };
                    if uscs.is_null() || !responds_to_selector(uscs, "count") {
                        continue;
                    }
                    let usc_count = send_u64(uscs, "count").unwrap_or(0).min(256);
                    for usc_index in 0..usc_count as usize {
                        let Ok(usc) = send_id_usize(uscs, "objectAtIndex:", usc_index) else {
                            continue;
                        };
                        if usc.is_null()
                            || !responds_to_selector(usc, "cliquesCount")
                            || !responds_to_selector(usc, "cliques")
                        {
                            continue;
                        }
                        let clique_count = send_u64(usc, "cliquesCount").unwrap_or(0);
                        if clique_count == 0 || clique_count > 5_000_000 {
                            continue;
                        }
                        let Ok(cliques) = send_ptr(usc, "cliques") else {
                            continue;
                        };
                        if cliques.is_null() {
                            continue;
                        }
                        let cliques = cliques.cast::<RawGtmioUSCCliqueMetadata>();
                        let mut aggregates = BTreeMap::<
                            (usize, usize, &'static str, u32, usize),
                            CliqueAggregate,
                        >::new();
                        for clique_index in 0..clique_count as usize {
                            let clique = *cliques.add(clique_index);
                            for (field_offset, value) in clique.u32_fields().iter().enumerate() {
                                let Some(targets) = target_values.get(value) else {
                                    continue;
                                };
                                let field_index = field_offset + 2;
                                for (match_kind, command_index) in targets {
                                    let key = (
                                        usc_index,
                                        field_index,
                                        *match_kind,
                                        *value,
                                        *command_index,
                                    );
                                    aggregates
                                        .entry(key)
                                        .and_modify(|aggregate| {
                                            aggregate.push(clique_index, clique);
                                        })
                                        .or_insert_with(|| {
                                            CliqueAggregate::new(clique_index, clique)
                                        });
                                }
                            }
                        }

                        for (
                            (usc_index, field_index, match_kind, matched_value, command_index),
                            aggregate,
                        ) in aggregates
                        {
                            let Some(command) = gpu_commands.get(command_index) else {
                                continue;
                            };
                            let span_duration_ns = aggregate
                                .max_timestamp_ns
                                .saturating_sub(aggregate.min_timestamp_ns);
                            probes.push(XcodeMioUSCCliqueProbe {
                                source: source.source,
                                usc_index,
                                field_index,
                                data_master: None,
                                match_kind,
                                matched_value,
                                command_index: command.index,
                                function_index: command.function_index,
                                sub_command_index: command.sub_command_index,
                                encoder_index: command.encoder_index,
                                pipeline_index: command.pipeline_index,
                                function_name: command.function_name.clone(),
                                clique_count: aggregate.clique_count,
                                first_clique_index: aggregate.first_clique_index,
                                last_clique_index: aggregate.last_clique_index,
                                duration_sum_ns: aggregate.duration_sum_ns,
                                span_duration_ns,
                                min_duration_ns: aggregate.min_duration_ns,
                                max_duration_ns: aggregate.max_duration_ns,
                                min_timestamp_ns: aggregate.min_timestamp_ns,
                                max_timestamp_ns: aggregate.max_timestamp_ns,
                                sample_raw0: aggregate.sample.raw0,
                                sample_raw1: aggregate.sample.raw1,
                                sample_u32_fields: aggregate.sample.u32_fields().to_vec(),
                                sample_u16_fields: aggregate.sample.u16_fields().to_vec(),
                            });
                        }

                        if responds_to_selector(
                            usc,
                            "enumerateKickCliquesAtFunctionIndex:dataMaster:enumerator:",
                        ) {
                            for command_index in &command_indices {
                                let Some(command) = gpu_commands.get(*command_index) else {
                                    continue;
                                };
                                let Ok(function_index) = u32::try_from(command.function_index)
                                else {
                                    continue;
                                };
                                for data_master in function_time_data_master_candidates() {
                                    let aggregate = Arc::new(Mutex::new(None::<CliqueAggregate>));
                                    let callback_aggregate = Arc::clone(&aggregate);
                                    let block =
                                        RcBlock::new(move |clique: *const c_void, _usc: Id| {
                                            if clique.is_null() {
                                                return;
                                            }
                                            let clique =
                                                *clique.cast::<RawGtmioUSCCliqueMetadata>();
                                            let Ok(mut aggregate) = callback_aggregate.lock()
                                            else {
                                                return;
                                            };
                                            let next_index = aggregate
                                                .as_ref()
                                                .map(|aggregate| aggregate.clique_count)
                                                .unwrap_or(0);
                                            if let Some(aggregate) = aggregate.as_mut() {
                                                aggregate.push(next_index, clique);
                                            } else {
                                                *aggregate =
                                                    Some(CliqueAggregate::new(next_index, clique));
                                            }
                                        });
                                    let block_ptr = RcBlock::as_ptr(&block).cast::<c_void>();
                                    if send_void_u32_u16_id(
                                        usc,
                                        "enumerateKickCliquesAtFunctionIndex:dataMaster:enumerator:",
                                        function_index,
                                        data_master,
                                        block_ptr.cast(),
                                    )
                                    .is_err()
                                    {
                                        continue;
                                    }
                                    let aggregate =
                                        aggregate.lock().ok().and_then(|aggregate| *aggregate);
                                    let Some(aggregate) = aggregate else {
                                        continue;
                                    };
                                    let span_duration_ns = aggregate
                                        .max_timestamp_ns
                                        .saturating_sub(aggregate.min_timestamp_ns);
                                    probes.push(XcodeMioUSCCliqueProbe {
                                        source: source.source,
                                        usc_index,
                                        field_index: 999,
                                        data_master: Some(data_master),
                                        match_kind: "enumerate",
                                        matched_value: function_index,
                                        command_index: command.index,
                                        function_index: command.function_index,
                                        sub_command_index: command.sub_command_index,
                                        encoder_index: command.encoder_index,
                                        pipeline_index: command.pipeline_index,
                                        function_name: command.function_name.clone(),
                                        clique_count: aggregate.clique_count,
                                        first_clique_index: aggregate.first_clique_index,
                                        last_clique_index: aggregate.last_clique_index,
                                        duration_sum_ns: aggregate.duration_sum_ns,
                                        span_duration_ns,
                                        min_duration_ns: aggregate.min_duration_ns,
                                        max_duration_ns: aggregate.max_duration_ns,
                                        min_timestamp_ns: aggregate.min_timestamp_ns,
                                        max_timestamp_ns: aggregate.max_timestamp_ns,
                                        sample_raw0: aggregate.sample.raw0,
                                        sample_raw1: aggregate.sample.raw1,
                                        sample_u32_fields: aggregate.sample.u32_fields().to_vec(),
                                        sample_u16_fields: aggregate.sample.u16_fields().to_vec(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
            probes.sort_by(|left, right| {
                left.command_index
                    .cmp(&right.command_index)
                    .then_with(|| right.duration_sum_ns.cmp(&left.duration_sum_ns))
                    .then_with(|| left.source.cmp(right.source))
                    .then_with(|| left.usc_index.cmp(&right.usc_index))
                    .then_with(|| left.field_index.cmp(&right.field_index))
                    .then_with(|| left.match_kind.cmp(right.match_kind))
            });
            probes
        }

        unsafe fn decode_encoder_quad_probes(
            &mut self,
            trace_data: Id,
            timeline: Option<Id>,
            pipelines: &[XcodeMioPipeline],
            encoders: &[XcodeMioEncoder],
            gpu_commands: &[XcodeMioGpuCommand],
            _timeline_pipeline_state_ids: &[u64],
        ) -> Vec<XcodeMioEncoderQuadProbe> {
            let Ok(quad_class) = (unsafe { lookup_class("GTMioEncoderQuadData") }) else {
                return Vec::new();
            };
            let mut probes = Vec::new();
            let mut seen = BTreeSet::new();
            unsafe {
                for source in function_time_trace_data_sources(trace_data, timeline) {
                    if source.source != "mio" {
                        continue;
                    }
                    for encoder in encoders {
                        let Ok(encoder_function_index) = u32::try_from(encoder.function_index)
                        else {
                            continue;
                        };
                        for program_type in encoder_quad_program_type_candidates(None) {
                            for options in encoder_quad_options_candidates() {
                                if !seen.insert((
                                    source.source,
                                    "encoder",
                                    encoder.index,
                                    usize::MAX,
                                    0_u64,
                                    u32::MAX,
                                    program_type,
                                    options,
                                )) {
                                    continue;
                                }
                                let Ok(quad_data) = init_encoder_quad_for_encoder(
                                    quad_class,
                                    source.object,
                                    encoder_function_index,
                                    program_type,
                                    options,
                                ) else {
                                    continue;
                                };
                                push_encoder_quad_probe(
                                    &mut probes,
                                    quad_data,
                                    source.source,
                                    "encoder",
                                    Some(encoder.index),
                                    encoder_function_index,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    None,
                                    program_type,
                                    options,
                                );
                            }
                        }
                    }

                    for pipeline in pipelines {
                        let mut pipeline_candidate_seen = BTreeSet::new();
                        let pipeline_ids = pipeline_function_time_candidates(
                            pipeline,
                            source.pipeline_sources,
                            &[],
                        );
                        for (pipeline_id_kind, pipeline_id) in pipeline_ids {
                            if !pipeline_candidate_seen.insert(pipeline_id) {
                                continue;
                            }
                            let draw_indices =
                                enumerate_draw_indices_for_pipeline(source.object, pipeline_id);
                            if draw_indices.is_empty() {
                                continue;
                            }
                            let sampled_draws = sample_draw_indices(&draw_indices, 8);

                            for encoder in encoders {
                                let Ok(encoder_function_index) =
                                    u32::try_from(encoder.function_index)
                                else {
                                    continue;
                                };
                                for program_type in
                                    encoder_quad_program_type_candidates(Some(pipeline))
                                {
                                    for options in encoder_quad_options_candidates() {
                                        if !seen.insert((
                                            source.source,
                                            "pipeline",
                                            encoder.index,
                                            pipeline.index,
                                            pipeline_id,
                                            u32::MAX,
                                            program_type,
                                            options,
                                        )) {
                                            continue;
                                        }
                                        let Ok(quad_data) = init_encoder_quad_for_pipeline(
                                            quad_class,
                                            source.object,
                                            encoder_function_index,
                                            pipeline_id,
                                            program_type,
                                            options,
                                        ) else {
                                            continue;
                                        };
                                        push_encoder_quad_probe(
                                            &mut probes,
                                            quad_data,
                                            source.source,
                                            "pipeline",
                                            Some(encoder.index),
                                            encoder_function_index,
                                            Some(pipeline.index),
                                            Some(pipeline_id_kind),
                                            Some(pipeline_id),
                                            None,
                                            None,
                                            pipeline.function_name.clone(),
                                            program_type,
                                            options,
                                        );
                                    }
                                }
                            }

                            for draw_index in selected_draw_indices(&draw_indices, 8) {
                                let Some(draw_index_u32) = u32::try_from(draw_index).ok() else {
                                    continue;
                                };
                                let command = command_for_enumerated_draw(gpu_commands, draw_index);
                                let mut draw_id_candidates =
                                    vec![("enumerated_draw", draw_index_u32)];
                                if let Some(command) = command
                                    && let Ok(command_index) = u32::try_from(command.index)
                                    && command_index != draw_index_u32
                                {
                                    draw_id_candidates.push(("command_index", command_index));
                                }
                                for (draw_id_kind, draw_id) in draw_id_candidates {
                                    for encoder in encoders {
                                        if let Some(command) = command
                                            && command.encoder_index != encoder.index
                                        {
                                            continue;
                                        }
                                        let Ok(encoder_function_index) =
                                            u32::try_from(encoder.function_index)
                                        else {
                                            continue;
                                        };
                                        for program_type in
                                            encoder_quad_program_type_candidates(Some(pipeline))
                                        {
                                            for options in encoder_quad_options_candidates() {
                                                if !seen.insert((
                                                    source.source,
                                                    draw_id_kind,
                                                    encoder.index,
                                                    pipeline.index,
                                                    pipeline_id,
                                                    draw_id,
                                                    program_type,
                                                    options,
                                                )) {
                                                    continue;
                                                }
                                                let Ok(quad_data) = init_encoder_quad_for_draw(
                                                    quad_class,
                                                    source.object,
                                                    encoder_function_index,
                                                    draw_id,
                                                    program_type,
                                                    options,
                                                ) else {
                                                    continue;
                                                };
                                                push_encoder_quad_probe(
                                                    &mut probes,
                                                    quad_data,
                                                    source.source,
                                                    "draw",
                                                    Some(encoder.index),
                                                    encoder_function_index,
                                                    Some(pipeline.index),
                                                    Some(pipeline_id_kind),
                                                    Some(pipeline_id),
                                                    Some(draw_id_kind),
                                                    Some(draw_id),
                                                    command
                                                        .and_then(|command| {
                                                            command.function_name.clone()
                                                        })
                                                        .or_else(|| pipeline.function_name.clone()),
                                                    program_type,
                                                    options,
                                                );
                                            }
                                        }
                                    }
                                }
                            }

                            if !sampled_draws.is_empty() {
                                for probe in probes.iter_mut().rev().take_while(|probe| {
                                    probe.source == source.source
                                        && probe.pipeline_index == Some(pipeline.index)
                                        && probe.pipeline_id == Some(pipeline_id)
                                }) {
                                    if probe.sampled_draws.is_empty() {
                                        probe.sampled_draws = sampled_draws.clone();
                                    }
                                }
                            }
                        }
                    }
                }
            }
            probes.sort_by(|left, right| {
                left.source
                    .cmp(right.source)
                    .then_with(|| left.mode.cmp(right.mode))
                    .then_with(|| {
                        left.encoder_function_index
                            .cmp(&right.encoder_function_index)
                    })
                    .then_with(|| left.pipeline_index.cmp(&right.pipeline_index))
                    .then_with(|| left.draw_index.cmp(&right.draw_index))
                    .then_with(|| left.program_type.cmp(&right.program_type))
                    .then_with(|| left.options.cmp(&right.options))
            });
            probes
        }

        unsafe fn decode_draw_execution_history_probes(
            &mut self,
            trace_data: Id,
            timeline: Option<Id>,
            pipelines: &[XcodeMioPipeline],
            gpu_commands: &[XcodeMioGpuCommand],
        ) -> Vec<XcodeMioDrawExecutionHistoryProbe> {
            let Ok(history_class) = (unsafe { lookup_class("GTMioShaderExecutionHistory") }) else {
                return Vec::new();
            };
            let delegate = unsafe {
                lookup_class("GTMioShaderExecutionHistoryDefaultDelegate")
                    .ok()
                    .and_then(|class| send_id_allow_nil(class, "shared").ok())
                    .unwrap_or(std::ptr::null_mut())
            };
            let mut probes = Vec::new();
            let draw_indices = draw_execution_history_probe_indices(pipelines, gpu_commands);
            unsafe {
                for source in function_time_trace_data_sources(trace_data, timeline) {
                    if !matches!(source.source, "mio" | "requestCostTimeline") {
                        continue;
                    }
                    for command_index in &draw_indices {
                        let Some(command) = gpu_commands.get(*command_index as usize) else {
                            continue;
                        };
                        let Some(pipeline) = pipelines.get(command.pipeline_index) else {
                            continue;
                        };
                        let draw_index_candidates =
                            draw_execution_history_draw_index_candidates(command);
                        for style in draw_execution_history_style_candidates() {
                            for options in draw_execution_history_options_candidates() {
                                for program_type in
                                    encoder_quad_program_type_candidates(Some(pipeline))
                                {
                                    for (draw_id_kind, draw_index) in &draw_index_candidates {
                                        let Ok(history) = new_shader_execution_history(
                                            history_class,
                                            source.object,
                                            style,
                                            options,
                                            delegate,
                                        ) else {
                                            continue;
                                        };
                                        let trace_generated = if responds_to_selector(
                                            source.object,
                                            "executionHistoryForDraw:programType:delegate:progressController:",
                                        ) {
                                            send_void_u32_u16_id_id(
                                                source.object,
                                                "executionHistoryForDraw:programType:delegate:progressController:",
                                                *draw_index,
                                                program_type,
                                                history,
                                                std::ptr::null_mut(),
                                            )
                                            .is_ok()
                                        } else {
                                            false
                                        };
                                        push_draw_execution_history_nodes(
                                            &mut probes,
                                            history,
                                            source.source,
                                            match *draw_id_kind {
                                                "function_index" => "trace_call.function",
                                                _ => "trace_call.index",
                                            },
                                            command,
                                            *draw_index,
                                            style,
                                            options,
                                            program_type,
                                            trace_generated,
                                        );

                                        let Ok(history) = new_shader_execution_history(
                                            history_class,
                                            source.object,
                                            style,
                                            options,
                                            delegate,
                                        ) else {
                                            continue;
                                        };
                                        let generated = if responds_to_selector(
                                            history,
                                            "generateDrawIndex:programType:",
                                        ) {
                                            send_i8_u32_u16(
                                                history,
                                                "generateDrawIndex:programType:",
                                                *draw_index,
                                                program_type,
                                            )
                                            .unwrap_or(0)
                                                != 0
                                        } else {
                                            false
                                        };
                                        push_draw_execution_history_nodes(
                                            &mut probes,
                                            history,
                                            source.source,
                                            match *draw_id_kind {
                                                "function_index" => "hist_gen.function",
                                                _ => "hist_gen.index",
                                            },
                                            command,
                                            *draw_index,
                                            style,
                                            options,
                                            program_type,
                                            generated,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
            probes.sort_by(|left, right| {
                left.source
                    .cmp(right.source)
                    .then_with(|| left.mode.cmp(right.mode))
                    .then_with(|| left.command_index.cmp(&right.command_index))
                    .then_with(|| left.style.cmp(&right.style))
                    .then_with(|| left.options.cmp(&right.options))
                    .then_with(|| left.program_type.cmp(&right.program_type))
                    .then_with(|| left.node_source.cmp(right.node_source))
            });
            probes
        }

        unsafe fn decode_timeline_cost_records(
            &mut self,
            timeline: Id,
        ) -> Vec<XcodeMioDecodedCostRecord> {
            let Ok(cost_count) = (unsafe { send_u64(timeline, "costCount") }) else {
                return Vec::new();
            };
            let Ok(costs) = (unsafe { send_ptr(timeline, "costs") }) else {
                return Vec::new();
            };
            if costs.is_null() {
                return Vec::new();
            }
            let costs = costs.cast::<RawGtmioCostInfo>();
            let mut records = Vec::with_capacity(cost_count as usize);
            for index in 0..cost_count as usize {
                let raw = unsafe { *costs.add(index) };
                let mut populated = RawGtmioCostInfo::default();
                let cost = unsafe {
                    send_i8_u16_u64_cost_mut(
                        timeline,
                        "costForScope:scopeIdentifier:cost:",
                        raw.context.scope,
                        raw.context.scope_identifier,
                        &mut populated,
                    )
                }
                .ok()
                .filter(|found| *found != 0)
                .filter(|_| !populated.is_empty())
                .map(|_| populated)
                .unwrap_or_else(|| {
                    let mut populated = RawGtmioCostInfo::default();
                    let by_context = unsafe {
                        send_i8_context_cost_mut(
                            timeline,
                            "costForContext:cost:",
                            &raw.context,
                            &mut populated,
                        )
                    };
                    by_context
                        .ok()
                        .filter(|found| *found != 0)
                        .filter(|_| !populated.is_empty())
                        .map(|_| populated)
                        .unwrap_or_else(|| {
                            let mut populated = RawGtmioCostInfo::default();
                            let by_level = unsafe {
                                send_i8_u16_u32_u16_u64_cost_mut(
                                    timeline,
                                    "costForLevel:levelIdentifier:scope:scopeIdentifier:cost:",
                                    raw.context.level,
                                    raw.context.level_identifier,
                                    raw.context.scope,
                                    raw.context.scope_identifier,
                                    &mut populated,
                                )
                            };
                            by_level
                                .ok()
                                .filter(|found| *found != 0)
                                .filter(|_| !populated.is_empty())
                                .map(|_| populated)
                                .unwrap_or(raw)
                        })
                });
                records.push(XcodeMioDecodedCostRecord {
                    index,
                    level: cost.context.level,
                    scope: cost.context.scope,
                    level_identifier: cost.context.level_identifier,
                    scope_identifier: cost.context.scope_identifier,
                    alu_cost: cost.alu_cost,
                    non_alu_cost: cost.non_alu_cost,
                    total_cost: cost.alu_cost + cost.non_alu_cost,
                    instruction_count: cost.instruction_count,
                    threads_executing_instruction: cost.threads_executing_instruction,
                    cpi_weighted_instruction_count: cost.cpi_weighted_instruction_count,
                    active_thread_instruction_count: cost.active_thread_instruction_count,
                });
            }
            records
        }

        unsafe fn timeline_binary_count(&mut self, timeline: Id) -> usize {
            unsafe {
                let Ok(binaries) = send_id(timeline, "binaries") else {
                    return 0;
                };
                send_u64(binaries, "count").unwrap_or(0) as usize
            }
        }

        unsafe fn decode_timeline_binaries(&mut self, timeline: Id) -> Vec<XcodeMioTimelineBinary> {
            unsafe {
                let Ok(binaries) = send_id(timeline, "binaries") else {
                    return Vec::new();
                };
                let binary_count = send_u64(binaries, "count").unwrap_or(0);
                let mut decoded = Vec::with_capacity(binary_count as usize);
                for index in 0..binary_count as usize {
                    let Ok(binary) = send_id_usize(binaries, "objectAtIndex:", index) else {
                        continue;
                    };
                    decoded.push(decode_timeline_binary(binary, index as u64));
                }
                decoded
            }
        }

        unsafe fn decode_shader_binary_info(
            &mut self,
            timeline: Id,
        ) -> Vec<XcodeMioShaderBinaryInfo> {
            let Ok(info_count) = (unsafe { send_u64(timeline, "shaderBinaryInfoCount") }) else {
                return Vec::new();
            };
            let Ok(info) = (unsafe { send_ptr(timeline, "shaderBinaryInfo") }) else {
                return Vec::new();
            };
            if info.is_null() {
                return Vec::new();
            }
            let info = info.cast::<u8>();
            let mut records = Vec::with_capacity(info_count as usize);
            for index in 0..info_count as usize {
                let raw = unsafe { std::slice::from_raw_parts(info.add(index * 30), 30) };
                records.push(XcodeMioShaderBinaryInfo {
                    index,
                    raw0: read_u32(raw, 0),
                    raw1: read_u32(raw, 4),
                    raw2: read_u64(raw, 8),
                    raw3: read_u64(raw, 16),
                    raw4: read_u16(raw, 24),
                    raw5: read_u16(raw, 26),
                    raw6: read_u16(raw, 28),
                });
            }
            records
        }

        unsafe fn decode_timeline_pipeline_state_ids(&mut self, timeline: Id) -> Vec<u64> {
            let values = Arc::new(Mutex::new(Vec::new()));
            let callback_values = Arc::clone(&values);
            let block = RcBlock::new(move |pipeline_state_id: u64| {
                if let Ok(mut values) = callback_values.lock() {
                    values.push(pipeline_state_id);
                }
            });
            let block_ptr = RcBlock::as_ptr(&block).cast::<c_void>();
            if unsafe { send_void_id(timeline, "enumeratePipelineStates:", block_ptr.cast()) }
                .is_err()
            {
                return Vec::new();
            }
            let mut values = values
                .lock()
                .map(|values| values.clone())
                .unwrap_or_default();
            values.sort_unstable();
            values.dedup();
            values
        }

        unsafe fn decode_draw_timeline_records(
            &mut self,
            timeline: Id,
        ) -> Vec<XcodeMioDrawTimelineRecord> {
            let Ok(draw_count) = (unsafe { send_u64(timeline, "drawCount") }) else {
                return Vec::new();
            };
            let Ok(draws) = (unsafe { send_ptr(timeline, "drawTraces") }) else {
                return Vec::new();
            };
            if draws.is_null() {
                return Vec::new();
            }
            let draws = draws.cast::<RawGtmioDrawTrace>();
            let mut records = Vec::with_capacity(draw_count as usize);
            for index in 0..draw_count as usize {
                let draw = unsafe { *draws.add(index) };
                records.push(XcodeMioDrawTimelineRecord {
                    index,
                    raw0: draw.raw0,
                    raw1: draw.raw1,
                    raw2: draw.raw2,
                    raw3: draw.raw3,
                });
            }
            records
        }

        unsafe fn decode_draw_metadata_records(
            &mut self,
            timeline: Id,
        ) -> Vec<XcodeMioDrawMetadataRecord> {
            let Ok(draw_count) = (unsafe { send_u64(timeline, "drawCount") }) else {
                return Vec::new();
            };
            let Ok(draws) = (unsafe { send_ptr(timeline, "draws") }) else {
                return Vec::new();
            };
            if draws.is_null() {
                return Vec::new();
            }
            let draws = draws.cast::<u8>();
            let mut records = Vec::with_capacity(draw_count as usize);
            for index in 0..draw_count as usize {
                records.push(unsafe { decode_draw_metadata_record(draws, index) });
            }
            records
        }

        unsafe fn decode_pipeline_draw_timeline(
            &mut self,
            trace_data: Id,
            timeline: Id,
            draw_records: &[XcodeMioDrawTimelineRecord],
            draw_metadata: &[XcodeMioDrawMetadataRecord],
            gpu_commands: &[XcodeMioGpuCommand],
            pipelines: &mut [XcodeMioPipeline],
        ) {
            for draw in draw_metadata {
                let command_index = draw.raw0 as usize;
                let Some(command) = gpu_commands.get(command_index) else {
                    continue;
                };
                let Some(pipeline) = pipelines.get_mut(command.pipeline_index) else {
                    continue;
                };
                pipeline.timeline_draw_count += 1;
                let mut duration_ns = 0;
                if let Some(trace) = draw_records.get(draw.index) {
                    duration_ns = trace.raw1.saturating_sub(trace.raw0);
                }
                if duration_ns == 0 {
                    duration_ns =
                        unsafe { draw_duration(timeline, draw.index as u32, draw.raw7 as u16) };
                }
                if duration_ns == 0 {
                    duration_ns =
                        unsafe { draw_duration(trace_data, draw.index as u32, draw.raw7 as u16) };
                }
                pipeline.timeline_duration_ns += duration_ns;
                pipeline.timeline_total_cost +=
                    unsafe { draw_scope_cost(timeline, draw.raw0 as u64, draw.raw7) };
            }
        }

        unsafe fn decode_pipeline_shader_stats(
            &mut self,
            timeline: Id,
            pipelines: &mut [XcodeMioPipeline],
        ) -> Result<()> {
            unsafe {
                let stats_class = lookup_class("GTMioTraceDataStats")?;
                let stats = send_id(stats_class, "alloc")?;
                let stats = send_id_id(stats, "initWithTraceData:", timeline)?;
                send_void(stats, "build")?;

                for pipeline in pipelines {
                    let candidates = [
                        ("object_id", pipeline.object_id),
                        ("pointer_id", pipeline.pointer_id),
                        ("function_index", pipeline.function_index),
                    ];
                    for (shader_id_kind, shader_id) in candidates {
                        for program_type in program_type_candidates(pipeline) {
                            let stat = send_id_u64_u16_allow_nil(
                                stats,
                                "shaderStatForShader:programType:",
                                shader_id,
                                program_type,
                            )?;
                            if stat.is_null() {
                                continue;
                            }
                            let number_of_cliques = send_u64(stat, "numberOfCliques")?;
                            let total_gpu_cycles = send_u64(stat, "totalGPUCycles")?;
                            let total_latency = send_u64(stat, "totalLatency")?;
                            if number_of_cliques == 0 && total_gpu_cycles == 0 && total_latency == 0
                            {
                                continue;
                            }
                            pipeline.shader_stats.push(XcodeMioPipelineShaderStat {
                                shader_id_kind,
                                shader_id,
                                program_type,
                                number_of_cliques,
                                total_gpu_cycles,
                                total_latency,
                            });
                        }
                    }
                }
            }
            Ok(())
        }

        fn decode_pipeline_scope_costs(
            &mut self,
            cost_records: &[XcodeMioDecodedCostRecord],
            pipelines: &mut [XcodeMioPipeline],
        ) {
            for pipeline in pipelines {
                let identifiers = [
                    ("object_id", pipeline.object_id),
                    ("pointer_id", pipeline.pointer_id),
                    ("function_index", pipeline.function_index),
                    ("pipeline_index", pipeline.index as u64),
                ];
                for (identifier_kind, identifier) in identifiers {
                    for cost in cost_records {
                        if cost.scope_identifier != identifier {
                            continue;
                        }
                        if !cost.total_cost.is_finite() || cost.total_cost <= 0.0 {
                            continue;
                        }
                        pipeline.scope_costs.push(XcodeMioPipelineScopeCost {
                            scope: cost.scope,
                            identifier_kind,
                            identifier,
                            level: cost.level,
                            level_identifier: cost.level_identifier,
                            alu_cost: cost.alu_cost,
                            non_alu_cost: cost.non_alu_cost,
                            total_cost: cost.total_cost,
                            instruction_count: cost.instruction_count,
                        });
                    }
                }
            }
        }

        unsafe fn decode_pipeline_shader_tracks(
            &mut self,
            trace_data: Id,
            pipelines: &mut [XcodeMioPipeline],
            source: &'static str,
        ) -> Result<()> {
            unsafe {
                let helper_class = lookup_class("GTMioTraceDataHelper")?;
                let helper = send_id(helper_class, "alloc")?;
                let helper = send_id_id(helper, "initWithTraceData:", trace_data)?;
                let _ = send_void_i8(helper, "setShowDriverInternalShaders:", 1);
                let _ = send_void_i8(helper, "setShowDriverIntersectionShaders:", 1);
                let _ = send_void_i8(helper, "setShowESLShaders:", 1);

                for pipeline in pipelines {
                    let candidates = [
                        ("object_id", pipeline.object_id),
                        ("pointer_id", pipeline.pointer_id),
                        ("function_index", pipeline.function_index),
                        ("pipeline_index", pipeline.index as u64),
                    ];
                    for (pipeline_id_kind, pipeline_id) in candidates {
                        for program_type in program_type_candidates(pipeline) {
                            let track = send_id_u64_u16_allow_nil(
                                helper,
                                "generateAggregatedShaderTrackForPipelineState:programType:",
                                pipeline_id,
                                program_type,
                            )?;
                            if track.is_null() {
                                continue;
                            }
                            if let Some(decoded) = decode_shader_track(
                                track,
                                source,
                                pipeline_id_kind,
                                pipeline_id,
                                program_type,
                            )? {
                                pipeline.shader_tracks.push(decoded);
                            }
                        }
                        let tracks = send_id_u64_allow_nil(
                            helper,
                            "generateShaderTracksForPipelineState:",
                            pipeline_id,
                        )?;
                        if tracks.is_null() {
                            continue;
                        }
                        let track_count = send_u64(tracks, "count").unwrap_or(0);
                        for index in 0..track_count as usize {
                            let track = send_id_usize(tracks, "objectAtIndex:", index)?;
                            if let Some(decoded) = decode_shader_track(
                                track,
                                "shader_tracks_for_pipeline_state",
                                pipeline_id_kind,
                                pipeline_id,
                                u16::MAX,
                            )? {
                                pipeline.shader_tracks.push(decoded);
                            }
                        }
                    }
                }
            }
            Ok(())
        }

        unsafe fn decode_pipeline_shader_binaries(
            &mut self,
            timeline: Id,
            pipelines: &mut [XcodeMioPipeline],
        ) -> Result<()> {
            unsafe {
                let mut seen = BTreeSet::new();
                let binaries = send_id(timeline, "binaries")?;
                let binary_count = send_u64(binaries, "count")?;
                for binary_index in 0..binary_count as usize {
                    let binary = send_id_usize(binaries, "objectAtIndex:", binary_index)?;
                    let address = send_u64(binary, "address").unwrap_or(0);
                    for pipeline in pipelines.iter_mut() {
                        let candidates = [
                            ("object_id", pipeline.object_id),
                            ("pointer_id", pipeline.pointer_id),
                            ("function_index", pipeline.function_index),
                            ("pipeline_index", pipeline.index as u64),
                        ];
                        for (pipeline_id_kind, pipeline_id) in candidates {
                            let used = send_i8_u64(binary, "usedInPipelineState:", pipeline_id)
                                .unwrap_or(0)
                                != 0;
                            let joined_by_key =
                                pipeline.binary_keys.contains(&(binary_index as u64))
                                    || pipeline.all_binary_keys.contains(&(binary_index as u64));
                            let joined_by_address = address != 0
                                && (pipeline.binary_keys.contains(&address)
                                    || pipeline.all_binary_keys.contains(&address));
                            if !used && !joined_by_key && !joined_by_address {
                                continue;
                            }
                            let pipeline_id_kind = if used {
                                pipeline_id_kind
                            } else if joined_by_key {
                                "binary_key"
                            } else {
                                "binary_key_address"
                            };
                            let pipeline_id = if used {
                                pipeline_id
                            } else if joined_by_key {
                                binary_index as u64
                            } else {
                                address
                            };
                            push_decoded_binary(
                                pipeline,
                                &mut seen,
                                binary,
                                pipeline_id_kind,
                                pipeline_id,
                                binary_index as u64,
                            );
                            break;
                        }
                    }
                }
            }
            Ok(())
        }

        unsafe fn decode_pipeline_execution_history(
            &mut self,
            timeline: Id,
            pipelines: &mut [XcodeMioPipeline],
            timeline_pipeline_state_ids: &[u64],
        ) -> Result<()> {
            unsafe {
                if !responds_to_selector(
                    timeline,
                    "executionHistoryForPipelineState:programType:delegate:progressController:",
                ) {
                    return Ok(());
                }
                let history_class = lookup_class("GTMioShaderExecutionHistory")?;
                let delegate = lookup_class("GTMioShaderExecutionHistoryDefaultDelegate")
                    .ok()
                    .and_then(|class| send_id_allow_nil(class, "shared").ok())
                    .unwrap_or(std::ptr::null_mut());

                for style in [1_u32, 2, 4] {
                    for options in 0_u32..=15 {
                        let history = send_id(history_class, "alloc")?;
                        let history = send_id_id_u32_u32_id(
                            history,
                            "initWithTraceData:style:options:delegate:",
                            timeline,
                            style,
                            options,
                            delegate,
                        )?;
                        for pipeline in pipelines.iter_mut() {
                            let mut seen = BTreeSet::new();
                            let identifiers = execution_history_pipeline_ids(
                                pipeline,
                                timeline_pipeline_state_ids,
                            );
                            for (pipeline_id_kind, pipeline_id) in identifiers {
                                for program_type in program_type_candidates(pipeline) {
                                    if !seen.insert((pipeline_id_kind, pipeline_id, program_type)) {
                                        continue;
                                    }
                                    let generated = send_void_u64_u16_id_id(
                                        timeline,
                                        "executionHistoryForPipelineState:programType:delegate:progressController:",
                                        pipeline_id,
                                        program_type,
                                        history,
                                        std::ptr::null_mut(),
                                    )
                                    .is_ok();
                                    if !generated {
                                        continue;
                                    }
                                    let node =
                                        send_id_u32_allow_nil(history, "nodeForStyle:", style)
                                            .unwrap_or(std::ptr::null_mut());
                                    if node.is_null() {
                                        continue;
                                    }
                                    let decoded_node = decode_execution_history_node(node);
                                    let top_cost_percentage = decoded_node.top_cost_percentage;
                                    let duration_percentage = decoded_node.duration_percentage;
                                    let total_duration_ns = decoded_node.total_duration_ns;
                                    let total_cost = decoded_node.total_cost;
                                    if top_cost_percentage == 0.0
                                        && duration_percentage == 0.0
                                        && total_cost == 0.0
                                        && total_duration_ns == 0
                                    {
                                        continue;
                                    }
                                    pipeline.execution_history.push(
                                        XcodeMioPipelineExecutionHistory {
                                            style,
                                            options,
                                            program_type,
                                            pipeline_id_kind,
                                            pipeline_id,
                                            top_cost_percentage,
                                            duration_percentage,
                                            total_duration_ns,
                                            total_cost,
                                            instruction_count: decoded_node.instruction_count,
                                        },
                                    );
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        }
    }

    struct DecodedTimeline {
        object: Id,
        summary: XcodeMioCostTimeline,
    }

    #[derive(Clone, Copy, Default)]
    #[repr(C)]
    struct RawGtmioCostContext {
        level: u16,
        scope: u16,
        level_identifier: u32,
        scope_identifier: u64,
    }

    #[derive(Clone, Copy, Default)]
    #[repr(C)]
    struct RawGtmioCostInfo {
        context: RawGtmioCostContext,
        alu_cost: f64,
        alu_cost_per_dm: [f64; 10],
        non_alu_cost: f64,
        non_alu_cost_per_dm: [f64; 10],
        instruction_count: u64,
        instruction_count_per_dm: [u64; 10],
        threads_executing_instruction: u64,
        cpi_weighted_instruction_count: u64,
        active_thread_instruction_count: u64,
    }

    #[derive(Clone, Copy, Default)]
    #[repr(C)]
    struct RawGtStatistics {
        average: f64,
        minimum: f64,
        maximum: f64,
    }

    #[derive(Clone, Copy, Default)]
    #[repr(C)]
    struct RawGtShaderProfilerTiming {
        cycles: RawGtStatistics,
        time: RawGtStatistics,
        percentage: RawGtStatistics,
        surplus_cycles: f64,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct RawGtmioBinaryTrace {
        start_timestamp_ns: u64,
        end_timestamp_ns: u64,
        raw_identifier: u64,
        raw_index: u32,
        raw_count: u32,
        raw_program_type: u16,
    }

    #[derive(Clone, Copy, Default)]
    #[repr(C)]
    struct RawGtmioDrawTrace {
        raw0: u64,
        raw1: u64,
        raw2: u32,
        raw3: u16,
    }

    #[derive(Clone, Copy, Default)]
    #[repr(C)]
    struct RawGtmioUSCCliqueMetadata {
        raw0: u64,
        raw1: u64,
        raw2: u32,
        raw3: u32,
        raw4: u32,
        raw5: u32,
        raw6: u32,
        raw7: u32,
        raw8: u32,
        raw9: u32,
        raw10: u32,
        raw11: u32,
        raw12: u32,
        raw13: u32,
        raw14: u32,
        raw15: u32,
        raw16: u16,
        raw17: u16,
        raw18: u16,
    }

    impl RawGtmioUSCCliqueMetadata {
        fn duration_ns(&self) -> u64 {
            self.raw1.saturating_sub(self.raw0)
        }

        fn u32_fields(&self) -> [u32; 14] {
            [
                self.raw2, self.raw3, self.raw4, self.raw5, self.raw6, self.raw7, self.raw8,
                self.raw9, self.raw10, self.raw11, self.raw12, self.raw13, self.raw14, self.raw15,
            ]
        }

        fn u16_fields(&self) -> [u16; 3] {
            [self.raw16, self.raw17, self.raw18]
        }
    }

    impl RawGtmioCostInfo {
        fn is_empty(&self) -> bool {
            self.context.level == 0
                && self.context.scope == 0
                && self.context.level_identifier == 0
                && self.context.scope_identifier == 0
                && self.alu_cost == 0.0
                && self.non_alu_cost == 0.0
                && self.instruction_count == 0
                && self.threads_executing_instruction == 0
                && self.cpi_weighted_instruction_count == 0
                && self.active_thread_instruction_count == 0
        }
    }

    impl RawGtShaderProfilerTiming {
        fn has_useful_values(&self) -> bool {
            [
                self.cycles.average,
                self.cycles.minimum,
                self.cycles.maximum,
                self.time.average,
                self.time.minimum,
                self.time.maximum,
                self.percentage.average,
                self.percentage.minimum,
                self.percentage.maximum,
                self.surplus_cycles,
            ]
            .into_iter()
            .any(|value| value.is_finite() && value != 0.0)
        }
    }

    #[derive(Clone, Copy, Default)]
    struct DecodedCostAggregate {
        record_count: u64,
        nonzero_record_count: u64,
        total_cost: f64,
        total_instruction_count: u64,
        alu_cost: f64,
        non_alu_cost: f64,
    }

    #[derive(Clone, Copy, Default)]
    struct DecodedExecutionHistoryNode {
        top_cost_percentage: f64,
        duration_percentage: f64,
        total_duration_ns: u64,
        total_cost: f64,
        instruction_count: u64,
    }

    #[derive(Clone, Copy, Default)]
    struct DecodedTimingInfo {
        time: u64,
        vertex_time: u64,
        fragment_time: u64,
        compute_time: u64,
    }

    unsafe fn decode_shader_track(
        track: Id,
        source: &'static str,
        pipeline_id_kind: &'static str,
        pipeline_id: u64,
        program_type: u16,
    ) -> Result<Option<XcodeMioPipelineShaderTrack>> {
        unsafe {
            let duration_ns = send_u64(track, "duration")?;
            let trace_count = send_u64(track, "traceCount")?;
            if duration_ns == 0 && trace_count == 0 {
                return Ok(None);
            }
            let traces = decode_binary_traces(track, trace_count)?;
            Ok(Some(XcodeMioPipelineShaderTrack {
                source,
                pipeline_id_kind,
                pipeline_id,
                program_type,
                track_id: send_i32(track, "trackId")?,
                first_index: send_u64(track, "firstIndex").unwrap_or(0),
                start_timestamp_ns: send_u64(track, "startTimestamp")?,
                end_timestamp_ns: send_u64(track, "endTimestamp")?,
                duration_ns,
                trace_count,
                traces,
            }))
        }
    }

    unsafe fn decode_timing_info(receiver: Id) -> Option<DecodedTimingInfo> {
        unsafe {
            if !responds_to_selector(receiver, "timingInfo") {
                return None;
            }
            let timing_info = send_id_allow_nil(receiver, "timingInfo").ok()?;
            if timing_info.is_null() {
                return None;
            }
            let decoded = DecodedTimingInfo {
                time: send_u64(timing_info, "time").unwrap_or(0),
                vertex_time: send_u64(timing_info, "vertexTime").unwrap_or(0),
                fragment_time: send_u64(timing_info, "fragmentTime").unwrap_or(0),
                compute_time: send_u64(timing_info, "computeTime").unwrap_or(0),
            };
            (decoded.time != 0
                || decoded.vertex_time != 0
                || decoded.fragment_time != 0
                || decoded.compute_time != 0)
                .then_some(decoded)
        }
    }

    fn shader_profiler_command_duration(timing: DecodedTimingInfo) -> (u64, u16) {
        [
            (timing.compute_time, 28_u16),
            (timing.vertex_time, 1_u16),
            (timing.fragment_time, 2_u16),
            (timing.time, 0_u16),
        ]
        .into_iter()
        .max_by_key(|(duration_ns, _)| *duration_ns)
        .unwrap_or((0, 0))
    }

    unsafe fn decode_pipeline_profiler_timings(
        pipeline: Id,
    ) -> Vec<XcodeMioPipelineProfilerTiming> {
        let mut timings = Vec::new();
        for (selector, source) in [
            ("timing", "total"),
            ("vertexTiming", "vertex"),
            ("fragmentTiming", "fragment"),
            ("computeTiming", "compute"),
        ] {
            if !unsafe { responds_to_selector(pipeline, selector) } {
                continue;
            }
            let Ok(raw) = (unsafe { send_profiler_timing(pipeline, selector) }) else {
                continue;
            };
            if !raw.has_useful_values() {
                continue;
            }
            timings.push(XcodeMioPipelineProfilerTiming {
                source,
                cycle_average: raw.cycles.average,
                cycle_min: raw.cycles.minimum,
                cycle_max: raw.cycles.maximum,
                time_average: raw.time.average,
                time_min: raw.time.minimum,
                time_max: raw.time.maximum,
                percentage_average: raw.percentage.average,
                percentage_min: raw.percentage.minimum,
                percentage_max: raw.percentage.maximum,
                surplus_cycles: raw.surplus_cycles,
            });
        }
        timings
    }

    fn dedup_profiler_timings(timings: &mut Vec<XcodeMioPipelineProfilerTiming>) {
        let mut seen = BTreeSet::new();
        timings.retain(|timing| {
            seen.insert((
                timing.source,
                timing.cycle_average.to_bits(),
                timing.time_average.to_bits(),
                timing.percentage_average.to_bits(),
                timing.surplus_cycles.to_bits(),
            ))
        });
    }

    unsafe fn decode_binary_traces(
        track: Id,
        trace_count: u64,
    ) -> Result<Vec<XcodeMioBinaryTrace>> {
        unsafe {
            let traces = send_ptr(track, "traces")?;
            if traces.is_null() || trace_count == 0 {
                return Ok(Vec::new());
            }
            let traces = traces.cast::<RawGtmioBinaryTrace>();
            let mut decoded = Vec::with_capacity(trace_count as usize);
            for index in 0..trace_count as usize {
                let raw = *traces.add(index);
                decoded.push(XcodeMioBinaryTrace {
                    index,
                    start_timestamp_ns: raw.start_timestamp_ns,
                    end_timestamp_ns: raw.end_timestamp_ns,
                    raw_identifier: raw.raw_identifier,
                    raw_index: raw.raw_index,
                    raw_count: raw.raw_count,
                    raw_program_type: raw.raw_program_type,
                });
            }
            Ok(decoded)
        }
    }

    fn decode_agxps_timing_trace_costs(
        profiler_directory: &Path,
        pipelines: &mut [XcodeMioPipeline],
    ) -> Result<()> {
        // Map an ESL shader address to every pipeline that references it as an
        // executable shader binary (raw5 == 6 && raw6 == 28). Multiple pipelines
        // can share an ESL — see e.g. matvec_prerot_qa and fused_rmsnorm_activation_quantize
        // on `qa-decode-ar-legacy.gputrace`, which share 8 such addresses.
        //
        // Previously this map was `BTreeMap<u64, usize>`, so colliding addresses
        // resolved to the last pipeline iterated (highest index), and all clique
        // work for a shared address was attributed to one pipeline rather than
        // distributed. That produced a +4.47pp / -3.73pp swap between matvec
        // and fused_rmsnorm vs Xcode's reference Cost % column.
        //
        // We now keep all referencing pipelines and fan each timing record's
        // contributions across them, weighted by `gpu_command_count` (how many
        // GPU commands of that pipeline used the binary). The weighting scheme
        // is selectable via `GPUTRACE_MIO_ATTRIB`:
        //   - "min_commands" (default): route exclusively to the candidate
        //                      with the smallest `gpu_command_count`.
        //                      Empirically this matches Xcode's Cost % column
        //                      best on `qa-decode-ar-legacy.gputrace`,
        //                      cutting matvec error from +4.47pp to +2.94pp
        //                      and fused_rmsnorm from -3.73pp to -2.71pp.
        //                      Rationale: shared shader binaries are usually
        //                      utility/dispatch fragments characteristic of
        //                      the lighter pipeline, not the heavyweight one.
        //   - "work_addr":     two-pass: pin `work_shader_address` -> pipeline
        //                      using unique-ESL records, then disambiguate
        //                      shared-ESL records by their work address
        //   - "commands":      weight by `gpu_command_count`
        //   - "inv_commands":  weight by 1/gpu_command_count (favor smaller pipelines)
        //   - "equal":         uniform split across referencing pipelines
        //   - "first_idx":     route exclusively to lowest-indexed candidate
        //   - "legacy":        last-write-wins (pre-fix behavior, for A/B)
        let attrib_mode_raw = std::env::var("GPUTRACE_MIO_ATTRIB").unwrap_or_default();
        let attrib_mode = if attrib_mode_raw.is_empty() {
            "min_commands"
        } else {
            attrib_mode_raw.as_str()
        };
        // Capture per-pipeline command counts; we need them post-iteration
        // for the inverse-weighting and min-selection modes.
        let pipeline_cmd_counts: Vec<u64> = pipelines
            .iter()
            .map(|p| p.gpu_command_count as u64)
            .collect();
        let mut address_to_pipelines = BTreeMap::<u64, Vec<(usize, u64)>>::new();
        for (pipeline_index, pipeline) in pipelines.iter().enumerate() {
            // Pipelines with zero recorded GPU commands shouldn't get any clique
            // budget — this avoids ghost pipelines that were loaded but never
            // dispatched from absorbing samples. We still want a non-zero weight
            // floor in `equal` mode where the count is intentionally ignored.
            let weight = pipeline.gpu_command_count as u64;
            if attrib_mode != "equal" && weight == 0 {
                continue;
            }
            for reference in &pipeline.shader_binary_references {
                if reference.raw5 == 6 && reference.raw6 == 28 {
                    address_to_pipelines
                        .entry(reference.address)
                        .or_default()
                        .push((pipeline_index, weight.max(1)));
                }
            }
        }
        if address_to_pipelines.is_empty() {
            return Ok(());
        }
        // Build the routing list per address based on the selected mode.
        let address_routes: BTreeMap<u64, Vec<(usize, u64)>> = address_to_pipelines
            .into_iter()
            .map(|(addr, mut routes)| {
                let routes = match attrib_mode {
                    "legacy" => {
                        // Reproduce the pre-fix behavior: keep only the
                        // highest-indexed pipeline and give it full weight.
                        let last = routes.into_iter().max_by_key(|(idx, _)| *idx);
                        last.map(|(idx, _)| vec![(idx, 1)]).unwrap_or_default()
                    }
                    "equal" => {
                        // Drop the count-based weights; every referencing
                        // pipeline gets an equal slice.
                        routes.into_iter().map(|(idx, _)| (idx, 1)).collect()
                    }
                    "first_idx" => {
                        // Send the entire timing record to the *lowest*-indexed
                        // candidate. Empirically the highest-indexed (legacy)
                        // is wrong; this tests the opposite extreme.
                        let first = routes.into_iter().min_by_key(|(idx, _)| *idx);
                        first.map(|(idx, _)| vec![(idx, 1)]).unwrap_or_default()
                    }
                    "min_commands" => {
                        // Send to whichever candidate has the smallest
                        // gpu_command_count (ties broken by lowest index).
                        let pick = routes.into_iter().min_by_key(|(idx, _)| {
                            (pipeline_cmd_counts.get(*idx).copied().unwrap_or(0), *idx)
                        });
                        pick.map(|(idx, _)| vec![(idx, 1)]).unwrap_or_default()
                    }
                    "inv_commands" => {
                        // Weight by 1/gpu_command_count (scaled). Pipelines
                        // with fewer commands receive proportionally more of
                        // the shared budget, on the theory that the shared
                        // binaries are more characteristic of the lighter
                        // pipeline rather than the heavier one. We multiply
                        // the inverse by the LCM-ish constant 1_000_000 to
                        // keep everything in u64 with reasonable precision.
                        const SCALE: u64 = 1_000_000;
                        routes
                            .into_iter()
                            .map(|(idx, _)| {
                                let n = pipeline_cmd_counts.get(idx).copied().unwrap_or(0).max(1);
                                (idx, SCALE / n)
                            })
                            .collect()
                    }
                    "commands" => {
                        // Weight by gpu_command_count. Ensure the route list
                        // is sorted for determinism.
                        routes.sort_by_key(|(idx, _)| *idx);
                        routes
                    }
                    _ => {
                        // Default / unknown mode: same as `min_commands`.
                        let pick = routes.into_iter().min_by_key(|(idx, _)| {
                            (pipeline_cmd_counts.get(*idx).copied().unwrap_or(0), *idx)
                        });
                        pick.map(|(idx, _)| vec![(idx, 1)]).unwrap_or_default()
                    }
                };
                (addr, routes)
            })
            .filter(|(_, r)| !r.is_empty())
            .collect();

        let paths = profiling_raw_paths(profiler_directory)?;
        if paths.is_empty() {
            return Ok(());
        }

        let loaded = agxps_sys::load()
            .map_err(|error| Error::InvalidInput(format!("agxps load failed: {error}")))?;
        let generation = u32_env("AGXPS_GEN", 16);
        let variant = u32_env("AGXPS_VARIANT", 3);
        let rev = u32_env("AGXPS_REV", 1);
        let mut groups = BTreeMap::<(usize, u64), XcodeMioPipelineAgxpsTraceCost>::new();

        for path in paths {
            let bytes = fs::read(&path)?;
            let raw = unsafe { parse_agxps_profile(&loaded, generation, variant, rev, &bytes) }?;
            let records = unsafe { agxps_timing_records(&loaded, raw.profile_data) }?;
            // First pass (work_addr mode only): discover which
            // `work_shader_address` maps to which pipeline by looking only at
            // records whose ESL is non-shared. The result is a "trusted" work
            // address -> set-of-pipelines map we use to disambiguate the
            // shared-ESL records below. Common dispatch shim addresses
            // (e.g. 0x1000000b540) appear under many pipelines and remain
            // ambiguous; for those we fall back to `min_commands` style
            // exclusive routing.
            let work_to_pipelines: BTreeMap<u64, BTreeSet<usize>> = if attrib_mode == "work_addr" {
                let mut map = BTreeMap::<u64, BTreeSet<usize>>::new();
                for record in &records {
                    let Some(routes) = address_routes.get(&record.esl_shader_address) else {
                        continue;
                    };
                    if routes.len() != 1 {
                        continue;
                    }
                    let pidx = routes[0].0;
                    map.entry(record.work_shader_address)
                        .or_default()
                        .insert(pidx);
                }
                map
            } else {
                BTreeMap::new()
            };
            let relevant_records = records
                .into_iter()
                .filter_map(|record| {
                    let routes = address_routes.get(&record.esl_shader_address)?;
                    if routes.is_empty() {
                        return None;
                    }
                    let routes = if attrib_mode == "work_addr" && routes.len() > 1 {
                        // Disambiguate using the work address: keep only
                        // candidate pipelines that we observed using this
                        // work_shader_address in unique-ESL records.
                        let known = work_to_pipelines.get(&record.work_shader_address);
                        let filtered: Vec<(usize, u64)> = match known {
                            Some(set) => routes
                                .iter()
                                .copied()
                                .filter(|(idx, _)| set.contains(idx))
                                .collect(),
                            None => Vec::new(),
                        };
                        if filtered.is_empty() {
                            // No work-address match. Fall back to the
                            // candidate with the smallest gpu_command_count
                            // (matches `min_commands` heuristic).
                            routes
                                .iter()
                                .copied()
                                .min_by_key(|(idx, _)| {
                                    (pipeline_cmd_counts.get(*idx).copied().unwrap_or(0), *idx)
                                })
                                .map(|(idx, _)| vec![(idx, 1u64)])
                                .unwrap_or_default()
                        } else if filtered.len() == 1 {
                            vec![(filtered[0].0, 1u64)]
                        } else {
                            // Multiple candidates still match; equal split.
                            filtered.into_iter().map(|(idx, _)| (idx, 1u64)).collect()
                        }
                    } else {
                        routes.clone()
                    };
                    let total_weight = routes.iter().map(|(_, w)| *w).sum::<u64>();
                    if total_weight == 0 {
                        return None;
                    }
                    Some(AgxpsPipelineTimingRecord {
                        routes,
                        total_weight,
                        record,
                    })
                })
                .collect::<Vec<_>>();
            if relevant_records.is_empty() {
                continue;
            }

            for record in &relevant_records {
                let weighted_duration = saturating_u128_to_u64(
                    u128::from(record.record.work_cliques)
                        * u128::from(record.record.avg_clique_duration),
                );
                let duration_ns = record.record.duration_ns();
                for &(pipeline_index, weight) in &record.routes {
                    let entry = agxps_trace_group(
                        &mut groups,
                        pipeline_index,
                        record.record.esl_shader_address,
                        record.record.work_shader_address,
                    );
                    entry.command_count += 1;
                    entry.record_cliques = entry.record_cliques.saturating_add(scale_u64(
                        record.record.work_cliques,
                        weight,
                        record.total_weight,
                    ));
                    entry.analyzer_avg_duration_sum =
                        entry.analyzer_avg_duration_sum.saturating_add(scale_u64(
                            record.record.avg_clique_duration,
                            weight,
                            record.total_weight,
                        ));
                    entry.analyzer_weighted_duration = entry
                        .analyzer_weighted_duration
                        .saturating_add(scale_u64(weighted_duration, weight, record.total_weight));
                    entry.duration_ns = entry.duration_ns.saturating_add(scale_u64(
                        duration_ns,
                        weight,
                        record.total_weight,
                    ));
                }
            }

            let work = unsafe { agxps_work_cliques(&loaded, raw.profile_data) }?;
            for index in 0..work.traces.len() {
                if work.missing_ends[index] != 0 {
                    continue;
                }
                let start_ns = unsafe {
                    (loaded.api.get_system_timestamp)(raw.profile_data, work.starts[index])
                };
                let end_ns = unsafe {
                    (loaded.api.get_system_timestamp)(raw.profile_data, work.ends[index])
                };
                let Some(record_index) =
                    find_agxps_timing_record(&relevant_records, start_ns, end_ns)
                else {
                    continue;
                };
                let record = &relevant_records[record_index];
                let execution_events = unsafe {
                    (loaded.api.instruction_trace_get_execution_events_num)(
                        raw.profile_data,
                        work.traces[index],
                    )
                };
                let stats = unsafe {
                    (loaded.api.instruction_trace_get_instruction_stats)(
                        raw.gpu,
                        raw.profile_data,
                        work.traces[index],
                    )
                };
                for &(pipeline_index, weight) in &record.routes {
                    let entry = agxps_trace_group(
                        &mut groups,
                        pipeline_index,
                        record.record.esl_shader_address,
                        record.record.work_shader_address,
                    );
                    entry.matched_work_cliques += 1;
                    entry.execution_events = entry.execution_events.saturating_add(scale_u64(
                        execution_events,
                        weight,
                        record.total_weight,
                    ));
                    entry.stats_word0 = entry.stats_word0.saturating_add(scale_u64(
                        stats.words[0],
                        weight,
                        record.total_weight,
                    ));
                    entry.stats_word1 = entry.stats_word1.saturating_add(scale_u64(
                        stats.words[1],
                        weight,
                        record.total_weight,
                    ));
                }
            }

            let _ = raw;
        }

        for ((pipeline_index, _), cost) in groups {
            if let Some(pipeline) = pipelines.get_mut(pipeline_index) {
                pipeline.agxps_trace_costs.push(cost);
            }
        }
        for pipeline in pipelines {
            pipeline.agxps_trace_costs.sort_by(|left, right| {
                right
                    .stats_word1
                    .cmp(&left.stats_word1)
                    .then_with(|| left.shader_address.cmp(&right.shader_address))
            });
        }

        Ok(())
    }

    fn profiling_raw_paths(profiler_directory: &Path) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();
        for entry in fs::read_dir(profiler_directory)? {
            let entry = entry?;
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if name.starts_with("Profiling_f_") && name.ends_with(".raw") {
                paths.push(path);
            }
        }
        paths.sort();
        Ok(paths)
    }

    fn u32_env(name: &str, default: u32) -> u32 {
        std::env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn agxps_trace_group<'a>(
        groups: &'a mut BTreeMap<(usize, u64), XcodeMioPipelineAgxpsTraceCost>,
        pipeline_index: usize,
        esl_shader_address: u64,
        work_shader_address: u64,
    ) -> &'a mut XcodeMioPipelineAgxpsTraceCost {
        groups
            .entry((pipeline_index, esl_shader_address))
            .or_insert_with(|| XcodeMioPipelineAgxpsTraceCost {
                source: "agxps-timing-trace",
                shader_address: esl_shader_address,
                work_shader_address,
                command_count: 0,
                record_cliques: 0,
                analyzer_weighted_duration: 0,
                analyzer_avg_duration_sum: 0,
                matched_work_cliques: 0,
                duration_ns: 0,
                execution_events: 0,
                stats_word0: 0,
                stats_word1: 0,
            })
    }

    /// Scale `value * weight / total`, computed in u128 to avoid overflow,
    /// returning a saturating u64. `total` of 0 yields 0.
    fn scale_u64(value: u64, weight: u64, total: u64) -> u64 {
        if total == 0 {
            return 0;
        }
        let product = u128::from(value).saturating_mul(u128::from(weight));
        let scaled = product / u128::from(total);
        if scaled > u128::from(u64::MAX) {
            u64::MAX
        } else {
            scaled as u64
        }
    }

    unsafe fn parse_agxps_profile(
        loaded: &agxps_sys::LoadedApi,
        generation: u32,
        variant: u32,
        rev: u32,
        bytes: &[u8],
    ) -> Result<AgxpsRawProfile> {
        let api = &loaded.api;
        let gpu = unsafe { (api.gpu_create)(generation, variant, rev, false) };
        if gpu.is_null() {
            return Err(Error::InvalidInput(format!(
                "agxps_gpu_create({generation}, {variant}, {rev}) failed"
            )));
        }

        let descriptor = agxps_sys::AgxpsApsDescriptor::defaults_for(gpu);
        let parser = unsafe { (api.parser_create)(&descriptor) };
        if parser.is_null() {
            return Err(Error::InvalidInput(
                "agxps_aps_parser_create returned NULL".to_owned(),
            ));
        }

        let mut out = vec![0u8; 4096];
        let profile_data = unsafe {
            (api.parser_parse)(
                parser,
                bytes.as_ptr(),
                bytes.len() as c_long,
                agxps_sys::APS_PROFILING_TYPE_USC_SAMPLES,
                out.as_mut_ptr().cast(),
            )
        };
        let err_code = u64::from_le_bytes(out[..8].try_into().unwrap());
        if err_code != 0 {
            let message = unsafe {
                let ptr = (api.parse_error_string)(err_code);
                if ptr.is_null() {
                    "(null)".to_owned()
                } else {
                    CStr::from_ptr(ptr).to_string_lossy().into_owned()
                }
            };
            return Err(Error::InvalidInput(format!(
                "agxps parser error {err_code}: {message}"
            )));
        }

        let _ = parser;
        Ok(AgxpsRawProfile { gpu, profile_data })
    }

    unsafe fn agxps_timing_records(
        loaded: &agxps_sys::LoadedApi,
        profile_data: agxps_sys::AgxpsApsProfileData,
    ) -> Result<Vec<AgxpsTimingRecord>> {
        const KIND: u32 = 1;
        let api = &loaded.api;
        let analyzer = unsafe { (api.timing_analyzer_create)(KIND) };
        if analyzer.is_null() {
            return Err(Error::InvalidInput(format!(
                "agxps_aps_timing_analyzer_create({KIND}) returned NULL"
            )));
        }
        unsafe {
            (api.timing_analyzer_process_usc)(analyzer, profile_data);
            (api.timing_analyzer_finish)(analyzer);
        }
        let count = unsafe { (api.timing_analyzer_get_num_commands)(analyzer, KIND) } as usize;
        let result = unsafe { agxps_fetch_timing_records(loaded, analyzer, count) };
        unsafe { (api.timing_analyzer_destroy)(analyzer) };
        result
    }

    unsafe fn agxps_fetch_timing_records(
        loaded: &agxps_sys::LoadedApi,
        analyzer: agxps_sys::AgxpsApsTimingAnalyzer,
        count: usize,
    ) -> Result<Vec<AgxpsTimingRecord>> {
        const KIND: u32 = 1;
        let api = &loaded.api;
        let mut starts = vec![0u64; count];
        let mut ends = vec![0u64; count];
        let mut work_shaders = vec![0u64; count];
        let mut esl_shaders = vec![0u64; count];
        let mut avg_durations = vec![0u64; count];
        let mut cliques = vec![0u64; count];
        if count > 0 {
            let ok = unsafe {
                (api.timing_analyzer_get_work_start)(
                    analyzer,
                    KIND,
                    starts.as_mut_ptr(),
                    0,
                    count as u64,
                ) != 0
                    && (api.timing_analyzer_get_work_end)(
                        analyzer,
                        KIND,
                        ends.as_mut_ptr(),
                        0,
                        count as u64,
                    ) != 0
                    && (api.timing_analyzer_get_work_shader_address)(
                        analyzer,
                        KIND,
                        work_shaders.as_mut_ptr(),
                        0,
                        count as u64,
                    ) != 0
                    && (api.timing_analyzer_get_esl_shader_address)(
                        analyzer,
                        KIND,
                        esl_shaders.as_mut_ptr(),
                        0,
                        count as u64,
                    ) != 0
                    && (api.timing_analyzer_get_work_cliques_average_duration)(
                        analyzer,
                        KIND,
                        avg_durations.as_mut_ptr(),
                        0,
                        count as u64,
                    ) != 0
                    && (api.timing_analyzer_get_num_work_cliques)(
                        analyzer,
                        KIND,
                        cliques.as_mut_ptr(),
                        0,
                        count as u64,
                    ) != 0
            };
            if !ok {
                return Err(Error::InvalidInput(
                    "AGXPS timing-analyzer range getter failed".to_owned(),
                ));
            }
        }

        Ok((0..count)
            .map(|index| AgxpsTimingRecord {
                start_ns: starts[index],
                end_ns: ends[index],
                work_shader_address: work_shaders[index],
                esl_shader_address: esl_shaders[index],
                avg_clique_duration: avg_durations[index],
                work_cliques: cliques[index],
            })
            .collect())
    }

    fn saturating_u128_to_u64(value: u128) -> u64 {
        value.min(u128::from(u64::MAX)) as u64
    }

    unsafe fn agxps_work_cliques(
        loaded: &agxps_sys::LoadedApi,
        profile_data: agxps_sys::AgxpsApsProfileData,
    ) -> Result<AgxpsWorkCliques> {
        let api = &loaded.api;
        let count = unsafe { (api.get_work_cliques_num)(profile_data) } as usize;
        let mut starts = vec![0u64; count];
        let mut ends = vec![0u64; count];
        let mut missing_ends = vec![0u8; count];
        let mut traces = vec![0u64; count];
        if count > 0 {
            let ok = unsafe {
                (api.get_work_clique_start)(profile_data, starts.as_mut_ptr(), 0, count as u64) != 0
                    && (api.get_work_clique_end)(profile_data, ends.as_mut_ptr(), 0, count as u64)
                        != 0
                    && (api.get_work_clique_missing_end)(
                        profile_data,
                        missing_ends.as_mut_ptr(),
                        0,
                        count as u64,
                    ) != 0
                    && (api.get_work_clique_instruction_trace)(
                        profile_data,
                        traces.as_mut_ptr(),
                        0,
                        count as u64,
                    ) != 0
            };
            if !ok {
                return Err(Error::InvalidInput(
                    "AGXPS work-clique range getter failed".to_owned(),
                ));
            }
        }
        Ok(AgxpsWorkCliques {
            starts,
            ends,
            missing_ends,
            traces,
        })
    }

    fn find_agxps_timing_record(
        records: &[AgxpsPipelineTimingRecord],
        start_ns: u64,
        end_ns: u64,
    ) -> Option<usize> {
        let mut best = None;
        for (index, record) in records.iter().enumerate() {
            let timing = record.record;
            let contains_range = timing.start_ns <= start_ns && end_ns <= timing.end_ns;
            let contains_start = timing.start_ns <= start_ns && start_ns <= timing.end_ns;
            let overlaps = timing.start_ns <= end_ns && start_ns <= timing.end_ns;
            if !contains_range && !contains_start && !overlaps {
                continue;
            }
            let rank = if contains_range {
                0
            } else if contains_start {
                1
            } else {
                2
            };
            let duration = timing.duration_ns();
            if best
                .map(|(_, best_rank, best_duration)| (rank, duration) < (best_rank, best_duration))
                .unwrap_or(true)
            {
                best = Some((index, rank, duration));
            }
        }
        best.map(|(index, _, _)| index)
    }

    struct AgxpsRawProfile {
        gpu: agxps_sys::AgxpsGpu,
        profile_data: agxps_sys::AgxpsApsProfileData,
    }

    #[derive(Clone, Copy)]
    struct AgxpsTimingRecord {
        start_ns: u64,
        end_ns: u64,
        work_shader_address: u64,
        esl_shader_address: u64,
        avg_clique_duration: u64,
        work_cliques: u64,
    }

    impl AgxpsTimingRecord {
        fn duration_ns(self) -> u64 {
            self.end_ns.saturating_sub(self.start_ns)
        }
    }

    #[derive(Clone)]
    struct AgxpsPipelineTimingRecord {
        /// (pipeline_index, weight) for every pipeline whose ESL binary
        /// references the timing record's `esl_shader_address`. Multiple
        /// entries here mean the timing record's contributions are split
        /// across pipelines proportional to weight.
        routes: Vec<(usize, u64)>,
        /// Sum of all weights in `routes`. Cached so the per-record loops
        /// don't have to re-sum.
        total_weight: u64,
        record: AgxpsTimingRecord,
    }

    struct AgxpsWorkCliques {
        starts: Vec<u64>,
        ends: Vec<u64>,
        missing_ends: Vec<u8>,
        traces: Vec<agxps_sys::AgxpsApsCliqueInstructionTrace>,
    }

    fn attach_shader_binary_references(
        shader_binary_info: &[XcodeMioShaderBinaryInfo],
        timeline_binaries: &[XcodeMioTimelineBinary],
        gpu_commands: &[XcodeMioGpuCommand],
        pipelines: &mut [XcodeMioPipeline],
    ) {
        let binary_by_address = timeline_binaries
            .iter()
            .map(|binary| (binary.address, binary.index))
            .collect::<BTreeMap<_, _>>();
        let mut groups =
            BTreeMap::<(usize, u16, u16, u32, u64), XcodeMioPipelineShaderBinaryReference>::new();
        for info in shader_binary_info {
            let command_index = info.raw3 as usize;
            let Some(command) = gpu_commands.get(command_index) else {
                continue;
            };
            let key = (
                command.pipeline_index,
                info.raw5,
                info.raw6,
                info.raw1,
                info.raw2,
            );
            groups
                .entry(key)
                .and_modify(|reference| {
                    reference.record_count += 1;
                    reference.first_command_index =
                        reference.first_command_index.min(command_index);
                    reference.last_command_index = reference.last_command_index.max(command_index);
                })
                .or_insert_with(|| XcodeMioPipelineShaderBinaryReference {
                    raw5: info.raw5,
                    raw6: info.raw6,
                    raw1: info.raw1,
                    address: info.raw2,
                    timeline_binary_index: binary_by_address.get(&info.raw2).copied(),
                    record_count: 1,
                    first_command_index: command_index,
                    last_command_index: command_index,
                });
        }
        for ((pipeline_index, _, _, _, _), reference) in groups {
            if let Some(pipeline) = pipelines.get_mut(pipeline_index) {
                pipeline.shader_binary_references.push(reference);
            }
        }
    }

    fn program_type_candidates(pipeline: &XcodeMioPipeline) -> Vec<u16> {
        let mut candidates = BTreeSet::from([8_u16, 28_u16]);
        for binary in &pipeline.shader_binaries {
            candidates.insert(binary.program_type);
        }
        candidates.into_iter().collect()
    }

    fn execution_history_pipeline_ids(
        pipeline: &XcodeMioPipeline,
        timeline_pipeline_state_ids: &[u64],
    ) -> Vec<(&'static str, u64)> {
        let mut values = vec![
            ("object_id", pipeline.object_id),
            ("pointer_id", pipeline.pointer_id),
            ("function_index", pipeline.function_index),
            ("pipeline_index", pipeline.index as u64),
        ];
        let raw_ids = pipeline
            .shader_binary_references
            .iter()
            .filter(|reference| reference.raw5 == 6 && reference.raw6 == 28)
            .map(|reference| reference.raw1 as u64)
            .collect::<BTreeSet<_>>();
        values.extend(raw_ids.into_iter().map(|raw1| ("shader_binary_raw1", raw1)));
        values.extend(
            timeline_pipeline_state_ids
                .iter()
                .copied()
                .map(|pipeline_state_id| ("timeline_pipeline_state_id", pipeline_state_id)),
        );
        values
    }

    unsafe fn decode_timeline_binary(binary: Id, fallback_index: u64) -> XcodeMioTimelineBinary {
        unsafe {
            let index = send_u64(binary, "index").unwrap_or(fallback_index);
            let address = send_u64(binary, "address").unwrap_or(0);
            let program_type = send_u16(binary, "programType").unwrap_or(0);
            let instruction_info_count = send_u64(binary, "instructionInfoCount").unwrap_or(0);
            let instruction_executed = send_u64(binary, "instructionExecuted").unwrap_or(0);
            let duration_ns = send_u64(binary, "duration").unwrap_or(0);
            let trace_count = send_u64(binary, "traceCount").unwrap_or(0);
            let cost_count = send_u64(binary, "costCount").unwrap_or(0);
            let (total_cost, total_instruction_count) =
                decode_binary_total_cost(binary, cost_count, instruction_info_count);
            let instruction_costs =
                decode_binary_instruction_costs(binary, instruction_info_count, "instructionCosts");
            XcodeMioTimelineBinary {
                index,
                address,
                program_type,
                instruction_info_count,
                instruction_executed,
                duration_ns,
                trace_count,
                cost_count,
                total_cost,
                total_instruction_count,
                instruction_cost_record_count: instruction_costs.record_count,
                instruction_nonzero_record_count: instruction_costs.nonzero_record_count,
                instruction_total_cost: instruction_costs.total_cost,
                instruction_total_count: instruction_costs.total_instruction_count,
            }
        }
    }

    unsafe fn push_decoded_binary(
        pipeline: &mut XcodeMioPipeline,
        seen: &mut BTreeSet<(usize, u64, u16, u64)>,
        binary: Id,
        pipeline_id_kind: &'static str,
        pipeline_id: u64,
        fallback_binary_index: u64,
    ) {
        unsafe {
            let binary_index = send_u64(binary, "index").unwrap_or(fallback_binary_index);
            let address = send_u64(binary, "address").unwrap_or(0);
            let program_type = send_u16(binary, "programType").unwrap_or(0);
            if !seen.insert((pipeline.index, binary_index, program_type, address)) {
                return;
            }
            let instruction_info_count = send_u64(binary, "instructionInfoCount").unwrap_or(0);
            let instruction_executed = send_u64(binary, "instructionExecuted").unwrap_or(0);
            let duration_ns = send_u64(binary, "duration").unwrap_or(0);
            let trace_count = send_u64(binary, "traceCount").unwrap_or(0);
            let cost_count = send_u64(binary, "costCount").unwrap_or(0);
            let (total_cost, total_instruction_count) =
                decode_binary_total_cost(binary, cost_count, instruction_info_count);
            for (source, costs) in [
                (
                    "instructionCostsForPipelineState",
                    decode_binary_instruction_costs_for_pipeline_state(
                        binary,
                        pipeline_id,
                        instruction_info_count,
                    ),
                ),
                (
                    "instructionCosts",
                    decode_binary_instruction_costs(
                        binary,
                        instruction_info_count,
                        "instructionCosts",
                    ),
                ),
            ] {
                if costs.total_cost > 0.0 || costs.total_instruction_count > 0 {
                    pipeline
                        .shader_binary_costs
                        .push(XcodeMioPipelineShaderBinaryCost {
                            source,
                            pipeline_id_kind,
                            pipeline_id,
                            binary_index,
                            address,
                            program_type,
                            record_count: costs.record_count,
                            nonzero_record_count: costs.nonzero_record_count,
                            total_cost: costs.total_cost,
                            total_instruction_count: costs.total_instruction_count,
                            alu_cost: costs.alu_cost,
                            non_alu_cost: costs.non_alu_cost,
                        });
                }
            }
            pipeline.shader_binaries.push(XcodeMioPipelineShaderBinary {
                pipeline_id_kind,
                pipeline_id,
                binary_index,
                address,
                program_type,
                instruction_info_count,
                instruction_executed,
                duration_ns,
                trace_count,
                cost_count,
                total_cost,
                total_instruction_count,
            });
        }
    }

    unsafe fn decode_binary_total_cost(
        binary: Id,
        cost_count: u64,
        _instruction_info_count: u64,
    ) -> (f64, u64) {
        unsafe {
            if cost_count == 0 {
                return (0.0, 0);
            }
            let Ok(costs) = send_ptr(binary, "costs") else {
                return (0.0, 0);
            };
            if costs.is_null() {
                return (0.0, 0);
            }
            decode_cost_info_sum(costs.cast(), cost_count as usize)
        }
    }

    unsafe fn decode_gpu_cost(trace_data: Id) -> (f64, u64) {
        unsafe {
            let Ok(cost) = send_ptr(trace_data, "gpuCost") else {
                return (0.0, 0);
            };
            if cost.is_null() {
                return (0.0, 0);
            }
            let cost = *cost.cast::<RawGtmioCostInfo>();
            (cost.alu_cost + cost.non_alu_cost, cost.instruction_count)
        }
    }

    unsafe fn decode_binary_instruction_costs(
        binary: Id,
        instruction_info_count: u64,
        selector: &str,
    ) -> DecodedCostAggregate {
        unsafe {
            let Ok(costs) = send_ptr(binary, selector) else {
                return DecodedCostAggregate::default();
            };
            decode_instruction_cost_pointer(costs, instruction_info_count)
        }
    }

    unsafe fn decode_binary_instruction_costs_for_pipeline_state(
        binary: Id,
        pipeline_state_id: u64,
        instruction_info_count: u64,
    ) -> DecodedCostAggregate {
        unsafe {
            let Ok(costs) = send_ptr_u64(
                binary,
                "instructionCostsForPipelineState:",
                pipeline_state_id,
            ) else {
                return DecodedCostAggregate::default();
            };
            decode_instruction_cost_pointer(costs, instruction_info_count)
        }
    }

    unsafe fn decode_instruction_cost_pointer(
        costs: *const c_void,
        instruction_info_count: u64,
    ) -> DecodedCostAggregate {
        if costs.is_null() || instruction_info_count == 0 || instruction_info_count > 1_000_000 {
            return DecodedCostAggregate::default();
        }
        let costs = costs.cast::<RawGtmioCostInfo>();
        let mut aggregate = DecodedCostAggregate {
            record_count: instruction_info_count,
            ..DecodedCostAggregate::default()
        };
        for index in 0..instruction_info_count as usize {
            let cost = unsafe { *costs.add(index) };
            let total_cost = cost.alu_cost + cost.non_alu_cost;
            if !cost.is_empty() {
                aggregate.nonzero_record_count += 1;
            }
            if total_cost.is_finite() {
                aggregate.total_cost += total_cost;
                aggregate.alu_cost += cost.alu_cost;
                aggregate.non_alu_cost += cost.non_alu_cost;
            }
            aggregate.total_instruction_count = aggregate
                .total_instruction_count
                .saturating_add(cost.instruction_count);
        }
        aggregate
    }

    unsafe fn decode_execution_history_node_self(node: Id) -> DecodedExecutionHistoryNode {
        unsafe {
            let top_cost_percentage = send_f64(node, "topCostPercentage").unwrap_or(0.0);
            let duration_percentage = send_f64(node, "durationPercentage").unwrap_or(0.0);
            let total_duration_ns = send_u64(node, "totalDuration").unwrap_or(0);
            let mut cost = RawGtmioCostInfo::default();
            let found = send_i8_u16_u64_cost_mut(
                node,
                "costForScope:scopeIdentifier:cost:",
                0,
                0,
                &mut cost,
            )
            .unwrap_or(0);
            let total_cost = if found != 0 {
                cost.alu_cost + cost.non_alu_cost
            } else {
                0.0
            };
            DecodedExecutionHistoryNode {
                top_cost_percentage,
                duration_percentage,
                total_duration_ns,
                total_cost,
                instruction_count: cost.instruction_count,
            }
        }
    }

    unsafe fn decode_execution_history_node(root: Id) -> DecodedExecutionHistoryNode {
        unsafe {
            let mut best = DecodedExecutionHistoryNode::default();
            let mut stack = vec![root];
            let mut visited = 0_usize;
            while let Some(node) = stack.pop() {
                visited += 1;
                if visited > 20_000 {
                    break;
                }
                let candidate = decode_execution_history_node_self(node);
                if candidate
                    .top_cost_percentage
                    .partial_cmp(&best.top_cost_percentage)
                    .unwrap_or(std::cmp::Ordering::Less)
                    .is_gt()
                    || (candidate.top_cost_percentage == best.top_cost_percentage
                        && candidate.total_cost > best.total_cost)
                {
                    best = candidate;
                }
                let Ok(children) = send_id_allow_nil(node, "children") else {
                    continue;
                };
                if children.is_null() {
                    continue;
                }
                let child_count = send_u64(children, "count").unwrap_or(0).min(20_000);
                for index in 0..child_count as usize {
                    if let Ok(child) = send_id_usize(children, "objectAtIndex:", index) {
                        stack.push(child);
                    }
                }
            }
            best
        }
    }

    unsafe fn decode_cost_info_sum(costs: *const RawGtmioCostInfo, count: usize) -> (f64, u64) {
        let mut total_cost = 0.0;
        let mut total_instruction_count = 0;
        for index in 0..count {
            let cost = unsafe { *costs.add(index) };
            total_cost += cost.alu_cost + cost.non_alu_cost;
            total_instruction_count += cost.instruction_count;
        }
        (total_cost, total_instruction_count)
    }

    unsafe fn draw_scope_cost(timeline: Id, draw_index: u64, metadata_program_type: u32) -> f64 {
        let mut best = 0.0_f64;
        let mut candidates = [u16::MAX; 18];
        candidates[0] = metadata_program_type as u16;
        for (index, value) in (0_u16..=16).enumerate() {
            candidates[index + 1] = value;
        }
        for scope in 0_u16..=16 {
            for program_type in candidates {
                if program_type == u16::MAX {
                    continue;
                }
                for selector in [
                    "totalCostForScope:scopeIdentifier:programType:",
                    "totalCostForScope:scopeIdentifier:dataMaster:",
                ] {
                    let cost = unsafe {
                        send_f64_u16_u64_u16(timeline, selector, scope, draw_index, program_type)
                    }
                    .unwrap_or(0.0);
                    if cost.is_finite() && cost > best {
                        best = cost;
                    }
                }
            }
        }
        best
    }

    unsafe fn function_time_trace_data_sources(
        trace_data: Id,
        requested_timeline: Option<Id>,
    ) -> Vec<FunctionTimeTraceDataSource> {
        unsafe {
            let mut sources = Vec::new();
            let mut seen = BTreeSet::new();
            push_function_time_source(
                &mut sources,
                &mut seen,
                trace_data,
                "mio",
                "mio.function_index",
                "mio.command_index",
                EnumFunctionTimeSources {
                    object_id: "mio.object_id",
                    pointer_id: "mio.pointer_id",
                    function_index: "mio.function_index",
                    pipeline_index: "mio.pipeline_index",
                },
            );
            if let Some(timeline) = requested_timeline {
                push_function_time_source(
                    &mut sources,
                    &mut seen,
                    timeline,
                    "requestCostTimeline",
                    "request.function_index",
                    "request.command_index",
                    EnumFunctionTimeSources {
                        object_id: "request.object_id",
                        pointer_id: "request.pointer_id",
                        function_index: "request.function_index",
                        pipeline_index: "request.pipeline_index",
                    },
                );
            }
            for (selector, source, function_index_source, command_index_source, pipeline_sources) in [
                (
                    "costTimeline",
                    "mio.costTimeline",
                    "cost.function_index",
                    "cost.command_index",
                    EnumFunctionTimeSources {
                        object_id: "cost.object_id",
                        pointer_id: "cost.pointer_id",
                        function_index: "cost.function_index",
                        pipeline_index: "cost.pipeline_index",
                    },
                ),
                (
                    "overlappingTimeline",
                    "mio.overlappingTimeline",
                    "overlap.function_index",
                    "overlap.command_index",
                    EnumFunctionTimeSources {
                        object_id: "overlap.object_id",
                        pointer_id: "overlap.pointer_id",
                        function_index: "overlap.function_index",
                        pipeline_index: "overlap.pipeline_index",
                    },
                ),
                (
                    "nonOverlappingTimeline",
                    "mio.nonOverlappingTimeline",
                    "nonoverlap.function_index",
                    "nonoverlap.command_index",
                    EnumFunctionTimeSources {
                        object_id: "nonoverlap.object_id",
                        pointer_id: "nonoverlap.pointer_id",
                        function_index: "nonoverlap.function_index",
                        pipeline_index: "nonoverlap.pipeline_index",
                    },
                ),
            ] {
                if !responds_to_selector(trace_data, selector) {
                    continue;
                }
                let Ok(object) = send_id_allow_nil(trace_data, selector) else {
                    continue;
                };
                push_function_time_source(
                    &mut sources,
                    &mut seen,
                    object,
                    source,
                    function_index_source,
                    command_index_source,
                    pipeline_sources,
                );
            }
            sources
        }
    }

    fn push_function_time_source(
        sources: &mut Vec<FunctionTimeTraceDataSource>,
        seen: &mut BTreeSet<usize>,
        object: Id,
        source: &'static str,
        function_index_source: &'static str,
        command_index_source: &'static str,
        pipeline_sources: EnumFunctionTimeSources,
    ) {
        if object.is_null() || !seen.insert(object as usize) {
            return;
        }
        sources.push(FunctionTimeTraceDataSource {
            object,
            source,
            function_index_source,
            command_index_source,
            pipeline_sources,
        });
    }

    unsafe fn push_gpu_command_function_time_source(
        rows: &mut Vec<XcodeMioGpuCommandFunctionTime>,
        trace_data: Id,
        function_index_source: &'static str,
        command_index_source: &'static str,
        gpu_commands: &[XcodeMioGpuCommand],
    ) {
        for command in gpu_commands {
            let mut candidates = Vec::with_capacity(2);
            if let Ok(draw_index) = u32::try_from(command.function_index) {
                candidates.push((function_index_source, draw_index));
            }
            if let Ok(draw_index) = u32::try_from(command.index) {
                candidates.push((command_index_source, draw_index));
            }
            for (source, draw_index) in candidates {
                let mut best_duration_ns = 0_u64;
                let mut best_data_master = 0_u16;
                for data_master in function_time_data_master_candidates() {
                    let duration_ns = unsafe {
                        send_u64_u32_u16(
                            trace_data,
                            "durationForDraw:dataMaster:",
                            draw_index,
                            data_master,
                        )
                    }
                    .unwrap_or(0);
                    if duration_ns > best_duration_ns {
                        best_duration_ns = duration_ns;
                        best_data_master = data_master;
                    }
                }
                if best_duration_ns == 0 {
                    continue;
                }
                rows.push(XcodeMioGpuCommandFunctionTime {
                    source,
                    command_index: command.index,
                    function_index: command.function_index,
                    sub_command_index: command.sub_command_index,
                    encoder_index: command.encoder_index,
                    pipeline_index: command.pipeline_index,
                    function_name: command.function_name.clone(),
                    draw_index,
                    data_master: best_data_master,
                    duration_ns: best_duration_ns,
                });
            }
        }
    }

    unsafe fn push_enumerated_gpu_command_function_time_source(
        rows: &mut Vec<XcodeMioGpuCommandFunctionTime>,
        probes: &mut Vec<XcodeMioFunctionTimeProbe>,
        source: FunctionTimeTraceDataSource,
        pipelines: &[XcodeMioPipeline],
        gpu_commands: &[XcodeMioGpuCommand],
        timeline_pipeline_state_ids: &[u64],
    ) {
        let mut seen = BTreeSet::new();
        for pipeline in pipelines {
            for (target_id_kind, pipeline_state_id) in pipeline_function_time_candidates(
                pipeline,
                source.pipeline_sources,
                timeline_pipeline_state_ids,
            ) {
                let draw_count = if unsafe {
                    responds_to_selector(source.object, "numDrawsForPipelineState:")
                } {
                    unsafe {
                        send_u64_u64(
                            source.object,
                            "numDrawsForPipelineState:",
                            pipeline_state_id,
                        )
                    }
                    .unwrap_or(0)
                } else {
                    0
                };
                let draw_indices = if unsafe {
                    responds_to_selector(
                        source.object,
                        "enumerateDrawsForPipelineState:enumerator:",
                    )
                } {
                    unsafe { enumerate_draw_indices_for_pipeline(source.object, pipeline_state_id) }
                } else {
                    Vec::new()
                };
                let mut best_draw_index = None;
                let mut best_data_master = None;
                let mut best_duration_ns = 0_u64;
                for draw_index in draw_indices.iter().copied() {
                    let Ok(draw_index_u32) = u32::try_from(draw_index) else {
                        continue;
                    };
                    let Some((duration_ns, data_master)) =
                        (unsafe { best_duration_for_draw(source.object, draw_index_u32) })
                    else {
                        continue;
                    };
                    if duration_ns > best_duration_ns {
                        best_draw_index = Some(draw_index_u32);
                        best_data_master = Some(data_master);
                        best_duration_ns = duration_ns;
                    }
                    if !seen.insert((source.source, pipeline.index, draw_index_u32)) {
                        continue;
                    }
                    if let Some(command) = command_for_enumerated_draw(gpu_commands, draw_index) {
                        rows.push(XcodeMioGpuCommandFunctionTime {
                            source: source.source,
                            command_index: command.index,
                            function_index: command.function_index,
                            sub_command_index: command.sub_command_index,
                            encoder_index: command.encoder_index,
                            pipeline_index: command.pipeline_index,
                            function_name: command.function_name.clone(),
                            draw_index: draw_index_u32,
                            data_master,
                            duration_ns,
                        });
                    } else {
                        rows.push(XcodeMioGpuCommandFunctionTime {
                            source: source.source,
                            command_index: usize::MAX,
                            function_index: draw_index,
                            sub_command_index: -1,
                            encoder_index: usize::MAX,
                            pipeline_index: pipeline.index,
                            function_name: pipeline.function_name.clone(),
                            draw_index: draw_index_u32,
                            data_master,
                            duration_ns,
                        });
                    }
                }
                probes.push(XcodeMioFunctionTimeProbe {
                    source: source.source,
                    target_kind: "pipeline",
                    target_id_kind,
                    target_id: pipeline_state_id,
                    pipeline_index: Some(pipeline.index),
                    encoder_index: None,
                    function_name: pipeline.function_name.clone(),
                    reported_draw_count: draw_count,
                    enumerated_draw_count: draw_indices.len(),
                    sampled_draws: draw_indices
                        .iter()
                        .filter_map(|value| u32::try_from(*value).ok())
                        .take(8)
                        .collect(),
                    best_draw_index,
                    best_data_master,
                    best_duration_ns,
                    kick_duration_ns: 0,
                });
            }
        }
    }

    unsafe fn push_enumerated_encoder_function_time_source(
        rows: &mut Vec<XcodeMioGpuCommandFunctionTime>,
        probes: &mut Vec<XcodeMioFunctionTimeProbe>,
        source: FunctionTimeTraceDataSource,
        encoders: &[XcodeMioEncoder],
        gpu_commands: &[XcodeMioGpuCommand],
    ) {
        let mut seen = BTreeSet::new();
        for encoder in encoders {
            for (target_id_kind, encoder_function_index) in
                encoder_function_time_candidates(encoder)
            {
                let Ok(encoder_function_index_u32) = u32::try_from(encoder_function_index) else {
                    continue;
                };
                let draw_count =
                    if unsafe { responds_to_selector(source.object, "numDrawsForEncoder:") } {
                        unsafe {
                            send_u64_u32(
                                source.object,
                                "numDrawsForEncoder:",
                                encoder_function_index_u32,
                            )
                        }
                        .unwrap_or(0)
                    } else {
                        0
                    };
                let draw_indices = if unsafe {
                    responds_to_selector(source.object, "enumerateDrawsForEncoder:enumerator:")
                } {
                    unsafe {
                        enumerate_draw_indices_for_encoder(
                            source.object,
                            encoder_function_index_u32,
                        )
                    }
                } else {
                    Vec::new()
                };
                let kick_duration_ns = unsafe {
                    best_kick_duration_for_encoder(source.object, encoder_function_index_u32)
                };
                let mut best_draw_index = None;
                let mut best_data_master = None;
                let mut best_duration_ns = 0_u64;
                for (draw_offset, draw_index) in draw_indices.iter().copied().enumerate() {
                    let Ok(draw_index_u32) = u32::try_from(draw_index) else {
                        continue;
                    };
                    let Some((duration_ns, data_master)) =
                        (unsafe { best_duration_for_draw(source.object, draw_index_u32) })
                    else {
                        continue;
                    };
                    if duration_ns > best_duration_ns {
                        best_draw_index = Some(draw_index_u32);
                        best_data_master = Some(data_master);
                        best_duration_ns = duration_ns;
                    }
                    if !seen.insert((source.source, encoder.index, draw_index_u32)) {
                        continue;
                    }
                    let command =
                        command_for_enumerated_draw(gpu_commands, draw_index).or_else(|| {
                            gpu_commands.get(encoder.gpu_command_start_index + draw_offset)
                        });
                    if let Some(command) = command {
                        rows.push(XcodeMioGpuCommandFunctionTime {
                            source: source.source,
                            command_index: command.index,
                            function_index: command.function_index,
                            sub_command_index: command.sub_command_index,
                            encoder_index: command.encoder_index,
                            pipeline_index: command.pipeline_index,
                            function_name: command.function_name.clone(),
                            draw_index: draw_index_u32,
                            data_master,
                            duration_ns,
                        });
                    } else {
                        rows.push(XcodeMioGpuCommandFunctionTime {
                            source: source.source,
                            command_index: usize::MAX,
                            function_index: draw_index,
                            sub_command_index: -1,
                            encoder_index: encoder.index,
                            pipeline_index: usize::MAX,
                            function_name: None,
                            draw_index: draw_index_u32,
                            data_master,
                            duration_ns,
                        });
                    }
                }
                probes.push(XcodeMioFunctionTimeProbe {
                    source: source.source,
                    target_kind: "encoder",
                    target_id_kind,
                    target_id: encoder_function_index,
                    pipeline_index: None,
                    encoder_index: Some(encoder.index),
                    function_name: None,
                    reported_draw_count: draw_count,
                    enumerated_draw_count: draw_indices.len(),
                    sampled_draws: draw_indices
                        .iter()
                        .filter_map(|value| u32::try_from(*value).ok())
                        .take(8)
                        .collect(),
                    best_draw_index,
                    best_data_master,
                    best_duration_ns,
                    kick_duration_ns,
                });
            }
        }
    }

    unsafe fn init_encoder_quad_for_encoder(
        quad_class: Class,
        trace_data: Id,
        encoder_function_index: u32,
        program_type: u16,
        options: u64,
    ) -> Result<Id> {
        unsafe {
            let quad_data = send_id(quad_class, "alloc")?;
            let quad_data = send_id_id_u32_u16_u64(
                quad_data,
                "initWithTraceData:encoderFunctionIndex:programType:options:",
                trace_data,
                encoder_function_index,
                program_type,
                options,
            )?;
            build_encoder_quad_data(quad_data, trace_data, encoder_function_index);
            Ok(quad_data)
        }
    }

    unsafe fn init_encoder_quad_for_pipeline(
        quad_class: Class,
        trace_data: Id,
        encoder_function_index: u32,
        pipeline_id: u64,
        program_type: u16,
        options: u64,
    ) -> Result<Id> {
        unsafe {
            let quad_data = send_id(quad_class, "alloc")?;
            let quad_data = send_id_id_u32_u64_u16_u64(
                quad_data,
                "initWithTraceData:encoderFunctionIndex:pipelineStateId:programType:options:",
                trace_data,
                encoder_function_index,
                pipeline_id,
                program_type,
                options,
            )?;
            build_encoder_quad_data(quad_data, trace_data, encoder_function_index);
            Ok(quad_data)
        }
    }

    unsafe fn init_encoder_quad_for_draw(
        quad_class: Class,
        trace_data: Id,
        encoder_function_index: u32,
        draw_index: u32,
        program_type: u16,
        options: u64,
    ) -> Result<Id> {
        unsafe {
            let quad_data = send_id(quad_class, "alloc")?;
            let quad_data = send_id_id_u32_u32_u16_u64(
                quad_data,
                "initWithTraceData:encoderFunctionIndex:drawIndex:programType:options:",
                trace_data,
                encoder_function_index,
                draw_index,
                program_type,
                options,
            )?;
            build_encoder_quad_data(quad_data, trace_data, encoder_function_index);
            Ok(quad_data)
        }
    }

    unsafe fn build_encoder_quad_data(quad_data: Id, trace_data: Id, encoder_function_index: u32) {
        if unsafe { responds_to_selector(quad_data, "build:encoderFunctionIndex:cliqueFilter:") } {
            let _ = unsafe {
                send_i8_id_u32_id(
                    quad_data,
                    "build:encoderFunctionIndex:cliqueFilter:",
                    trace_data,
                    encoder_function_index,
                    std::ptr::null_mut(),
                )
            };
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn push_encoder_quad_probe(
        probes: &mut Vec<XcodeMioEncoderQuadProbe>,
        quad_data: Id,
        source: &'static str,
        mode: &'static str,
        encoder_index: Option<usize>,
        encoder_function_index: u32,
        pipeline_index: Option<usize>,
        pipeline_id_kind: Option<&'static str>,
        pipeline_id: Option<u64>,
        draw_id_kind: Option<&'static str>,
        draw_index: Option<u32>,
        function_name: Option<String>,
        program_type: u16,
        options: u64,
    ) {
        unsafe {
            let draw_count = send_u64_if_supported(quad_data, "drawCount");
            let quad_count = send_u64_if_supported(quad_data, "quadCount");
            let min_timestamp_ns = send_u64_if_supported(quad_data, "minTimestamp");
            let max_timestamp_ns = send_u64_if_supported(quad_data, "maxTimestamp");
            let duration_ns = max_timestamp_ns.saturating_sub(min_timestamp_ns);
            let max_cost = send_f64_if_supported(quad_data, "maxCost");
            let min_cost = send_f64_if_supported(quad_data, "minCost");
            let sampled_draws = encoder_quad_draw_samples(quad_data, draw_count, 8);
            if draw_count == 0
                && quad_count == 0
                && duration_ns == 0
                && max_timestamp_ns == 0
                && min_timestamp_ns == 0
                && (!max_cost.is_finite() || max_cost == 0.0)
                && sampled_draws.is_empty()
            {
                return;
            }
            probes.push(XcodeMioEncoderQuadProbe {
                source,
                mode,
                encoder_index,
                encoder_function_index,
                pipeline_index,
                pipeline_id_kind,
                pipeline_id,
                draw_id_kind,
                draw_index,
                function_name,
                program_type,
                options,
                draw_count,
                quad_count,
                min_timestamp_ns,
                max_timestamp_ns,
                duration_ns,
                max_cost,
                min_cost,
                sampled_draws,
            });
        }
    }

    unsafe fn encoder_quad_draw_samples(quad_data: Id, draw_count: u64, limit: usize) -> Vec<u32> {
        if draw_count == 0 || limit == 0 {
            return Vec::new();
        }
        if !unsafe { responds_to_selector(quad_data, "drawIndexes") } {
            return Vec::new();
        }
        let Ok(draw_indexes) = (unsafe { send_ptr(quad_data, "drawIndexes") }) else {
            return Vec::new();
        };
        if draw_indexes.is_null() {
            return Vec::new();
        }
        let count = (draw_count as usize).min(10_000);
        let values = unsafe { std::slice::from_raw_parts(draw_indexes.cast::<u32>(), count) };
        values.iter().copied().take(limit).collect()
    }

    fn encoder_quad_program_type_candidates(pipeline: Option<&XcodeMioPipeline>) -> Vec<u16> {
        let mut candidates = BTreeSet::from([0_u16, 6_u16, 8_u16, 28_u16]);
        if let Some(pipeline) = pipeline {
            candidates.extend(program_type_candidates(pipeline));
        }
        candidates.into_iter().collect()
    }

    fn encoder_quad_options_candidates() -> impl Iterator<Item = u64> {
        0_u64..=15
    }

    fn sample_draw_indices(draw_indices: &[u64], limit: usize) -> Vec<u32> {
        draw_indices
            .iter()
            .filter_map(|value| u32::try_from(*value).ok())
            .take(limit)
            .collect()
    }

    fn selected_draw_indices(draw_indices: &[u64], limit: usize) -> Vec<u64> {
        if draw_indices.len() <= limit {
            return draw_indices.to_vec();
        }
        let front = limit / 2;
        let back = limit.saturating_sub(front);
        let mut values = draw_indices.iter().copied().take(front).collect::<Vec<_>>();
        values.extend(draw_indices.iter().copied().rev().take(back));
        values.sort_unstable();
        values.dedup();
        values
    }

    fn draw_array_probe_command_indices(
        pipelines: &[XcodeMioPipeline],
        gpu_commands: &[XcodeMioGpuCommand],
    ) -> Vec<usize> {
        let mut values = BTreeSet::new();
        for command in gpu_commands.iter().take(16) {
            values.insert(command.index);
        }
        for command in gpu_commands.iter().rev().take(16) {
            values.insert(command.index);
        }

        let mut by_pipeline = BTreeMap::<usize, Vec<usize>>::new();
        for command in gpu_commands {
            by_pipeline
                .entry(command.pipeline_index)
                .or_default()
                .push(command.index);
        }
        for (pipeline_index, mut command_indices) in by_pipeline {
            command_indices.sort_unstable();
            command_indices.dedup();
            let pipeline_command_count = pipelines
                .get(pipeline_index)
                .map(|pipeline| pipeline.gpu_command_count)
                .unwrap_or(command_indices.len());
            if pipeline_command_count <= 2 || command_indices.len() <= 6 {
                values.extend(command_indices);
                continue;
            }
            values.extend(command_indices.iter().copied().take(3));
            values.extend(command_indices.iter().copied().rev().take(3));
        }

        values.into_iter().collect()
    }

    unsafe fn new_shader_execution_history(
        history_class: Class,
        trace_data: Id,
        style: u32,
        options: u32,
        delegate: Id,
    ) -> Result<Id> {
        unsafe {
            let history = send_id(history_class, "alloc")?;
            send_id_id_u32_u32_id(
                history,
                "initWithTraceData:style:options:delegate:",
                trace_data,
                style,
                options,
                delegate,
            )
        }
    }

    fn draw_execution_history_probe_indices(
        pipelines: &[XcodeMioPipeline],
        gpu_commands: &[XcodeMioGpuCommand],
    ) -> Vec<u32> {
        let mut values = BTreeSet::new();
        for command in gpu_commands.iter().take(16) {
            if let Ok(index) = u32::try_from(command.index) {
                values.insert(index);
            }
        }
        for command in gpu_commands.iter().rev().take(16) {
            if let Ok(index) = u32::try_from(command.index) {
                values.insert(index);
            }
        }
        for command in gpu_commands {
            if pipelines
                .get(command.pipeline_index)
                .is_some_and(|pipeline| pipeline.gpu_command_count <= 2)
                && let Ok(index) = u32::try_from(command.index)
            {
                values.insert(index);
            }
        }
        values.into_iter().collect()
    }

    fn draw_execution_history_draw_index_candidates(
        command: &XcodeMioGpuCommand,
    ) -> Vec<(&'static str, u32)> {
        let mut values = Vec::new();
        let mut seen = BTreeSet::new();
        if let Ok(index) = u32::try_from(command.index)
            && seen.insert(index)
        {
            values.push(("command_index", index));
        }
        if let Ok(function_index) = u32::try_from(command.function_index)
            && seen.insert(function_index)
        {
            values.push(("function_index", function_index));
        }
        values
    }

    fn draw_execution_history_style_candidates() -> impl Iterator<Item = u32> {
        [0_u32, 1, 2, 4].into_iter()
    }

    fn draw_execution_history_options_candidates() -> impl Iterator<Item = u32> {
        [0_u32, 1, 2, 4, 8, 15].into_iter()
    }

    unsafe fn push_draw_execution_history_nodes(
        probes: &mut Vec<XcodeMioDrawExecutionHistoryProbe>,
        history: Id,
        source: &'static str,
        mode: &'static str,
        command: &XcodeMioGpuCommand,
        draw_index: u32,
        style: u32,
        options: u32,
        program_type: u16,
        generated: bool,
    ) {
        unsafe {
            if let Ok(node) = send_id_u32_allow_nil(history, "nodeForStyle:", style)
                && !node.is_null()
            {
                push_draw_execution_history_node(
                    probes,
                    node,
                    source,
                    mode,
                    "nodeForStyle",
                    command,
                    draw_index,
                    style,
                    options,
                    program_type,
                    generated,
                    false,
                );
                push_draw_execution_history_node(
                    probes,
                    node,
                    source,
                    mode,
                    "nodeForStyle.best",
                    command,
                    draw_index,
                    style,
                    options,
                    program_type,
                    generated,
                    true,
                );
            }
            for (selector, node_source) in [
                ("callStack", "callStack"),
                ("compact", "compact"),
                ("full", "full"),
            ] {
                if !responds_to_selector(history, selector) {
                    continue;
                }
                let Ok(node) = send_id_allow_nil(history, selector) else {
                    continue;
                };
                if node.is_null() {
                    continue;
                }
                push_draw_execution_history_node(
                    probes,
                    node,
                    source,
                    mode,
                    node_source,
                    command,
                    draw_index,
                    style,
                    options,
                    program_type,
                    generated,
                    false,
                );
                push_draw_execution_history_node(
                    probes,
                    node,
                    source,
                    mode,
                    match node_source {
                        "callStack" => "callStack.best",
                        "compact" => "compact.best",
                        "full" => "full.best",
                        _ => "best",
                    },
                    command,
                    draw_index,
                    style,
                    options,
                    program_type,
                    generated,
                    true,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn push_draw_execution_history_node(
        probes: &mut Vec<XcodeMioDrawExecutionHistoryProbe>,
        node: Id,
        source: &'static str,
        mode: &'static str,
        node_source: &'static str,
        command: &XcodeMioGpuCommand,
        draw_index: u32,
        style: u32,
        options: u32,
        program_type: u16,
        generated: bool,
        best_descendant: bool,
    ) {
        let decoded = if best_descendant {
            unsafe { decode_execution_history_node(node) }
        } else {
            unsafe { decode_execution_history_node_self(node) }
        };
        if decoded.top_cost_percentage == 0.0
            && decoded.duration_percentage == 0.0
            && decoded.total_duration_ns == 0
            && decoded.total_cost == 0.0
            && decoded.instruction_count == 0
        {
            return;
        }
        probes.push(XcodeMioDrawExecutionHistoryProbe {
            source,
            mode,
            node_source,
            command_index: command.index,
            draw_index,
            pipeline_index: command.pipeline_index,
            function_name: command.function_name.clone(),
            style,
            options,
            program_type,
            generated,
            top_cost_percentage: decoded.top_cost_percentage,
            duration_percentage: decoded.duration_percentage,
            total_duration_ns: decoded.total_duration_ns,
            total_cost: decoded.total_cost,
            instruction_count: decoded.instruction_count,
        });
    }

    unsafe fn enumerate_draw_indices_for_pipeline(
        trace_data: Id,
        pipeline_state_id: u64,
    ) -> Vec<u64> {
        let values = Arc::new(Mutex::new(Vec::<u64>::new()));
        let callback_values = Arc::clone(&values);
        let block = RcBlock::new(move |draw_index: u64| {
            if let Ok(mut values) = callback_values.lock() {
                values.push(draw_index);
            }
        });
        let block_ptr = RcBlock::as_ptr(&block).cast::<c_void>();
        if unsafe {
            send_void_u64_id(
                trace_data,
                "enumerateDrawsForPipelineState:enumerator:",
                pipeline_state_id,
                block_ptr.cast(),
            )
        }
        .is_err()
        {
            return Vec::new();
        }
        let mut values = values
            .lock()
            .map(|values| values.clone())
            .unwrap_or_default();
        values.sort_unstable();
        values.dedup();
        values
    }

    fn pipeline_function_time_candidates(
        pipeline: &XcodeMioPipeline,
        sources: EnumFunctionTimeSources,
        timeline_pipeline_state_ids: &[u64],
    ) -> Vec<(&'static str, u64)> {
        let mut candidates = vec![
            (sources.object_id, pipeline.object_id),
            (sources.pointer_id, pipeline.pointer_id),
            (sources.function_index, pipeline.function_index),
            (sources.pipeline_index, pipeline.index as u64),
        ];
        let raw_ids = pipeline
            .shader_binary_references
            .iter()
            .filter(|reference| reference.raw5 == 6 && reference.raw6 == 28)
            .map(|reference| reference.raw1 as u64)
            .collect::<BTreeSet<_>>();
        candidates.extend(raw_ids.into_iter().map(|raw1| ("shader_binary_raw1", raw1)));
        if let Some(pipeline_state_id) = timeline_pipeline_state_ids.get(pipeline.index) {
            candidates.push(("timeline_pipeline_state_id", *pipeline_state_id));
        }
        let mut seen = BTreeSet::new();
        candidates
            .into_iter()
            .filter(|candidate| seen.insert(candidate.1))
            .collect()
    }

    fn encoder_function_time_candidates(encoder: &XcodeMioEncoder) -> Vec<(&'static str, u64)> {
        let candidates = [
            ("encoder.index", encoder.index as u64),
            ("encoder.function_index", encoder.function_index),
            (
                "encoder.gpu_command_start_index",
                encoder.gpu_command_start_index as u64,
            ),
        ];
        let mut seen = BTreeSet::new();
        candidates
            .into_iter()
            .filter(|candidate| seen.insert(candidate.1))
            .collect()
    }

    unsafe fn enumerate_draw_indices_for_encoder(
        trace_data: Id,
        encoder_function_index: u32,
    ) -> Vec<u64> {
        let values = Arc::new(Mutex::new(Vec::<u64>::new()));
        let callback_values = Arc::clone(&values);
        let block = RcBlock::new(move |draw_index: u64| {
            if let Ok(mut values) = callback_values.lock() {
                values.push(draw_index);
            }
        });
        let block_ptr = RcBlock::as_ptr(&block).cast::<c_void>();
        if unsafe {
            send_void_u32_id(
                trace_data,
                "enumerateDrawsForEncoder:enumerator:",
                encoder_function_index,
                block_ptr.cast(),
            )
        }
        .is_err()
        {
            return Vec::new();
        }
        let values = values
            .lock()
            .map(|values| values.clone())
            .unwrap_or_default();
        let mut seen = BTreeSet::new();
        values
            .into_iter()
            .filter(|value| seen.insert(*value))
            .collect()
    }

    fn command_for_enumerated_draw(
        gpu_commands: &[XcodeMioGpuCommand],
        draw_index: u64,
    ) -> Option<&XcodeMioGpuCommand> {
        gpu_commands
            .iter()
            .find(|command| command.function_index == draw_index)
            .or_else(|| {
                usize::try_from(draw_index).ok().and_then(|index| {
                    gpu_commands
                        .get(index)
                        .filter(|command| command.index == index)
                })
            })
    }

    fn command_for_top_draw_track(
        first_index: u64,
        trace: Option<RawGtmioDrawTrace>,
        gpu_commands: &[XcodeMioGpuCommand],
    ) -> Option<&XcodeMioGpuCommand> {
        trace
            .and_then(|trace| command_for_enumerated_draw(gpu_commands, trace.raw2 as u64))
            .or_else(|| command_for_enumerated_draw(gpu_commands, first_index))
    }

    #[allow(clippy::too_many_arguments)]
    fn top_draw_track_row(
        source: &'static str,
        track_index: usize,
        trace_index: usize,
        track_id: i32,
        first_index: u64,
        start_timestamp_ns: u64,
        end_timestamp_ns: u64,
        duration_ns: u64,
        trace_count: u64,
        trace: RawGtmioDrawTrace,
        command: Option<&XcodeMioGpuCommand>,
    ) -> XcodeMioTopDrawTrack {
        XcodeMioTopDrawTrack {
            source,
            track_index,
            trace_index,
            track_id,
            first_index,
            start_timestamp_ns,
            end_timestamp_ns,
            duration_ns,
            trace_count,
            trace_raw0: trace.raw0,
            trace_raw1: trace.raw1,
            trace_raw2: trace.raw2,
            trace_raw3: trace.raw3,
            trace_duration_ns: trace.raw1.saturating_sub(trace.raw0),
            command_index: command.map(|command| command.index),
            function_index: command.map(|command| command.function_index),
            sub_command_index: command.map(|command| command.sub_command_index),
            encoder_index: command.map(|command| command.encoder_index),
            pipeline_index: command.map(|command| command.pipeline_index),
            function_name: command.and_then(|command| command.function_name.clone()),
        }
    }

    unsafe fn best_duration_for_draw(trace_data: Id, draw_index: u32) -> Option<(u64, u16)> {
        let mut best_duration_ns = 0_u64;
        let mut best_data_master = 0_u16;
        for data_master in function_time_data_master_candidates() {
            let duration_ns = unsafe {
                send_u64_u32_u16(
                    trace_data,
                    "durationForDraw:dataMaster:",
                    draw_index,
                    data_master,
                )
            }
            .unwrap_or(0);
            if duration_ns > best_duration_ns {
                best_duration_ns = duration_ns;
                best_data_master = data_master;
            }
        }
        (best_duration_ns > 0).then_some((best_duration_ns, best_data_master))
    }

    unsafe fn best_kick_duration_for_encoder(trace_data: Id, encoder_function_index: u32) -> u64 {
        let mut best = 0_u64;
        if unsafe { responds_to_selector(trace_data, "kickDurationForEncoder:") } {
            best = best.max(
                unsafe {
                    send_u64_u32(
                        trace_data,
                        "kickDurationForEncoder:",
                        encoder_function_index,
                    )
                }
                .unwrap_or(0),
            );
        }
        if unsafe { responds_to_selector(trace_data, "kickDurationForEncoder:dataMaster:") } {
            for data_master in function_time_data_master_candidates() {
                best = best.max(
                    unsafe {
                        send_u64_u32_u16(
                            trace_data,
                            "kickDurationForEncoder:dataMaster:",
                            encoder_function_index,
                            data_master,
                        )
                    }
                    .unwrap_or(0),
                );
            }
        }
        best
    }

    fn function_time_data_master_candidates() -> impl Iterator<Item = u16> {
        [
            0_u16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 28,
        ]
        .into_iter()
    }

    unsafe fn draw_duration(timeline: Id, draw_index: u32, metadata_data_master: u16) -> u64 {
        let mut best = 0_u64;
        let mut candidates = BTreeSet::from([metadata_data_master, 0, 1, 2, 8, 28]);
        candidates.extend(0_u16..=16);
        for data_master in candidates {
            let duration = unsafe {
                send_u64_u32_u16(
                    timeline,
                    "durationForDraw:dataMaster:",
                    draw_index,
                    data_master,
                )
            }
            .unwrap_or(0);
            best = best.max(duration);
        }
        best
    }

    const _: () = assert!(std::mem::size_of::<RawGtmioCostContext>() == 16);
    const _: () = assert!(std::mem::size_of::<RawGtmioCostInfo>() == 304);
    const _: () = assert!(std::mem::size_of::<RawGtStatistics>() == 24);
    const _: () = assert!(std::mem::size_of::<RawGtShaderProfilerTiming>() == 80);
    const _: () = assert!(std::mem::size_of::<RawGtmioDrawTrace>() == 24);
    const _: () = assert!(std::mem::size_of::<RawGtmioBinaryTrace>() == 40);
    const _: () = assert!(std::mem::size_of::<RawGtmioUSCCliqueMetadata>() == 80);

    struct FdSilencer {
        stdout_fd: c_int,
        stderr_fd: c_int,
    }

    impl FdSilencer {
        fn new() -> Self {
            unsafe {
                let dev_null = c"/dev/null";
                let null_fd = open(dev_null.as_ptr(), 1);
                if null_fd < 0 {
                    return Self {
                        stdout_fd: -1,
                        stderr_fd: -1,
                    };
                }
                let stdout_fd = dup(1);
                let stderr_fd = dup(2);
                if stdout_fd >= 0 {
                    let _ = dup2(null_fd, 1);
                }
                if stderr_fd >= 0 {
                    let _ = dup2(null_fd, 2);
                }
                let _ = close(null_fd);
                Self {
                    stdout_fd,
                    stderr_fd,
                }
            }
        }
    }

    impl Drop for FdSilencer {
        fn drop(&mut self) {
            unsafe {
                if self.stdout_fd >= 0 {
                    let _ = dup2(self.stdout_fd, 1);
                    let _ = close(self.stdout_fd);
                }
                if self.stderr_fd >= 0 {
                    let _ = dup2(self.stderr_fd, 2);
                    let _ = close(self.stderr_fd);
                }
            }
        }
    }

    impl Drop for Runtime {
        fn drop(&mut self) {
            if !self.pool.is_null() {
                unsafe {
                    let _ = send_void(self.pool, "drain");
                }
            }
        }
    }

    struct CfString {
        ptr: CfStringRef,
    }

    impl CfString {
        unsafe fn new(value: *const c_char) -> Result<Self> {
            let ptr = unsafe {
                CFStringCreateWithCString(std::ptr::null(), value, K_CF_STRING_ENCODING_UTF8)
            };
            if ptr.is_null() {
                Err(Error::InvalidInput("failed to create CFString".to_owned()))
            } else {
                Ok(Self { ptr })
            }
        }
    }

    impl Drop for CfString {
        fn drop(&mut self) {
            unsafe {
                CFRelease(self.ptr);
            }
        }
    }

    fn read_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_ne_bytes(bytes[offset..offset + 4].try_into().expect("u32 slice"))
    }

    fn read_i32(bytes: &[u8], offset: usize) -> i32 {
        i32::from_ne_bytes(bytes[offset..offset + 4].try_into().expect("i32 slice"))
    }

    fn read_u64(bytes: &[u8], offset: usize) -> u64 {
        u64::from_ne_bytes(bytes[offset..offset + 8].try_into().expect("u64 slice"))
    }

    fn read_u16(bytes: &[u8], offset: usize) -> u16 {
        u16::from_ne_bytes(bytes[offset..offset + 2].try_into().expect("u16 slice"))
    }

    unsafe fn decode_draw_metadata_record(
        draws: *const u8,
        index: usize,
    ) -> XcodeMioDrawMetadataRecord {
        let draw = unsafe { std::slice::from_raw_parts(draws.add(index * 44), 44) };
        XcodeMioDrawMetadataRecord {
            index,
            raw0: read_u32(draw, 0),
            raw1: read_u32(draw, 4),
            raw2: read_u32(draw, 8),
            raw3: read_u32(draw, 12),
            raw4: read_i32(draw, 16),
            raw5: read_u32(draw, 20),
            raw6: read_u64(draw, 24),
            raw7: read_u32(draw, 32),
            raw8: read_u32(draw, 36),
            raw9: read_u32(draw, 40),
        }
    }

    unsafe fn nsstring_to_string(value: Id) -> Option<String> {
        let bytes = unsafe { send_ptr(value, "UTF8String").ok()? };
        if bytes.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(bytes.cast::<c_char>()) }
            .to_str()
            .ok()
            .map(ToOwned::to_owned)
    }

    unsafe fn object_ivar(object: Id, ivar_name: &str) -> Option<Id> {
        if object.is_null() {
            return None;
        }
        let class = unsafe { object_getClass(object) };
        if class.is_null() {
            return None;
        }
        let name = CString::new(ivar_name).ok()?;
        let ivar = unsafe { class_getInstanceVariable(class, name.as_ptr()) };
        if ivar.is_null() || !unsafe { ivar_is_object(ivar) } {
            return None;
        }
        let value = unsafe { object_getIvar(object, ivar) };
        (!value.is_null()).then_some(value)
    }

    unsafe fn object_class_name(object: Id) -> Option<String> {
        if object.is_null() {
            return None;
        }
        let class = unsafe { object_getClass(object) };
        if class.is_null() {
            return None;
        }
        let name = unsafe { class_getName(class) };
        if name.is_null() {
            return None;
        }
        unsafe { CStr::from_ptr(name) }
            .to_str()
            .ok()
            .map(ToOwned::to_owned)
    }

    unsafe fn object_ivar_assume_object(object: Id, ivar_name: &str) -> Option<Id> {
        if object.is_null() {
            return None;
        }
        let class = unsafe { object_getClass(object) };
        if class.is_null() {
            return None;
        }
        let name = CString::new(ivar_name).ok()?;
        let ivar = unsafe { class_getInstanceVariable(class, name.as_ptr()) };
        if ivar.is_null() {
            return None;
        }
        let value = unsafe { object_getIvar(object, ivar) };
        (!value.is_null()).then_some(value)
    }

    unsafe fn ivar_is_object(ivar: Ivar) -> bool {
        let encoding = unsafe { ivar_getTypeEncoding(ivar) };
        if encoding.is_null() {
            return false;
        }
        unsafe { CStr::from_ptr(encoding).to_bytes().first() == Some(&b'@') }
    }

    unsafe fn push_numeric_array_probe(
        out: &mut Vec<XcodeMioPrivateNumericArray>,
        receiver: Id,
        selector_or_ivar: &str,
        source: &'static str,
    ) {
        let value = if selector_or_ivar.starts_with('_') {
            unsafe { object_ivar(receiver, selector_or_ivar) }
        } else if unsafe { responds_to_selector(receiver, selector_or_ivar) } {
            unsafe { send_id_allow_nil(receiver, selector_or_ivar).ok() }
                .filter(|value| !value.is_null())
        } else {
            None
        };
        let Some(value) = value else {
            return;
        };
        let rows = unsafe { decode_numeric_rows(value) };
        if !rows.is_empty() {
            out.push(XcodeMioPrivateNumericArray { source, rows });
        }
    }

    unsafe fn push_parent_processor_numeric_probes(
        out: &mut Vec<XcodeMioPrivateNumericArray>,
        processor: Id,
    ) {
        let Some(shader_processor) =
            (unsafe { object_ivar_assume_object(processor, "_shaderProfilerProcessor") })
        else {
            return;
        };
        unsafe {
            push_numeric_ivar_probe(
                out,
                shader_processor,
                "_effectivePerEncoderDrawKickTimes",
                "stream_processor._shaderProfilerProcessor._effectivePerEncoderDrawKickTimes",
            );
            push_numeric_ivar_probe(
                out,
                shader_processor,
                "_shaderProfilerFrameTimes",
                "stream_processor._shaderProfilerProcessor._shaderProfilerFrameTimes",
            );
            push_numeric_array_probe(
                out,
                shader_processor,
                "effectivePerEncoderDrawKickTimes",
                "stream_processor._shaderProfilerProcessor.effectivePerEncoderDrawKickTimes",
            );
        }
        if let Some(shader_profiler) =
            unsafe { object_ivar_assume_object(shader_processor, "_shaderProfiler") }
        {
            unsafe {
                push_numeric_array_probe(
                    out,
                    shader_profiler,
                    "effectiveKickTimes",
                    "stream_processor._shaderProfilerProcessor._shaderProfiler.effectiveKickTimes",
                );
                push_numeric_array_probe(
                    out,
                    shader_profiler,
                    "averagePerDrawKickDurations",
                    "stream_processor._shaderProfilerProcessor._shaderProfiler.averagePerDrawKickDurations",
                );
            }
        }
    }

    unsafe fn push_numeric_ivar_probe(
        out: &mut Vec<XcodeMioPrivateNumericArray>,
        receiver: Id,
        ivar_name: &str,
        source: &'static str,
    ) {
        let Some(value) = (unsafe { object_ivar_assume_object(receiver, ivar_name) }) else {
            return;
        };
        let rows = unsafe { decode_numeric_rows(value) };
        if !rows.is_empty() {
            out.push(XcodeMioPrivateNumericArray { source, rows });
        }
    }

    unsafe fn decode_numeric_rows(value: Id) -> Vec<Vec<f64>> {
        unsafe { decode_numeric_rows_at(value, 0) }
    }

    unsafe fn decode_numeric_rows_at(value: Id, depth: usize) -> Vec<Vec<f64>> {
        if value.is_null() || depth > 8 {
            return Vec::new();
        }
        if let Some(value) = unsafe { objc_number_value(value) } {
            return vec![vec![value]];
        }
        if unsafe { responds_to_selector(value, "allValues") } {
            if let Ok(values) = unsafe { send_id_allow_nil(value, "allValues") }
                && !values.is_null()
            {
                let rows = unsafe { decode_numeric_rows_at(values, depth + 1) };
                if !rows.is_empty() {
                    return rows;
                }
            }
        }
        if !unsafe { responds_to_selector(value, "count") }
            || !unsafe { responds_to_selector(value, "objectAtIndex:") }
        {
            return Vec::new();
        }
        let count = unsafe { send_u64(value, "count").unwrap_or(0) as usize }.min(20_000);
        let mut rows = Vec::new();
        let mut scalar_row = Vec::new();
        for index in 0..count {
            let Ok(item) = (unsafe { send_id_usize(value, "objectAtIndex:", index) }) else {
                continue;
            };
            if let Some(number) = unsafe { objc_number_value(item) } {
                scalar_row.push(number);
                continue;
            }
            let nested_rows = unsafe { decode_numeric_rows_at(item, depth + 1) };
            if nested_rows.is_empty() {
                continue;
            }
            if !scalar_row.is_empty() {
                rows.push(std::mem::take(&mut scalar_row));
            }
            rows.extend(nested_rows);
        }
        if !scalar_row.is_empty() {
            rows.push(scalar_row);
        }
        rows
    }

    unsafe fn objc_number_value(value: Id) -> Option<f64> {
        if !unsafe { is_kind_of_class(value, "NSNumber") }
            || !unsafe { responds_to_selector(value, "doubleValue") }
        {
            return None;
        }
        unsafe { send_f64(value, "doubleValue").ok() }
    }

    unsafe fn is_kind_of_class(receiver: Id, class_name: &str) -> bool {
        if receiver.is_null() {
            return false;
        }
        let Ok(class) = (unsafe { lookup_class(class_name) }) else {
            return false;
        };
        let Ok(sel) = (unsafe { selector("isKindOfClass:") }) else {
            return false;
        };
        let f: extern "C" fn(Id, Sel, Class) -> i8 =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        f(receiver, sel, class) != 0
    }

    unsafe fn load_framework(path: &str) -> Result<()> {
        let path = CString::new(path).expect("framework path contains no NUL");
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
        if handle.is_null() {
            Err(Error::NotFound(PathBuf::from(
                path.to_string_lossy().into_owned(),
            )))
        } else {
            Ok(())
        }
    }

    unsafe fn lookup_class(name: &str) -> Result<Class> {
        let name = CString::new(name).expect("class name contains no NUL");
        let class = unsafe { objc_lookUpClass(name.as_ptr()) };
        if class.is_null() {
            Err(Error::InvalidInput(format!(
                "Objective-C class {} is not available",
                name.to_string_lossy()
            )))
        } else {
            Ok(class)
        }
    }

    unsafe fn selector(name: &str) -> Result<Sel> {
        let name = CString::new(name)
            .map_err(|_| Error::InvalidInput("selector contains NUL".to_owned()))?;
        let sel = unsafe { sel_registerName(name.as_ptr()) };
        if sel.is_null() {
            Err(Error::InvalidInput(format!(
                "Objective-C selector {} is not available",
                name.to_string_lossy()
            )))
        } else {
            Ok(sel)
        }
    }

    unsafe fn responds_to_selector(receiver: Id, selector_name: &str) -> bool {
        if receiver.is_null() {
            return false;
        }
        let Ok(responds_to_selector) = (unsafe { selector("respondsToSelector:") }) else {
            return false;
        };
        let Ok(target_selector) = (unsafe { selector(selector_name) }) else {
            return false;
        };
        let f: extern "C" fn(Id, Sel, Sel) -> i8 =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        f(receiver, responds_to_selector, target_selector) != 0
    }

    unsafe fn send_id(receiver: Id, sel: &str) -> Result<Id> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel) -> Id = unsafe { mem::transmute(objc_msgSend as *const ()) };
        let value = f(receiver, sel);
        if value.is_null() {
            Err(Error::InvalidInput(
                "Objective-C message returned nil".to_owned(),
            ))
        } else {
            Ok(value)
        }
    }

    unsafe fn send_id_allow_nil(receiver: Id, sel: &str) -> Result<Id> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel) -> Id = unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel))
    }

    unsafe fn send_id_id(receiver: Id, sel: &str, arg: Id) -> Result<Id> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, Id) -> Id =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        let value = f(receiver, sel, arg);
        if value.is_null() {
            Err(Error::InvalidInput(
                "Objective-C message returned nil".to_owned(),
            ))
        } else {
            Ok(value)
        }
    }

    unsafe fn send_id_id_allow_nil(receiver: Id, sel: &str, arg: Id) -> Result<Id> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, Id) -> Id =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, arg))
    }

    unsafe fn send_id_id_id(receiver: Id, sel: &str, left: Id, right: Id) -> Result<Id> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, Id, Id) -> Id =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        let value = f(receiver, sel, left, right);
        if value.is_null() {
            Err(Error::InvalidInput(
                "Objective-C message returned nil".to_owned(),
            ))
        } else {
            Ok(value)
        }
    }

    unsafe fn send_id_id_u32_u32_id(
        receiver: Id,
        sel: &str,
        first: Id,
        second: u32,
        third: u32,
        fourth: Id,
    ) -> Result<Id> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, Id, u32, u32, Id) -> Id =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        let value = f(receiver, sel, first, second, third, fourth);
        if value.is_null() {
            Err(Error::InvalidInput(
                "Objective-C message returned nil".to_owned(),
            ))
        } else {
            Ok(value)
        }
    }

    unsafe fn send_id_id_u32_u16_u64(
        receiver: Id,
        sel: &str,
        first: Id,
        second: u32,
        third: u16,
        fourth: u64,
    ) -> Result<Id> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, Id, u32, u16, u64) -> Id =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        let value = f(receiver, sel, first, second, third, fourth);
        if value.is_null() {
            Err(Error::InvalidInput(
                "Objective-C message returned nil".to_owned(),
            ))
        } else {
            Ok(value)
        }
    }

    unsafe fn send_id_id_u32_u64_u16_u64(
        receiver: Id,
        sel: &str,
        first: Id,
        second: u32,
        third: u64,
        fourth: u16,
        fifth: u64,
    ) -> Result<Id> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, Id, u32, u64, u16, u64) -> Id =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        let value = f(receiver, sel, first, second, third, fourth, fifth);
        if value.is_null() {
            Err(Error::InvalidInput(
                "Objective-C message returned nil".to_owned(),
            ))
        } else {
            Ok(value)
        }
    }

    unsafe fn send_id_id_u32_u32_u16_u64(
        receiver: Id,
        sel: &str,
        first: Id,
        second: u32,
        third: u32,
        fourth: u16,
        fifth: u64,
    ) -> Result<Id> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, Id, u32, u32, u16, u64) -> Id =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        let value = f(receiver, sel, first, second, third, fourth, fifth);
        if value.is_null() {
            Err(Error::InvalidInput(
                "Objective-C message returned nil".to_owned(),
            ))
        } else {
            Ok(value)
        }
    }

    unsafe fn send_id_u64_u16_allow_nil(
        receiver: Id,
        sel: &str,
        left: u64,
        right: u16,
    ) -> Result<Id> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u64, u16) -> Id =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, left, right))
    }

    unsafe fn send_id_u32_i32_allow_nil(
        receiver: Id,
        sel: &str,
        left: u32,
        right: i32,
    ) -> Result<Id> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u32, i32) -> Id =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, left, right))
    }

    unsafe fn send_id_u64_allow_nil(receiver: Id, sel: &str, arg: u64) -> Result<Id> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u64) -> Id =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, arg))
    }

    unsafe fn send_id_u32_allow_nil(receiver: Id, sel: &str, arg: u32) -> Result<Id> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u32) -> Id =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, arg))
    }

    unsafe fn send_id_usize(receiver: Id, sel: &str, arg: usize) -> Result<Id> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, usize) -> Id =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        let value = f(receiver, sel, arg);
        if value.is_null() {
            Err(Error::InvalidInput(
                "Objective-C message returned nil".to_owned(),
            ))
        } else {
            Ok(value)
        }
    }

    unsafe fn send_void(receiver: Id, sel: &str) -> Result<()> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel) = unsafe { mem::transmute(objc_msgSend as *const ()) };
        f(receiver, sel);
        Ok(())
    }

    unsafe fn send_void_id(receiver: Id, sel: &str, arg: Id) -> Result<()> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, Id) = unsafe { mem::transmute(objc_msgSend as *const ()) };
        f(receiver, sel, arg);
        Ok(())
    }

    unsafe fn send_void_u64_id(receiver: Id, sel: &str, first: u64, second: Id) -> Result<()> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u64, Id) =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        f(receiver, sel, first, second);
        Ok(())
    }

    unsafe fn send_void_u32_id(receiver: Id, sel: &str, first: u32, second: Id) -> Result<()> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u32, Id) =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        f(receiver, sel, first, second);
        Ok(())
    }

    unsafe fn send_void_u32_u16_id(
        receiver: Id,
        sel: &str,
        first: u32,
        second: u16,
        third: Id,
    ) -> Result<()> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u32, u16, Id) =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        f(receiver, sel, first, second, third);
        Ok(())
    }

    unsafe fn send_void_u32_u16_id_id(
        receiver: Id,
        sel: &str,
        first: u32,
        second: u16,
        third: Id,
        fourth: Id,
    ) -> Result<()> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u32, u16, Id, Id) =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        f(receiver, sel, first, second, third, fourth);
        Ok(())
    }

    unsafe fn send_void_u64_u16_id_id(
        receiver: Id,
        sel: &str,
        first: u64,
        second: u16,
        third: Id,
        fourth: Id,
    ) -> Result<()> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u64, u16, Id, Id) =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        f(receiver, sel, first, second, third, fourth);
        Ok(())
    }

    unsafe fn send_u64(receiver: Id, sel: &str) -> Result<u64> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel) -> u64 = unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel))
    }

    unsafe fn send_u64_if_supported(receiver: Id, sel: &str) -> u64 {
        if unsafe { responds_to_selector(receiver, sel) } {
            unsafe { send_u64(receiver, sel).unwrap_or(0) }
        } else {
            0
        }
    }

    unsafe fn send_u64_u64(receiver: Id, sel: &str, arg: u64) -> Result<u64> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u64) -> u64 =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, arg))
    }

    unsafe fn send_u64_u32(receiver: Id, sel: &str, arg: u32) -> Result<u64> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u32) -> u64 =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, arg))
    }

    unsafe fn send_u64_u32_u16(receiver: Id, sel: &str, first: u32, second: u16) -> Result<u64> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u32, u16) -> u64 =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, first, second))
    }

    unsafe fn send_u32(receiver: Id, sel: &str) -> Result<u32> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel) -> u32 = unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel))
    }

    unsafe fn send_u16(receiver: Id, sel: &str) -> Result<u16> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel) -> u16 = unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel))
    }

    unsafe fn send_i32(receiver: Id, sel: &str) -> Result<i32> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel) -> i32 = unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel))
    }

    unsafe fn send_i8_u64(receiver: Id, sel: &str, arg: u64) -> Result<i8> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u64) -> i8 =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, arg))
    }

    unsafe fn send_i8_u32_u16(receiver: Id, sel: &str, first: u32, second: u16) -> Result<i8> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u32, u16) -> i8 =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, first, second))
    }

    unsafe fn send_i8_id_u32_id(
        receiver: Id,
        sel: &str,
        first: Id,
        second: u32,
        third: Id,
    ) -> Result<i8> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, Id, u32, Id) -> i8 =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, first, second, third))
    }

    unsafe fn send_f64(receiver: Id, sel: &str) -> Result<f64> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel) -> f64 = unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel))
    }

    unsafe fn send_f64_if_supported(receiver: Id, sel: &str) -> f64 {
        if unsafe { responds_to_selector(receiver, sel) } {
            unsafe { send_f64(receiver, sel).unwrap_or(0.0) }
        } else {
            0.0
        }
    }

    unsafe fn send_f64_u32(receiver: Id, sel: &str, arg: u32) -> Result<f64> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u32) -> f64 =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, arg))
    }

    unsafe fn send_profiler_timing(receiver: Id, sel: &str) -> Result<RawGtShaderProfilerTiming> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel) -> RawGtShaderProfilerTiming =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel))
    }

    unsafe fn send_void_i8(receiver: Id, sel: &str, arg: i8) -> Result<()> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, i8) = unsafe { mem::transmute(objc_msgSend as *const ()) };
        f(receiver, sel, arg);
        Ok(())
    }

    unsafe fn send_ptr(receiver: Id, sel: &str) -> Result<*const c_void> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel) -> *const c_void =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel))
    }

    unsafe fn send_ptr_u64(receiver: Id, sel: &str, arg: u64) -> Result<*const c_void> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u64) -> *const c_void =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, arg))
    }

    unsafe fn send_ptr_u64_u32(
        receiver: Id,
        sel: &str,
        first: u64,
        second: u32,
    ) -> Result<*const c_void> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u64, u32) -> *const c_void =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, first, second))
    }

    unsafe fn send_ptr_u64_i32(
        receiver: Id,
        sel: &str,
        first: u64,
        second: i32,
    ) -> Result<*const c_void> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u64, i32) -> *const c_void =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, first, second))
    }

    unsafe fn send_i8_u16_u64_cost_mut(
        receiver: Id,
        sel: &str,
        first: u16,
        second: u64,
        third: *mut RawGtmioCostInfo,
    ) -> Result<i8> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u16, u64, *mut RawGtmioCostInfo) -> i8 =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, first, second, third))
    }

    unsafe fn send_f64_u16_u64_u16(
        receiver: Id,
        sel: &str,
        first: u16,
        second: u64,
        third: u16,
    ) -> Result<f64> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u16, u64, u16) -> f64 =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, first, second, third))
    }

    unsafe fn send_i8_context_cost_mut(
        receiver: Id,
        sel: &str,
        context: *const RawGtmioCostContext,
        cost: *mut RawGtmioCostInfo,
    ) -> Result<i8> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, *const RawGtmioCostContext, *mut RawGtmioCostInfo) -> i8 =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(receiver, sel, context, cost))
    }

    unsafe fn send_i8_u16_u32_u16_u64_cost_mut(
        receiver: Id,
        sel: &str,
        level: u16,
        level_identifier: u32,
        scope: u16,
        scope_identifier: u64,
        cost: *mut RawGtmioCostInfo,
    ) -> Result<i8> {
        if receiver.is_null() {
            return Err(Error::InvalidInput(format!(
                "nil Objective-C receiver for {sel}"
            )));
        }
        let sel = unsafe { selector(sel)? };
        let f: extern "C" fn(Id, Sel, u16, u32, u16, u64, *mut RawGtmioCostInfo) -> i8 =
            unsafe { mem::transmute(objc_msgSend as *const ()) };
        Ok(f(
            receiver,
            sel,
            level,
            level_identifier,
            scope,
            scope_identifier,
            cost,
        ))
    }
}

#[cfg(not(target_os = "macos"))]
mod platform {
    use std::path::PathBuf;

    use super::{XcodeMioDecodeOptions, XcodeMioReport, XcodeMioTimings};
    use crate::error::{Error, Result};
    use crate::profiler;

    pub fn decode(
        _trace_source: PathBuf,
        _profiler_directory: PathBuf,
        _stream_data_path: PathBuf,
        _profiler_summary: Option<&profiler::ProfilerStreamDataSummary>,
        _timings: XcodeMioTimings,
        _options: XcodeMioDecodeOptions,
    ) -> Result<XcodeMioReport> {
        Err(Error::Unsupported(
            "xcode-mio is only available on macOS with Xcode installed",
        ))
    }
}
