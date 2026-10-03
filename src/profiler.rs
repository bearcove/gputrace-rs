use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Cursor;
use std::mem;
use std::path::{Path, PathBuf};

use plist::{Dictionary, Uid, Value};
use serde::Serialize;

use crate::counter;
use crate::error::{Error, Result};
use crate::trace::TraceBundle;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProfilerFileEntry {
    pub name: String,
    pub size: u64,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProfilerPipeline {
    pub pipeline_id: i64,
    pub pipeline_address: u64,
    pub function_name: Option<String>,
    pub stats: Option<ProfilerPipelineStats>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProfilerPipelineStats {
    pub temporary_register_count: i64,
    pub uniform_register_count: i64,
    pub spilled_bytes: i64,
    pub threadgroup_memory: i64,
    pub instruction_count: i64,
    pub alu_instruction_count: i64,
    pub branch_instruction_count: i64,
    pub compilation_time_ms: f64,
    pub line_instruction_counts: BTreeMap<usize, u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProfilerExecutionCost {
    pub pipeline_id: i64,
    pub function_name: Option<String>,
    pub sample_count: usize,
    pub cost_percent: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProfilerDispatch {
    pub index: usize,
    pub pipeline_index: usize,
    pub pipeline_id: Option<i64>,
    pub function_name: Option<String>,
    pub encoder_index: usize,
    pub cumulative_us: u64,
    pub duration_us: u64,
    pub sample_count: usize,
    pub sampling_density: f64,
    pub start_ticks: u64,
    pub end_ticks: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProfilerEncoderTiming {
    pub index: usize,
    pub sequence_id: u64,
    /// MTL encoder address (u64 at offset 8 of `encoderInfoData`). Matches the
    /// addresses surfaced by byte-stream `CS` records — used to bridge streamData
    /// dispatches to the byte-stream encoder topology when attributing kernel
    /// dispatch counts to named encoders in `encoders.md`.
    pub address: u64,
    pub end_offset_micros: u64,
    pub duration_micros: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProfilerRawEncoderTiming {
    pub index: usize,
    pub duration_ns: u64,
    pub confidence_milli: u16,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProfilerCommandBufferTimestamp {
    pub index: usize,
    pub start_ticks: u64,
    pub end_ticks: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProfilerTimelineInfo {
    pub command_buffer_timestamps: Vec<ProfilerCommandBufferTimestamp>,
    pub encoder_profiles: Vec<ProfilerEncoderProfile>,
    pub timebase_numer: u64,
    pub timebase_denom: u64,
    pub absolute_time: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProfilerEncoderProfile {
    pub index: usize,
    pub source: String,
    pub ring_buffer_index: usize,
    pub sample_count: usize,
    pub timestamps: Vec<GprwcntrTimestamp>,
    pub start_ticks: u64,
    pub end_ticks: u64,
    pub duration_ns: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct GprwcntrTimestamp {
    pub timestamp: u64,
    pub size: u64,
    pub count: u64,
    pub flags: u32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProfilerStreamDataSummary {
    pub function_names: Vec<String>,
    pub pipelines: Vec<ProfilerPipeline>,
    pub execution_costs: Vec<ProfilerExecutionCost>,
    pub dispatches: Vec<ProfilerDispatch>,
    pub encoder_timings: Vec<ProfilerEncoderTiming>,
    pub timeline: Option<ProfilerTimelineInfo>,
    pub num_pipelines: usize,
    pub num_gpu_commands: usize,
    pub num_encoders: usize,
    pub total_time_us: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProfilerReport {
    pub input_path: PathBuf,
    pub profiler_directory: PathBuf,
    pub stream_data_present: bool,
    pub stream_data_summary: Option<ProfilerStreamDataSummary>,
    pub limiter_metrics: Vec<counter::CounterLimiter>,
    pub timeline_file_count: usize,
    pub counter_file_count: usize,
    pub profiling_file_count: usize,
    pub kdebug_file_count: usize,
    pub other_file_count: usize,
    pub total_bytes: u64,
    pub files: Vec<ProfilerFileEntry>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProfilerCoverageReport {
    pub input_path: PathBuf,
    pub profiler_directory: PathBuf,
    pub total_bytes: u64,
    pub total_files: usize,
    pub groups: Vec<ProfilerCoverageGroup>,
    pub top_opaque_files: Vec<ProfilerCoverageFile>,
    pub decoded_summary: ProfilerCoverageDecodedSummary,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ProfilerCoverageGroup {
    pub kind: String,
    pub file_count: usize,
    pub byte_count: u64,
    pub byte_percent: f64,
    pub decoder_status: String,
    pub decoded_signals: Vec<String>,
    pub remaining_work: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProfilerCoverageFile {
    pub name: String,
    pub kind: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ProfilerCoverageDecodedSummary {
    pub stream_data_present: bool,
    pub aggregate_metadata_blocks: usize,
    pub counter_info_entries: usize,
    pub counter_schemas: usize,
    pub decoded_counter_streams: usize,
    pub decoded_counter_metrics: usize,
    pub derived_metric_groups: usize,
    pub encoder_sample_metrics: usize,
    pub trace_map_entries: usize,
    pub stream_archives: usize,
    pub aps_data_archives: usize,
    pub aps_counter_data_archives: usize,
    pub aps_timeline_data_archives: usize,
    pub data_file_references: Vec<String>,
}

pub fn report<P: AsRef<Path>>(path: P) -> Result<ProfilerReport> {
    report_with_stream_data_summary(path, None)
}

pub fn report_with_stream_data_summary<P: AsRef<Path>>(
    path: P,
    precomputed_stream_data_summary: Option<ProfilerStreamDataSummary>,
) -> Result<ProfilerReport> {
    let input_path = path.as_ref().to_path_buf();
    let profiler_directory =
        find_profiler_directory(&input_path).ok_or_else(|| Error::NotFound(input_path.clone()))?;

    let mut files = Vec::new();
    let mut stream_data_present = false;
    let mut timeline_file_count = 0;
    let mut counter_file_count = 0;
    let mut profiling_file_count = 0;
    let mut kdebug_file_count = 0;
    let mut other_file_count = 0;
    let mut total_bytes = 0;

    for entry in fs::read_dir(&profiler_directory)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if !metadata.is_file() {
            continue;
        }

        let name = entry.file_name().to_string_lossy().into_owned();
        let kind = classify_file(&name);
        match kind.as_str() {
            "streamData" => stream_data_present = true,
            "timeline" => timeline_file_count += 1,
            "counter" => counter_file_count += 1,
            "profiling" => profiling_file_count += 1,
            "kdebug" => kdebug_file_count += 1,
            _ => other_file_count += 1,
        }

        total_bytes += metadata.len();
        files.push(ProfilerFileEntry {
            name,
            size: metadata.len(),
            kind,
        });
    }

    files.sort_by(|left, right| left.name.cmp(&right.name));

    let stream_data_summary = if let Some(summary) = precomputed_stream_data_summary {
        Some(summary)
    } else if stream_data_present {
        let stream_data_path = profiler_directory.join("streamData");
        Some(parse_stream_data(&stream_data_path)?)
    } else {
        None
    };
    let limiter_metrics = counter::extract_limiters(&profiler_directory);

    let mut notes = Vec::new();
    if !stream_data_present {
        notes.push(
            "streamData is missing, so dispatch-level profiler joins are unavailable.".to_owned(),
        );
    } else {
        notes.push(
            "streamData timing and dispatch summaries are real profiler data from the bundle."
                .to_owned(),
        );
    }
    notes.push(
        "Encoder limiters come from the limiter-pass hardware counters (agxps; see counters.md); pipeline-id byte scans are retained as a debug signal, not treated as Xcode Cost."
            .to_owned(),
    );

    Ok(ProfilerReport {
        input_path,
        profiler_directory,
        stream_data_present,
        stream_data_summary,
        limiter_metrics,
        timeline_file_count,
        counter_file_count,
        profiling_file_count,
        kdebug_file_count,
        other_file_count,
        total_bytes,
        files,
        notes,
    })
}

pub fn coverage_report(trace: &TraceBundle) -> Result<ProfilerCoverageReport> {
    coverage_report_with_decoded(trace, None, None)
}

pub fn coverage_report_with_decoded(
    trace: &TraceBundle,
    precomputed_raw_counters: Option<&counter::RawCountersReport>,
    precomputed_probe: Option<&counter::RawCounterProbeReport>,
) -> Result<ProfilerCoverageReport> {
    let input_path = trace.path.clone();
    let profiler_directory =
        find_profiler_directory(&input_path).ok_or_else(|| Error::NotFound(input_path.clone()))?;
    let files = profiler_files(&profiler_directory)?;
    let total_bytes = files.iter().map(|file| file.size).sum::<u64>();
    let mut warnings = Vec::new();

    let raw_counters_owned;
    let raw_counters = if let Some(report) = precomputed_raw_counters {
        Some(report)
    } else {
        raw_counters_owned = match counter::raw_counters_report(trace) {
            Ok(report) => Some(report),
            Err(error) => {
                warnings.push(format!("raw-counters decoder failed: {error}"));
                None
            }
        };
        raw_counters_owned.as_ref()
    };
    let probe_owned;
    let probe = if let Some(report) = precomputed_probe {
        Some(report)
    } else {
        probe_owned = match counter::probe_raw_counters(trace, None, None, false) {
            Ok(report) => Some(report),
            Err(error) => {
                warnings.push(format!("streamData archive probe failed: {error}"));
                None
            }
        };
        probe_owned.as_ref()
    };

    let stream_data_present = files.iter().any(|file| file.name == "streamData");
    let decoded_summary = ProfilerCoverageDecodedSummary {
        stream_data_present,
        aggregate_metadata_blocks: raw_counters
            .as_ref()
            .map_or(0, |report| report.aggregate_metadata.len()),
        counter_info_entries: raw_counters
            .as_ref()
            .map_or(0, |report| report.counter_info.len()),
        counter_schemas: raw_counters
            .as_ref()
            .map_or(0, |report| report.schemas.len()),
        decoded_counter_streams: raw_counters
            .as_ref()
            .map_or(0, |report| report.streams.len()),
        decoded_counter_metrics: raw_counters
            .as_ref()
            .map_or(0, |report| report.metrics.len()),
        derived_metric_groups: raw_counters
            .as_ref()
            .map_or(0, |report| report.grouped_derived_metrics.len()),
        encoder_sample_metrics: raw_counters
            .as_ref()
            .map_or(0, |report| report.encoder_sample_metrics.len()),
        trace_map_entries: raw_counters
            .as_ref()
            .map_or(0, |report| report.trace_maps.len()),
        stream_archives: probe.map_or(0, |report| report.stream_archives.len()),
        aps_data_archives: archive_count(probe, "APSData"),
        aps_counter_data_archives: archive_count(probe, "APSCounterData"),
        aps_timeline_data_archives: archive_count(probe, "APSTimelineData"),
        data_file_references: data_file_references(probe),
    };

    let mut groups = Vec::new();
    for kind in [
        "streamData",
        "profiling",
        "counter",
        "timeline",
        "kdebug",
        "other",
    ] {
        let matching = files
            .iter()
            .filter(|file| file.kind == kind)
            .collect::<Vec<_>>();
        if matching.is_empty() {
            continue;
        }
        let byte_count = matching.iter().map(|file| file.size).sum::<u64>();
        groups.push(coverage_group(
            kind,
            matching.len(),
            byte_count,
            total_bytes,
            &decoded_summary,
        ));
    }

    let mut top_opaque_files = files
        .iter()
        .filter(|file| {
            matches!(
                file.kind.as_str(),
                "profiling" | "counter" | "timeline" | "kdebug"
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    top_opaque_files.sort_by(|left, right| {
        right
            .size
            .cmp(&left.size)
            .then_with(|| left.name.cmp(&right.name))
    });
    top_opaque_files.truncate(12);

    if total_bytes > 0 {
        let opaque_bytes = top_opaque_kind_bytes(&files);
        if opaque_bytes > 0 {
            warnings.push(format!(
                "{} bytes ({:.2}%) are in raw file families that are not byte-complete decoded yet",
                opaque_bytes,
                percent(opaque_bytes, total_bytes)
            ));
        }
    }

    Ok(ProfilerCoverageReport {
        input_path,
        profiler_directory,
        total_bytes,
        total_files: files.len(),
        groups,
        top_opaque_files,
        decoded_summary,
        warnings,
    })
}

pub fn format_coverage_report(report: &ProfilerCoverageReport) -> String {
    let mut out = String::new();
    out.push_str("GPU Profiler Coverage\n");
    out.push_str("=====================\n");
    out.push_str(&format!("input={}\n", report.input_path.display()));
    out.push_str(&format!(
        "profiler_directory={}\n",
        report.profiler_directory.display()
    ));
    out.push_str(&format!(
        "files={} total_bytes={}\n\n",
        report.total_files, report.total_bytes
    ));

    out.push_str("byte coverage by family\n");
    out.push_str("-----------------------\n");
    for group in &report.groups {
        out.push_str(&format!(
            "  {:<10} files={:<3} bytes={:<12} {:>6.2}% status={}\n",
            group.kind,
            group.file_count,
            group.byte_count,
            group.byte_percent,
            group.decoder_status
        ));
        if !group.decoded_signals.is_empty() {
            out.push_str(&format!(
                "    decoded: {}\n",
                group.decoded_signals.join("; ")
            ));
        }
        if !group.remaining_work.is_empty() {
            out.push_str(&format!(
                "    remaining: {}\n",
                group.remaining_work.join("; ")
            ));
        }
    }

    out.push_str("\ndecoded streamData structures\n");
    out.push_str("-----------------------------\n");
    let summary = &report.decoded_summary;
    out.push_str(&format!(
        "present={} archives={} APSData={} APSCounterData={} APSTimelineData={}\n",
        summary.stream_data_present,
        summary.stream_archives,
        summary.aps_data_archives,
        summary.aps_counter_data_archives,
        summary.aps_timeline_data_archives
    ));
    out.push_str(&format!(
        "aggregate_metadata={} counter_info={} schemas={} streams={} metrics={} encoder_sample_metrics={} trace_map_entries={} derived_groups={}\n",
        summary.aggregate_metadata_blocks,
        summary.counter_info_entries,
        summary.counter_schemas,
        summary.decoded_counter_streams,
        summary.decoded_counter_metrics,
        summary.encoder_sample_metrics,
        summary.trace_map_entries,
        summary.derived_metric_groups
    ));
    if !summary.data_file_references.is_empty() {
        out.push_str("data_file_references:");
        for reference in &summary.data_file_references {
            out.push_str(&format!(" {reference}"));
        }
        out.push('\n');
    }

    if !report.top_opaque_files.is_empty() {
        out.push_str("\nlargest not-byte-complete raw files\n");
        out.push_str("-----------------------------------\n");
        for file in &report.top_opaque_files {
            out.push_str(&format!(
                "  {:<10} {:>12} {}\n",
                file.kind, file.size, file.name
            ));
        }
    }

    for warning in &report.warnings {
        out.push_str(&format!("~ {warning}\n"));
    }

    out
}

pub fn stream_data_summary<P: AsRef<Path>>(path: P) -> Result<ProfilerStreamDataSummary> {
    let input_path = path.as_ref().to_path_buf();
    let profiler_directory =
        find_profiler_directory(&input_path).ok_or_else(|| Error::NotFound(input_path.clone()))?;
    let stream_data_path = profiler_directory.join("streamData");
    if !stream_data_path.is_file() {
        return Err(Error::MissingFile(stream_data_path));
    }
    let mut summary = parse_stream_data(&stream_data_path)?;
    if let Some(names) = trace_function_names(&input_path) {
        summary.fill_function_names(&names);
    }
    Ok(summary)
}

/// Raw profiler file families a complete MTLReplayer profile writes next to
/// `streamData` that are absent (None when there is no profile at all).
pub fn missing_profile_families(path: &Path) -> Option<Vec<&'static str>> {
    let directory = find_profiler_directory(path)?;
    if !directory.join("streamData").is_file() {
        return None;
    }
    let names: Vec<String> = fs::read_dir(&directory)
        .ok()?
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    Some(
        ["Profiling_f_", "Counters_f_"]
            .into_iter()
            .filter(|prefix| !names.iter().any(|name| name.starts_with(prefix)))
            .map(|prefix| prefix.trim_end_matches('_'))
            .collect(),
    )
}

/// The capture's own pipeline → function map for the bundle at `path`, built
/// once per process (every report section asks for the stream summary).
fn trace_function_names(path: &Path) -> Option<std::sync::Arc<crate::trace::PipelineFunctionMap>> {
    use std::sync::{Arc, Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<BTreeMap<PathBuf, Option<Arc<crate::trace::PipelineFunctionMap>>>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut cache = cache.lock().ok()?;
    cache
        .entry(path.to_path_buf())
        .or_insert_with(|| {
            crate::trace::TraceBundle::open(path)
                .and_then(|trace| trace.pipeline_function_map())
                .map(Arc::new)
                .ok()
        })
        .clone()
}

impl ProfilerStreamDataSummary {
    /// streamData leaves functions specialized from a descriptor (function
    /// constants: MLX's attention, RoPE, ...) unnamed; take their names from
    /// the capture's pipeline → function map, by pipeline address.
    pub fn fill_function_names(&mut self, names: &BTreeMap<u64, String>) {
        let unnamed = |name: &Option<String>| name.as_deref().is_none_or(str::is_empty);
        for pipeline in &mut self.pipelines {
            if unnamed(&pipeline.function_name)
                && let Some(name) = names.get(&pipeline.pipeline_address)
            {
                pipeline.function_name = Some(name.clone());
            }
        }
        for dispatch in &mut self.dispatches {
            if unnamed(&dispatch.function_name) {
                dispatch.function_name = self
                    .pipelines
                    .get(dispatch.pipeline_index)
                    .and_then(|pipeline| pipeline.function_name.clone());
            }
        }
        for cost in &mut self.execution_costs {
            if unnamed(&cost.function_name) {
                cost.function_name = self
                    .pipelines
                    .iter()
                    .find(|pipeline| pipeline.pipeline_id == cost.pipeline_id)
                    .and_then(|pipeline| pipeline.function_name.clone());
            }
        }
    }
}

fn profiler_files(profiler_directory: &Path) -> Result<Vec<ProfilerCoverageFile>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(profiler_directory)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        if !metadata.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        files.push(ProfilerCoverageFile {
            kind: classify_file(&name),
            name,
            size: metadata.len(),
        });
    }
    files.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(files)
}

fn archive_count(report: Option<&counter::RawCounterProbeReport>, group: &str) -> usize {
    report.map_or(0, |report| {
        report
            .stream_archives
            .iter()
            .filter(|archive| archive.group == group)
            .count()
    })
}

fn data_file_references(report: Option<&counter::RawCounterProbeReport>) -> Vec<String> {
    let mut references = report
        .into_iter()
        .flat_map(|report| report.stream_archives.iter())
        .filter_map(|archive| archive.data_file.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    references.sort();
    references
}

fn coverage_group(
    kind: &str,
    file_count: usize,
    byte_count: u64,
    total_bytes: u64,
    summary: &ProfilerCoverageDecodedSummary,
) -> ProfilerCoverageGroup {
    let (decoder_status, decoded_signals, remaining_work) = match kind {
        "streamData" => (
            "partial-semantic",
            vec![
                "binary plist root/keyed archives".to_owned(),
                format!(
                    "{} APSCounterData schemas, {} decoded streams, {} decoded metrics",
                    summary.counter_schemas,
                    summary.decoded_counter_streams,
                    summary.decoded_counter_metrics
                ),
                format!(
                    "{} dispatch/timeline counter sample rows",
                    summary.encoder_sample_metrics
                ),
            ],
            vec![
                "not every embedded data blob has a named record layout yet".to_owned(),
                "cost/source-line MIO/USC structures are still being mapped".to_owned(),
            ],
        ),
        "profiling" => (
            "heuristic-only",
            vec![
                "current profiler command can scan pipeline ids".to_owned(),
            ],
            vec![
                "decode MIO/USC profiling records from Profiling_f_*".to_owned(),
                "map per-shader/per-line cost records instead of byte scans".to_owned(),
            ],
        ),
        "counter" => (
            "decoded",
            vec![
                "limiter-pass Counters_f_* (APS_USC) and RDE GPRWCNTR samples are decoded with agxps and attributed to encoders, dispatches and kernels (hw_counters, counters.md)"
                    .to_owned(),
            ],
            vec![
                "per-dispatch values for dispatches that overlap on the same cores are estimates (split by residency)"
                    .to_owned(),
            ],
        ),
        "timeline" => (
            "partial-structural",
            vec!["APSTimelineData GPRWCNTR timestamps are decoded from streamData".to_owned()],
            vec![
                "decode external Timeline_f_* record families byte-completely".to_owned(),
                "join all timeline samples back to command buffers/encoders".to_owned(),
            ],
        ),
        "kdebug" => (
            "opaque",
            Vec::new(),
            vec!["decode kdebug raw event payloads or explicitly mark as out of scope".to_owned()],
        ),
        _ => (
            "unknown",
            Vec::new(),
            vec!["classify and decode this file family".to_owned()],
        ),
    };

    ProfilerCoverageGroup {
        kind: kind.to_owned(),
        file_count,
        byte_count,
        byte_percent: percent(byte_count, total_bytes),
        decoder_status: decoder_status.to_owned(),
        decoded_signals,
        remaining_work,
    }
}

fn top_opaque_kind_bytes(files: &[ProfilerCoverageFile]) -> u64 {
    files
        .iter()
        .filter(|file| {
            matches!(
                file.kind.as_str(),
                "profiling" | "counter" | "timeline" | "kdebug"
            )
        })
        .map(|file| file.size)
        .sum()
}

fn percent(value: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        value as f64 / total as f64 * 100.0
    }
}

pub fn raw_encoder_timings<P: AsRef<Path>>(path: P) -> Result<Vec<ProfilerRawEncoderTiming>> {
    let input_path = path.as_ref().to_path_buf();
    let profiler_directory =
        find_profiler_directory(&input_path).ok_or_else(|| Error::NotFound(input_path.clone()))?;

    let mut files = fs::read_dir(&profiler_directory)?
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            let index = name
                .strip_prefix("Counters_f_")
                .and_then(|rest| rest.strip_suffix(".raw"))
                .and_then(|rest| rest.parse::<usize>().ok())?;
            Some((index, entry.path()))
        })
        .collect::<Vec<_>>();
    files.sort_by_key(|(index, _)| *index);

    let mut timings = Vec::new();
    for (index, path) in files {
        let data = fs::read(path)?;
        let records = raw_counter_records(&data);
        if records.is_empty() {
            continue;
        }
        let mut relative_duration = 0f64;
        let mut sample_records = 0usize;
        for record in records {
            if record.len() != 464 {
                continue;
            }
            let limiter_values = raw_record_limiter_values(record, 30);
            if limiter_values.is_empty() {
                continue;
            }
            relative_duration += limiter_values.iter().sum::<f64>();
            sample_records += 1;
        }
        if relative_duration <= f64::EPSILON {
            continue;
        }
        timings.push(ProfilerRawEncoderTiming {
            index,
            duration_ns: (relative_duration * 1_000_000.0) as u64,
            confidence_milli: if sample_records >= 4 { 300 } else { 200 },
        });
    }

    timings.sort_by_key(|timing| timing.index);
    Ok(timings)
}

fn raw_counter_records(data: &[u8]) -> Vec<&[u8]> {
    let mut records = Vec::new();
    let mut i = 0usize;
    while i < data.len().saturating_sub(mem::size_of::<u32>()) {
        if data[i..].starts_with(&[0x4e, 0x00, 0x00, 0x00]) {
            let record_size = estimate_raw_counter_record_size(data, i);
            if i + record_size <= data.len() {
                records.push(&data[i..i + record_size]);
                i += record_size;
            } else {
                i += mem::size_of::<u32>();
            }
        } else {
            i += 1;
        }
    }
    records
}

fn estimate_raw_counter_record_size(data: &[u8], offset: usize) -> usize {
    if offset + 8 <= data.len() {
        let potential_size = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
        if potential_size > 0 && potential_size < 100_000 {
            return potential_size as usize + 8;
        }
    }

    let search_start = offset + mem::size_of::<u32>();
    let search_end = data.len().min(offset.saturating_add(100_000));
    for i in search_start..search_end.saturating_sub(mem::size_of::<u32>()) {
        if data[i..].starts_with(&[0x4e, 0x00, 0x00, 0x00]) {
            return i - offset;
        }
    }

    69
}

fn raw_record_limiter_values(record: &[u8], max_count: usize) -> Vec<f64> {
    let mut values = Vec::new();
    let mut seen = BTreeSet::new();
    for offset in (0..record.len().saturating_sub(mem::size_of::<u32>())).step_by(4) {
        let bits = u32::from_le_bytes(record[offset..offset + 4].try_into().unwrap());
        let value = f32::from_bits(bits);
        if value.is_finite() && (0.001..=10.0).contains(&value) && seen.insert(bits) {
            values.push(value as f64);
            if values.len() >= max_count {
                break;
            }
        }
    }
    values
}

pub fn format_report(report: &ProfilerReport) -> String {
    let mut out = String::new();
    out.push_str("GPU Profiler Inventory\n");
    out.push_str("======================\n");
    out.push_str(&format!(
        "profiler_directory={}\n",
        report.profiler_directory.display()
    ));
    out.push_str(&format!(
        "files={} total_bytes={} streamData={}\n",
        report.files.len(),
        report.total_bytes,
        if report.stream_data_present {
            "present"
        } else {
            "missing"
        }
    ));
    out.push_str(&format!(
        "timeline={} counter={} profiling={} kdebug={} other={}\n",
        report.timeline_file_count,
        report.counter_file_count,
        report.profiling_file_count,
        report.kdebug_file_count,
        report.other_file_count
    ));

    if let Some(summary) = &report.stream_data_summary {
        out.push_str("\nstreamData summary\n");
        out.push_str("------------------\n");
        out.push_str(&format!(
            "pipelines={} dispatches={} encoders={} total_time={} us functions={}\n",
            summary.num_pipelines,
            summary.num_gpu_commands,
            summary.num_encoders,
            summary.total_time_us,
            summary.function_names.len()
        ));
        if let Some(timeline) = &summary.timeline {
            out.push_str(&format!(
                "command_buffers={} encoder_profiles={} timebase={}/{} absolute_time={}\n",
                timeline.command_buffer_timestamps.len(),
                timeline.encoder_profiles.len(),
                timeline.timebase_numer,
                timeline.timebase_denom,
                timeline.absolute_time
            ));
        }
        if !summary.execution_costs.is_empty() {
            out.push_str("top functions by execution cost\n");
            for cost in summary.execution_costs.iter().take(5) {
                let name = cost
                    .function_name
                    .clone()
                    .unwrap_or_else(|| format!("pipeline_{}", cost.pipeline_id));
                out.push_str(&format!(
                    "  - {name}: {:.2}% ({} samples)\n",
                    cost.cost_percent, cost.sample_count
                ));
            }
        }
        let mut pipelines_with_stats = summary
            .pipelines
            .iter()
            .filter_map(|pipeline| pipeline.stats.as_ref().map(|stats| (pipeline, stats)))
            .collect::<Vec<_>>();
        pipelines_with_stats.sort_by(|left, right| {
            right
                .1
                .instruction_count
                .cmp(&left.1.instruction_count)
                .then_with(|| {
                    right
                        .1
                        .temporary_register_count
                        .cmp(&left.1.temporary_register_count)
                })
        });
        if !pipelines_with_stats.is_empty() {
            out.push_str("top pipeline compilation stats\n");
            for (pipeline, stats) in pipelines_with_stats.into_iter().take(5) {
                let name = pipeline
                    .function_name
                    .clone()
                    .unwrap_or_else(|| format!("pipeline_{}", pipeline.pipeline_id));
                out.push_str(&format!(
                    "  - {name}: regs={} spills={} tgmem={} inst={} compile={:.2} ms\n",
                    stats.temporary_register_count,
                    stats.spilled_bytes,
                    stats.threadgroup_memory,
                    stats.instruction_count,
                    stats.compilation_time_ms
                ));
            }
        }

        let top = top_dispatch_functions(summary);
        if !top.is_empty() {
            out.push_str("top functions by dispatch time\n");
            for (name, count, time) in top.into_iter().take(5) {
                out.push_str(&format!("  - {name}: {count} dispatches, {time} us\n"));
            }
        }
    }

    if !report.limiter_metrics.is_empty() {
        out.push_str("\ncounter limiter summary\n");
        out.push_str("-----------------------\n");
        for limiter in report.limiter_metrics.iter().take(8) {
            out.push_str(&format!(
                "  - encoder {}: occ={} alu={} launch={} instr={} int_complex={} ctrl={} f32={} l1={} llc={} dev_bw={} l1r_bw={} l1w_bw={}\n",
                limiter.encoder_index,
                limiter
                    .occupancy
                    .map(|value| format!("{value:.2}%"))
                    .unwrap_or_else(|| "-".to_owned()),
                limiter
                    .alu_utilization
                    .map(|value| format!("{value:.2}%"))
                    .unwrap_or_else(|| "-".to_owned()),
                limiter
                    .compute_shader_launch
                    .map(|value| format!("{value:.3}%"))
                    .unwrap_or_else(|| "-".to_owned()),
                limiter
                    .instruction_throughput
                    .map(|value| format!("{value:.2}%"))
                    .unwrap_or_else(|| "-".to_owned()),
                limiter
                    .integer_complex
                    .map(|value| format!("{value:.2}%"))
                    .unwrap_or_else(|| "-".to_owned()),
                limiter
                    .control_flow
                    .map(|value| format!("{value:.2}%"))
                    .unwrap_or_else(|| "-".to_owned()),
                limiter
                    .f32_limiter
                    .map(|value| format!("{value:.2}%"))
                    .unwrap_or_else(|| "-".to_owned()),
                limiter
                    .l1_cache
                    .map(|value| format!("{value:.2}%"))
                    .unwrap_or_else(|| "-".to_owned()),
                limiter
                    .last_level_cache
                    .map(|value| format!("{value:.2}%"))
                    .unwrap_or_else(|| "-".to_owned()),
                limiter
                    .device_memory_bandwidth_gbps
                    .map(|value| format!("{value:.2} GB/s"))
                    .unwrap_or_else(|| "-".to_owned()),
                limiter
                    .buffer_l1_read_bandwidth_gbps
                    .map(|value| format!("{value:.2} GB/s"))
                    .unwrap_or_else(|| "-".to_owned()),
                limiter
                    .buffer_l1_write_bandwidth_gbps
                    .map(|value| format!("{value:.2} GB/s"))
                    .unwrap_or_else(|| "-".to_owned())
            ));
        }
    }

    for note in &report.notes {
        out.push_str(&format!("~ {note}\n"));
    }

    for file in &report.files {
        out.push_str(&format!(
            "  {:<10} {:>10} {}\n",
            file.kind, file.size, file.name
        ));
    }

    out
}

fn top_dispatch_functions(summary: &ProfilerStreamDataSummary) -> Vec<(String, usize, u64)> {
    let mut by_name = BTreeMap::<String, (usize, u64)>::new();
    for dispatch in &summary.dispatches {
        let name = dispatch
            .function_name
            .clone()
            .unwrap_or_else(|| format!("pipeline_{}", dispatch.pipeline_index));
        let entry = by_name.entry(name).or_default();
        entry.0 += 1;
        entry.1 += dispatch.duration_us;
    }

    let mut rows = by_name
        .into_iter()
        .map(|(name, (count, time))| (name, count, time))
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| right.2.cmp(&left.2).then_with(|| left.0.cmp(&right.0)));
    rows
}

fn parse_stream_data(path: &Path) -> Result<ProfilerStreamDataSummary> {
    let plist = Value::from_file(path)?;
    let archive = plist
        .as_dictionary()
        .ok_or(Error::InvalidTrace("invalid streamData archive"))?;
    let objects = archive
        .get("$objects")
        .and_then(Value::as_array)
        .ok_or(Error::InvalidTrace("streamData archive missing $objects"))?;
    let root = objects
        .get(1)
        .and_then(Value::as_dictionary)
        .ok_or(Error::InvalidTrace("streamData archive missing object 1"))?;

    let function_names = extract_function_names(objects, root);
    let (pipeline_addresses, pipeline_functions) =
        extract_pipeline_info(objects, root, &function_names);
    let pipelines = extract_pipelines(objects, root, &pipeline_addresses, &pipeline_functions);
    let encoder_timings = extract_encoder_timings(objects, root);
    let mut dispatches = extract_dispatches(objects, root, &pipelines);
    let timeline = extract_timeline(objects, root);
    if let Some(timeline) = &timeline {
        correlate_dispatch_samples(&mut dispatches, timeline);
    }

    Ok(ProfilerStreamDataSummary {
        function_names,
        num_pipelines: pipelines.len(),
        num_gpu_commands: dispatches.len(),
        num_encoders: encoder_timings.len(),
        total_time_us: encoder_timings
            .iter()
            .map(|encoder| encoder.duration_micros)
            .sum(),
        pipelines,
        execution_costs: Vec::new(),
        dispatches,
        encoder_timings,
        timeline,
    })
}

fn extract_function_names(objects: &[Value], root: &Dictionary) -> Vec<String> {
    ns_objects_from_root_key(objects, root, "strings")
        .into_iter()
        .filter_map(|value| value.as_string().map(ToOwned::to_owned))
        .collect()
}

fn extract_pipeline_info(
    objects: &[Value],
    root: &Dictionary,
    function_names: &[String],
) -> (Vec<u64>, Vec<Option<String>>) {
    let Some(data) = ns_data_from_root_key(objects, root, "pipelineStateInfoData") else {
        return (Vec::new(), Vec::new());
    };
    let record_size = root
        .get("pipelineStateInfoSize")
        .and_then(as_u64)
        .unwrap_or(40) as usize;
    if record_size == 0 {
        return (Vec::new(), Vec::new());
    }

    let function_info_data = ns_data_from_root_key(objects, root, "functionInfoData");
    let function_info_size = root.get("functionInfoSize").and_then(as_u64).unwrap_or(48) as usize;

    let record_count = data.len() / record_size;
    let function_info_count = function_info_data
        .map(|bytes| bytes.len() / function_info_size)
        .unwrap_or(0);

    let mut addresses = vec![0; record_count];
    let mut functions = vec![None; record_count];
    for index in 0..record_count {
        let offset = index * record_size;
        let record = &data[offset..offset + record_size];
        if record.len() >= 16 {
            addresses[index] = read_u64(record, 8);
        }

        if let Some(function_info_data) = function_info_data
            && index < function_info_count
            && function_info_size >= 32
        {
            let info_offset = index * function_info_size;
            let info = &function_info_data[info_offset..info_offset + function_info_size];
            let string_index = read_u32(info, 28) as usize;
            if let Some(name) = function_names.get(string_index) {
                functions[index] = Some(name.clone());
                continue;
            }
        }

        if let Some(name) = function_names.get(index) {
            functions[index] = Some(name.clone());
        }
    }

    (addresses, functions)
}

fn extract_pipelines(
    objects: &[Value],
    root: &Dictionary,
    pipeline_addresses: &[u64],
    pipeline_functions: &[Option<String>],
) -> Vec<ProfilerPipeline> {
    let stats_map = extract_pipeline_stats(objects, root);
    let Some(uid) = root.get("pipelinePerformanceStatistics").and_then(as_uid) else {
        return pipeline_addresses
            .iter()
            .enumerate()
            .map(|(index, address)| ProfilerPipeline {
                pipeline_id: index as i64,
                pipeline_address: *address,
                function_name: pipeline_functions.get(index).cloned().flatten(),
                stats: None,
            })
            .collect();
    };

    let Some(stats_dict) = object_dictionary(objects, uid) else {
        return Vec::new();
    };
    let Some(keys) = stats_dict.get("NS.keys").and_then(Value::as_array) else {
        return Vec::new();
    };
    let Some(values) = stats_dict.get("NS.objects").and_then(Value::as_array) else {
        return Vec::new();
    };

    let mut pipelines = Vec::new();
    for (index, key) in keys.iter().enumerate() {
        if index >= values.len() {
            break;
        }

        let pipeline_id = match key {
            Value::Uid(uid) => object(objects, *uid)
                .and_then(as_i64)
                .unwrap_or(index as i64),
            _ => as_i64(key).unwrap_or(index as i64),
        };

        pipelines.push(ProfilerPipeline {
            pipeline_id,
            pipeline_address: pipeline_addresses.get(index).copied().unwrap_or_default(),
            function_name: pipeline_functions.get(index).cloned().flatten(),
            stats: stats_map.get(&pipeline_id).cloned(),
        });
    }

    pipelines
}

fn extract_pipeline_stats(
    objects: &[Value],
    root: &Dictionary,
) -> BTreeMap<i64, ProfilerPipelineStats> {
    let Some(uid) = root.get("pipelinePerformanceStatistics").and_then(as_uid) else {
        return BTreeMap::new();
    };
    let Some(stats_dict) = object_dictionary(objects, uid) else {
        return BTreeMap::new();
    };
    let Some(keys) = stats_dict.get("NS.keys").and_then(Value::as_array) else {
        return BTreeMap::new();
    };
    let Some(values) = stats_dict.get("NS.objects").and_then(Value::as_array) else {
        return BTreeMap::new();
    };
    if keys.len() != values.len() {
        return BTreeMap::new();
    }

    let mut pipelines = BTreeMap::new();
    for (key, value) in keys.iter().zip(values.iter()) {
        let pipeline_id = match resolve_value(objects, key).and_then(as_i64) {
            Some(id) => id,
            None => continue,
        };
        let Some(stats_value) = resolve_value(objects, value) else {
            continue;
        };
        let Some(stat_dict) = stats_value.as_dictionary() else {
            continue;
        };
        let Some(stat_keys) = stat_dict.get("NS.keys").and_then(Value::as_array) else {
            continue;
        };
        let Some(stat_values) = stat_dict.get("NS.objects").and_then(Value::as_array) else {
            continue;
        };
        if stat_keys.len() != stat_values.len() {
            continue;
        }

        let mut key_map = BTreeMap::<String, &Value>::new();
        for (stat_key, stat_value) in stat_keys.iter().zip(stat_values.iter()) {
            let Some(key_name) = resolve_value(objects, stat_key).and_then(Value::as_string) else {
                continue;
            };
            if let Some(resolved) = resolve_value(objects, stat_value) {
                key_map.insert(key_name.to_owned(), resolved);
            }
        }

        let remarks = key_map.get("Remarks").and_then(|value| value.as_string());
        pipelines.insert(
            pipeline_id,
            ProfilerPipelineStats {
                temporary_register_count: key_map
                    .get("Temporary register count")
                    .and_then(|value| as_i64(value))
                    .unwrap_or(0),
                uniform_register_count: key_map
                    .get("Uniform register count")
                    .and_then(|value| as_i64(value))
                    .unwrap_or(0),
                spilled_bytes: key_map
                    .get("Spilled bytes")
                    .and_then(|value| as_i64(value))
                    .unwrap_or(0),
                threadgroup_memory: key_map
                    .get("Threadgroup memory")
                    .and_then(|value| as_i64(value))
                    .unwrap_or(0),
                instruction_count: key_map
                    .get("Instruction count")
                    .and_then(|value| as_i64(value))
                    .unwrap_or(0),
                alu_instruction_count: key_map
                    .get("ALU instruction count")
                    .and_then(|value| as_i64(value))
                    .unwrap_or(0),
                branch_instruction_count: key_map
                    .get("Branch instruction count")
                    .and_then(|value| as_i64(value))
                    .unwrap_or(0),
                compilation_time_ms: key_map
                    .get("Compilation time in milliseconds")
                    .and_then(|value| as_f64(value))
                    .unwrap_or(0.0),
                line_instruction_counts: remarks
                    .map(parse_instruction_mix_by_line)
                    .unwrap_or_default(),
            },
        );
    }

    pipelines
}

pub(crate) fn parse_instruction_mix_by_line(remarks: &str) -> BTreeMap<usize, u64> {
    let mut counts = BTreeMap::new();
    for block in remarks.split("--- !") {
        if !block.contains("Name:            InstructionMix")
            || !block.contains("Function:        agc.main")
            || block.contains("Function:        agc.main.constant_program")
        {
            continue;
        }
        let Some(line) = debugloc_program_source_line(block) else {
            continue;
        };
        if line == 0 {
            continue;
        }
        let Some(inst) = instruction_mix_count(block) else {
            continue;
        };
        *counts.entry(line).or_default() += inst;
    }
    counts
}

fn debugloc_program_source_line(block: &str) -> Option<usize> {
    let debugloc = block.lines().find(|line| line.contains("DebugLoc:"))?;
    if !debugloc.contains("program_source") {
        return None;
    }
    let line_marker = "Line:";
    let start = debugloc.find(line_marker)? + line_marker.len();
    let rest = debugloc[start..].trim_start();
    let digits = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    digits.parse().ok()
}

fn instruction_mix_count(block: &str) -> Option<u64> {
    let marker = "INST_:";
    let line = block.lines().find(|line| line.contains(marker))?;
    let start = line.find(marker)? + marker.len();
    let rest = line[start..].trim();
    let rest = rest.trim_matches('\'').trim_matches('"');
    rest.parse().ok()
}

fn extract_encoder_timings(objects: &[Value], root: &Dictionary) -> Vec<ProfilerEncoderTiming> {
    let Some(data) = ns_data_from_root_key(objects, root, "encoderInfoData") else {
        return Vec::new();
    };
    let record_size = root.get("encoderInfoSize").and_then(as_u64).unwrap_or(40) as usize;
    if record_size == 0 {
        return Vec::new();
    }

    let record_count = data.len() / record_size;
    let mut encoders = Vec::with_capacity(record_count);
    let mut previous_end = 0;
    for index in 0..record_count {
        let offset = index * record_size;
        let record = &data[offset..offset + record_size];
        if record.len() < 24 {
            continue;
        }
        let end_offset = read_u64(record, 16);
        let duration = if index == 0 {
            end_offset
        } else {
            end_offset.saturating_sub(previous_end)
        };
        previous_end = end_offset;
        encoders.push(ProfilerEncoderTiming {
            index,
            sequence_id: read_u64(record, 0),
            address: read_u64(record, 8),
            end_offset_micros: end_offset,
            duration_micros: duration,
        });
    }
    encoders
}

fn extract_dispatches(
    objects: &[Value],
    root: &Dictionary,
    pipelines: &[ProfilerPipeline],
) -> Vec<ProfilerDispatch> {
    let Some(data) = ns_data_from_root_key(objects, root, "gpuCommandInfoData") else {
        return Vec::new();
    };
    let record_size = root
        .get("gpuCommandInfoSize")
        .and_then(as_u64)
        .unwrap_or(32) as usize;
    if record_size == 0 {
        return Vec::new();
    }

    let record_count = data.len() / record_size;
    let mut dispatches = Vec::with_capacity(record_count);
    let mut previous_cumulative_us = None;
    for index in 0..record_count {
        let offset = index * record_size;
        let record = &data[offset..offset + record_size];
        if record.len() < 28 {
            continue;
        }

        let pipeline_index = (read_u64(record, 8) >> 32) as usize;
        let cumulative_us = read_u64(record, 16);
        // `cumulative_us` is a running end-time stamp into a global timeline that
        // includes idle time before the first dispatch (typically several hundred us
        // of command-buffer scheduling overhead). It is *not* zero-based, so the
        // very first record's value is the offset from CB/encoder origin to the end
        // of dispatch 0 — not its duration. Without a known encoder-start offset we
        // can't recover dispatch 0's true duration here, so attribute 0us rather
        // than absorbing the leading idle gap into whatever kernel happens to land
        // first.
        let duration_us = match previous_cumulative_us {
            Some(previous) => cumulative_us.saturating_sub(previous),
            None => 0,
        };
        previous_cumulative_us = Some(cumulative_us);

        dispatches.push(ProfilerDispatch {
            index,
            pipeline_index,
            pipeline_id: pipelines
                .get(pipeline_index)
                .map(|pipeline| pipeline.pipeline_id),
            function_name: pipelines
                .get(pipeline_index)
                .and_then(|pipeline| pipeline.function_name.clone()),
            encoder_index: read_u32(record, 24) as usize,
            cumulative_us,
            duration_us,
            sample_count: 0,
            sampling_density: 0.0,
            start_ticks: 0,
            end_ticks: 0,
        });
    }
    dispatches
}

fn extract_timeline(objects: &[Value], root: &Dictionary) -> Option<ProfilerTimelineInfo> {
    let blobs = ns_data_array_from_root_key(objects, root, "APSTimelineData");
    (!blobs.is_empty())
        .then(|| parse_aps_timeline_data(&blobs))
        .flatten()
}

fn parse_aps_timeline_data(blobs: &[Vec<u8>]) -> Option<ProfilerTimelineInfo> {
    let mut info = ProfilerTimelineInfo {
        command_buffer_timestamps: Vec::new(),
        encoder_profiles: Vec::new(),
        timebase_numer: 1,
        timebase_denom: 1,
        absolute_time: 0,
    };
    let mut found = false;

    for blob in blobs.iter().rev() {
        if blob.len() > 1000 && parse_timeline_metadata_blob(blob, &mut info) {
            found = true;
            break;
        }
    }

    if !found {
        for blob in blobs.iter().rev() {
            if parse_timeline_metadata_blob(blob, &mut info) {
                found = true;
                break;
            }
        }
    }

    if found {
        info.encoder_profiles =
            parse_encoder_profile_blobs(blobs, info.timebase_numer, info.timebase_denom);
        Some(info)
    } else {
        None
    }
}

fn parse_timeline_metadata_blob(data: &[u8], info: &mut ProfilerTimelineInfo) -> bool {
    let Ok(plist) = Value::from_reader(Cursor::new(data)) else {
        return false;
    };
    let Some(archive) = plist.as_dictionary() else {
        return false;
    };
    let Some(objects) = archive.get("$objects").and_then(Value::as_array) else {
        return false;
    };
    let Some(top) = archive.get("$top").and_then(Value::as_dictionary) else {
        return false;
    };
    let Some(root_uid) = top.get("root").and_then(as_uid) else {
        return false;
    };
    let Some(root) = object_dictionary(objects, root_uid) else {
        return false;
    };
    let Some(keys) = root.get("NS.keys").and_then(Value::as_array) else {
        return false;
    };
    let Some(values) = root.get("NS.objects").and_then(Value::as_array) else {
        return false;
    };
    if keys.len() != values.len() {
        return false;
    }

    let mut found = false;
    for (key, value) in keys.iter().zip(values.iter()) {
        let Some(key_uid) = as_uid(key) else {
            continue;
        };
        let Some(key_name) = object(objects, key_uid).and_then(Value::as_string) else {
            continue;
        };
        let Some(resolved) = resolve_value(objects, value) else {
            continue;
        };

        match key_name {
            "Command Buffer Timestamps" => {
                if let Some(data) = ns_data_from_value(resolved) {
                    info.command_buffer_timestamps = parse_command_buffer_timestamps(data);
                    found = !info.command_buffer_timestamps.is_empty();
                }
            }
            "Absolute Time" => {
                info.absolute_time = extract_scalar_u64(objects, resolved).unwrap_or_default();
            }
            "Timebase" => {
                if let Some((numer, denom)) = extract_timebase(objects, resolved) {
                    info.timebase_numer = numer.max(1);
                    info.timebase_denom = denom.max(1);
                }
            }
            _ => {}
        }
    }

    found
}

fn parse_command_buffer_timestamps(data: &[u8]) -> Vec<ProfilerCommandBufferTimestamp> {
    let mut timestamps = Vec::with_capacity(data.len() / 16);
    for (index, chunk) in data.chunks_exact(16).enumerate() {
        timestamps.push(ProfilerCommandBufferTimestamp {
            index,
            start_ticks: read_u64(chunk, 0),
            end_ticks: read_u64(chunk, 8),
        });
    }
    timestamps
}

fn parse_encoder_profile_blobs(
    blobs: &[Vec<u8>],
    timebase_numer: u64,
    timebase_denom: u64,
) -> Vec<ProfilerEncoderProfile> {
    let mut profiles = Vec::new();
    let max_encoder_blob = blobs.len().min(12);
    let mut encoder_index = 0usize;

    for blob in blobs.iter().take(max_encoder_blob).skip(1) {
        let Some((source, ring_buffer_index, shader_profiler_data)) =
            extract_encoder_blob_data(blob)
        else {
            continue;
        };
        if source != "RDE_0" {
            continue;
        }
        if let Some(mut profile) = parse_gprwcntr_blob(
            &shader_profiler_data,
            encoder_index,
            timebase_numer,
            timebase_denom,
        ) {
            profile.source = source;
            profile.ring_buffer_index = ring_buffer_index;
            profiles.push(profile);
            encoder_index += 1;
        }
    }

    profiles
}

fn extract_encoder_blob_data(data: &[u8]) -> Option<(String, usize, Vec<u8>)> {
    let plist = Value::from_reader(Cursor::new(data)).ok()?;
    let archive = plist.as_dictionary()?;
    let objects = archive.get("$objects").and_then(Value::as_array)?;
    let top = archive.get("$top").and_then(Value::as_dictionary)?;
    let root_uid = top.get("root").and_then(as_uid)?;
    let root = object_dictionary(objects, root_uid)?;
    let keys = root.get("NS.keys").and_then(Value::as_array)?;
    let values = root.get("NS.objects").and_then(Value::as_array)?;
    if keys.len() != values.len() {
        return None;
    }

    let mut source = None;
    let mut ring_buffer_index = 0usize;
    let mut shader_profiler_data = None;
    for (key, value) in keys.iter().zip(values.iter()) {
        let Some(key_uid) = as_uid(key) else {
            continue;
        };
        let Some(key_name) = object(objects, key_uid).and_then(Value::as_string) else {
            continue;
        };
        let Some(resolved) = resolve_value(objects, value) else {
            continue;
        };
        match key_name {
            "Source" => {
                source = resolved.as_string().map(ToOwned::to_owned);
            }
            "RingBufferIndex" => {
                ring_buffer_index =
                    extract_scalar_u64(objects, resolved).unwrap_or_default() as usize;
            }
            "ShaderProfilerData" => {
                shader_profiler_data = ns_data_from_value(resolved).map(ToOwned::to_owned);
            }
            _ => {}
        }
    }

    Some((source?, ring_buffer_index, shader_profiler_data?))
}

fn parse_gprwcntr_blob(
    data: &[u8],
    encoder_index: usize,
    timebase_numer: u64,
    timebase_denom: u64,
) -> Option<ProfilerEncoderProfile> {
    if data.len() < 8 || &data[..8] != b"GPRWCNTR" {
        return None;
    }

    let repeated_magic = data.get(80..88).is_some_and(|magic| magic == b"GPRWCNTR");
    let record_size = if repeated_magic { 80usize } else { 168usize };
    let sample_count = if repeated_magic {
        data.len() / record_size
    } else {
        data[8..].len() / record_size
    };
    let mut timestamps = Vec::with_capacity(sample_count);
    let mut min_timestamp = u64::MAX;
    let mut max_timestamp = 0u64;

    for chunk in gprwcntr_records(data, repeated_magic, record_size) {
        let entry = if repeated_magic {
            if chunk.get(..8) != Some(b"GPRWCNTR") {
                continue;
            }
            GprwcntrTimestamp {
                timestamp: read_u64(chunk, 8),
                size: read_u64(chunk, 16),
                count: read_u64(chunk, 24),
                flags: read_u32(chunk, 48),
            }
        } else {
            GprwcntrTimestamp {
                timestamp: read_u64(chunk, 0),
                size: read_u64(chunk, 8),
                count: read_u64(chunk, 16),
                flags: read_u32(chunk, 24),
            }
        };
        if entry.timestamp > 0 {
            min_timestamp = min_timestamp.min(entry.timestamp);
        }
        max_timestamp = max_timestamp.max(entry.timestamp);
        timestamps.push(entry);
    }

    let (start_ticks, end_ticks, duration_ns) = if min_timestamp == u64::MAX {
        (0, 0, 0)
    } else {
        (
            min_timestamp,
            max_timestamp,
            ticks_to_ns(
                max_timestamp.saturating_sub(min_timestamp),
                timebase_numer,
                timebase_denom,
            ),
        )
    };

    Some(ProfilerEncoderProfile {
        index: encoder_index,
        source: String::new(),
        ring_buffer_index: 0,
        sample_count,
        timestamps,
        start_ticks,
        end_ticks,
        duration_ns,
    })
}

fn gprwcntr_records(
    data: &[u8],
    repeated_magic: bool,
    record_size: usize,
) -> std::slice::ChunksExact<'_, u8> {
    if repeated_magic {
        data.chunks_exact(record_size)
    } else {
        data[8..].chunks_exact(record_size)
    }
}

fn correlate_dispatch_samples(
    dispatches: &mut [ProfilerDispatch],
    timeline: &ProfilerTimelineInfo,
) {
    if dispatches.is_empty() || timeline.command_buffer_timestamps.is_empty() {
        return;
    }

    let mut sample_timestamps = timeline
        .encoder_profiles
        .iter()
        .flat_map(|profile| profile.timestamps.iter().map(|entry| entry.timestamp))
        .collect::<Vec<_>>();
    sample_timestamps.sort_unstable();
    sample_timestamps.dedup();
    if sample_timestamps.is_empty() {
        return;
    }

    let command_buffer = &timeline.command_buffer_timestamps[0];
    let command_buffer_duration_ticks = command_buffer
        .end_ticks
        .saturating_sub(command_buffer.start_ticks);
    let total_dispatch_us = dispatches
        .last()
        .map(|dispatch| dispatch.cumulative_us)
        .unwrap_or_default();
    if total_dispatch_us == 0 || timeline.timebase_numer == 0 {
        return;
    }

    let ticks_per_us = (timeline.timebase_denom as f64 * 1_000.0) / timeline.timebase_numer as f64;
    let scale = command_buffer_duration_ticks as f64 / total_dispatch_us as f64 / ticks_per_us;

    for index in 0..dispatches.len() {
        // The first dispatch's `cumulative_us` is the offset from CB start to
        // dispatch-0-end (it includes leading idle), not its duration. Using
        // [0, cumulative_us[0]) as a sample window would attribute every
        // pre-first-dispatch sample to whichever kernel happens to land first.
        // Match the convention from `extract_dispatches` and give dispatch 0 a
        // zero-width window so those samples stay unattributed.
        let start_us = if index == 0 {
            dispatches[index].cumulative_us
        } else {
            dispatches[index - 1].cumulative_us
        };
        let end_us = dispatches[index].cumulative_us;
        let start_ticks =
            command_buffer.start_ticks + (start_us as f64 * ticks_per_us * scale) as u64;
        let end_ticks = command_buffer.start_ticks + (end_us as f64 * ticks_per_us * scale) as u64;
        let sample_count = sample_timestamps
            .iter()
            .filter(|timestamp| **timestamp >= start_ticks && **timestamp < end_ticks)
            .count();

        let dispatch = &mut dispatches[index];
        dispatch.start_ticks = start_ticks;
        dispatch.end_ticks = end_ticks;
        dispatch.sample_count = sample_count;
        if dispatch.duration_us > 0 {
            dispatch.sampling_density = sample_count as f64 / dispatch.duration_us as f64;
        }
    }
}

fn ticks_to_ns(ticks: u64, numer: u64, denom: u64) -> u64 {
    ticks.saturating_mul(numer.max(1)) / denom.max(1)
}

pub(crate) fn find_profiler_directory(path: &Path) -> Option<PathBuf> {
    if let Some(override_path) = std::env::var_os("GPUTRACE_PROFILER_DIR") {
        let override_path = PathBuf::from(override_path);
        if override_path.is_dir() {
            return Some(override_path);
        }
    }

    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext == "gpuprofiler_raw")
    {
        return path.is_dir().then(|| path.to_path_buf());
    }

    let adjacent = PathBuf::from(format!("{}.gpuprofiler_raw", path.display()));
    if adjacent.is_dir() {
        return Some(adjacent);
    }

    let metadata = fs::metadata(path).ok()?;
    if !metadata.is_dir() {
        return system_landing_zone_for(path).filter(|landing| landing.is_dir());
    }

    if let Some(embedded) = embedded_landing_zone_for(path).filter(|landing| landing.is_dir()) {
        return Some(embedded);
    }

    let nested = fs::read_dir(path)
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .find(|entry| {
            entry.is_dir()
                && entry
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| ext == "gpuprofiler_raw")
        });
    nested.or_else(|| system_landing_zone_for(path).filter(|landing| landing.is_dir()))
}

fn embedded_landing_zone_for(trace_path: &Path) -> Option<PathBuf> {
    let stem = trace_path.file_stem()?.to_str()?;
    Some(
        trace_path
            .join("gputrace-profile")
            .join(format!("{stem}.gpuprofiler_raw")),
    )
}

/// Xcode 26.x writes profile output to `/private/tmp/com.apple.gputools.profiling/`
/// when the user clicks Profile in Xcode.
fn system_landing_zone_for(trace_path: &Path) -> Option<PathBuf> {
    let stem = trace_path.file_stem()?.to_str()?;
    Some(PathBuf::from(format!(
        "/private/tmp/com.apple.gputools.profiling/{stem}_stream.gpuprofiler_raw"
    )))
}

fn classify_file(name: &str) -> String {
    if name == "streamData" {
        "streamData".to_owned()
    } else if name.starts_with("Timeline_f_") && name.ends_with(".raw") {
        "timeline".to_owned()
    } else if name.starts_with("Counters_f_") && name.ends_with(".raw") {
        "counter".to_owned()
    } else if name.starts_with("Profiling_f_") && name.ends_with(".raw") {
        "profiling".to_owned()
    } else if name.starts_with("kdebug") && name.ends_with(".raw") {
        "kdebug".to_owned()
    } else {
        "other".to_owned()
    }
}

fn object(objects: &[Value], uid: Uid) -> Option<&Value> {
    objects.get(uid.get() as usize)
}

fn object_dictionary(objects: &[Value], uid: Uid) -> Option<&Dictionary> {
    object(objects, uid).and_then(Value::as_dictionary)
}

fn ns_data_from_root_key<'a>(
    objects: &'a [Value],
    root: &Dictionary,
    key: &str,
) -> Option<&'a [u8]> {
    root.get(key)
        .and_then(as_uid)
        .and_then(|uid| object_dictionary(objects, uid))
        .and_then(|dict| dict.get("NS.data"))
        .and_then(Value::as_data)
}

fn ns_objects_from_root_key<'a>(
    objects: &'a [Value],
    root: &Dictionary,
    key: &str,
) -> Vec<&'a Value> {
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
        .filter_map(|value| match value {
            Value::Uid(uid) => object(objects, *uid),
            _ => Some(value),
        })
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
        .filter_map(|value| {
            value
                .as_dictionary()
                .and_then(|dict| dict.get("NS.data"))
                .and_then(Value::as_data)
                .map(|bytes| bytes.to_vec())
        })
        .collect()
}

fn ns_data_from_value(value: &Value) -> Option<&[u8]> {
    value
        .as_dictionary()
        .and_then(|dict| dict.get("NS.data"))
        .and_then(Value::as_data)
        .or_else(|| value.as_data())
}

fn extract_scalar_u64(objects: &[Value], value: &Value) -> Option<u64> {
    as_u64(value).or_else(|| {
        value
            .as_dictionary()
            .and_then(|dict| dict.get("NS.objects"))
            .and_then(Value::as_array)
            .and_then(|entries| entries.first())
            .and_then(|entry| resolve_value(objects, entry))
            .and_then(as_u64)
    })
}

fn extract_timebase(objects: &[Value], value: &Value) -> Option<(u64, u64)> {
    let entries = value
        .as_dictionary()
        .and_then(|dict| dict.get("NS.objects"))
        .and_then(Value::as_array)
        .or_else(|| value.as_array())?;

    let numer = entries
        .first()
        .and_then(|entry| resolve_value(objects, entry))
        .and_then(|entry| extract_scalar_u64(objects, entry))?;
    let denom = entries
        .get(1)
        .and_then(|entry| resolve_value(objects, entry))
        .and_then(|entry| extract_scalar_u64(objects, entry))?;
    Some((numer, denom))
}

fn resolve_value<'a>(objects: &'a [Value], value: &'a Value) -> Option<&'a Value> {
    match value {
        Value::Uid(uid) => object(objects, *uid),
        other => Some(other),
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
        Value::Integer(value) => value
            .as_unsigned()
            .or_else(|| value.as_signed().map(|v| v as u64)),
        Value::Real(value) => Some(*value as u64),
        _ => None,
    }
}

fn as_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Integer(value) => value
            .as_signed()
            .or_else(|| value.as_unsigned().map(|v| v as i64)),
        Value::Real(value) => Some(*value as i64),
        _ => None,
    }
}

fn as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Integer(value) => value
            .as_signed()
            .map(|v| v as f64)
            .or_else(|| value.as_unsigned().map(|v| v as f64)),
        Value::Real(value) => Some(*value),
        _ => None,
    }
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    data.get(offset..offset + 4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .unwrap_or_default()
}

fn read_u64(data: &[u8], offset: usize) -> u64 {
    data.get(offset..offset + 8)
        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    use plist::Integer;
    use tempfile::tempdir;

    fn uid(value: u64) -> Value {
        Value::Uid(Uid::new(value))
    }

    fn string(value: &str) -> Value {
        Value::String(value.to_owned())
    }

    fn integer(value: i64) -> Value {
        Value::Integer(Integer::from(value))
    }

    fn dict(entries: &[(&str, Value)]) -> Value {
        let mut dict = Dictionary::new();
        for (key, value) in entries {
            dict.insert((*key).to_owned(), value.clone());
        }
        Value::Dictionary(dict)
    }

    fn data(bytes: Vec<u8>) -> Value {
        dict(&[("NS.data", Value::Data(bytes))])
    }

    fn array(entries: &[Value]) -> Value {
        dict(&[("NS.objects", Value::Array(entries.to_vec()))])
    }

    fn array_uids(values: &[u64]) -> Value {
        dict(&[(
            "NS.objects",
            Value::Array(values.iter().copied().map(uid).collect()),
        )])
    }

    fn dict_uids(keys: &[u64], values: &[u64]) -> Value {
        dict(&[
            (
                "NS.keys",
                Value::Array(keys.iter().copied().map(uid).collect()),
            ),
            (
                "NS.objects",
                Value::Array(values.iter().copied().map(uid).collect()),
            ),
        ])
    }

    fn streamdata_fixture() -> Value {
        let mut pipeline_state = vec![0u8; 40];
        pipeline_state[8..16].copy_from_slice(&0x1111_u64.to_le_bytes());

        let mut function_info = vec![0u8; 48];
        function_info[28..32].copy_from_slice(&(0_u32).to_le_bytes());

        let mut encoder_info = vec![0u8; 40];
        encoder_info[0..8].copy_from_slice(&7_u64.to_le_bytes());
        encoder_info[8..16].copy_from_slice(&100_u64.to_le_bytes());
        encoder_info[16..24].copy_from_slice(&250_u64.to_le_bytes());

        // Two GPU command records so we can exercise both the first-dispatch
        // (zero-width window) path and the per-dispatch diff path.
        let mut gpu_command = vec![0u8; 64];
        gpu_command[8..16].copy_from_slice(&(0_u64 << 32).to_le_bytes());
        gpu_command[16..24].copy_from_slice(&90_u64.to_le_bytes());
        gpu_command[24..28].copy_from_slice(&0_u32.to_le_bytes());
        gpu_command[40..48].copy_from_slice(&(0_u64 << 32).to_le_bytes());
        gpu_command[48..56].copy_from_slice(&130_u64.to_le_bytes());
        gpu_command[56..60].copy_from_slice(&0_u32.to_le_bytes());

        let objects = vec![
            string("$null"),
            dict(&[
                ("strings", uid(2)),
                ("pipelineStateInfoData", uid(4)),
                ("pipelineStateInfoSize", integer(40)),
                ("functionInfoData", uid(5)),
                ("functionInfoSize", integer(48)),
                ("pipelinePerformanceStatistics", uid(6)),
                ("encoderInfoData", uid(10)),
                ("encoderInfoSize", integer(40)),
                ("gpuCommandInfoData", uid(11)),
                ("gpuCommandInfoSize", integer(32)),
            ]),
            array_uids(&[3]),
            string("kernel_main"),
            data(pipeline_state),
            data(function_info),
            dict_uids(&[7], &[8]),
            integer(27),
            dict(&[
                ("NS.keys", Value::Array(vec![])),
                ("NS.objects", Value::Array(vec![])),
            ]),
            dict(&[
                ("NS.keys", Value::Array(vec![])),
                ("NS.objects", Value::Array(vec![])),
            ]),
            data(encoder_info),
            data(gpu_command),
        ];

        dict(&[("$objects", Value::Array(objects))])
    }

    fn timeline_blob() -> Vec<u8> {
        let mut timestamps = Vec::new();
        timestamps.extend_from_slice(&100_u64.to_le_bytes());
        timestamps.extend_from_slice(&160_u64.to_le_bytes());
        timestamps.extend_from_slice(&200_u64.to_le_bytes());
        timestamps.extend_from_slice(&320_u64.to_le_bytes());

        let metadata = dict(&[
            ("$top", dict(&[("root", uid(1))])),
            (
                "$objects",
                Value::Array(vec![
                    string("$null"),
                    dict_uids(&[2, 3, 4], &[5, 9, 6]),
                    string("Command Buffer Timestamps"),
                    string("Absolute Time"),
                    string("Timebase"),
                    data(timestamps),
                    array(&[uid(7), uid(8)]),
                    integer(125),
                    integer(3),
                    integer(99_999),
                ]),
            ),
        ]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.plist");
        metadata.to_file_binary(&path).unwrap();
        fs::read(path).unwrap()
    }

    fn encoder_blob() -> Vec<u8> {
        let mut shader_profiler_data = b"GPRWCNTR".to_vec();
        let mut record = vec![0u8; 168];
        // Sample timestamp lands inside dispatch[1]'s tick window
        // (start_us=90, end_us=130, scaled to ticks ~141..160 in the test
        // fixture's CB coordinate system).
        record[0..8].copy_from_slice(&150_u64.to_le_bytes());
        record[8..16].copy_from_slice(&4096_u64.to_le_bytes());
        record[16..24].copy_from_slice(&6_u64.to_le_bytes());
        record[24..28].copy_from_slice(&0xffff_ffff_u32.to_le_bytes());
        shader_profiler_data.extend_from_slice(&record);

        let blob = dict(&[
            ("$top", dict(&[("root", uid(1))])),
            (
                "$objects",
                Value::Array(vec![
                    string("$null"),
                    dict_uids(&[2, 3, 4], &[5, 6, 7]),
                    string("Source"),
                    string("RingBufferIndex"),
                    string("ShaderProfilerData"),
                    string("RDE_0"),
                    integer(2),
                    data(shader_profiler_data),
                ]),
            ),
        ]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("encoder.plist");
        blob.to_file_binary(&path).unwrap();
        fs::read(path).unwrap()
    }

    fn repeated_gprwcntr_encoder_blob() -> Vec<u8> {
        let mut shader_profiler_data = Vec::new();
        for (timestamp, size, count, flags) in
            [(120_u64, 4096_u64, 6_u64, 3_u32), (150, 8192, 7, 4)]
        {
            let mut record = vec![0u8; 80];
            record[0..8].copy_from_slice(b"GPRWCNTR");
            record[8..16].copy_from_slice(&timestamp.to_le_bytes());
            record[16..24].copy_from_slice(&size.to_le_bytes());
            record[24..32].copy_from_slice(&count.to_le_bytes());
            record[48..52].copy_from_slice(&flags.to_le_bytes());
            shader_profiler_data.extend_from_slice(&record);
        }

        let blob = dict(&[
            ("$top", dict(&[("root", uid(1))])),
            (
                "$objects",
                Value::Array(vec![
                    string("$null"),
                    dict_uids(&[2, 3, 4], &[5, 6, 7]),
                    string("Source"),
                    string("RingBufferIndex"),
                    string("ShaderProfilerData"),
                    string("RDE_0"),
                    integer(2),
                    data(shader_profiler_data),
                ]),
            ),
        ]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("encoder-repeated.plist");
        blob.to_file_binary(&path).unwrap();
        fs::read(path).unwrap()
    }

    fn streamdata_fixture_with_timeline() -> Value {
        let mut fixture = streamdata_fixture();
        let objects = fixture
            .as_dictionary_mut()
            .unwrap()
            .get_mut("$objects")
            .and_then(Value::as_array_mut)
            .unwrap();
        objects[1]
            .as_dictionary_mut()
            .unwrap()
            .insert("APSTimelineData".to_owned(), uid(12));
        objects.push(array(&[
            data(vec![0u8; 32]),
            data(encoder_blob()),
            data(timeline_blob()),
        ]));
        fixture
    }

    fn streamdata_fixture_with_repeated_gprwcntr() -> Value {
        let mut fixture = streamdata_fixture();
        let objects = fixture
            .as_dictionary_mut()
            .unwrap()
            .get_mut("$objects")
            .and_then(Value::as_array_mut)
            .unwrap();
        objects[1]
            .as_dictionary_mut()
            .unwrap()
            .insert("APSTimelineData".to_owned(), uid(12));
        objects.push(array(&[
            data(vec![0u8; 32]),
            data(repeated_gprwcntr_encoder_blob()),
            data(timeline_blob()),
        ]));
        fixture
    }

    #[test]
    fn finds_adjacent_profiler_directory() {
        let dir = tempdir().unwrap();
        let trace_path = dir.path().join("sample.gputrace");
        fs::create_dir(&trace_path).unwrap();
        let profiler_dir = dir.path().join("sample.gputrace.gpuprofiler_raw");
        fs::create_dir(&profiler_dir).unwrap();
        fs::write(profiler_dir.join("streamData"), [0u8; 8]).unwrap();
        fs::write(profiler_dir.join("Timeline_f_0.raw"), [0u8; 16]).unwrap();

        let report = find_profiler_directory(&trace_path).unwrap();
        assert_eq!(report, profiler_dir);
    }

    #[test]
    fn parses_streamdata_summary_from_fixture() {
        let dir = tempdir().unwrap();
        let stream_data_path = dir.path().join("streamData");
        streamdata_fixture()
            .to_file_binary(&stream_data_path)
            .unwrap();

        let summary = parse_stream_data(&stream_data_path).unwrap();
        assert_eq!(summary.function_names, vec!["kernel_main".to_owned()]);
        assert_eq!(summary.num_pipelines, 1);
        assert_eq!(summary.num_encoders, 1);
        assert_eq!(summary.num_gpu_commands, 2);
        assert_eq!(summary.total_time_us, 250);
        assert_eq!(summary.pipelines[0].pipeline_id, 27);
        assert_eq!(summary.pipelines[0].pipeline_address, 0x1111);
        assert_eq!(
            summary.pipelines[0].function_name.as_deref(),
            Some("kernel_main")
        );
        assert_eq!(
            summary.dispatches[0].function_name.as_deref(),
            Some("kernel_main")
        );
        // First dispatch's `cumulative_us` is the offset from CB/encoder origin
        // (it includes leading idle), not its duration, so we attribute 0us to it.
        assert_eq!(summary.dispatches[0].cumulative_us, 90);
        assert_eq!(summary.dispatches[0].duration_us, 0);
        assert_eq!(summary.dispatches[0].sample_count, 0);
        // Subsequent dispatches recover their duration from the cumulative diff.
        assert_eq!(summary.dispatches[1].cumulative_us, 130);
        assert_eq!(summary.dispatches[1].duration_us, 40);
        assert!(summary.execution_costs.is_empty());
        assert!(summary.timeline.is_none());
    }

    #[test]
    fn parses_timeline_metadata_from_streamdata() {
        let dir = tempdir().unwrap();
        let stream_data_path = dir.path().join("streamData");
        streamdata_fixture_with_timeline()
            .to_file_binary(&stream_data_path)
            .unwrap();

        let summary = parse_stream_data(&stream_data_path).unwrap();
        let timeline = summary.timeline.expect("timeline metadata");
        assert_eq!(timeline.timebase_numer, 125);
        assert_eq!(timeline.timebase_denom, 3);
        assert_eq!(timeline.absolute_time, 99_999);
        assert_eq!(timeline.command_buffer_timestamps.len(), 2);
        assert_eq!(timeline.encoder_profiles.len(), 1);
        assert_eq!(timeline.encoder_profiles[0].sample_count, 1);
        assert_eq!(timeline.encoder_profiles[0].start_ticks, 150);
        assert_eq!(timeline.command_buffer_timestamps[0].start_ticks, 100);
        assert_eq!(timeline.command_buffer_timestamps[0].end_ticks, 160);
        assert_eq!(timeline.command_buffer_timestamps[1].start_ticks, 200);
        assert_eq!(timeline.command_buffer_timestamps[1].end_ticks, 320);
        // dispatch[0]'s sample window is zero-width (its cumulative_us is the
        // pre-first-dispatch idle gap, not a real time range), so the sample
        // at timestamp 120 stays unattributed and dispatch[1] picks up the
        // GPU-active samples within [start_us(=90), end_us(=130)].
        assert_eq!(summary.dispatches[0].sample_count, 0);
        assert_eq!(summary.dispatches[0].sampling_density, 0.0);
        assert_eq!(
            summary.dispatches[0].start_ticks,
            summary.dispatches[0].end_ticks
        );
        assert!(summary.dispatches[1].sample_count >= 1);
        assert!(summary.dispatches[1].sampling_density > 0.0);
    }

    #[test]
    fn parses_repeated_magic_gprwcntr_records() {
        let dir = tempdir().unwrap();
        let stream_data_path = dir.path().join("streamData");
        streamdata_fixture_with_repeated_gprwcntr()
            .to_file_binary(&stream_data_path)
            .unwrap();

        let summary = parse_stream_data(&stream_data_path).unwrap();
        let timeline = summary.timeline.expect("timeline metadata");
        assert_eq!(timeline.encoder_profiles.len(), 1);
        let profile = &timeline.encoder_profiles[0];
        assert_eq!(profile.sample_count, 2);
        assert_eq!(profile.start_ticks, 120);
        assert_eq!(profile.end_ticks, 150);
        assert_eq!(profile.timestamps[0].size, 4096);
        assert_eq!(profile.timestamps[1].count, 7);
        assert_eq!(profile.timestamps[1].flags, 4);
    }

    #[test]
    fn reports_streamdata_summary_when_present() {
        let dir = tempdir().unwrap();
        let trace_path = dir.path().join("sample.gputrace");
        fs::create_dir(&trace_path).unwrap();
        let profiler_dir = dir.path().join("sample.gputrace.gpuprofiler_raw");
        fs::create_dir(&profiler_dir).unwrap();
        streamdata_fixture()
            .to_file_binary(profiler_dir.join("streamData"))
            .unwrap();
        fs::write(profiler_dir.join("Timeline_f_0.raw"), [0u8; 16]).unwrap();

        let report = report(&trace_path).unwrap();
        assert!(report.stream_data_present);
        let summary = report.stream_data_summary.as_ref().unwrap();
        assert!(summary.execution_costs.is_empty());
        let text = format_report(&report);
        assert!(text.contains("streamData summary"));
        assert!(text.contains("kernel_main"));
    }

    #[test]
    fn finds_nested_profiler_directory_inside_trace_bundle() {
        let dir = tempdir().unwrap();
        let trace_path = dir.path().join("sample.gputrace");
        let profiler_dir = trace_path.join("capture.gpuprofiler_raw");
        fs::create_dir_all(&profiler_dir).unwrap();
        fs::write(profiler_dir.join("Counters_f_4.raw"), [0u8; 4]).unwrap();
        fs::write(profiler_dir.join("Profiling_f_1.raw"), [0u8; 12]).unwrap();

        let report = report(&trace_path).unwrap();
        assert_eq!(report.profiler_directory, profiler_dir);
        assert_eq!(report.counter_file_count, 1);
        assert_eq!(report.profiling_file_count, 1);
        assert!(!report.stream_data_present);
    }

    #[test]
    fn formats_inventory_report() {
        let report = ProfilerReport {
            input_path: PathBuf::from("trace.gputrace"),
            profiler_directory: PathBuf::from("trace.gputrace.gpuprofiler_raw"),
            stream_data_present: true,
            stream_data_summary: Some(ProfilerStreamDataSummary {
                function_names: vec!["kernel_main".to_owned()],
                pipelines: vec![ProfilerPipeline {
                    pipeline_id: 27,
                    pipeline_address: 0x1111,
                    function_name: Some("kernel_main".to_owned()),
                    stats: Some(ProfilerPipelineStats {
                        temporary_register_count: 32,
                        uniform_register_count: 4,
                        spilled_bytes: 128,
                        threadgroup_memory: 2048,
                        instruction_count: 640,
                        alu_instruction_count: 512,
                        branch_instruction_count: 12,
                        compilation_time_ms: 2.5,
                        line_instruction_counts: BTreeMap::new(),
                    }),
                }],
                execution_costs: vec![ProfilerExecutionCost {
                    pipeline_id: 27,
                    function_name: Some("kernel_main".to_owned()),
                    sample_count: 5,
                    cost_percent: 62.5,
                }],
                dispatches: vec![ProfilerDispatch {
                    index: 0,
                    pipeline_index: 0,
                    pipeline_id: Some(27),
                    function_name: Some("kernel_main".to_owned()),
                    encoder_index: 0,
                    cumulative_us: 90,
                    duration_us: 90,
                    sample_count: 1,
                    sampling_density: 0.011,
                    start_ticks: 100,
                    end_ticks: 160,
                }],
                encoder_timings: vec![ProfilerEncoderTiming {
                    index: 0,
                    sequence_id: 7,
                    address: 0x942f15720,
                    end_offset_micros: 250,
                    duration_micros: 250,
                }],
                timeline: Some(ProfilerTimelineInfo {
                    command_buffer_timestamps: vec![ProfilerCommandBufferTimestamp {
                        index: 0,
                        start_ticks: 100,
                        end_ticks: 160,
                    }],
                    encoder_profiles: vec![ProfilerEncoderProfile {
                        index: 0,
                        source: "RDE_0".to_owned(),
                        ring_buffer_index: 2,
                        sample_count: 1,
                        timestamps: vec![GprwcntrTimestamp {
                            timestamp: 120,
                            size: 4096,
                            count: 6,
                            flags: 0xffff_ffff,
                        }],
                        start_ticks: 120,
                        end_ticks: 120,
                        duration_ns: 0,
                    }],
                    timebase_numer: 125,
                    timebase_denom: 3,
                    absolute_time: 99_999,
                }),
                num_pipelines: 1,
                num_gpu_commands: 1,
                num_encoders: 1,
                total_time_us: 250,
            }),
            limiter_metrics: vec![counter::CounterLimiter {
                encoder_index: 0,
                occupancy: Some(72.0),
                alu_utilization: Some(61.0),
                compute_shader_launch: Some(0.18),
                instruction_throughput: Some(1.2),
                integer_complex: Some(2.4),
                control_flow: Some(0.09),
                f32_limiter: Some(6.5),
                l1_cache: Some(0.8),
                last_level_cache: Some(0.04),
                device_memory_bandwidth_gbps: Some(8.2),
                buffer_l1_read_bandwidth_gbps: Some(2.3),
                buffer_l1_write_bandwidth_gbps: Some(0.7),
            }],
            timeline_file_count: 1,
            counter_file_count: 2,
            profiling_file_count: 1,
            kdebug_file_count: 0,
            other_file_count: 1,
            total_bytes: 42,
            files: vec![ProfilerFileEntry {
                name: "streamData".to_owned(),
                size: 42,
                kind: "streamData".to_owned(),
            }],
            notes: vec![
                "Counters_f_* is mapped through Xcode counter names and summarized per encoder."
                    .to_owned(),
            ],
        };

        let text = format_report(&report);
        assert!(text.contains("GPU Profiler Inventory"));
        assert!(text.contains("streamData=present"));
        assert!(text.contains("kernel_main"));
        assert!(text.contains("command_buffers=1"));
        assert!(text.contains("encoder_profiles=1"));
        assert!(text.contains("execution cost"));
        assert!(text.contains("counter limiter summary"));
        assert!(text.contains("occ=72.00%"));
        assert!(text.contains("regs=32"));
    }

    #[test]
    fn formats_coverage_report_with_decoder_status() {
        let report = ProfilerCoverageReport {
            input_path: PathBuf::from("trace.gputrace"),
            profiler_directory: PathBuf::from("trace.gputrace.gpuprofiler_raw"),
            total_bytes: 1024,
            total_files: 3,
            groups: vec![
                ProfilerCoverageGroup {
                    kind: "streamData".to_owned(),
                    file_count: 1,
                    byte_count: 256,
                    byte_percent: 25.0,
                    decoder_status: "partial-semantic".to_owned(),
                    decoded_signals: vec!["1 APSCounterData schemas".to_owned()],
                    remaining_work: vec!["cost/source-line MIO/USC structures".to_owned()],
                },
                ProfilerCoverageGroup {
                    kind: "profiling".to_owned(),
                    file_count: 2,
                    byte_count: 768,
                    byte_percent: 75.0,
                    decoder_status: "heuristic-only".to_owned(),
                    decoded_signals: vec!["pipeline ids".to_owned()],
                    remaining_work: vec!["decode MIO/USC profiling records".to_owned()],
                },
            ],
            top_opaque_files: vec![ProfilerCoverageFile {
                name: "Profiling_f_0.raw".to_owned(),
                kind: "profiling".to_owned(),
                size: 512,
            }],
            decoded_summary: ProfilerCoverageDecodedSummary {
                stream_data_present: true,
                aggregate_metadata_blocks: 1,
                counter_info_entries: 2,
                counter_schemas: 1,
                decoded_counter_streams: 3,
                decoded_counter_metrics: 4,
                derived_metric_groups: 5,
                encoder_sample_metrics: 6,
                trace_map_entries: 7,
                stream_archives: 7,
                aps_data_archives: 8,
                aps_counter_data_archives: 9,
                aps_timeline_data_archives: 10,
                data_file_references: vec!["Profiling_f_0.raw".to_owned()],
            },
            warnings: vec!["raw file families are not byte-complete decoded yet".to_owned()],
        };

        let text = format_coverage_report(&report);
        assert!(text.contains("GPU Profiler Coverage"));
        assert!(text.contains("status=partial-semantic"));
        assert!(text.contains("status=heuristic-only"));
        assert!(text.contains("decode MIO/USC profiling records"));
        assert!(text.contains("APSCounterData"));
        assert!(text.contains("Profiling_f_0.raw"));
        assert!(text.contains("not byte-complete"));
    }

    #[test]
    fn extracts_raw_encoder_timings_from_counter_files() {
        let dir = tempfile::tempdir().unwrap();
        let profiler_dir = dir.path().join("trace.gputrace.gpuprofiler_raw");
        fs::create_dir_all(&profiler_dir).unwrap();

        let mut data = vec![0u8; 2400];
        data[0..4].copy_from_slice(&0x4e_u32.to_le_bytes());
        let mut record = vec![0u8; 464];
        record[0..4].copy_from_slice(&0x4e_u32.to_le_bytes());
        record[4..8].copy_from_slice(&456u32.to_le_bytes());
        for (index, value) in [1.5f32, 2.0, 0.5].iter().enumerate() {
            let offset = 8 + index * 4;
            record[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        data.extend_from_slice(&record);
        fs::write(profiler_dir.join("Counters_f_0.raw"), data).unwrap();

        let timings = raw_encoder_timings(dir.path().join("trace.gputrace")).unwrap();
        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0].index, 0);
        assert!(timings[0].duration_ns > 0);
        assert_eq!(timings[0].confidence_milli, 200);
    }

    #[test]
    fn raw_encoder_timings_use_go_record_size_and_dedup_limiter_values() {
        let dir = tempfile::tempdir().unwrap();
        let profiler_dir = dir.path().join("trace.gputrace.gpuprofiler_raw");
        fs::create_dir_all(&profiler_dir).unwrap();

        let mut record = vec![0u8; 464];
        record[0..4].copy_from_slice(&0x4e_u32.to_le_bytes());
        record[4..8].copy_from_slice(&456u32.to_le_bytes());
        for (index, value) in [2.0f32, 2.0, 3.0].iter().enumerate() {
            let offset = 8 + index * 4;
            record[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        fs::write(profiler_dir.join("Counters_f_7.raw"), record).unwrap();

        let timings = raw_encoder_timings(dir.path().join("trace.gputrace")).unwrap();

        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0].index, 7);
        assert_eq!(timings[0].duration_ns, 5_000_000);
    }
}
