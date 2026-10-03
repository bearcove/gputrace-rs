use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{Value, json};

use crate::counter;
use crate::error::Result;
use crate::profiler;
use crate::trace::{BoundBuffer, CommandBufferRegion, TraceBundle};

#[derive(Debug, Clone, Serialize)]
pub struct TimelineReport {
    pub synthetic: bool,
    pub source: String,
    pub command_buffers_profiler_backed: bool,
    pub start_time_ns: u64,
    pub end_time_ns: u64,
    pub duration_ns: u64,
    pub command_buffer_count: usize,
    pub encoder_count: usize,
    pub dispatch_count: usize,
    pub counter_track_count: usize,
    pub command_buffers: Vec<TimelineCommandBuffer>,
    pub encoders: Vec<TimelineEncoder>,
    pub dispatches: Vec<TimelineDispatch>,
    pub counter_tracks: Vec<TimelineCounterTrack>,
    pub events: Vec<TimelineEvent>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimelineCommandBuffer {
    pub index: usize,
    pub timestamp_ns: u64,
    pub duration_ns: Option<u64>,
    pub encoder_count: usize,
    pub dispatch_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimelineEncoder {
    pub index: usize,
    pub command_buffer_index: usize,
    pub label: String,
    pub address: u64,
    pub dispatch_count: usize,
    pub start_time_ns: u64,
    pub end_time_ns: u64,
    pub duration_ns: Option<u64>,
    pub synthetic: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimelineDispatch {
    pub index: usize,
    pub command_buffer_index: usize,
    pub encoder_index: Option<usize>,
    pub encoder_address: Option<u64>,
    pub encoder_label: Option<String>,
    pub kernel_name: Option<String>,
    pub start_time_ns: u64,
    pub end_time_ns: u64,
    pub duration_ns: Option<u64>,
    pub grid_size: [u32; 3],
    pub group_size: [u32; 3],
    pub buffers: Vec<TimelineBufferBinding>,
    pub synthetic: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimelineBufferBinding {
    pub index: usize,
    pub address: u64,
    pub name: Option<String>,
    pub usage: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimelineCounterTrack {
    pub name: String,
    pub unit: String,
    pub samples: Vec<TimelineCounterSample>,
    pub min_value: f64,
    pub max_value: f64,
    pub average_value: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimelineCounterSample {
    pub timestamp_ns: u64,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TimelineEvent {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    pub phase: String,
    pub timestamp_us: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_us: Option<u64>,
    pub process_id: u32,
    pub thread_id: u32,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub args: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RawTimelineReport {
    pub profiler_directory: Option<PathBuf>,
    pub file_count: usize,
    pub files: Vec<RawTimelineFile>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RawTimelineFile {
    pub path: PathBuf,
    pub file_index: Option<usize>,
    pub file_size: u64,
    pub header: Option<RawTimelineHeader>,
    pub header_timestamps: Vec<u64>,
    pub timestamp_span: Option<RawTimestampSpan>,
    pub heuristic_confidence: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RawTimelineHeader {
    pub magic: u64,
    pub flags: u64,
    pub counter_count: u32,
    pub data_offset: u64,
    pub entry_count: u64,
    pub gpu_timestamp: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RawTimestampSpan {
    pub first: u64,
    pub last: u64,
    pub count: usize,
}

#[derive(Debug)]
struct DispatchSpan {
    index: usize,
    encoder_address: Option<u64>,
    encoder_label: Option<String>,
    kernel_name: Option<String>,
    start_time_ns: u64,
    end_time_ns: u64,
    duration_ns: Option<u64>,
    grid_size: [u32; 3],
    group_size: [u32; 3],
    buffers: Vec<TimelineBufferBinding>,
    synthetic: bool,
}

pub fn report(trace: &TraceBundle) -> Result<TimelineReport> {
    let profiler_summary = profiler::stream_data_summary(&trace.path).ok();
    report_with_profiler_summary(trace, profiler_summary.as_ref())
}

pub fn report_with_profiler_summary(
    trace: &TraceBundle,
    profiler_summary: Option<&profiler::ProfilerStreamDataSummary>,
) -> Result<TimelineReport> {
    let regions = trace.command_buffer_regions()?;
    if let Some(profiler_summary) = profiler_summary {
        let counter_limiters = counter::extract_limiters_for_trace(&trace.path);
        return Ok(build(
            &regions,
            Some(profiler_summary),
            None,
            Some(&counter_limiters),
        ));
    }
    let raw_timings = profiler::raw_encoder_timings(&trace.path).ok();
    let counter_limiters = counter::extract_limiters_for_trace(&trace.path);
    Ok(build(
        &regions,
        None,
        raw_timings.as_deref(),
        Some(&counter_limiters),
    ))
}

pub fn build(
    regions: &[CommandBufferRegion],
    profiler_summary: Option<&profiler::ProfilerStreamDataSummary>,
    raw_encoder_timings: Option<&[profiler::ProfilerRawEncoderTiming]>,
    counter_limiters: Option<&[counter::CounterLimiter]>,
) -> TimelineReport {
    let mut command_buffers = Vec::new();
    let mut encoders = Vec::new();
    let mut dispatches = Vec::new();
    let mut events = Vec::new();
    let mut start_time_ns = u64::MAX;
    let mut end_time_ns = 0u64;
    let profiler_timeline = profiler_summary.and_then(|summary| summary.timeline.as_ref());
    let profiler_dispatches = profiler_summary.map(|summary| &summary.dispatches[..]);
    let command_buffers_profiler_backed =
        profiler_timeline.is_some_and(|timeline| !timeline.command_buffer_timestamps.is_empty());

    events.push(metadata_event(1, 0, "process_name", "gputrace timeline"));
    events.push(metadata_event(1, 1, "thread_name", "Command Buffers"));

    for (region_pos, region) in regions.iter().enumerate() {
        let (timestamp_ns, duration_ns, command_buffer_synthetic) =
            profiler_command_buffer_span(region_pos, regions, profiler_timeline);

        command_buffers.push(TimelineCommandBuffer {
            index: region.command_buffer.index,
            timestamp_ns,
            duration_ns,
            encoder_count: region.encoders.len(),
            dispatch_count: region.dispatches.len(),
        });

        start_time_ns = start_time_ns.min(timestamp_ns);
        let cb_end_ns = duration_ns
            .map(|duration| timestamp_ns.saturating_add(duration))
            .unwrap_or(timestamp_ns);
        end_time_ns = end_time_ns.max(cb_end_ns);

        events.push(TimelineEvent {
            name: format!("CB#{}", region.command_buffer.index),
            category: Some("command_buffer".to_owned()),
            phase: "X".to_owned(),
            timestamp_us: ns_to_us(timestamp_ns),
            duration_us: duration_ns.map(ns_to_us),
            process_id: 1,
            thread_id: 1,
            args: BTreeMap::from([
                ("index".to_owned(), json!(region.command_buffer.index)),
                ("encoder_count".to_owned(), json!(region.encoders.len())),
                ("dispatch_count".to_owned(), json!(region.dispatches.len())),
                (
                    "profiler_backed".to_owned(),
                    json!(!command_buffer_synthetic),
                ),
                ("synthetic".to_owned(), json!(command_buffer_synthetic)),
            ]),
        });

        let dispatch_spans = build_dispatch_spans(
            region,
            command_buffer_start_ticks(profiler_timeline, region_pos),
            timestamp_ns,
            duration_ns,
            profiler_timeline,
            profiler_dispatches,
            raw_encoder_timings,
        );
        let raw_encoder_spans =
            build_raw_encoder_spans(region, timestamp_ns, duration_ns, raw_encoder_timings);
        let mut encoder_dispatches: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for (dispatch_pos, span) in dispatch_spans.iter().enumerate() {
            if let Some(address) = span.encoder_address {
                encoder_dispatches
                    .entry(address)
                    .or_default()
                    .push(dispatch_pos);
            }
        }

        for encoder in &region.encoders {
            let label = encoder_label(&encoder.label, encoder.address);
            let thread_id = 1_000u32.saturating_add(encoder.index as u32);
            events.push(metadata_event(1, thread_id, "thread_name", &label));

            let dispatch_positions = encoder_dispatches.get(&encoder.address);
            let (encoder_start_ns, encoder_end_ns, encoder_duration_ns, synthetic) =
                if let Some(positions) = dispatch_positions {
                    let first = &dispatch_spans[*positions.first().unwrap()];
                    let last = &dispatch_spans[*positions.last().unwrap()];
                    let start = first.start_time_ns;
                    let end = last.end_time_ns.max(start);
                    let duration = end.checked_sub(start).filter(|duration| *duration > 0);
                    let synthetic = positions
                        .iter()
                        .any(|position| dispatch_spans[*position].synthetic());
                    (start, end, duration, synthetic)
                } else if let Some((start, end, duration)) = raw_encoder_spans.get(&encoder.address)
                {
                    (*start, *end, *duration, false)
                } else if let Some(cb_duration_ns) = duration_ns {
                    let slice = if region.encoders.is_empty() {
                        0
                    } else {
                        cb_duration_ns / region.encoders.len() as u64
                    };
                    let slot = region
                        .encoders
                        .iter()
                        .position(|candidate| candidate.address == encoder.address)
                        .unwrap_or(0) as u64;
                    let start = timestamp_ns.saturating_add(slice.saturating_mul(slot));
                    let end = start.saturating_add(slice);
                    let duration = (slice > 0).then_some(slice);
                    (start, end, duration, true)
                } else {
                    (timestamp_ns, timestamp_ns, None, true)
                };

            start_time_ns = start_time_ns.min(encoder_start_ns);
            end_time_ns = end_time_ns.max(encoder_end_ns);

            encoders.push(TimelineEncoder {
                index: encoder.index,
                command_buffer_index: region.command_buffer.index,
                label: encoder.label.clone(),
                address: encoder.address,
                dispatch_count: dispatch_positions.map_or(0, Vec::len),
                start_time_ns: encoder_start_ns,
                end_time_ns: encoder_end_ns,
                duration_ns: encoder_duration_ns,
                synthetic,
            });

            let mut args = BTreeMap::new();
            args.insert("index".to_owned(), json!(encoder.index));
            args.insert(
                "command_buffer_index".to_owned(),
                json!(region.command_buffer.index),
            );
            args.insert(
                "address".to_owned(),
                json!(format!("0x{:x}", encoder.address)),
            );
            args.insert(
                "dispatch_count".to_owned(),
                json!(dispatch_positions.map_or(0, Vec::len)),
            );
            args.insert("synthetic".to_owned(), json!(synthetic));

            events.push(TimelineEvent {
                name: label,
                category: Some("encoder".to_owned()),
                phase: "X".to_owned(),
                timestamp_us: ns_to_us(encoder_start_ns),
                duration_us: encoder_duration_ns.map(ns_to_us),
                process_id: 1,
                thread_id,
                args,
            });
        }

        for span in dispatch_spans {
            let encoder_index = span.encoder_address.and_then(|address| {
                region
                    .encoders
                    .iter()
                    .find(|encoder| encoder.address == address)
                    .map(|encoder| encoder.index)
            });
            let thread_id = 2_000u32
                .saturating_add(encoder_index.unwrap_or(region.command_buffer.index) as u32);
            let thread_name = span
                .encoder_label
                .as_deref()
                .map(|label| format!("Dispatches: {label}"))
                .unwrap_or_else(|| format!("Dispatches: CB#{}", region.command_buffer.index));
            events.push(metadata_event(1, thread_id, "thread_name", &thread_name));

            let event_name = span
                .kernel_name
                .clone()
                .or_else(|| span.encoder_label.clone())
                .unwrap_or_else(|| format!("dispatch_{}", span.index));

            let mut args = BTreeMap::new();
            args.insert("index".to_owned(), json!(span.index));
            args.insert(
                "command_buffer_index".to_owned(),
                json!(region.command_buffer.index),
            );
            if let Some(address) = span.encoder_address {
                args.insert(
                    "encoder_address".to_owned(),
                    json!(format!("0x{address:x}")),
                );
            }
            if let Some(label) = &span.encoder_label {
                args.insert("encoder_label".to_owned(), json!(label));
            }
            args.insert("grid_size".to_owned(), json!(span.grid_size));
            args.insert("group_size".to_owned(), json!(span.group_size));
            args.insert("buffer_count".to_owned(), json!(span.buffers.len()));
            args.insert("synthetic".to_owned(), json!(span.synthetic()));

            start_time_ns = start_time_ns.min(span.start_time_ns);
            end_time_ns = end_time_ns.max(span.end_time_ns);

            dispatches.push(TimelineDispatch {
                index: span.index,
                command_buffer_index: region.command_buffer.index,
                encoder_index,
                encoder_address: span.encoder_address,
                encoder_label: span.encoder_label.clone(),
                kernel_name: span.kernel_name.clone(),
                start_time_ns: span.start_time_ns,
                end_time_ns: span.end_time_ns,
                duration_ns: span.duration_ns,
                grid_size: span.grid_size,
                group_size: span.group_size,
                buffers: span.buffers.clone(),
                synthetic: span.synthetic(),
            });

            events.push(TimelineEvent {
                name: event_name,
                category: Some("dispatch".to_owned()),
                phase: "X".to_owned(),
                timestamp_us: ns_to_us(span.start_time_ns),
                duration_us: span.duration_ns.map(ns_to_us),
                process_id: 1,
                thread_id,
                args,
            });
        }
    }

    if start_time_ns == u64::MAX {
        start_time_ns = 0;
    }

    let duration_ns = end_time_ns.saturating_sub(start_time_ns);

    events.sort_by(|left, right| {
        left.timestamp_us
            .cmp(&right.timestamp_us)
            .then_with(|| left.thread_id.cmp(&right.thread_id))
            .then_with(|| left.name.cmp(&right.name))
    });

    encoders.sort_by(|left, right| {
        left.command_buffer_index
            .cmp(&right.command_buffer_index)
            .then_with(|| left.index.cmp(&right.index))
    });
    dispatches.sort_by_key(|left| left.index);

    let counter_tracks =
        build_counter_tracks(&encoders, &dispatches, profiler_summary, counter_limiters);
    append_counter_track_events(&mut events, &counter_tracks);
    events.sort_by(|left, right| {
        left.timestamp_us
            .cmp(&right.timestamp_us)
            .then_with(|| left.thread_id.cmp(&right.thread_id))
            .then_with(|| left.name.cmp(&right.name))
    });

    TimelineReport {
        synthetic: profiler_summary.is_none() && raw_encoder_timings.is_none(),
        source: if profiler_summary.is_some() {
            "streamData".to_owned()
        } else if raw_encoder_timings.is_some_and(|timings| !timings.is_empty()) {
            "raw-profiler-heuristic".to_owned()
        } else {
            "synthetic".to_owned()
        },
        command_buffers_profiler_backed,
        start_time_ns,
        end_time_ns,
        duration_ns,
        command_buffer_count: command_buffers.len(),
        encoder_count: encoders.len(),
        dispatch_count: dispatches.len(),
        counter_track_count: counter_tracks.len(),
        command_buffers,
        encoders,
        dispatches,
        counter_tracks,
        events,
    }
}

pub fn raw_report(trace: &TraceBundle) -> Result<RawTimelineReport> {
    build_raw_report_for_path(&trace.path)
}

pub fn export_json(report: &TimelineReport) -> Result<String> {
    Ok(serde_json::to_string_pretty(report)?)
}

pub fn export_raw_json(report: &RawTimelineReport) -> Result<String> {
    Ok(serde_json::to_string_pretty(report)?)
}

pub fn format_report(report: &TimelineReport) -> String {
    let mut out = String::new();
    if report.source == "streamData" && report.command_buffers_profiler_backed && !report.synthetic
    {
        out.push_str("Profiler-backed timeline report\n");
        out.push_str(
            "Command-buffer and dispatch spans come from profiler data when available; remaining gaps stay synthetic.\n\n",
        );
    } else if report.source == "streamData" && report.command_buffers_profiler_backed {
        out.push_str("Mixed timeline report\n");
        out.push_str(
            "Command-buffer spans come from profiler APSTimelineData; encoder and dispatch slices remain synthetic.\n\n",
        );
    } else if report.source == "raw-profiler-heuristic" {
        out.push_str("Raw-profiler timeline report\n");
        out.push_str(
            "Encoder spans come from Counters_f_* heuristic timing; command-buffer rows still follow trace ordering and dispatch slices stay synthetic inside encoder windows.\n\n",
        );
    } else {
        out.push_str("Synthetic timeline report\n");
        out.push_str("Derived from command-buffer ordering and adjacent timestamps.\n\n");
    }
    out.push_str(&format!(
        "start={} ns end={} ns duration={} ns command_buffers={} encoders={} dispatches={} counter_tracks={}\n\n",
        report.start_time_ns,
        report.end_time_ns,
        report.duration_ns,
        report.command_buffer_count,
        report.encoder_count,
        report.dispatch_count,
        report.counter_track_count
    ));

    for cb in &report.command_buffers {
        out.push_str(&format!(
            "CB#{} start={:.3} ms duration={} encoders={} dispatches={}\n",
            cb.index,
            ns_to_ms(cb.timestamp_ns),
            format_duration_ms(cb.duration_ns),
            cb.encoder_count,
            cb.dispatch_count
        ));

        for encoder in report
            .encoders
            .iter()
            .filter(|encoder| encoder.command_buffer_index == cb.index)
        {
            let label = encoder_label(&encoder.label, encoder.address);
            out.push_str(&format!(
                "  encoder {} start={:.3} ms duration={} dispatches={}\n",
                label,
                ns_to_ms(encoder.start_time_ns),
                format_duration_ms(encoder.duration_ns),
                encoder.dispatch_count
            ));

            for dispatch in report
                .dispatches
                .iter()
                .filter(|dispatch| dispatch.command_buffer_index == cb.index)
                .filter(|dispatch| dispatch.encoder_index == Some(encoder.index))
            {
                let name = dispatch
                    .kernel_name
                    .clone()
                    .unwrap_or_else(|| format!("dispatch_{}", dispatch.index));
                out.push_str(&format!(
                    "    dispatch {} start={:.3} ms duration={} grid={:?} group={:?}\n",
                    name,
                    ns_to_ms(dispatch.start_time_ns),
                    format_duration_ms(dispatch.duration_ns),
                    dispatch.grid_size,
                    dispatch.group_size
                ));
            }
        }

        for dispatch in report
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.command_buffer_index == cb.index)
            .filter(|dispatch| dispatch.encoder_index.is_none())
        {
            let name = dispatch
                .kernel_name
                .clone()
                .unwrap_or_else(|| format!("dispatch_{}", dispatch.index));
            out.push_str(&format!(
                "  dispatch {} start={:.3} ms duration={} grid={:?} group={:?}\n",
                name,
                ns_to_ms(dispatch.start_time_ns),
                format_duration_ms(dispatch.duration_ns),
                dispatch.grid_size,
                dispatch.group_size
            ));
        }
    }

    if !report.counter_tracks.is_empty() {
        out.push_str("\nCounter tracks:\n");
        for track in &report.counter_tracks {
            out.push_str(&format!(
                "  {} [{}] samples={} min={:.3} max={:.3} avg={:.3}\n",
                track.name,
                track.unit,
                track.samples.len(),
                track.min_value,
                track.max_value,
                track.average_value
            ));
        }
    }

    out
}

pub fn format_raw_report(report: &RawTimelineReport) -> String {
    let mut out = String::new();
    out.push_str("Raw timeline heuristic report\n");
    match &report.profiler_directory {
        Some(path) => out.push_str(&format!("profiler_directory={}\n", path.display())),
        None => out.push_str("profiler_directory=<not found>\n"),
    }
    out.push_str(&format!("files={}\n", report.file_count));
    for note in &report.notes {
        out.push_str(&format!("note: {note}\n"));
    }
    if !report.files.is_empty() {
        out.push('\n');
    }

    for file in &report.files {
        out.push_str(&format!("{}\n", file.path.display()));
        out.push_str(&format!(
            "  size={} confidence={}\n",
            file.file_size, file.heuristic_confidence
        ));
        if let Some(header) = &file.header {
            out.push_str(&format!(
                "  header magic=0x{:x} counters={} data_offset={} entries={} gpu_timestamp={}\n",
                header.magic,
                header.counter_count,
                header.data_offset,
                header.entry_count,
                header.gpu_timestamp
            ));
        } else {
            out.push_str("  header=<unavailable>\n");
        }
        if let Some(span) = &file.timestamp_span {
            out.push_str(&format!(
                "  header_timestamps={} first={} last={}\n",
                span.count, span.first, span.last
            ));
        } else {
            out.push_str("  header_timestamps=0\n");
        }
        for note in &file.notes {
            out.push_str(&format!("  note: {note}\n"));
        }
    }
    out
}

pub fn format_chrome_trace_json(report: &TimelineReport) -> Result<String> {
    format_trace_json(report)
}

pub fn format_perfetto_trace_json(report: &TimelineReport) -> Result<String> {
    format_trace_json(report)
}

fn format_trace_json(report: &TimelineReport) -> Result<String> {
    #[derive(Serialize)]
    struct ChromeTrace<'a> {
        #[serde(rename = "traceEvents")]
        trace_events: &'a [TimelineEvent],
        #[serde(rename = "displayTimeUnit")]
        display_time_unit: &'static str,
    }

    Ok(serde_json::to_string_pretty(&ChromeTrace {
        trace_events: &report.events,
        display_time_unit: "ms",
    })?)
}

fn metadata_event(process_id: u32, thread_id: u32, name: &str, value: &str) -> TimelineEvent {
    TimelineEvent {
        name: name.to_owned(),
        category: None,
        phase: "M".to_owned(),
        timestamp_us: 0,
        duration_us: None,
        process_id,
        thread_id,
        args: BTreeMap::from([("name".to_owned(), json!(value))]),
    }
}

fn profiler_command_buffer_span(
    region_pos: usize,
    regions: &[CommandBufferRegion],
    profiler_timeline: Option<&profiler::ProfilerTimelineInfo>,
) -> (u64, Option<u64>, bool) {
    if let Some(timeline) = profiler_timeline
        && let Some(entry) = timeline.command_buffer_timestamps.get(region_pos)
    {
        let first_start = timeline
            .command_buffer_timestamps
            .first()
            .map(|value| value.start_ticks)
            .unwrap_or(entry.start_ticks);
        let timestamp_ns = ticks_to_ns(
            entry.start_ticks.saturating_sub(first_start),
            timeline.timebase_numer,
            timeline.timebase_denom,
        );
        let duration_ns = ticks_to_ns(
            entry.end_ticks.saturating_sub(entry.start_ticks),
            timeline.timebase_numer,
            timeline.timebase_denom,
        );
        return (timestamp_ns, Some(duration_ns), false);
    }

    let timestamp_ns = regions[region_pos].command_buffer.timestamp;
    let next_timestamp = regions
        .get(region_pos + 1)
        .map(|next| next.command_buffer.timestamp);
    let duration_ns = next_timestamp.and_then(|next| {
        next.checked_sub(timestamp_ns)
            .filter(|duration| *duration > 0)
    });
    (timestamp_ns, duration_ns, true)
}

fn command_buffer_start_ticks(
    profiler_timeline: Option<&profiler::ProfilerTimelineInfo>,
    region_pos: usize,
) -> Option<u64> {
    profiler_timeline.and_then(|timeline| {
        timeline
            .command_buffer_timestamps
            .get(region_pos)
            .map(|entry| entry.start_ticks)
    })
}

fn build_dispatch_spans(
    region: &CommandBufferRegion,
    command_buffer_start_ticks: Option<u64>,
    command_buffer_start_ns: u64,
    duration_ns: Option<u64>,
    profiler_timeline: Option<&profiler::ProfilerTimelineInfo>,
    profiler_dispatches: Option<&[profiler::ProfilerDispatch]>,
    raw_encoder_timings: Option<&[profiler::ProfilerRawEncoderTiming]>,
) -> Vec<DispatchSpan> {
    let first_command_buffer_start_ticks = profiler_timeline.and_then(|timeline| {
        timeline
            .command_buffer_timestamps
            .first()
            .map(|entry| entry.start_ticks)
    });

    if let (
        Some(command_buffer_start_ticks),
        Some(first_start_ticks),
        Some(timeline),
        Some(dispatches),
    ) = (
        command_buffer_start_ticks,
        first_command_buffer_start_ticks,
        profiler_timeline,
        profiler_dispatches,
    ) {
        let command_buffer_end_ticks = timeline
            .command_buffer_timestamps
            .iter()
            .find(|entry| entry.start_ticks == command_buffer_start_ticks)
            .map(|entry| entry.end_ticks)
            .unwrap_or(command_buffer_start_ticks);
        let mut spans = Vec::with_capacity(region.dispatches.len());
        for dispatch in &region.dispatches {
            if let Some(profiler_dispatch) = dispatches.get(dispatch.index)
                && profiler_dispatch.end_ticks > profiler_dispatch.start_ticks
                && profiler_dispatch.start_ticks >= command_buffer_start_ticks
                && profiler_dispatch.end_ticks <= command_buffer_end_ticks
            {
                let start_time_ns = ticks_to_ns(
                    profiler_dispatch
                        .start_ticks
                        .saturating_sub(first_start_ticks),
                    timeline.timebase_numer,
                    timeline.timebase_denom,
                );
                let end_time_ns = ticks_to_ns(
                    profiler_dispatch
                        .end_ticks
                        .saturating_sub(first_start_ticks),
                    timeline.timebase_numer,
                    timeline.timebase_denom,
                );
                let (encoder_label, encoder_address) = dispatch
                    .encoder_id
                    .and_then(|address| {
                        region
                            .encoders
                            .iter()
                            .find(|encoder| encoder.address == address)
                            .map(|encoder| (Some(encoder.label.clone()), Some(encoder.address)))
                    })
                    .unwrap_or((None, dispatch.encoder_id));

                spans.push(DispatchSpan {
                    index: dispatch.index,
                    encoder_address,
                    encoder_label,
                    kernel_name: dispatch.kernel_name.clone(),
                    start_time_ns,
                    end_time_ns,
                    duration_ns: end_time_ns
                        .checked_sub(start_time_ns)
                        .filter(|duration| *duration > 0),
                    grid_size: dispatch.grid_size,
                    group_size: dispatch.group_size,
                    buffers: dispatch
                        .buffers
                        .iter()
                        .map(bound_buffer_to_timeline)
                        .collect(),
                    synthetic: false,
                });
                continue;
            }
        }
        if spans.len() == region.dispatches.len() {
            return spans;
        }
    }

    let raw_encoder_spans = build_raw_encoder_spans(
        region,
        command_buffer_start_ns,
        duration_ns,
        raw_encoder_timings,
    );
    if !raw_encoder_spans.is_empty() {
        let mut per_encoder_dispatches = BTreeMap::<u64, usize>::new();
        for dispatch in &region.dispatches {
            if let Some(address) = dispatch.encoder_id {
                *per_encoder_dispatches.entry(address).or_default() += 1;
            }
        }

        let mut spans = Vec::with_capacity(region.dispatches.len());
        let mut encoder_progress = BTreeMap::<u64, u64>::new();
        for dispatch in &region.dispatches {
            let (encoder_label, encoder_address) = dispatch
                .encoder_id
                .and_then(|address| {
                    region
                        .encoders
                        .iter()
                        .find(|encoder| encoder.address == address)
                        .map(|encoder| (Some(encoder.label.clone()), Some(encoder.address)))
                })
                .unwrap_or((None, dispatch.encoder_id));

            let (start_time_ns, end_time_ns, duration_ns, synthetic) = if let Some(address) =
                encoder_address
            {
                if let Some((encoder_start_ns, _encoder_end_ns, encoder_duration_ns)) =
                    raw_encoder_spans.get(&address)
                {
                    let dispatch_count = per_encoder_dispatches.get(&address).copied().unwrap_or(1);
                    let per_dispatch_duration = encoder_duration_ns
                        .map(|duration| duration / dispatch_count as u64)
                        .filter(|duration| *duration > 0);
                    let offset = encoder_progress.entry(address).or_default();
                    let start = encoder_start_ns.saturating_add(*offset);
                    let end = per_dispatch_duration
                        .map(|duration| start.saturating_add(duration))
                        .unwrap_or(start);
                    if let Some(duration) = per_dispatch_duration {
                        *offset = offset.saturating_add(duration);
                    }
                    (start, end, per_dispatch_duration, false)
                } else {
                    (command_buffer_start_ns, command_buffer_start_ns, None, true)
                }
            } else {
                (command_buffer_start_ns, command_buffer_start_ns, None, true)
            };

            spans.push(DispatchSpan {
                index: dispatch.index,
                encoder_address,
                encoder_label,
                kernel_name: dispatch.kernel_name.clone(),
                start_time_ns,
                end_time_ns,
                duration_ns,
                grid_size: dispatch.grid_size,
                group_size: dispatch.group_size,
                buffers: dispatch
                    .buffers
                    .iter()
                    .map(bound_buffer_to_timeline)
                    .collect(),
                synthetic,
            });
        }
        return spans;
    }

    let dispatch_count = region.dispatches.len();
    let per_dispatch_duration = duration_ns.and_then(|duration| {
        if dispatch_count == 0 {
            None
        } else {
            let slice = duration / dispatch_count as u64;
            (slice > 0).then_some(slice)
        }
    });

    let mut current_start_ns = command_buffer_start_ns;
    let mut spans = Vec::with_capacity(dispatch_count);
    for dispatch in &region.dispatches {
        let span_duration_ns = per_dispatch_duration;
        let span_end_ns = span_duration_ns
            .map(|duration| current_start_ns.saturating_add(duration))
            .unwrap_or(current_start_ns);
        let (encoder_label, encoder_address) = dispatch
            .encoder_id
            .and_then(|address| {
                region
                    .encoders
                    .iter()
                    .find(|encoder| encoder.address == address)
                    .map(|encoder| (Some(encoder.label.clone()), Some(encoder.address)))
            })
            .unwrap_or((None, dispatch.encoder_id));

        spans.push(DispatchSpan {
            index: dispatch.index,
            encoder_address,
            encoder_label,
            kernel_name: dispatch.kernel_name.clone(),
            start_time_ns: current_start_ns,
            end_time_ns: span_end_ns,
            duration_ns: span_duration_ns,
            grid_size: dispatch.grid_size,
            group_size: dispatch.group_size,
            buffers: dispatch
                .buffers
                .iter()
                .map(bound_buffer_to_timeline)
                .collect(),
            synthetic: true,
        });
        current_start_ns = span_end_ns;
    }
    spans
}

fn build_raw_encoder_spans(
    region: &CommandBufferRegion,
    command_buffer_start_ns: u64,
    duration_ns: Option<u64>,
    raw_encoder_timings: Option<&[profiler::ProfilerRawEncoderTiming]>,
) -> BTreeMap<u64, (u64, u64, Option<u64>)> {
    let Some(raw_encoder_timings) = raw_encoder_timings else {
        return BTreeMap::new();
    };

    let duration_by_index = raw_encoder_timings
        .iter()
        .map(|timing| (timing.index, timing.duration_ns))
        .collect::<BTreeMap<_, _>>();
    let known_total = region
        .encoders
        .iter()
        .filter_map(|encoder| duration_by_index.get(&encoder.index).copied())
        .sum::<u64>();
    let unknown_count = region
        .encoders
        .iter()
        .filter(|encoder| !duration_by_index.contains_key(&encoder.index))
        .count();
    let fallback_per_unknown = duration_ns
        .and_then(|duration| duration.checked_sub(known_total))
        .map(|remaining| remaining / unknown_count.max(1) as u64)
        .filter(|duration| *duration > 0);

    let mut current_start_ns = command_buffer_start_ns;
    let mut spans = BTreeMap::new();
    for encoder in &region.encoders {
        let encoder_duration_ns = duration_by_index
            .get(&encoder.index)
            .copied()
            .or(fallback_per_unknown);
        let end_time_ns = encoder_duration_ns
            .map(|duration| current_start_ns.saturating_add(duration))
            .unwrap_or(current_start_ns);
        spans.insert(
            encoder.address,
            (current_start_ns, end_time_ns, encoder_duration_ns),
        );
        current_start_ns = end_time_ns;
    }
    spans
}

fn bound_buffer_to_timeline(buffer: &BoundBuffer) -> TimelineBufferBinding {
    TimelineBufferBinding {
        index: buffer.index,
        address: buffer.address,
        name: buffer.name.clone(),
        usage: buffer.usage.to_string(),
    }
}

fn build_counter_tracks(
    encoders: &[TimelineEncoder],
    dispatches: &[TimelineDispatch],
    profiler_summary: Option<&profiler::ProfilerStreamDataSummary>,
    counter_limiters: Option<&[counter::CounterLimiter]>,
) -> Vec<TimelineCounterTrack> {
    if encoders.is_empty() {
        return Vec::new();
    }

    let encoder_by_index = encoders
        .iter()
        .map(|encoder| (encoder.index, encoder))
        .collect::<BTreeMap<_, _>>();
    let mut tracks = Vec::new();

    if let Some(counter_limiters) = counter_limiters
        && !counter_limiters.is_empty()
    {
        let mut limiter_tracks = vec![
            CounterTrackBuilder::new("Compute Occupancy", "%"),
            CounterTrackBuilder::new("ALU Utilization", "%"),
            CounterTrackBuilder::new("Shader Launch Limiter", "%"),
            CounterTrackBuilder::new("Instruction Throughput", "%"),
            CounterTrackBuilder::new("Integer Complex", "%"),
            CounterTrackBuilder::new("F32 Limiter", "%"),
            CounterTrackBuilder::new("L1 Cache", "%"),
            CounterTrackBuilder::new("Last Level Cache", "%"),
            CounterTrackBuilder::new("Control Flow", "%"),
            CounterTrackBuilder::new("Device Memory Bandwidth", "GB/s"),
            CounterTrackBuilder::new("Buffer L1 Read Bandwidth", "GB/s"),
            CounterTrackBuilder::new("Buffer L1 Write Bandwidth", "GB/s"),
        ];

        for limiter in counter_limiters {
            let Some(encoder) = encoder_by_index.get(&limiter.encoder_index) else {
                continue;
            };
            let start = encoder.start_time_ns;
            let end = encoder.end_time_ns.max(start);
            if let Some(value) = limiter.occupancy {
                limiter_tracks[0].push(start, end, value);
            }
            if let Some(value) = limiter.alu_utilization {
                limiter_tracks[1].push(start, end, value);
            }
            if let Some(value) = limiter.compute_shader_launch {
                limiter_tracks[2].push(start, end, value);
            }
            if let Some(value) = limiter.instruction_throughput {
                limiter_tracks[3].push(start, end, value);
            }
            if let Some(value) = limiter.integer_complex {
                limiter_tracks[4].push(start, end, value);
            }
            if let Some(value) = limiter.f32_limiter {
                limiter_tracks[5].push(start, end, value);
            }
            if let Some(value) = limiter.l1_cache {
                limiter_tracks[6].push(start, end, value);
            }
            if let Some(value) = limiter.last_level_cache {
                limiter_tracks[7].push(start, end, value);
            }
            if let Some(value) = limiter.control_flow {
                limiter_tracks[8].push(start, end, value);
            }
            if let Some(value) = limiter.device_memory_bandwidth_gbps {
                limiter_tracks[9].push(start, end, value);
            }
            if let Some(value) = limiter.buffer_l1_read_bandwidth_gbps {
                limiter_tracks[10].push(start, end, value);
            }
            if let Some(value) = limiter.buffer_l1_write_bandwidth_gbps {
                limiter_tracks[11].push(start, end, value);
            }
        }
        tracks.extend(
            limiter_tracks
                .into_iter()
                .filter_map(CounterTrackBuilder::build),
        );
    }

    if let Some(profiler_summary) = profiler_summary {
        let mut encoder_kernel_counts = BTreeMap::<usize, BTreeMap<String, usize>>::new();
        for dispatch in dispatches {
            if let (Some(encoder_index), Some(kernel_name)) =
                (dispatch.encoder_index, dispatch.kernel_name.as_ref())
            {
                *encoder_kernel_counts
                    .entry(encoder_index)
                    .or_default()
                    .entry(kernel_name.clone())
                    .or_default() += 1;
            }
        }

        let pipeline_stats_by_name = profiler_summary
            .pipelines
            .iter()
            .filter_map(|pipeline| {
                Some((
                    pipeline.function_name.clone()?,
                    pipeline.stats.as_ref()?.clone(),
                ))
            })
            .collect::<BTreeMap<_, _>>();

        let mut pipeline_tracks = vec![
            CounterTrackBuilder::new("Total Instructions", "count"),
            CounterTrackBuilder::new("ALU Instructions", "count"),
            CounterTrackBuilder::new("Branch Instructions", "count"),
            CounterTrackBuilder::new("Threadgroup Memory", "bytes"),
            CounterTrackBuilder::new("Temporary Registers", "count"),
            CounterTrackBuilder::new("Uniform Registers", "count"),
            CounterTrackBuilder::new("Spilled Bytes", "bytes"),
        ];

        for encoder in encoders {
            let Some(kernel_counts) = encoder_kernel_counts.get(&encoder.index) else {
                continue;
            };
            let Some((kernel_name, _)) = kernel_counts
                .iter()
                .max_by(|left, right| left.1.cmp(right.1).then_with(|| left.0.cmp(right.0)))
            else {
                continue;
            };
            let Some(stats) = pipeline_stats_by_name.get(kernel_name) else {
                continue;
            };
            let start = encoder.start_time_ns;
            let end = encoder.end_time_ns.max(start);
            pipeline_tracks[0].push(start, end, stats.instruction_count as f64);
            pipeline_tracks[1].push(start, end, stats.alu_instruction_count as f64);
            pipeline_tracks[2].push(start, end, stats.branch_instruction_count as f64);
            pipeline_tracks[3].push(start, end, stats.threadgroup_memory as f64);
            pipeline_tracks[4].push(start, end, stats.temporary_register_count as f64);
            pipeline_tracks[5].push(start, end, stats.uniform_register_count as f64);
            pipeline_tracks[6].push(start, end, stats.spilled_bytes as f64);
        }

        tracks.extend(
            pipeline_tracks
                .into_iter()
                .filter_map(CounterTrackBuilder::build),
        );
    }
    tracks
}

fn append_counter_track_events(
    events: &mut Vec<TimelineEvent>,
    counter_tracks: &[TimelineCounterTrack],
) {
    let mut thread_id = 10_000u32;
    for track in counter_tracks {
        events.push(metadata_event(1, thread_id, "thread_name", &track.name));
        for sample in &track.samples {
            let mut args = BTreeMap::new();
            args.insert("value".to_owned(), json!(sample.value));
            args.insert("unit".to_owned(), json!(track.unit));
            events.push(TimelineEvent {
                name: track.name.clone(),
                category: Some("counter".to_owned()),
                phase: "C".to_owned(),
                timestamp_us: ns_to_us(sample.timestamp_ns),
                duration_us: None,
                process_id: 1,
                thread_id,
                args,
            });
        }
        thread_id = thread_id.saturating_add(1);
    }
}

#[derive(Debug)]
struct CounterTrackBuilder {
    name: String,
    unit: String,
    samples: Vec<TimelineCounterSample>,
}

impl CounterTrackBuilder {
    fn new(name: &str, unit: &str) -> Self {
        Self {
            name: name.to_owned(),
            unit: unit.to_owned(),
            samples: Vec::new(),
        }
    }

    fn push(&mut self, start_time_ns: u64, end_time_ns: u64, value: f64) {
        self.samples.push(TimelineCounterSample {
            timestamp_ns: start_time_ns,
            value,
        });
        if end_time_ns != start_time_ns {
            self.samples.push(TimelineCounterSample {
                timestamp_ns: end_time_ns,
                value,
            });
        }
    }

    fn build(mut self) -> Option<TimelineCounterTrack> {
        if self.samples.is_empty() {
            return None;
        }
        self.samples.sort_by_key(|left| left.timestamp_ns);
        let mut min_value = f64::INFINITY;
        let mut max_value = f64::NEG_INFINITY;
        let mut total = 0.0;
        for sample in &self.samples {
            min_value = min_value.min(sample.value);
            max_value = max_value.max(sample.value);
            total += sample.value;
        }
        Some(TimelineCounterTrack {
            name: self.name,
            unit: self.unit,
            average_value: total / self.samples.len() as f64,
            min_value,
            max_value,
            samples: self.samples,
        })
    }
}

fn build_raw_report_for_path(trace_path: &Path) -> Result<RawTimelineReport> {
    let profiler_directory = find_profiler_directory(trace_path);
    let mut files = Vec::new();
    let mut notes = vec![
        "Header timestamps are low-confidence profiler sampling candidates, not GPU execution spans."
            .to_owned(),
        "Command-buffer and dispatch durations in the main timeline remain synthetic unless real profiler timing is added."
            .to_owned(),
    ];

    if let Some(dir) = &profiler_directory {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if !name.starts_with("Timeline_f_") || !name.ends_with(".raw") {
                continue;
            }
            files.push(parse_raw_timeline_file(&entry.path())?);
        }
        files.sort_by(|left, right| left.path.cmp(&right.path));
        if files.is_empty() {
            notes
                .push("No Timeline_f_*.raw files were found in the profiler directory.".to_owned());
        }
    } else {
        notes.push(
            "No .gpuprofiler_raw directory was found next to or inside the trace bundle."
                .to_owned(),
        );
    }

    Ok(RawTimelineReport {
        profiler_directory,
        file_count: files.len(),
        files,
        notes,
    })
}

fn find_profiler_directory(trace_path: &Path) -> Option<PathBuf> {
    if trace_path.is_dir()
        && trace_path
            .extension()
            .is_some_and(|ext| ext == "gpuprofiler_raw")
    {
        return Some(trace_path.to_path_buf());
    }

    let direct = PathBuf::from(format!("{}.gpuprofiler_raw", trace_path.display()));
    if direct.is_dir() {
        return Some(direct);
    }

    fs::read_dir(trace_path)
        .ok()?
        .filter_map(|entry| entry.ok())
        .find_map(|entry| {
            if entry.file_type().ok()?.is_dir()
                && entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext == "gpuprofiler_raw")
            {
                Some(entry.path())
            } else {
                None
            }
        })
}

fn parse_raw_timeline_file(path: &Path) -> Result<RawTimelineFile> {
    let data = fs::read(path)?;
    let file_size = data.len() as u64;
    let header = parse_raw_timeline_header(&data);
    let header_timestamps = extract_header_timestamps(&data);
    let timestamp_span = match (header_timestamps.first(), header_timestamps.last()) {
        (Some(first), Some(last)) => Some(RawTimestampSpan {
            first: *first,
            last: *last,
            count: header_timestamps.len(),
        }),
        _ => None,
    };
    let mut notes = Vec::new();
    if header.is_none() {
        notes.push(format!(
            "file too small: {} bytes (expected > 0x3c000, matching Go ParseTimelineRaw)",
            data.len()
        ));
    }
    if header_timestamps.is_empty() {
        notes.push(
            "No timestamp-like values were found in the 500..800 byte heuristic scan window."
                .to_owned(),
        );
    }
    notes.push("Parsing stops at header/sample discovery; packed payload decoding is intentionally not guessed.".to_owned());

    Ok(RawTimelineFile {
        path: path.to_path_buf(),
        file_index: parse_timeline_file_index(path),
        file_size,
        header,
        header_timestamps,
        timestamp_span,
        heuristic_confidence: "low".to_owned(),
        notes,
    })
}

fn parse_timeline_file_index(path: &Path) -> Option<usize> {
    let name = path.file_name()?.to_str()?;
    let index = name.strip_prefix("Timeline_f_")?.strip_suffix(".raw")?;
    index.parse().ok()
}

fn parse_raw_timeline_header(data: &[u8]) -> Option<RawTimelineHeader> {
    if data.len() < 0x3c000 {
        return None;
    }
    Some(RawTimelineHeader {
        magic: u64::from_le_bytes(data[0..8].try_into().ok()?),
        flags: u64::from_le_bytes(data[8..16].try_into().ok()?),
        counter_count: u32::from_le_bytes(data[12..16].try_into().ok()?),
        data_offset: u32::from_le_bytes(data[32..36].try_into().ok()?) as u64,
        entry_count: u64::from_le_bytes(data[80..88].try_into().ok()?),
        gpu_timestamp: u64::from_le_bytes(data[104..112].try_into().ok()?),
    })
}

fn extract_header_timestamps(data: &[u8]) -> Vec<u64> {
    let start = 500usize.min(data.len());
    let end = 800usize.min(data.len());
    if end <= start {
        return Vec::new();
    }

    let mut timestamps = Vec::new();
    let window = &data[start..end];
    let mut offset = 0usize;
    while offset + 8 <= window.len() {
        let candidate = u64::from_le_bytes(window[offset..offset + 8].try_into().unwrap());
        if candidate > 100_000 {
            timestamps.push(candidate);
        }
        offset += 8;
    }
    timestamps
}

fn encoder_label(label: &str, address: u64) -> String {
    if label.is_empty() {
        format!("0x{address:x}")
    } else {
        label.to_owned()
    }
}

fn ns_to_us(value: u64) -> u64 {
    value / 1_000
}

fn ticks_to_ns(ticks: u64, numer: u64, denom: u64) -> u64 {
    ticks.saturating_mul(numer.max(1)) / denom.max(1)
}

fn ns_to_ms(value: u64) -> f64 {
    value as f64 / 1_000_000.0
}

fn format_duration_ms(duration_ns: Option<u64>) -> String {
    duration_ns
        .map(|duration| format!("{:.3} ms", ns_to_ms(duration)))
        .unwrap_or_else(|| "?".to_owned())
}

impl DispatchSpan {
    fn synthetic(&self) -> bool {
        self.synthetic
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use tempfile::tempdir;

    use crate::trace::{CommandBuffer, ComputeEncoder, DispatchCall, MTLResourceUsage};

    fn profiler_summary_with_timeline(
        timeline: profiler::ProfilerTimelineInfo,
        dispatches: Vec<profiler::ProfilerDispatch>,
    ) -> profiler::ProfilerStreamDataSummary {
        profiler::ProfilerStreamDataSummary {
            function_names: vec![],
            pipelines: vec![],
            execution_costs: vec![],
            dispatches,
            encoder_timings: vec![],
            timeline: Some(timeline),
            num_pipelines: 0,
            num_gpu_commands: 0,
            num_encoders: 0,
            total_time_us: 0,
        }
    }

    #[test]
    fn builds_synthetic_timeline_report() {
        let regions = vec![
            CommandBufferRegion {
                command_buffer: CommandBuffer {
                    index: 0,
                    timestamp: 1_000_000,
                    offset: 0,
                    artifact_hashes: Vec::new(),
                },
                end_offset: 100,
                encoders: vec![ComputeEncoder {
                    index: 0,
                    address: 0x10,
                    label: "main".to_owned(),
                    offset: 10,
                }],
                pipeline_events: vec![],
                dispatches: vec![
                    DispatchCall {
                        index: 0,
                        offset: 20,
                        encoder_id: Some(0x10),
                        pipeline_addr: Some(0xaa),
                        kernel_name: Some("k0".to_owned()),
                        buffers: vec![BoundBuffer {
                            address: 0x99,
                            name: Some("buf0".to_owned()),
                            index: 0,
                            usage: MTLResourceUsage::READ,
                        }],
                        grid_size: [8, 1, 1],
                        group_size: [4, 1, 1],
                    },
                    DispatchCall {
                        index: 1,
                        offset: 30,
                        encoder_id: Some(0x10),
                        pipeline_addr: Some(0xbb),
                        kernel_name: Some("k1".to_owned()),
                        buffers: vec![],
                        grid_size: [16, 1, 1],
                        group_size: [8, 1, 1],
                    },
                ],
            },
            CommandBufferRegion {
                command_buffer: CommandBuffer {
                    index: 1,
                    timestamp: 5_000_000,
                    offset: 100,
                    artifact_hashes: Vec::new(),
                },
                end_offset: 200,
                encoders: vec![],
                pipeline_events: vec![],
                dispatches: vec![],
            },
        ];

        let report = build(&regions, None, None, None);

        assert!(report.synthetic);
        assert_eq!(report.source, "synthetic");
        assert!(!report.command_buffers_profiler_backed);
        assert_eq!(report.command_buffer_count, 2);
        assert_eq!(report.encoder_count, 1);
        assert_eq!(report.dispatch_count, 2);
        assert_eq!(report.duration_ns, 4_000_000);
        assert_eq!(report.command_buffers[0].duration_ns, Some(4_000_000));
        assert_eq!(report.encoders[0].duration_ns, Some(4_000_000));
        assert_eq!(report.dispatches[0].duration_ns, Some(2_000_000));
        assert_eq!(report.dispatches[1].start_time_ns, 3_000_000);

        let text = format_report(&report);
        assert!(text.contains("CB#0"));
        assert!(text.contains("dispatch k0"));

        let json = export_json(&report).unwrap();
        assert!(json.contains("\"command_buffer_count\": 2"));

        let chrome = format_chrome_trace_json(&report).unwrap();
        assert!(chrome.contains("\"traceEvents\""));
        assert!(chrome.contains("\"displayTimeUnit\": \"ms\""));
    }

    #[test]
    fn builds_profiler_backed_command_buffer_spans() {
        let regions = vec![
            CommandBufferRegion {
                command_buffer: CommandBuffer {
                    index: 0,
                    timestamp: 1_000_000,
                    offset: 0,
                    artifact_hashes: Vec::new(),
                },
                end_offset: 100,
                encoders: vec![],
                pipeline_events: vec![],
                dispatches: vec![],
            },
            CommandBufferRegion {
                command_buffer: CommandBuffer {
                    index: 1,
                    timestamp: 5_000_000,
                    offset: 100,
                    artifact_hashes: Vec::new(),
                },
                end_offset: 200,
                encoders: vec![],
                pipeline_events: vec![],
                dispatches: vec![],
            },
        ];

        let timeline = profiler::ProfilerTimelineInfo {
            command_buffer_timestamps: vec![
                profiler::ProfilerCommandBufferTimestamp {
                    index: 0,
                    start_ticks: 100,
                    end_ticks: 160,
                },
                profiler::ProfilerCommandBufferTimestamp {
                    index: 1,
                    start_ticks: 200,
                    end_ticks: 320,
                },
            ],
            encoder_profiles: vec![],
            timebase_numer: 125,
            timebase_denom: 5,
            absolute_time: 99_999,
        };

        let summary = profiler_summary_with_timeline(timeline, vec![]);
        let report = build(&regions, Some(&summary), None, None);

        assert!(!report.synthetic);
        assert_eq!(report.source, "streamData");
        assert!(report.command_buffers_profiler_backed);
        assert_eq!(report.command_buffers[0].timestamp_ns, 0);
        assert_eq!(report.command_buffers[0].duration_ns, Some(1_500));
        assert_eq!(report.command_buffers[1].timestamp_ns, 2_500);

        let text = format_report(&report);
        assert!(text.contains("Profiler-backed timeline report"));
    }

    #[test]
    fn builds_profiler_backed_dispatch_spans() {
        let regions = vec![CommandBufferRegion {
            command_buffer: CommandBuffer {
                index: 0,
                timestamp: 1_000_000,
                offset: 0,
                artifact_hashes: Vec::new(),
            },
            end_offset: 100,
            encoders: vec![ComputeEncoder {
                index: 0,
                address: 0x10,
                label: "main".to_owned(),
                offset: 10,
            }],
            pipeline_events: vec![],
            dispatches: vec![
                DispatchCall {
                    index: 0,
                    offset: 20,
                    encoder_id: Some(0x10),
                    pipeline_addr: Some(0xaa),
                    kernel_name: Some("k0".to_owned()),
                    buffers: vec![],
                    grid_size: [8, 1, 1],
                    group_size: [4, 1, 1],
                },
                DispatchCall {
                    index: 1,
                    offset: 30,
                    encoder_id: Some(0x10),
                    pipeline_addr: Some(0xbb),
                    kernel_name: Some("k1".to_owned()),
                    buffers: vec![],
                    grid_size: [16, 1, 1],
                    group_size: [8, 1, 1],
                },
            ],
        }];

        let timeline = profiler::ProfilerTimelineInfo {
            command_buffer_timestamps: vec![profiler::ProfilerCommandBufferTimestamp {
                index: 0,
                start_ticks: 100,
                end_ticks: 220,
            }],
            encoder_profiles: vec![],
            timebase_numer: 10,
            timebase_denom: 1,
            absolute_time: 0,
        };
        let dispatches = vec![
            profiler::ProfilerDispatch {
                index: 0,
                pipeline_index: 0,
                pipeline_id: None,
                function_name: Some("k0".to_owned()),
                encoder_index: 0,
                cumulative_us: 50,
                duration_us: 50,
                sample_count: 2,
                sampling_density: 0.04,
                start_ticks: 100,
                end_ticks: 140,
            },
            profiler::ProfilerDispatch {
                index: 1,
                pipeline_index: 1,
                pipeline_id: None,
                function_name: Some("k1".to_owned()),
                encoder_index: 0,
                cumulative_us: 120,
                duration_us: 70,
                sample_count: 3,
                sampling_density: 0.043,
                start_ticks: 140,
                end_ticks: 220,
            },
        ];

        let summary = profiler_summary_with_timeline(timeline, dispatches);
        let report = build(&regions, Some(&summary), None, None);

        assert!(!report.synthetic);
        assert_eq!(report.source, "streamData");
        assert!(report.command_buffers_profiler_backed);
        assert_eq!(report.dispatches[0].start_time_ns, 0);
        assert_eq!(report.dispatches[0].duration_ns, Some(400));
        assert_eq!(report.dispatches[1].start_time_ns, 400);
        assert_eq!(report.dispatches[1].duration_ns, Some(800));
        assert!(!report.dispatches[0].synthetic);
        assert!(!report.encoders[0].synthetic);

        let text = format_report(&report);
        assert!(text.contains("Profiler-backed timeline report"));
    }

    #[test]
    fn builds_raw_profiler_backed_encoder_and_dispatch_spans() {
        let regions = vec![
            CommandBufferRegion {
                command_buffer: CommandBuffer {
                    index: 0,
                    timestamp: 1_000_000,
                    offset: 0,
                    artifact_hashes: Vec::new(),
                },
                end_offset: 100,
                encoders: vec![
                    ComputeEncoder {
                        index: 0,
                        address: 0x10,
                        label: "main".to_owned(),
                        offset: 10,
                    },
                    ComputeEncoder {
                        index: 1,
                        address: 0x20,
                        label: "aux".to_owned(),
                        offset: 11,
                    },
                ],
                pipeline_events: vec![],
                dispatches: vec![
                    DispatchCall {
                        index: 0,
                        offset: 20,
                        encoder_id: Some(0x10),
                        pipeline_addr: Some(0xaa),
                        kernel_name: Some("k0".to_owned()),
                        buffers: vec![],
                        grid_size: [8, 1, 1],
                        group_size: [4, 1, 1],
                    },
                    DispatchCall {
                        index: 1,
                        offset: 30,
                        encoder_id: Some(0x10),
                        pipeline_addr: Some(0xbb),
                        kernel_name: Some("k1".to_owned()),
                        buffers: vec![],
                        grid_size: [16, 1, 1],
                        group_size: [8, 1, 1],
                    },
                    DispatchCall {
                        index: 2,
                        offset: 40,
                        encoder_id: Some(0x20),
                        pipeline_addr: Some(0xcc),
                        kernel_name: Some("k2".to_owned()),
                        buffers: vec![],
                        grid_size: [4, 1, 1],
                        group_size: [4, 1, 1],
                    },
                ],
            },
            CommandBufferRegion {
                command_buffer: CommandBuffer {
                    index: 1,
                    timestamp: 5_000_000,
                    offset: 100,
                    artifact_hashes: Vec::new(),
                },
                end_offset: 200,
                encoders: vec![],
                pipeline_events: vec![],
                dispatches: vec![],
            },
        ];

        let raw_timings = vec![
            profiler::ProfilerRawEncoderTiming {
                index: 0,
                duration_ns: 2_000_000,
                confidence_milli: 300,
            },
            profiler::ProfilerRawEncoderTiming {
                index: 1,
                duration_ns: 1_000_000,
                confidence_milli: 300,
            },
        ];

        let report = build(&regions, None, Some(&raw_timings), None);

        assert!(!report.synthetic);
        assert_eq!(report.source, "raw-profiler-heuristic");
        assert!(!report.command_buffers_profiler_backed);
        assert_eq!(report.encoders[0].duration_ns, Some(2_000_000));
        assert_eq!(report.encoders[1].duration_ns, Some(1_000_000));
        assert!(!report.encoders[0].synthetic);
        assert!(!report.dispatches[0].synthetic);
        assert_eq!(report.dispatches[0].start_time_ns, 1_000_000);
        assert_eq!(report.dispatches[0].duration_ns, Some(1_000_000));
        assert_eq!(report.dispatches[1].start_time_ns, 2_000_000);
        assert_eq!(report.dispatches[2].start_time_ns, 3_000_000);

        let text = format_report(&report);
        assert!(text.contains("Raw-profiler timeline report"));
    }

    #[test]
    fn builds_counter_tracks_from_real_counter_limiters() {
        let encoders = vec![
            TimelineEncoder {
                index: 0,
                command_buffer_index: 0,
                label: "main".into(),
                address: 0x10,
                dispatch_count: 2,
                start_time_ns: 1_000,
                end_time_ns: 5_000,
                duration_ns: Some(4_000),
                synthetic: false,
            },
            TimelineEncoder {
                index: 1,
                command_buffer_index: 0,
                label: "aux".into(),
                address: 0x20,
                dispatch_count: 1,
                start_time_ns: 5_000,
                end_time_ns: 8_000,
                duration_ns: Some(3_000),
                synthetic: false,
            },
        ];
        let limiters = vec![
            counter::CounterLimiter {
                encoder_index: 0,
                occupancy: Some(80.0),
                alu_utilization: Some(62.0),
                compute_shader_launch: Some(0.12),
                instruction_throughput: Some(2.4),
                integer_complex: Some(1.1),
                control_flow: Some(0.09),
                f32_limiter: Some(6.5),
                l1_cache: Some(0.08),
                last_level_cache: Some(0.07),
                device_memory_bandwidth_gbps: Some(3.2),
                buffer_l1_read_bandwidth_gbps: Some(1.4),
                buffer_l1_write_bandwidth_gbps: Some(0.8),
            },
            counter::CounterLimiter {
                encoder_index: 1,
                occupancy: Some(75.0),
                alu_utilization: Some(55.0),
                compute_shader_launch: Some(0.10),
                instruction_throughput: Some(2.0),
                integer_complex: Some(0.9),
                control_flow: Some(0.05),
                f32_limiter: Some(5.0),
                l1_cache: Some(0.06),
                last_level_cache: Some(0.04),
                device_memory_bandwidth_gbps: Some(2.8),
                buffer_l1_read_bandwidth_gbps: Some(1.2),
                buffer_l1_write_bandwidth_gbps: Some(0.6),
            },
        ];

        let tracks = build_counter_tracks(&encoders, &[], None, Some(&limiters));
        assert!(!tracks.is_empty());
        assert_eq!(tracks[0].name, "Compute Occupancy");
        assert_eq!(tracks[0].samples.len(), 4);
        assert_eq!(tracks[0].min_value, 75.0);
        assert_eq!(tracks[0].max_value, 80.0);

        let mut events = Vec::new();
        append_counter_track_events(&mut events, &tracks);
        assert!(
            events
                .iter()
                .any(|event| event.category.as_deref() == Some("counter"))
        );
        assert!(events.iter().any(|event| event.phase == "C"));
    }

    #[test]
    fn builds_pipeline_stat_tracks_from_profiler_summary() {
        let encoders = vec![TimelineEncoder {
            index: 0,
            command_buffer_index: 0,
            label: "main".into(),
            address: 0x10,
            dispatch_count: 2,
            start_time_ns: 1_000,
            end_time_ns: 5_000,
            duration_ns: Some(4_000),
            synthetic: false,
        }];
        let dispatches = vec![
            TimelineDispatch {
                index: 0,
                command_buffer_index: 0,
                encoder_index: Some(0),
                encoder_address: Some(0x10),
                encoder_label: Some("main".into()),
                kernel_name: Some("blur".into()),
                start_time_ns: 1_000,
                end_time_ns: 3_000,
                duration_ns: Some(2_000),
                grid_size: [1, 1, 1],
                group_size: [1, 1, 1],
                buffers: vec![],
                synthetic: false,
            },
            TimelineDispatch {
                index: 1,
                command_buffer_index: 0,
                encoder_index: Some(0),
                encoder_address: Some(0x10),
                encoder_label: Some("main".into()),
                kernel_name: Some("blur".into()),
                start_time_ns: 3_000,
                end_time_ns: 5_000,
                duration_ns: Some(2_000),
                grid_size: [1, 1, 1],
                group_size: [1, 1, 1],
                buffers: vec![],
                synthetic: false,
            },
        ];
        let summary = profiler::ProfilerStreamDataSummary {
            function_names: vec!["blur".into()],
            pipelines: vec![profiler::ProfilerPipeline {
                pipeline_id: 1,
                pipeline_address: 0xaaa,
                function_name: Some("blur".into()),
                stats: Some(profiler::ProfilerPipelineStats {
                    temporary_register_count: 24,
                    uniform_register_count: 12,
                    spilled_bytes: 64,
                    threadgroup_memory: 128,
                    instruction_count: 1024,
                    alu_instruction_count: 700,
                    branch_instruction_count: 20,
                    compilation_time_ms: 1.2,
                    line_instruction_counts: BTreeMap::new(),
                }),
            }],
            execution_costs: vec![],
            dispatches: vec![],
            encoder_timings: vec![],
            timeline: None,
            num_pipelines: 1,
            num_gpu_commands: 0,
            num_encoders: 1,
            total_time_us: 0,
        };

        let tracks = build_counter_tracks(&encoders, &dispatches, Some(&summary), None);
        assert!(
            tracks
                .iter()
                .any(|track| track.name == "Total Instructions")
        );
        assert!(
            tracks
                .iter()
                .any(|track| track.name == "Threadgroup Memory")
        );
        let total_instructions = tracks
            .iter()
            .find(|track| track.name == "Total Instructions")
            .unwrap();
        assert_eq!(total_instructions.max_value, 1024.0);
    }

    #[test]
    fn parses_raw_timeline_header_and_timestamps() {
        let dir = tempdir().unwrap();
        let raw_path = dir.path().join("Timeline_f_3.raw");
        let mut data = vec![0u8; 0x3c100];
        data[0..8].copy_from_slice(&0x773d413b0016b551u64.to_le_bytes());
        data[8..16].copy_from_slice(&0x12u64.to_le_bytes());
        data[12..16].copy_from_slice(&752u32.to_le_bytes());
        data[32..36].copy_from_slice(&0x3c000u32.to_le_bytes());
        data[80..88].copy_from_slice(&42u64.to_le_bytes());
        data[104..112].copy_from_slice(&123456789u64.to_le_bytes());
        data[500..508].copy_from_slice(&111_111u64.to_le_bytes());
        data[508..516].copy_from_slice(&222_222u64.to_le_bytes());
        fs::write(&raw_path, data).unwrap();

        let file = parse_raw_timeline_file(&raw_path).unwrap();

        assert_eq!(file.file_index, Some(3));
        assert_eq!(file.header.as_ref().unwrap().counter_count, 752);
        assert_eq!(file.header_timestamps, vec![111_111, 222_222]);
        assert_eq!(file.timestamp_span.as_ref().unwrap().count, 2);
        assert_eq!(file.heuristic_confidence, "low");
    }

    #[test]
    fn raw_report_finds_profiler_directory_inside_bundle() {
        let dir = tempdir().unwrap();
        let trace_path = dir.path().join("sample.gputrace");
        let profiler_dir = trace_path.join("capture.gpuprofiler_raw");
        fs::create_dir_all(&profiler_dir).unwrap();
        fs::write(profiler_dir.join("Timeline_f_0.raw"), vec![0u8; 256]).unwrap();

        let report = build_raw_report_for_path(&trace_path).unwrap();

        assert_eq!(report.file_count, 1);
        assert_eq!(
            report.profiler_directory.as_deref(),
            Some(profiler_dir.as_path())
        );

        let text = format_raw_report(&report);
        assert!(text.contains("Raw timeline heuristic report"));
        assert!(text.contains("Timeline_f_0.raw"));
        assert!(text.contains("expected > 0x3c000"));

        let json = export_raw_json(&report).unwrap();
        assert!(json.contains("\"file_count\": 1"));
    }
}
