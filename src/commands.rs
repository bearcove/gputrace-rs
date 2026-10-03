use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;

use crate::analysis;
use crate::error::Result;
use crate::profiler;
use crate::timeline;
use crate::timing;
use crate::trace::TraceBundle;

#[derive(Debug, Clone, Serialize)]
pub struct KernelReport {
    pub total_kernels: usize,
    pub filter: Option<String>,
    pub kernels: Vec<KernelEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct KernelEntry {
    pub name: String,
    pub pipeline_addr: u64,
    pub dispatch_count: usize,
    pub encoder_labels: BTreeMap<String, usize>,
    pub buffers: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EncoderReport {
    pub total_encoders: usize,
    pub command_buffer_count: usize,
    pub average_encoders_per_command_buffer: f64,
    pub encoders: Vec<EncoderEntry>,
    pub command_buffers: Vec<CommandBufferEncoderSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EncoderEntry {
    pub index: usize,
    pub label: String,
    pub address: u64,
    pub dispatch_count: usize,
    pub command_buffer_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CommandBufferEncoderSummary {
    pub index: usize,
    pub encoder_count: usize,
    pub dispatch_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DependencyReport {
    pub source: String,
    pub total_nodes: usize,
    pub total_edges: usize,
    pub nodes: Vec<DependencyNode>,
    pub edges: Vec<DependencyEdge>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DependencyNode {
    pub id: usize,
    pub label: String,
    pub command_buffer_index: usize,
    pub encoder_label: Option<String>,
    pub kernel_name: Option<String>,
    pub start_time_ns: Option<u64>,
    pub end_time_ns: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DependencyEdge {
    pub from: usize,
    pub to: usize,
    pub buffers: Vec<String>,
    pub hazard: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CommandBuffersReport {
    pub total_command_buffers: usize,
    pub total_encoders: usize,
    pub total_dispatches: usize,
    pub average_encoders_per_buffer: f64,
    pub average_dispatches_per_buffer: f64,
    pub command_buffers: Vec<CommandBufferEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CommandBufferEntry {
    pub index: usize,
    pub offset: usize,
    pub timestamp_ns: u64,
    pub duration_ns: Option<u64>,
    pub encoder_count: usize,
    pub dispatch_count: usize,
    pub encoders: Vec<CommandBufferEncoderEntry>,
    pub kernels: Vec<String>,
    pub artifact_hashes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CommandBufferEncoderEntry {
    pub index: usize,
    pub label: String,
    pub address: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BufferAccessReport {
    pub total_buffers: usize,
    pub shared_buffers: usize,
    pub single_use_buffers: usize,
    pub short_lived_buffers: usize,
    pub long_lived_buffers: usize,
    pub total_encoders: usize,
    pub alias_count: usize,
    pub buffers: Vec<BufferAccessEntry>,
    pub encoders: Vec<BufferAccessEncoderEntry>,
    pub aliasing_instances: Vec<BufferAlias>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BufferAccessEntry {
    pub name: String,
    pub address: Option<u64>,
    pub use_count: usize,
    pub dispatch_count: usize,
    pub encoder_count: usize,
    pub command_buffer_count: usize,
    pub first_dispatch_index: usize,
    pub last_dispatch_index: usize,
    pub is_shared: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct BufferAccessEncoderEntry {
    pub label: String,
    pub address: u64,
    pub unique_buffers: usize,
    pub total_buffer_uses: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct BufferAlias {
    pub address: u64,
    pub names: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TreeReport {
    pub group_by: String,
    pub nodes: Vec<TreeNode>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TreeNode {
    pub kind: String,
    pub label: String,
    pub details: Vec<String>,
    pub children: Vec<TreeNode>,
}

pub fn kernels(trace: &TraceBundle, filter: Option<&str>) -> Result<KernelReport> {
    let filter_lower = filter.map(|value| value.to_ascii_lowercase());
    let mut kernels: Vec<_> = trace
        .analyze_kernels()?
        .into_values()
        .filter(|kernel| {
            filter_lower
                .as_ref()
                .is_none_or(|needle| kernel.name.to_ascii_lowercase().contains(needle))
        })
        .map(|kernel| KernelEntry {
            name: kernel.name,
            pipeline_addr: kernel.pipeline_addr,
            dispatch_count: kernel.dispatch_count,
            encoder_labels: kernel.encoder_labels,
            buffers: kernel.buffers,
        })
        .collect();
    kernels.sort_by(|left, right| {
        right
            .dispatch_count
            .cmp(&left.dispatch_count)
            .then_with(|| left.name.cmp(&right.name))
    });
    Ok(KernelReport {
        total_kernels: kernels.len(),
        filter: filter.map(ToOwned::to_owned),
        kernels,
    })
}

pub fn format_kernels(report: &KernelReport, verbose: bool) -> String {
    let mut out = String::new();
    if let Some(filter) = &report.filter {
        out.push_str(&format!(
            "{} kernels matching {:?}\n\n",
            report.total_kernels, filter
        ));
    } else {
        out.push_str(&format!("{} kernels\n\n", report.total_kernels));
    }
    out.push_str(&format!(
        "{:<36} {:<18} {:>10}\n",
        "Name", "Pipeline State", "Dispatches"
    ));
    for kernel in &report.kernels {
        out.push_str(&format!(
            "{:<36} 0x{:<16x} {:>10}\n",
            truncate(&kernel.name, 36),
            kernel.pipeline_addr,
            kernel.dispatch_count
        ));
        if verbose {
            if !kernel.encoder_labels.is_empty() {
                out.push_str(&format!(
                    "           encoders: {}\n",
                    summarize_counted_entries(kernel.encoder_labels.iter(), 5)
                ));
            }
            if !kernel.buffers.is_empty() {
                out.push_str(&format!(
                    "           buffers: {}\n",
                    summarize_counted_entries(kernel.buffers.iter(), 5)
                ));
            }
        }
    }
    out
}

pub fn encoders(trace: &TraceBundle) -> Result<EncoderReport> {
    encoders_with_profiler_summary(
        trace,
        profiler::stream_data_summary(&trace.path).ok().as_ref(),
    )
}

pub fn encoders_with_profiler_summary(
    trace: &TraceBundle,
    profiler_summary: Option<&profiler::ProfilerStreamDataSummary>,
) -> Result<EncoderReport> {
    let encoders = trace.compute_encoders()?;
    let regions = trace.command_buffer_regions()?;
    let timing = timing::report(trace).ok();
    let mut dispatch_counts: BTreeMap<u64, usize> = BTreeMap::new();
    let mut command_buffer_sets: BTreeMap<u64, BTreeSet<usize>> = BTreeMap::new();
    let mut command_buffers = Vec::new();

    for region in &regions {
        command_buffers.push(CommandBufferEncoderSummary {
            index: region.command_buffer.index,
            encoder_count: region.encoders.len(),
            dispatch_count: region.dispatches.len(),
        });
        for encoder in &region.encoders {
            command_buffer_sets
                .entry(encoder.address)
                .or_default()
                .insert(region.command_buffer.index);
        }
        for dispatch in &region.dispatches {
            if let Some(encoder_id) = dispatch.encoder_id {
                *dispatch_counts.entry(encoder_id).or_default() += 1;
            }
        }
    }

    // Bridge streamData dispatches to byte-stream encoders by matching the
    // encoderInfoData `address` field (the u64 at offset 8 of each record) to
    // the byte-stream encoder address. streamData dispatches reference
    // `encoder_index`, but that namespace isn't shared with byte-stream
    // encoder offsets — addresses are. With multiple streamData encoders we
    // assign each profiler dispatch to whichever encoder's cumulative_us
    // window contains it.
    if let Some(summary) = profiler_summary
        && !summary.dispatches.is_empty()
        && !summary.encoder_timings.is_empty()
    {
        let bytestream_addresses: BTreeSet<u64> =
            encoders.iter().map(|encoder| encoder.address).collect();
        let mut prev_end_us = 0u64;
        let mut encoder_windows: Vec<(u64, u64, u64)> = Vec::new();
        for encoder in &summary.encoder_timings {
            let start_us = prev_end_us;
            let end_us = encoder.end_offset_micros;
            prev_end_us = end_us;
            if bytestream_addresses.contains(&encoder.address) {
                encoder_windows.push((start_us, end_us, encoder.address));
            }
        }
        for dispatch in &summary.dispatches {
            let dispatch_us = dispatch.cumulative_us;
            if let Some((_, _, address)) = encoder_windows
                .iter()
                .find(|(start, end, _)| dispatch_us > *start && dispatch_us <= *end)
            {
                *dispatch_counts.entry(*address).or_default() += 1;
            } else if encoder_windows.len() == 1 {
                // Single-encoder traces sometimes carry dispatches whose
                // cumulative_us falls outside the encoder window because the
                // capture-relative anchor differs from the encoder-relative
                // anchor; attribute them anyway.
                *dispatch_counts.entry(encoder_windows[0].2).or_default() += 1;
            }
        }
    }

    let mut entries: Vec<_> = encoders
        .into_iter()
        .map(|encoder| EncoderEntry {
            index: encoder.index,
            label: encoder.label,
            address: encoder.address,
            dispatch_count: dispatch_counts
                .get(&encoder.address)
                .copied()
                .unwrap_or_default(),
            command_buffer_count: command_buffer_sets
                .get(&encoder.address)
                .map_or(0, BTreeSet::len),
        })
        .collect();
    // Drop byte-scan false positives: real compute encoders surface in at least
    // one command-buffer region. The MTLB stream contains buffer-label bytes that
    // sometimes match the `CS\0\0` marker, producing ghost rows with
    // dispatches=0/command_buffers=0.
    entries.retain(|encoder| encoder.dispatch_count > 0 || encoder.command_buffer_count > 0);
    if entries.is_empty()
        && let Some(timing) = &timing
        && timing.dispatch_count > 0
        && !timing.encoders.is_empty()
    {
        entries = timing
            .encoders
            .iter()
            .enumerate()
            .map(|(index, encoder)| EncoderEntry {
                index,
                label: encoder.label.clone(),
                address: encoder.address,
                dispatch_count: encoder.dispatch_count,
                command_buffer_count: 0,
            })
            .collect();
        command_buffers = timing
            .command_buffers
            .iter()
            .map(|cb| CommandBufferEncoderSummary {
                index: cb.index,
                encoder_count: cb.encoder_count,
                dispatch_count: cb.dispatch_count,
            })
            .collect();
    }
    let summed_region_dispatches: usize = command_buffers
        .iter()
        .map(|command_buffer| command_buffer.dispatch_count)
        .sum();
    if let Some(timing) = &timing
        && timing.dispatch_count > summed_region_dispatches
        && let Some(timing_cb) = timing
            .command_buffers
            .iter()
            .min_by_key(|cb| duration_distance(cb.duration_ns, timing.total_duration_ns))
        && let Some(summary) = command_buffers
            .iter_mut()
            .find(|summary| summary.index == timing_cb.index)
    {
        summary.dispatch_count = timing.dispatch_count;
    }
    entries.sort_by_key(|left| left.index);

    let command_buffer_count = command_buffers.len();
    let average_encoders_per_command_buffer = if command_buffer_count == 0 {
        0.0
    } else {
        entries.len() as f64 / command_buffer_count as f64
    };

    Ok(EncoderReport {
        total_encoders: entries.len(),
        command_buffer_count,
        average_encoders_per_command_buffer,
        encoders: entries,
        command_buffers,
    })
}

pub fn dependencies(trace: &TraceBundle) -> Result<DependencyReport> {
    let regions = trace.command_buffer_regions()?;
    let timeline = timeline::report(trace).ok();
    let source = timeline
        .as_ref()
        .map(|report| report.source.clone())
        .unwrap_or_else(|| "synthetic".to_owned());
    let dispatch_spans = timeline
        .as_ref()
        .map(|report| {
            report
                .dispatches
                .iter()
                .map(|dispatch| {
                    (
                        dispatch.index,
                        (dispatch.start_time_ns, dispatch.end_time_ns),
                    )
                })
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();

    #[derive(Clone)]
    struct OrderedDispatch {
        index: usize,
        command_buffer_index: usize,
        encoder_label: Option<String>,
        kernel_name: Option<String>,
        label: String,
        start_time_ns: Option<u64>,
        end_time_ns: Option<u64>,
        buffers: Vec<crate::trace::BoundBuffer>,
    }

    let mut ordered_dispatches = Vec::new();
    for region in regions {
        for dispatch in region.dispatches {
            let label = dispatch
                .kernel_name
                .clone()
                .or_else(|| {
                    region
                        .encoders
                        .iter()
                        .find(|encoder| Some(encoder.address) == dispatch.encoder_id)
                        .map(|encoder| encoder.label.clone())
                })
                .unwrap_or_else(|| format!("dispatch_{}", dispatch.index));
            let encoder_label = region
                .encoders
                .iter()
                .find(|encoder| Some(encoder.address) == dispatch.encoder_id)
                .map(|encoder| encoder.label.clone())
                .filter(|label| !label.is_empty());
            let (start_time_ns, end_time_ns) = dispatch_spans
                .get(&dispatch.index)
                .copied()
                .map(|(start, end)| (Some(start), Some(end)))
                .unwrap_or((None, None));

            ordered_dispatches.push(OrderedDispatch {
                index: dispatch.index,
                command_buffer_index: region.command_buffer.index,
                encoder_label,
                kernel_name: dispatch.kernel_name.clone(),
                label,
                start_time_ns,
                end_time_ns,
                buffers: dispatch.buffers,
            });
        }
    }
    ordered_dispatches.sort_by(|left, right| {
        left.start_time_ns
            .unwrap_or(u64::MAX)
            .cmp(&right.start_time_ns.unwrap_or(u64::MAX))
            .then_with(|| left.index.cmp(&right.index))
    });

    let mut nodes = Vec::new();
    let mut edges: BTreeMap<(usize, usize), DependencyAccumulator> = BTreeMap::new();
    let mut last_user: BTreeMap<String, (usize, crate::trace::MTLResourceUsage)> = BTreeMap::new();

    for dispatch in ordered_dispatches {
        nodes.push(DependencyNode {
            id: dispatch.index,
            label: dispatch.label.clone(),
            command_buffer_index: dispatch.command_buffer_index,
            encoder_label: dispatch.encoder_label.clone(),
            kernel_name: dispatch.kernel_name.clone(),
            start_time_ns: dispatch.start_time_ns,
            end_time_ns: dispatch.end_time_ns,
        });

        let mut seen_buffers = BTreeSet::new();
        for buffer in dispatch.buffers {
            let buffer_name = buffer
                .name
                .clone()
                .unwrap_or_else(|| format!("0x{:x}", buffer.address));
            if !seen_buffers.insert(buffer_name.clone()) {
                continue;
            }
            if let Some((previous, previous_usage)) =
                last_user.insert(buffer_name.clone(), (dispatch.index, buffer.usage))
                && previous != dispatch.index
            {
                edges
                    .entry((previous, dispatch.index))
                    .or_default()
                    .push(buffer_name, classify_hazard(previous_usage, buffer.usage));
            }
        }
    }

    let edges: Vec<_> = edges
        .into_iter()
        .map(|((from, to), edge)| DependencyEdge {
            from,
            to,
            buffers: edge.buffers.into_iter().collect(),
            hazard: edge.hazard,
        })
        .collect();

    Ok(DependencyReport {
        source,
        total_nodes: nodes.len(),
        total_edges: edges.len(),
        nodes,
        edges,
    })
}

#[derive(Debug, Default)]
struct DependencyAccumulator {
    buffers: BTreeSet<String>,
    hazard: String,
}

impl DependencyAccumulator {
    fn push(&mut self, buffer: String, hazard: &'static str) {
        self.buffers.insert(buffer);
        self.hazard = merge_hazards(&self.hazard, hazard);
    }
}

fn classify_hazard(
    previous: crate::trace::MTLResourceUsage,
    current: crate::trace::MTLResourceUsage,
) -> &'static str {
    let prev_read = previous.contains(crate::trace::MTLResourceUsage::READ)
        || previous.contains(crate::trace::MTLResourceUsage::SAMPLE);
    let prev_write = previous.contains(crate::trace::MTLResourceUsage::WRITE);
    let curr_read = current.contains(crate::trace::MTLResourceUsage::READ)
        || current.contains(crate::trace::MTLResourceUsage::SAMPLE);
    let curr_write = current.contains(crate::trace::MTLResourceUsage::WRITE);

    if prev_read && prev_write && curr_read && curr_write {
        return "RAW/WAR/WAW";
    }
    if prev_read && prev_write && curr_write {
        return "WAW/WAR";
    }

    match (prev_read, prev_write, curr_read, curr_write) {
        (_, true, true, false) => "RAW",
        (_, true, _, true) => "WAW",
        (true, false, _, true) => "WAR",
        (false, false, _, _) => "USE",
        _ => "RW",
    }
}

fn merge_hazards(existing: &str, new_hazard: &str) -> String {
    let mut hazards = BTreeSet::new();
    for hazard in existing.split('/') {
        if !hazard.is_empty() {
            hazards.insert(hazard.to_owned());
        }
    }
    for hazard in new_hazard.split('/') {
        if !hazard.is_empty() {
            hazards.insert(hazard.to_owned());
        }
    }
    if hazards.is_empty() {
        return new_hazard.to_owned();
    }
    hazards.into_iter().collect::<Vec<_>>().join("/")
}

fn duration_distance(duration: Option<u64>, target: u64) -> u64 {
    duration
        .map(|duration| duration.abs_diff(target))
        .unwrap_or(u64::MAX)
}

pub fn command_buffers(trace: &TraceBundle) -> Result<CommandBuffersReport> {
    let command_buffers = trace.command_buffers()?;
    let regions = trace.command_buffer_regions()?;
    let timing_report = timing::report(trace)?;
    let timing_dispatch_count = timing_report.dispatch_count;
    let timing_total_duration_ns = timing_report.total_duration_ns;
    let timing_kernels = timing_report
        .kernels
        .iter()
        .map(|kernel| kernel.name.clone())
        .collect::<Vec<_>>();
    let timing_by_index: BTreeMap<_, _> = timing_report
        .command_buffers
        .into_iter()
        .map(|cb| (cb.index, cb))
        .collect();

    let mut entries = Vec::new();
    let mut total_encoders = 0usize;
    let mut total_dispatches = 0usize;

    for (idx, region) in regions.into_iter().enumerate() {
        let cb = command_buffers.get(idx).unwrap_or(&region.command_buffer);
        let kernels: BTreeSet<_> = region
            .dispatches
            .iter()
            .filter_map(|dispatch| dispatch.kernel_name.clone())
            .collect();
        let timing = timing_by_index.get(&region.command_buffer.index);
        total_encoders += region.encoders.len();
        total_dispatches += region.dispatches.len();
        entries.push(CommandBufferEntry {
            index: region.command_buffer.index,
            offset: cb.offset,
            // Prefer the boot-relative absolute timestamp from APSTimelineData
            // when the profiler decoded one — it's meaningful even for
            // single-CB traces where the capture-relative `timestamp_ns` is
            // always 0. Fall back to the capture-relative value, then to the
            // raw `cb.timestamp` for synthetic timing without profiler data.
            timestamp_ns: timing
                .and_then(|timing_cb| {
                    (timing_cb.absolute_timestamp_ns != 0)
                        .then_some(timing_cb.absolute_timestamp_ns)
                        .or_else(|| (timing_cb.timestamp_ns != 0).then_some(timing_cb.timestamp_ns))
                })
                .unwrap_or(cb.timestamp),
            duration_ns: timing.and_then(|cb| cb.duration_ns),
            encoder_count: region.encoders.len(),
            dispatch_count: region.dispatches.len(),
            encoders: region
                .encoders
                .into_iter()
                .map(|encoder| CommandBufferEncoderEntry {
                    index: encoder.index,
                    label: encoder.label,
                    address: encoder.address,
                })
                .collect(),
            kernels: kernels.into_iter().collect(),
            artifact_hashes: cb.artifact_hashes.clone(),
        });
    }

    let total_command_buffers = entries.len();
    // streamData typically attributes far more dispatches than the MTLB byte-scan
    // surfaces (the capture stream only records a sparse subset of dispatch records).
    // Prefer the profiler's count whenever it's larger so the per-CB row matches the
    // kernel timing report.
    if timing_dispatch_count > total_dispatches {
        total_dispatches = timing_dispatch_count;
        if let Some(entry) = entries
            .iter_mut()
            .min_by_key(|entry| duration_distance(entry.duration_ns, timing_total_duration_ns))
        {
            entry.dispatch_count = timing_dispatch_count;
            entry.kernels = timing_kernels;
        }
    }
    let average_encoders_per_buffer = if total_command_buffers == 0 {
        0.0
    } else {
        total_encoders as f64 / total_command_buffers as f64
    };
    let average_dispatches_per_buffer = if total_command_buffers == 0 {
        0.0
    } else {
        total_dispatches as f64 / total_command_buffers as f64
    };

    Ok(CommandBuffersReport {
        total_command_buffers,
        total_encoders,
        total_dispatches,
        average_encoders_per_buffer,
        average_dispatches_per_buffer,
        command_buffers: entries,
    })
}

pub fn buffer_access(trace: &TraceBundle) -> Result<BufferAccessReport> {
    let analysis = analysis::analyze(trace);
    let regions = trace.command_buffer_regions()?;
    let buffer_name_map = trace.buffer_name_map()?;

    let mut encoder_buffers: BTreeMap<u64, BTreeSet<String>> = BTreeMap::new();
    let mut encoder_uses: BTreeMap<u64, usize> = BTreeMap::new();
    let mut encoder_labels: BTreeMap<u64, String> = BTreeMap::new();
    for region in &regions {
        for encoder in &region.encoders {
            encoder_labels
                .entry(encoder.address)
                .or_insert_with(|| encoder.label.clone());
        }
        for dispatch in &region.dispatches {
            if let Some(encoder_id) = dispatch.encoder_id {
                for buffer in &dispatch.buffers {
                    let name = buffer
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("0x{:x}", buffer.address));
                    encoder_buffers.entry(encoder_id).or_default().insert(name);
                    *encoder_uses.entry(encoder_id).or_default() += 1;
                }
            }
        }
    }

    let mut aliases_by_address: BTreeMap<u64, BTreeSet<String>> = BTreeMap::new();
    for (address, name) in buffer_name_map {
        aliases_by_address.entry(address).or_default().insert(name);
    }
    let aliasing_instances: Vec<_> = aliases_by_address
        .into_iter()
        .filter_map(|(address, names)| {
            (names.len() > 1).then(|| BufferAlias {
                address,
                names: names.into_iter().collect(),
            })
        })
        .collect();

    let buffers = analysis
        .buffer_stats
        .iter()
        .map(|buffer| BufferAccessEntry {
            name: buffer.name.clone(),
            address: buffer.address,
            use_count: buffer.use_count,
            dispatch_count: buffer.dispatch_count,
            encoder_count: buffer.encoder_count,
            command_buffer_count: buffer.command_buffer_count,
            first_dispatch_index: buffer.first_dispatch_index,
            last_dispatch_index: buffer.last_dispatch_index,
            is_shared: buffer.encoder_count > 1,
        })
        .collect();

    let mut encoders: Vec<_> = encoder_buffers
        .into_iter()
        .map(|(address, buffers)| BufferAccessEncoderEntry {
            label: encoder_labels
                .get(&address)
                .cloned()
                .unwrap_or_else(|| format!("0x{address:x}")),
            address,
            unique_buffers: buffers.len(),
            total_buffer_uses: encoder_uses.get(&address).copied().unwrap_or_default(),
        })
        .collect();
    encoders.sort_by(|left, right| {
        right
            .unique_buffers
            .cmp(&left.unique_buffers)
            .then_with(|| right.total_buffer_uses.cmp(&left.total_buffer_uses))
            .then_with(|| left.label.cmp(&right.label))
    });

    Ok(BufferAccessReport {
        total_buffers: analysis.buffer_count,
        shared_buffers: analysis.shared_buffer_count,
        single_use_buffers: analysis.single_use_buffer_count,
        short_lived_buffers: analysis.short_lived_buffer_count,
        long_lived_buffers: analysis.long_lived_buffer_count,
        total_encoders: encoders.len(),
        alias_count: aliasing_instances.len(),
        buffers,
        encoders,
        aliasing_instances,
    })
}

pub fn tree(trace: &TraceBundle, group_by: &str) -> Result<TreeReport> {
    let regions = trace.command_buffer_regions()?;
    let nodes = match group_by {
        "encoder" => tree_by_encoder(regions),
        "pipeline" => tree_by_pipeline(regions),
        _ => {
            return Err(crate::Error::InvalidInput(format!(
                "unknown tree grouping: {group_by} (expected encoder or pipeline)"
            )));
        }
    };
    Ok(TreeReport {
        group_by: group_by.to_owned(),
        nodes,
    })
}

pub fn format_encoders(report: &EncoderReport, verbose: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!("{} encoders\n", report.total_encoders));
    if verbose {
        out.push_str(&format!(
            "{} command buffers ({:.1} encoders/buffer avg)\n",
            report.command_buffer_count, report.average_encoders_per_command_buffer
        ));
    }
    out.push('\n');
    for encoder in &report.encoders {
        if encoder.label.is_empty() {
            out.push_str(&format!(
                "{:>4}: (unlabeled) 0x{:x}\n",
                encoder.index, encoder.address
            ));
        } else {
            out.push_str(&format!("{:>4}: {}\n", encoder.index, encoder.label));
        }
        if verbose {
            out.push_str(&format!(
                "      address=0x{:x} dispatches={} command_buffers={}\n",
                encoder.address, encoder.dispatch_count, encoder.command_buffer_count
            ));
        }
    }
    if verbose && !report.command_buffers.is_empty() {
        out.push_str("\nCommand buffers:\n");
        for cb in &report.command_buffers {
            out.push_str(&format!(
                "  CB {}: {} encoders, {} dispatches\n",
                cb.index, cb.encoder_count, cb.dispatch_count
            ));
        }
    }
    out
}

pub fn format_dependencies(report: &DependencyReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{} dispatch nodes, {} dependency edges\nordering={}\n\n",
        report.total_nodes, report.total_edges, report.source
    ));
    for node in &report.nodes {
        out.push_str(&format!("n{}: {}", node.id, node.label));
        if let Some(kernel_name) = &node.kernel_name
            && kernel_name != &node.label
        {
            out.push_str(&format!(" [kernel: {kernel_name}]"));
        }
        if let Some(encoder_label) = &node.encoder_label {
            out.push_str(&format!(" [encoder: {encoder_label}]"));
        }
        out.push_str(&format!(" (CB {})\n", node.command_buffer_index));
    }
    out.push_str("\nDependencies:\n");
    if report.edges.is_empty() {
        out.push_str("  none\n");
        return out;
    }
    for edge in &report.edges {
        out.push_str(&format!(
            "  n{} -> n{} [{}] via {}\n",
            edge.from,
            edge.to,
            edge.hazard,
            summarize_items(edge.buffers.iter().map(String::as_str), 4)
        ));
    }
    out
}

pub fn format_dependencies_dot(report: &DependencyReport) -> String {
    let mut out = String::new();
    out.push_str("digraph G {\n");
    out.push_str("  rankdir=LR;\n");
    out.push_str(&format!(
        "  label=\"dependency ordering: {}\";\n  labelloc=t;\n",
        escape_dot(&report.source)
    ));
    out.push_str("  node [shape=box, style=filled, fontname=\"Helvetica\"];\n");
    out.push_str("  edge [fontname=\"Helvetica\", fontsize=10];\n");
    for node in &report.nodes {
        let mut label = node.label.clone();
        if label.len() > 50 {
            label = format!("{}...", &label[..47]);
        }
        out.push_str(&format!(
            "  n{} [label=\"{}\"];\n",
            node.id,
            escape_dot(&label)
        ));
    }
    for edge in &report.edges {
        out.push_str(&format!(
            "  n{} -> n{} [label=\"{} ({})\"];\n",
            edge.from,
            edge.to,
            escape_dot(&edge.buffers.join(", ")),
            edge.hazard
        ));
    }
    out.push_str("}\n");
    out
}

pub fn format_command_buffers(report: &CommandBuffersReport, detailed: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{} command buffers\n",
        report.total_command_buffers
    ));
    out.push_str(&format!(
        "total encoders={}, total dispatches={}, {:.1} encoders/buffer, {:.1} dispatches/buffer\n\n",
        report.total_encoders,
        report.total_dispatches,
        report.average_encoders_per_buffer,
        report.average_dispatches_per_buffer
    ));
    for cb in &report.command_buffers {
        out.push_str(&format!(
            "{:>4}: offset=0x{:08x} ts={} duration={} encoders={} dispatches={}\n",
            cb.index,
            cb.offset,
            cb.timestamp_ns,
            cb.duration_ns
                .map(|value| value.to_string())
                .unwrap_or_else(|| "?".to_owned()),
            cb.encoder_count,
            cb.dispatch_count
        ));
        if detailed {
            for encoder in &cb.encoders {
                let label = if encoder.label.is_empty() {
                    format!("0x{:x}", encoder.address)
                } else {
                    encoder.label.clone()
                };
                out.push_str(&format!(
                    "       encoder {:>3}: {} (0x{:x})\n",
                    encoder.index, label, encoder.address
                ));
            }
            if !cb.kernels.is_empty() {
                out.push_str(&format!("       kernels: {}\n", cb.kernels.join(", ")));
            }
            if !cb.artifact_hashes.is_empty() {
                out.push_str(&format!(
                    "       artifacts: {}\n",
                    cb.artifact_hashes.join(", ")
                ));
            }
        }
    }
    out
}

pub fn format_buffer_access(report: &BufferAccessReport, verbose: bool) -> String {
    let mut out = String::new();
    out.push_str("=== Buffer Access Analysis ===\n\n");
    out.push_str("Summary:\n");
    out.push_str(&format!("  Total Buffers:   {}\n", report.total_buffers));
    out.push_str(&format!(
        "  Shared Buffers:  {} (accessed by multiple encoders)\n",
        report.shared_buffers
    ));
    out.push_str(&format!(
        "  Single-use:      {}\n",
        report.single_use_buffers
    ));
    out.push_str(&format!(
        "  Short-lived:     {}\n",
        report.short_lived_buffers
    ));
    out.push_str(&format!(
        "  Long-lived:      {}\n",
        report.long_lived_buffers
    ));
    out.push_str(&format!("  Total Encoders:  {}\n", report.total_encoders));
    out.push_str(&format!("  Alias Sets:      {}\n\n", report.alias_count));

    if !report.aliasing_instances.is_empty() {
        out.push_str("Potential Aliasing:\n");
        for (index, alias) in report.aliasing_instances.iter().take(10).enumerate() {
            out.push_str(&format!(
                "  [{}] 0x{:016x} -> {}\n",
                index + 1,
                alias.address,
                alias.names.join(", ")
            ));
        }
        out.push('\n');
    }

    let mut shared: Vec<_> = report
        .buffers
        .iter()
        .filter(|buffer| buffer.is_shared)
        .collect();
    shared.sort_by(|left, right| {
        right
            .encoder_count
            .cmp(&left.encoder_count)
            .then_with(|| right.use_count.cmp(&left.use_count))
            .then_with(|| left.name.cmp(&right.name))
    });
    if !shared.is_empty() {
        out.push_str("Top Shared Buffers:\n");
        for (index, buffer) in shared.iter().take(10).enumerate() {
            out.push_str(&format!(
                "  [{}] {} - {} encoders, {} uses\n",
                index + 1,
                buffer.name,
                buffer.encoder_count,
                buffer.use_count
            ));
        }
        out.push('\n');
    }

    if verbose && !report.encoders.is_empty() {
        out.push_str("Per-Encoder Statistics:\n");
        for encoder in &report.encoders {
            out.push_str(&format!(
                "  {}: {} unique buffers, {} total accesses\n",
                encoder.label, encoder.unique_buffers, encoder.total_buffer_uses
            ));
        }
        out.push('\n');
    }

    out.push_str("Optimization Opportunities:\n");
    if report.shared_buffers > 0 {
        out.push_str(&format!(
            "  • {} buffers are shared across encoders\n",
            report.shared_buffers
        ));
    }
    if report.single_use_buffers > 0 {
        out.push_str(&format!(
            "  • {} buffers are only touched once\n",
            report.single_use_buffers
        ));
    }
    if report.short_lived_buffers > 0 {
        out.push_str(&format!(
            "  • {} buffers have short lifetimes and may be poolable\n",
            report.short_lived_buffers
        ));
    }
    if report.alias_count > 0 {
        out.push_str(&format!(
            "  • {} potential alias sets deserve review\n",
            report.alias_count
        ));
    }
    if report.shared_buffers == 0
        && report.single_use_buffers == 0
        && report.short_lived_buffers == 0
        && report.alias_count == 0
    {
        out.push_str("  • No obvious access-pattern outliers detected\n");
    }
    out
}

pub fn format_tree(report: &TreeReport) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Execution tree grouped by {}\n\n",
        report.group_by
    ));
    for node in &report.nodes {
        render_tree_node(&mut out, node, 0);
    }
    out
}

fn tree_by_encoder(regions: Vec<crate::trace::CommandBufferRegion>) -> Vec<TreeNode> {
    regions
        .into_iter()
        .map(|region| {
            let encoder_children = region
                .encoders
                .iter()
                .map(|encoder| {
                    let dispatches: Vec<_> = region
                        .dispatches
                        .iter()
                        .filter(|dispatch| dispatch.encoder_id == Some(encoder.address))
                        .collect();
                    let mut kernels = BTreeSet::new();
                    let mut buffers = BTreeSet::new();
                    for dispatch in dispatches {
                        if let Some(kernel) = &dispatch.kernel_name {
                            kernels.insert(kernel.clone());
                        }
                        for buffer in &dispatch.buffers {
                            buffers.insert(
                                buffer
                                    .name
                                    .clone()
                                    .unwrap_or_else(|| format!("0x{:x}", buffer.address)),
                            );
                        }
                    }
                    TreeNode {
                        kind: "encoder".into(),
                        label: if encoder.label.is_empty() {
                            format!("encoder {}", encoder.index)
                        } else {
                            encoder.label.clone()
                        },
                        details: vec![
                            format!("index={}", encoder.index),
                            format!("address=0x{:x}", encoder.address),
                            format!(
                                "dispatches={}",
                                region
                                    .dispatches
                                    .iter()
                                    .filter(|dispatch| dispatch.encoder_id == Some(encoder.address))
                                    .count()
                            ),
                        ],
                        children: kernels
                            .into_iter()
                            .map(|kernel| TreeNode {
                                kind: "kernel".into(),
                                label: kernel,
                                details: vec![],
                                children: buffers
                                    .iter()
                                    .cloned()
                                    .map(|buffer| TreeNode {
                                        kind: "buffer".into(),
                                        label: buffer,
                                        details: vec![],
                                        children: vec![],
                                    })
                                    .collect(),
                            })
                            .collect(),
                    }
                })
                .collect();

            TreeNode {
                kind: "command_buffer".into(),
                label: format!("Command Buffer {}", region.command_buffer.index),
                details: vec![
                    format!("offset=0x{:x}", region.command_buffer.offset),
                    format!("timestamp={}", region.command_buffer.timestamp),
                    format!("dispatches={}", region.dispatches.len()),
                ],
                children: encoder_children,
            }
        })
        .collect()
}

fn tree_by_pipeline(regions: Vec<crate::trace::CommandBufferRegion>) -> Vec<TreeNode> {
    let mut kernels: BTreeMap<String, (BTreeSet<usize>, BTreeSet<String>)> = BTreeMap::new();
    for region in regions {
        for dispatch in region.dispatches {
            let kernel = dispatch
                .kernel_name
                .clone()
                .unwrap_or_else(|| "unknown".to_owned());
            let entry = kernels
                .entry(kernel)
                .or_insert_with(|| (BTreeSet::new(), BTreeSet::new()));
            entry.0.insert(region.command_buffer.index);
            for buffer in dispatch.buffers {
                entry.1.insert(
                    buffer
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("0x{:x}", buffer.address)),
                );
            }
        }
    }

    kernels
        .into_iter()
        .map(|(kernel, (command_buffers, buffers))| TreeNode {
            kind: "kernel".into(),
            label: kernel,
            details: vec![format!("command_buffers={}", command_buffers.len())],
            children: buffers
                .into_iter()
                .map(|buffer| TreeNode {
                    kind: "buffer".into(),
                    label: buffer,
                    details: vec![],
                    children: vec![],
                })
                .collect(),
        })
        .collect()
}

fn render_tree_node(out: &mut String, node: &TreeNode, depth: usize) {
    let indent = "  ".repeat(depth);
    out.push_str(&format!("{indent}- {}: {}", node.kind, node.label));
    if !node.details.is_empty() {
        out.push_str(&format!(" [{}]", node.details.join(", ")));
    }
    out.push('\n');
    for child in &node.children {
        render_tree_node(out, child, depth + 1);
    }
}

fn summarize_counted_entries<'a>(
    entries: impl IntoIterator<Item = (&'a String, &'a usize)>,
    limit: usize,
) -> String {
    let values: Vec<_> = entries
        .into_iter()
        .map(|(label, count)| format!("{label}({count})"))
        .collect();
    summarize_items(values.iter().map(String::as_str), limit)
}

fn summarize_items<'a>(items: impl IntoIterator<Item = &'a str>, limit: usize) -> String {
    let values: Vec<_> = items.into_iter().collect();
    if values.is_empty() {
        return "none".to_owned();
    }

    let shown = values.iter().take(limit).copied().collect::<Vec<_>>();
    let remaining = values.len().saturating_sub(shown.len());
    let mut summary = shown.join(", ");
    if remaining > 0 {
        summary.push_str(&format!(", +{remaining} more"));
    }
    summary
}

fn truncate(value: &str, width: usize) -> String {
    if value.len() <= width {
        return value.to_owned();
    }
    let keep = width.saturating_sub(3);
    format!("{}...", &value[..keep])
}

fn escape_dot(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_kernel_report() {
        let report = KernelReport {
            total_kernels: 1,
            filter: Some("copy".into()),
            kernels: vec![KernelEntry {
                name: "copy_kernel".into(),
                pipeline_addr: 0x1234,
                dispatch_count: 7,
                encoder_labels: [("copy".into(), 7)].into_iter().collect(),
                buffers: [("buf".into(), 3)].into_iter().collect(),
            }],
        };

        let rendered = format_kernels(&report, true);
        assert!(rendered.contains("copy_kernel"));
        assert!(rendered.contains("Dispatches"));
        assert!(rendered.contains("encoders:"));
        assert!(rendered.contains("buffers:"));
    }

    #[test]
    fn truncates_verbose_kernel_details() {
        let report = KernelReport {
            total_kernels: 1,
            filter: None,
            kernels: vec![KernelEntry {
                name: "copy_kernel".into(),
                pipeline_addr: 0x1234,
                dispatch_count: 7,
                encoder_labels: [
                    ("enc0".into(), 1),
                    ("enc1".into(), 1),
                    ("enc2".into(), 1),
                    ("enc3".into(), 1),
                    ("enc4".into(), 1),
                    ("enc5".into(), 1),
                ]
                .into_iter()
                .collect(),
                buffers: [
                    ("buf0".into(), 1),
                    ("buf1".into(), 1),
                    ("buf2".into(), 1),
                    ("buf3".into(), 1),
                    ("buf4".into(), 1),
                    ("buf5".into(), 1),
                ]
                .into_iter()
                .collect(),
            }],
        };

        let rendered = format_kernels(&report, true);
        assert!(rendered.contains("enc0(1), enc1(1), enc2(1), enc3(1), enc4(1), +1 more"));
        assert!(rendered.contains("buf0(1), buf1(1), buf2(1), buf3(1), buf4(1), +1 more"));
    }

    #[test]
    fn formats_encoder_report() {
        let report = EncoderReport {
            total_encoders: 1,
            command_buffer_count: 1,
            average_encoders_per_command_buffer: 1.0,
            encoders: vec![EncoderEntry {
                index: 0,
                label: "my_encoder".into(),
                address: 0x55,
                dispatch_count: 2,
                command_buffer_count: 1,
            }],
            command_buffers: vec![CommandBufferEncoderSummary {
                index: 0,
                encoder_count: 1,
                dispatch_count: 2,
            }],
        };

        let rendered = format_encoders(&report, true);
        assert!(rendered.contains("my_encoder"));
        assert!(rendered.contains("dispatches=2"));
        assert!(rendered.contains("CB 0"));
    }

    #[test]
    fn formats_dependency_dot() {
        let report = DependencyReport {
            source: "synthetic".into(),
            total_nodes: 2,
            total_edges: 1,
            nodes: vec![
                DependencyNode {
                    id: 0,
                    label: "first".into(),
                    command_buffer_index: 0,
                    encoder_label: Some("enc0".into()),
                    kernel_name: Some("first".into()),
                    start_time_ns: None,
                    end_time_ns: None,
                },
                DependencyNode {
                    id: 1,
                    label: "second".into(),
                    command_buffer_index: 0,
                    encoder_label: Some("enc1".into()),
                    kernel_name: Some("second".into()),
                    start_time_ns: None,
                    end_time_ns: None,
                },
            ],
            edges: vec![DependencyEdge {
                from: 0,
                to: 1,
                buffers: vec!["buf".into()],
                hazard: "RW".into(),
            }],
        };

        let rendered = format_dependencies_dot(&report);
        assert!(rendered.contains("digraph G"));
        assert!(rendered.contains("dependency ordering: synthetic"));
        assert!(rendered.contains("n0 -> n1"));
        assert!(rendered.contains("buf (RW)"));
    }

    #[test]
    fn classifies_hazards_from_usage_flags() {
        use crate::trace::MTLResourceUsage;

        assert_eq!(
            classify_hazard(MTLResourceUsage::WRITE, MTLResourceUsage::READ),
            "RAW"
        );
        assert_eq!(
            classify_hazard(MTLResourceUsage::WRITE, MTLResourceUsage::WRITE),
            "WAW"
        );
        assert_eq!(
            classify_hazard(MTLResourceUsage::READ, MTLResourceUsage::WRITE),
            "WAR"
        );
        assert_eq!(
            classify_hazard(
                MTLResourceUsage::READ | MTLResourceUsage::WRITE,
                MTLResourceUsage::READ | MTLResourceUsage::WRITE
            ),
            "RAW/WAR/WAW"
        );
    }

    #[test]
    fn merges_hazard_labels_without_duplicates() {
        assert_eq!(merge_hazards("", "RAW"), "RAW");
        assert_eq!(merge_hazards("RAW", "WAW"), "RAW/WAW");
        assert_eq!(merge_hazards("RAW/WAW", "RAW"), "RAW/WAW");
    }

    #[test]
    fn formats_dependency_report_with_fallbacks_and_truncation() {
        let report = DependencyReport {
            source: "raw-profiler-heuristic".into(),
            total_nodes: 2,
            total_edges: 1,
            nodes: vec![
                DependencyNode {
                    id: 2,
                    label: "dispatch_2".into(),
                    command_buffer_index: 3,
                    encoder_label: Some("main".into()),
                    kernel_name: Some("blur".into()),
                    start_time_ns: None,
                    end_time_ns: None,
                },
                DependencyNode {
                    id: 4,
                    label: "dispatch_4".into(),
                    command_buffer_index: 3,
                    encoder_label: None,
                    kernel_name: None,
                    start_time_ns: None,
                    end_time_ns: None,
                },
            ],
            edges: vec![DependencyEdge {
                from: 2,
                to: 4,
                buffers: vec![
                    "buf0".into(),
                    "buf1".into(),
                    "buf2".into(),
                    "buf3".into(),
                    "buf4".into(),
                ],
                hazard: "RW".into(),
            }],
        };

        let rendered = format_dependencies(&report);
        assert!(rendered.contains("2 dispatch nodes, 1 dependency edges"));
        assert!(rendered.contains("ordering=raw-profiler-heuristic"));
        assert!(rendered.contains("n2: dispatch_2 [kernel: blur] [encoder: main] (CB 3)"));
        assert!(rendered.contains("n2 -> n4 [RW] via buf0, buf1, buf2, buf3, +1 more"));
    }

    #[test]
    fn formats_dependency_report_without_edges() {
        let report = DependencyReport {
            source: "synthetic".into(),
            total_nodes: 1,
            total_edges: 0,
            nodes: vec![DependencyNode {
                id: 1,
                label: "only".into(),
                command_buffer_index: 0,
                encoder_label: None,
                kernel_name: None,
                start_time_ns: None,
                end_time_ns: None,
            }],
            edges: vec![],
        };

        let rendered = format_dependencies(&report);
        assert!(rendered.contains("Dependencies:\n  none"));
    }

    #[test]
    fn formats_command_buffer_report() {
        let report = CommandBuffersReport {
            total_command_buffers: 1,
            total_encoders: 1,
            total_dispatches: 2,
            average_encoders_per_buffer: 1.0,
            average_dispatches_per_buffer: 2.0,
            command_buffers: vec![CommandBufferEntry {
                index: 0,
                offset: 0x20,
                timestamp_ns: 100,
                duration_ns: Some(50),
                encoder_count: 1,
                dispatch_count: 2,
                encoders: vec![CommandBufferEncoderEntry {
                    index: 0,
                    label: "enc".into(),
                    address: 0x33,
                }],
                kernels: vec!["kernel".into()],
                artifact_hashes: vec!["F34AD02BAECBD543".into()],
            }],
        };

        let rendered = format_command_buffers(&report, true);
        assert!(rendered.contains("1 command buffers"));
        assert!(rendered.contains("offset=0x00000020"));
        assert!(rendered.contains("encoder   0"));
        assert!(rendered.contains("artifacts: F34AD02BAECBD543"));
        assert!(rendered.contains("kernels: kernel"));
    }

    #[test]
    fn formats_buffer_access_report() {
        let report = BufferAccessReport {
            total_buffers: 2,
            shared_buffers: 1,
            single_use_buffers: 1,
            short_lived_buffers: 1,
            long_lived_buffers: 0,
            total_encoders: 1,
            alias_count: 1,
            buffers: vec![
                BufferAccessEntry {
                    name: "buf".into(),
                    address: Some(1),
                    use_count: 4,
                    dispatch_count: 4,
                    encoder_count: 2,
                    command_buffer_count: 1,
                    first_dispatch_index: 0,
                    last_dispatch_index: 3,
                    is_shared: true,
                },
                BufferAccessEntry {
                    name: "tmp".into(),
                    address: Some(2),
                    use_count: 1,
                    dispatch_count: 1,
                    encoder_count: 1,
                    command_buffer_count: 1,
                    first_dispatch_index: 3,
                    last_dispatch_index: 3,
                    is_shared: false,
                },
            ],
            encoders: vec![BufferAccessEncoderEntry {
                label: "enc".into(),
                address: 1,
                unique_buffers: 2,
                total_buffer_uses: 5,
            }],
            aliasing_instances: vec![BufferAlias {
                address: 1,
                names: vec!["buf".into(), "buf_alias".into()],
            }],
        };

        let rendered = format_buffer_access(&report, true);
        assert!(rendered.contains("Top Shared Buffers"));
        assert!(rendered.contains("buf - 2 encoders, 4 uses"));
        assert!(rendered.contains("enc: 2 unique buffers, 5 total accesses"));
    }

    #[test]
    fn formats_tree_report() {
        let report = TreeReport {
            group_by: "encoder".into(),
            nodes: vec![TreeNode {
                kind: "command_buffer".into(),
                label: "Command Buffer 0".into(),
                details: vec!["dispatches=1".into()],
                children: vec![TreeNode {
                    kind: "encoder".into(),
                    label: "enc".into(),
                    details: vec![],
                    children: vec![TreeNode {
                        kind: "kernel".into(),
                        label: "kernel".into(),
                        details: vec![],
                        children: vec![],
                    }],
                }],
            }],
        };

        let rendered = format_tree(&report);
        assert!(rendered.contains("Execution tree grouped by encoder"));
        assert!(rendered.contains("- command_buffer: Command Buffer 0"));
        assert!(rendered.contains("- encoder: enc"));
        assert!(rendered.contains("- kernel: kernel"));
    }

    #[test]
    fn summarizes_items_with_overflow() {
        assert_eq!(
            summarize_items(["a", "b", "c", "d"].into_iter(), 3),
            "a, b, c, +1 more"
        );
    }
}
