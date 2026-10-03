//! Hardware performance counters for AGX G15+ GPUs (M3/M4 families).
//!
//! Where the numbers come from (details and the oracle evidence are in
//! `docs/COUNTERS_M4.md`):
//!
//! - MTLReplayer's *limiter pass* replays the capture once with two kinds of
//!   time-sampled counters enabled:
//!   - per shader core (USC): `Counters_f_<core>.raw`, decoded with agxps
//!     (`APS_USC`, one sample every `CountPeriod` core cycles);
//!   - GPU-global: `RDE_0` / `BMPR_RDE_0` `GPRWCNTR` records in `streamData`,
//!     one sample every ~10 us, plus one `Firmware` record per kick.
//! - Every sample is a delta since the previous one. Kicks carry the
//!   encoder trace id, and `TraceId to BatchId` maps it to the encoder index,
//!   so samples are attributed to encoders by time overlap with that
//!   encoder's kicks, inside the same replay pass.
//! - Raw counter sums are turned into named counters by agxps' own compiled
//!   derived-counter formulas for the exact GPU (generation + variant
//!   chosen to match the profile's core and mGPU count).
//!
//! The 16 "Derived Counter Sample Data" passes are *not* used: their kick
//! records only hold the counts since the last (unrecorded) periodic
//! sample, i.e. the final ~10 us of each kick.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use crate::error::{Error, Result};
use crate::keyed_archive::{self, ArchiveValue};
use crate::profiler;
use crate::trace::TraceBundle;

/// Counters surfaced in reports, in display order. Names are agxps derived
/// counter names; values are converted with [`MetricUnit`].
pub const METRICS: &[MetricSpec] = &[
    MetricSpec::new("ALU Utilization", "alu", MetricUnit::Percent),
    MetricSpec::new("F32 Utilization", "f32", MetricUnit::Percent),
    MetricSpec::new("F16 Utilization", "f16", MetricUnit::Percent),
    MetricSpec::new("Instruction Issue Limiter", "issue_lim", MetricUnit::Percent),
    MetricSpec::new("Shader Core Limiter", "core_lim", MetricUnit::Percent),
    MetricSpec::new("Compute Occupancy", "occ", MetricUnit::Percent),
    MetricSpec::new("Compute Shader Launch Limiter", "launch_lim", MetricUnit::Percent),
    MetricSpec::new("L1 Cache Limiter", "l1_lim", MetricUnit::Percent),
    MetricSpec::new("Buffer L1 Miss Rate", "l1_miss", MetricUnit::Percent),
    MetricSpec::new("L2 Cache Limiter", "l2_lim", MetricUnit::Percent),
    MetricSpec::new("MMU Limiter", "mmu_lim", MetricUnit::Percent),
    MetricSpec::new("AF Read Bandwidth", "dram_rd", MetricUnit::GigabytesPerSecond),
    MetricSpec::new("AF Write Bandwidth", "dram_wr", MetricUnit::GigabytesPerSecond),
    MetricSpec::new("BytesReadFromMainMemory", "dram_rd_bytes", MetricUnit::Bytes),
    MetricSpec::new("BytesWrittenToMainMemory", "dram_wr_bytes", MetricUnit::Bytes),
    MetricSpec::new("L2 Bandwidth", "l2_bw", MetricUnit::GigabytesPerSecond),
    MetricSpec::new("Buffer L1 Load Bandwidth", "l1_ld_bw", MetricUnit::GigabytesPerSecond),
    MetricSpec::new("Buffer L1 Store Bandwidth", "l1_st_bw", MetricUnit::GigabytesPerSecond),
    MetricSpec::new(
        "Threadgroup Memory L1 Load Bandwidth",
        "tg_ld_bw",
        MetricUnit::GigabytesPerSecond,
    ),
    MetricSpec::new(
        "Threadgroup Memory L1 Store Bandwidth",
        "tg_st_bw",
        MetricUnit::GigabytesPerSecond,
    ),
    MetricSpec::new("Compute Threads Launched", "threads", MetricUnit::Count),
    MetricSpec::new("ALU F32 Instructions", "f32_inst", MetricUnit::Count),
    MetricSpec::new("ALU F16 Instructions", "f16_inst", MetricUnit::Count),
    MetricSpec::new("Instructions Executed", "inst", MetricUnit::Count),
];

#[derive(Debug, Clone, Copy)]
pub struct MetricSpec {
    /// agxps derived counter name.
    pub name: &'static str,
    /// Short column label.
    pub label: &'static str,
    pub unit: MetricUnit,
}

impl MetricSpec {
    const fn new(name: &'static str, label: &'static str, unit: MetricUnit) -> Self {
        Self { name, label, unit }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricUnit {
    /// agxps returns a 0..1 fraction; shown as percent.
    Percent,
    /// agxps returns GiB/s-style 2^30 bytes per second; shown as GB/s (1e9).
    GigabytesPerSecond,
    Bytes,
    Count,
}

impl MetricUnit {
    /// Convert an agxps value into the unit shown in reports.
    pub fn display_value(self, value: f64) -> f64 {
        match self {
            Self::Percent => value * 100.0,
            Self::GigabytesPerSecond => value * (1u64 << 30) as f64 / 1e9,
            Self::Bytes | Self::Count => value,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HwCounterReport {
    pub gpu: HwGpu,
    /// Shader-core cycles per USC counter sample.
    pub usc_count_period_cycles: u64,
    pub usc_streams: usize,
    pub global_sample_period_ns: f64,
    pub encoders: Vec<HwCounterRow>,
    pub dispatches: Vec<HwCounterRow>,
    /// One row per kernel: per-dispatch raw counts summed by kernel name
    /// (or by kernel program address when no names are known).
    pub kernels: Vec<HwCounterRow>,
    /// Kicks in the limiter pass that belong to no encoder of this capture
    /// (other processes, or replayer-internal work).
    pub foreign_kicks: usize,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct HwGpu {
    pub gpu_type: String,
    pub generation: u32,
    pub variant: u32,
    pub num_cores: u64,
    pub num_mgpus: u64,
    pub peak_dram_gbps: f64,
}

#[derive(Debug, Clone, Default)]
pub struct HwCounterRow {
    pub label: String,
    pub encoder_index: Option<usize>,
    /// `drawCallIndex` of the dispatch in the capture.
    pub dispatch_index: Option<usize>,
    pub dispatches: usize,
    /// Wall time: kick time for encoders, union of the clique intervals for
    /// dispatches and kernels.
    pub gpu_time_ns: f64,
    /// Fraction of the row's time during which a kick that is not part of
    /// this capture was also running. Counters are GPU-wide or per core, not
    /// per process, so a large value makes the row unreliable.
    pub foreign_overlap: f64,
    /// Fraction of the row's counts taken from samples it shared with other
    /// rows of the same kind (split by clique time). 0 means every sample was
    /// exclusively this row's.
    pub shared: f64,
    /// agxps derived counter name -> value in agxps units.
    pub values: BTreeMap<String, f64>,
}

impl HwCounterRow {
    pub fn metric(&self, spec: &MetricSpec) -> Option<f64> {
        self.values
            .get(spec.name)
            .copied()
            .filter(|value| value.is_finite())
            .map(|value| spec.unit.display_value(value))
    }

    pub fn metric_named(&self, name: &str) -> Option<f64> {
        METRICS
            .iter()
            .find(|spec| spec.name == name)
            .and_then(|spec| self.metric(spec))
    }
}

/// A kick of the limiter pass, in one clock domain.
#[derive(Debug, Clone, Copy)]
struct Kick {
    start: u64,
    end: u64,
    software_id: u64,
    encoder: Option<usize>,
}

/// One clique interval: when a dispatch's work was resident on a USC.
#[derive(Debug, Clone, Copy)]
struct Activity {
    start: u64,
    end: u64,
    /// Index into the kick list of the stream being attributed.
    kick: usize,
    dispatch: Option<usize>,
}

/// One time-sampled counter stream.
struct SampleStream {
    /// Sample `i` covers `(ends[i - 1], ends[i]]`.
    ends: Vec<u64>,
    cycles: Vec<u64>,
    /// `values[counter][sample]`.
    values: Vec<Vec<u64>>,
    names: Vec<String>,
}

#[derive(Default, Clone)]
struct Accum {
    raw: BTreeMap<String, f64>,
    cycles: f64,
    seconds: f64,
    weight: f64,
    shared_weight: f64,
}

impl Accum {
    fn add_counts(&mut self, stream: &SampleStream, sample: usize, share: f64, shared: bool) {
        for (name, column) in stream.names.iter().zip(&stream.values) {
            if let Some(value) = column.get(sample) {
                *self.raw.entry(name.clone()).or_default() += *value as f64 * share;
            }
        }
        self.weight += share;
        if shared {
            self.shared_weight += share;
        }
    }

    fn add_time(&mut self, cycles: f64, seconds: f64) {
        self.cycles += cycles;
        self.seconds += seconds;
    }

    fn merge(&mut self, other: &Accum) {
        for (name, value) in &other.raw {
            *self.raw.entry(name.clone()).or_default() += value;
        }
        self.cycles += other.cycles;
        self.seconds += other.seconds;
        self.weight += other.weight;
        self.shared_weight += other.shared_weight;
    }

    fn scale_time(&mut self, factor: f64) {
        self.cycles *= factor;
        self.seconds *= factor;
    }
}

#[derive(Default, Clone)]
struct Accums {
    encoders: BTreeMap<usize, Accum>,
    dispatches: BTreeMap<usize, Accum>,
}

impl Accums {
    fn merge_counts(&mut self, other: &Accums, take_time: bool) {
        for (target, source) in [
            (&mut self.encoders, &other.encoders),
            (&mut self.dispatches, &other.dispatches),
        ] {
            for (key, accum) in source {
                let entry = target.entry(*key).or_default();
                for (name, value) in &accum.raw {
                    *entry.raw.entry(name.clone()).or_default() += value;
                }
                if take_time {
                    entry.cycles += accum.cycles;
                    entry.seconds += accum.seconds;
                    entry.weight += accum.weight;
                    entry.shared_weight += accum.shared_weight;
                }
            }
        }
    }
}

/// A dispatch as `Program Address Mappings` describes it.
#[derive(Debug, Clone, Copy)]
struct DispatchInfo {
    encoder: usize,
    /// Address of the kernel program, shared by every dispatch of a pipeline.
    kernel_address: Option<u64>,
}

pub fn report(trace: &TraceBundle) -> Result<HwCounterReport> {
    let profiler_dir = profiler::find_profiler_directory(&trace.path).ok_or_else(|| {
        Error::InvalidInput(format!(
            "no .gpuprofiler_raw directory for {}",
            trace.path.display()
        ))
    })?;
    let names = profiler::stream_data_summary(&trace.path)
        .map(|summary| dispatch_names(&summary))
        .unwrap_or_default();
    report_for_profiler_dir(&profiler_dir, &names)
}

/// Kernel name of every dispatch, keyed by its index in the capture.
pub fn dispatch_names(summary: &profiler::ProfilerStreamDataSummary) -> BTreeMap<usize, String> {
    summary
        .dispatches
        .iter()
        .filter_map(|dispatch| Some((dispatch.index, dispatch.function_name.clone()?)))
        .collect()
}

#[cfg(not(target_os = "macos"))]
pub fn report_for_profiler_dir(
    _profiler_dir: &Path,
    _dispatch_names: &BTreeMap<usize, String>,
) -> Result<HwCounterReport> {
    Err(Error::Unsupported("hardware counters require macOS and Xcode"))
}

#[cfg(target_os = "macos")]
pub fn report_for_profiler_dir(
    profiler_dir: &Path,
    dispatch_names: &BTreeMap<usize, String>,
) -> Result<HwCounterReport> {
    use agxps_sys::counters::{ApsParseSettings, counter_api};

    let stream_data = fs::read(profiler_dir.join("streamData"))?;
    let root = keyed_archive::decode(&stream_data)
        .ok_or_else(|| Error::InvalidInput("streamData is not a keyed archive".to_owned()))?;
    let entries = root
        .get("APSCounterData")
        .and_then(ArchiveValue::as_array)
        .ok_or_else(|| Error::InvalidInput("streamData has no APSCounterData".to_owned()))?
        .iter()
        .filter_map(ArchiveValue::nested)
        .collect::<Vec<_>>();
    let limiter = entries
        .iter()
        .find(|entry| entry.get("Limiter Counter List Map").is_some())
        .ok_or_else(|| {
            Error::InvalidInput("profile has no limiter pass (Limiter Counter List Map)".to_owned())
        })?;
    let metadata = entries
        .iter()
        .find(|entry| entry.get("Configuration Variables").is_some())
        .ok_or_else(|| Error::InvalidInput("profile has no APS metadata".to_owned()))?;
    let config = metadata.get("Configuration Variables").unwrap();
    let config_u64 = |key: &str| config.get(key).and_then(ArchiveValue::as_u64);
    let mut warnings = Vec::new();

    // --- GPU identity: the agxps variant whose core / mGPU counts match. ---
    let generation = config_u64("gpu_gen")
        .ok_or_else(|| Error::InvalidInput("Configuration Variables lack gpu_gen".to_owned()))?
        as u32;
    let num_cores = config_u64("num_cores").unwrap_or(0);
    let num_mgpus = config_u64("num_mgpus").unwrap_or(1);
    let num_gps = config_u64("num_gps").unwrap_or(1);
    let gpu_type = config
        .get("gpu_type")
        .and_then(ArchiveValue::as_str)
        .unwrap_or("?")
        .to_owned();
    let api = counter_api().map_err(|error| Error::InvalidInput(error.to_string()))?;
    let shape = api
        .variants(generation)
        .into_iter()
        .find(|shape| shape.num_cores == num_cores && shape.num_mgpus == num_mgpus)
        .ok_or_else(|| {
            Error::InvalidInput(format!(
                "agxps knows no generation-{generation} GPU with {num_cores} cores / {num_mgpus} mGPUs ({gpu_type})"
            ))
        })?;
    let gpu = api
        .gpu(generation, shape.variant)
        .map_err(|error| Error::InvalidInput(error.to_string()))?;
    tracing::debug!(
        gpu_type,
        generation,
        variant = shape.variant,
        num_cores,
        num_mgpus,
        "hw counters: agxps gpu"
    );

    let timebase_ns = metadata
        .get("Timebase")
        .and_then(ArchiveValue::as_array)
        .and_then(|parts| Some(parts.first()?.as_f64()? / parts.get(1)?.as_f64()?))
        .unwrap_or(125.0 / 3.0);
    let constants = [
        ("NUM_CORES", num_cores as f64),
        ("NUM_L2_BANKS", (2 * num_gps) as f64),
        ("NUM_AGCS", config_u64("num_agcs").unwrap_or(1) as f64),
        ("NUM_GPS", num_gps as f64),
        ("NSEC_PER_SEC", 1e9),
        ("TIME_SCALE", timebase_ns),
        (
            "OMU_EVAL_WINDOW_DEFAULT",
            config_u64("omu_eval_window").unwrap_or(2048) as f64,
        ),
    ];

    // --- Encoders: limiter-pass trace id -> encoder index. ---
    let encoder_of_trace = metadata
        .get("TraceId to BatchId")
        .and_then(ArchiveValue::nested)
        .and_then(|map| {
            Some(
                map.as_dictionary()?
                    .iter()
                    .filter_map(|(trace_id, batch)| {
                        Some((trace_id.parse::<u64>().ok()?, batch.as_u64()? as usize))
                    })
                    .collect::<BTreeMap<u64, usize>>(),
            )
        })
        .filter(|map| !map.is_empty())
        .ok_or_else(|| Error::InvalidInput("profile has no TraceId to BatchId map".to_owned()))?;
    let encoder_for = |software_id: u64| encoder_of_trace.get(&(software_id >> 32)).copied();

    // --- Dispatches: (encoder trace id, shader-launch program) -> dispatch. ---
    let mut dispatch_of_esl = BTreeMap::<(u64, u64), usize>::new();
    let mut dispatch_info = BTreeMap::<usize, DispatchInfo>::new();
    for mapping in metadata
        .get("Program Address Mappings")
        .and_then(ArchiveValue::as_array)
        .unwrap_or(&[])
    {
        let field = |key: &str| mapping.get(key).and_then(ArchiveValue::as_u64);
        let (Some(kind), Some(encoder_trace), Some(address), Some(dispatch)) = (
            mapping.get("type").and_then(ArchiveValue::as_str),
            field("encID"),
            field("mappedAddress"),
            field("drawCallIndex"),
        ) else {
            continue;
        };
        let dispatch = dispatch as usize;
        let Some(encoder) = encoder_of_trace.get(&encoder_trace).copied() else {
            continue;
        };
        match kind {
            "compute-sl" => {
                dispatch_of_esl.insert((encoder_trace, address), dispatch);
                dispatch_info
                    .entry(dispatch)
                    .or_insert(DispatchInfo {
                        encoder,
                        kernel_address: None,
                    })
                    .encoder = encoder;
            }
            "compute" => {
                dispatch_info
                    .entry(dispatch)
                    .or_insert(DispatchInfo {
                        encoder,
                        kernel_address: None,
                    })
                    .kernel_address = Some(address);
            }
            _ => {}
        }
    }

    // --- Limiter pass, GPU-global sources (absolute ticks). ---
    let list_map = limiter
        .get("Limiter Counter List Map")
        .and_then(ArchiveValue::as_dictionary)
        .unwrap();
    let names_for = |source: &str| -> Vec<String> {
        list_map
            .get(source)
            .and_then(ArchiveValue::as_array)
            .map(|names| {
                names
                    .iter()
                    .filter_map(ArchiveValue::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut global_streams = BTreeMap::<(String, u64), SampleStream>::new();
    let mut global_kicks = Vec::<Kick>::new();
    for entry in &entries {
        let (Some(source), Some(blob)) = (
            entry.get("Source").and_then(ArchiveValue::as_str),
            entry.get("ShaderProfilerData").and_then(ArchiveValue::as_data),
        ) else {
            continue;
        };
        let ring = entry
            .get("RingBufferIndex")
            .and_then(ArchiveValue::as_u64)
            .unwrap_or(0);
        let names = names_for(source);
        let records = gprw_records(blob, 8 + names.len());
        if source == "Firmware" {
            // [magic, ts, cycles, type, encoder id, kick id, slot, source,
            //  kick end, kick start] for every finished kick.
            for record in records {
                if record.len() >= 10 && record[3] == 5 {
                    let (a, b) = (record[8], record[9]);
                    let software_id = (record[4] << 32) | (record[5] & 0xffff_ffff);
                    global_kicks.push(Kick {
                        start: a.min(b),
                        end: a.max(b),
                        software_id,
                        encoder: encoder_for(software_id),
                    });
                }
            }
            continue;
        }
        let stream = global_streams
            .entry((source.to_owned(), ring))
            .or_insert_with(|| SampleStream {
                ends: Vec::new(),
                cycles: Vec::new(),
                values: vec![Vec::new(); names.len()],
                names: names.clone(),
            });
        for record in records {
            if record[3] != 6 {
                continue;
            }
            stream.ends.push(record[1]);
            stream.cycles.push(record[2]);
            for (index, column) in stream.values.iter_mut().enumerate() {
                column.push(record[8 + index]);
            }
        }
    }
    for stream in global_streams.values_mut() {
        sort_stream(stream);
    }
    let global_sample_period_ns = global_streams
        .values()
        .next()
        .map(|stream| median_period(&stream.ends) * timebase_ns)
        .unwrap_or(0.0);

    // --- Limiter pass, per-core APS streams (continuous ticks). ---
    let aps_options = metadata.get("APS Options");
    let kick_options = aps_options.and_then(|options| options.get("KickAndStateTracing"));
    let option_u64 = |value: Option<&ArchiveValue>, key: &str| {
        value
            .and_then(|value| value.get(key))
            .and_then(ArchiveValue::as_u64)
    };
    let usc_names = names_for("APS_USC");
    let settings = ApsParseSettings {
        pulse_period: option_u64(kick_options, "PulsePeriod").unwrap_or(2048) as u32,
        system_time_period: option_u64(aps_options, "SystemTimePeriod").unwrap_or(64) as u32,
        count_period: option_u64(kick_options, "CountPeriod").unwrap_or(32768) as u32,
        chunk_size: option_u64(aps_options, "ChunkSize").unwrap_or(0x1000),
        uarch_behaviour: usc_names.iter().any(|name| name == UARCH_TRIGGER_COUNTER),
    };
    let usc_files = entries
        .iter()
        .filter(|entry| entry.get("Source").and_then(ArchiveValue::as_str) == Some("APS_USC"))
        .filter_map(|entry| entry.get("APSTraceDataFile").and_then(ArchiveValue::as_str))
        .map(|file| profiler_dir.join(file))
        .collect::<Vec<_>>();
    let mut usc_profiles = Vec::with_capacity(usc_files.len());
    for path in &usc_files {
        let bytes = fs::read(path)?;
        match gpu.parse_aps_counters(settings, &bytes) {
            Ok(profile) => usc_profiles.push(profile),
            Err(error) => warnings.push(format!("{}: {error}", path.display())),
        }
    }

    // --- Clock join: APS kicks (continuous) vs Firmware kicks (absolute). ---
    let firmware_start = global_kicks
        .iter()
        .map(|kick| (kick.software_id, kick.start))
        .collect::<BTreeMap<_, _>>();
    let mut offsets = usc_profiles
        .iter()
        .flat_map(|profile| profile.kicks.iter())
        .filter_map(|kick| {
            let fw = *firmware_start.get(&kick.software_id)?;
            Some(kick.start_ticks as i128 - fw as i128)
        })
        .collect::<Vec<_>>();
    offsets.sort_unstable();
    let continuous_minus_absolute = offsets.get(offsets.len() / 2).copied().unwrap_or_else(|| {
        let field = |key: &str| metadata.get(key).and_then(ArchiveValue::as_i64).unwrap_or(0);
        field("Continuous Time") as i128 - field("Absolute Time") as i128
    });
    let to_absolute = |ticks: u64| (ticks as i128 - continuous_minus_absolute).max(0) as u64;

    // Kicks seen only by the USC streams (replayer-internal work, other
    // processes) still compete for the same samples.
    let mut known = global_kicks
        .iter()
        .map(|kick| kick.software_id)
        .collect::<BTreeSet<_>>();
    for profile in &usc_profiles {
        for kick in &profile.kicks {
            if kick.missing_end || kick.end_ticks <= kick.start_ticks {
                continue;
            }
            if known.insert(kick.software_id) {
                global_kicks.push(Kick {
                    start: to_absolute(kick.start_ticks),
                    end: to_absolute(kick.end_ticks),
                    software_id: kick.software_id,
                    encoder: encoder_for(kick.software_id),
                });
            }
        }
    }
    let global_kick_of = global_kicks
        .iter()
        .enumerate()
        .map(|(index, kick)| (kick.software_id, index))
        .collect::<BTreeMap<_, _>>();

    // --- Per-USC attribution, weighted by clique residency. ---
    let mut usc_accums = Accums::default();
    let mut global_activity = Vec::<Activity>::new();
    let mut unmapped_cliques = 0usize;
    let mut mapped_cliques = 0usize;
    for profile in &usc_profiles {
        let kicks = profile
            .kicks
            .iter()
            .map(|kick| Kick {
                start: kick.start_ticks,
                end: if kick.missing_end {
                    kick.start_ticks
                } else {
                    kick.end_ticks
                },
                software_id: kick.software_id,
                encoder: encoder_for(kick.software_id),
            })
            .collect::<Vec<_>>();
        let activity = usc_activity(profile, &kicks, &dispatch_of_esl);
        for item in &activity {
            if item.dispatch.is_some() {
                mapped_cliques += 1;
            } else if kicks[item.kick].encoder.is_some() {
                unmapped_cliques += 1;
            }
            if let Some(global) = global_kick_of.get(&kicks[item.kick].software_id) {
                global_activity.push(Activity {
                    start: to_absolute(item.start),
                    end: to_absolute(item.end),
                    kick: *global,
                    dispatch: item.dispatch,
                });
            }
        }
        let stream = SampleStream {
            ends: profile.sample_end_ticks.clone(),
            cycles: profile.sample_cycles.clone(),
            values: profile.values.clone(),
            names: profile.counter_names.clone(),
        };
        attribute(&stream, &kicks, &activity, timebase_ns, &mut usc_accums);
    }
    if unmapped_cliques > 0 {
        warnings.push(format!(
            "{unmapped_cliques} of {} work cliques of this capture's kicks matched no dispatch",
            unmapped_cliques + mapped_cliques
        ));
    }
    // Raw counts add up across cores; cycles and time are per core.
    let core_scale = 1.0 / usc_profiles.len().max(1) as f64;
    for accum in usc_accums
        .encoders
        .values_mut()
        .chain(usc_accums.dispatches.values_mut())
    {
        accum.scale_time(core_scale);
    }

    // --- GPU-global attribution: one family per source; rings are instances
    // of the same block (counts add), ring 0 carries the clock. ---
    global_activity.sort_by_key(|item| item.start);
    let mut global_families = BTreeMap::<String, Accums>::new();
    for ((source, ring), stream) in &global_streams {
        let mut accums = Accums::default();
        attribute(stream, &global_kicks, &global_activity, timebase_ns, &mut accums);
        // BTreeMap order visits each source's lowest ring first.
        let take_time = !global_families.contains_key(source);
        tracing::trace!(source, ring, take_time, "hw counters: global stream");
        global_families
            .entry(source.clone())
            .or_default()
            .merge_counts(&accums, take_time);
    }

    // --- Kernels: dispatch accumulations summed by kernel. ---
    let kernel_key = |dispatch: usize| -> String {
        dispatch_names.get(&dispatch).cloned().unwrap_or_else(|| {
            match dispatch_info.get(&dispatch).and_then(|info| info.kernel_address) {
                Some(address) => format!("kernel@{address:#x}"),
                None => format!("dispatch {dispatch}"),
            }
        })
    };
    let kernel_accums = |accums: &Accums| -> BTreeMap<String, Accum> {
        let mut out = BTreeMap::<String, Accum>::new();
        for (dispatch, accum) in &accums.dispatches {
            out.entry(kernel_key(*dispatch)).or_default().merge(accum);
        }
        out
    };

    // --- Derived counters per family, for every row kind. ---
    let derived = gpu.derived_counters();
    let families = std::iter::once(&usc_accums)
        .chain(global_families.values())
        .collect::<Vec<_>>();
    let mut encoder_values = BTreeMap::<usize, BTreeMap<String, f64>>::new();
    let mut dispatch_values = BTreeMap::<usize, BTreeMap<String, f64>>::new();
    let mut kernel_values = BTreeMap::<String, BTreeMap<String, f64>>::new();
    for family in &families {
        let kernels = kernel_accums(family);
        derive_into(&gpu, &derived, &constants, &family.encoders, &mut encoder_values)?;
        derive_into(&gpu, &derived, &constants, &family.dispatches, &mut dispatch_values)?;
        derive_into(&gpu, &derived, &constants, &kernels, &mut kernel_values)?;
    }

    // --- Timing and contamination. ---
    let foreign = merge_intervals(
        global_kicks
            .iter()
            .filter(|kick| kick.encoder.is_none())
            .map(|kick| (kick.start, kick.end))
            .collect(),
    );
    let foreign_kick_count = global_kicks
        .iter()
        .filter(|kick| kick.encoder.is_none())
        .count();
    let mut dispatch_intervals = BTreeMap::<usize, Vec<(u64, u64)>>::new();
    for item in &global_activity {
        if let Some(dispatch) = item.dispatch {
            dispatch_intervals
                .entry(dispatch)
                .or_default()
                .push((item.start, item.end));
        }
    }
    let dispatch_spans = dispatch_intervals
        .into_iter()
        .map(|(dispatch, intervals)| (dispatch, merge_intervals(intervals)))
        .collect::<BTreeMap<_, _>>();
    let span_ns = |spans: &[(u64, u64)]| {
        spans.iter().map(|(start, end)| end - start).sum::<u64>() as f64 * timebase_ns
    };
    let foreign_fraction = |spans: &[(u64, u64)]| {
        let total = spans.iter().map(|(start, end)| end - start).sum::<u64>();
        if total == 0 {
            return 0.0;
        }
        let shared = spans
            .iter()
            .map(|(start, end)| {
                foreign
                    .iter()
                    .map(|(f_start, f_end)| overlap(*start, *end, *f_start, *f_end))
                    .sum::<u64>()
            })
            .sum::<u64>();
        shared as f64 / total as f64
    };
    let shared_of = |accum: Option<&Accum>| {
        accum
            .filter(|accum| accum.weight > 0.0)
            .map(|accum| accum.shared_weight / accum.weight)
            .unwrap_or(0.0)
    };

    let mut encoders = Vec::new();
    for (trace_id, encoder_index) in &encoder_of_trace {
        let spans = merge_intervals(
            global_kicks
                .iter()
                .filter(|kick| kick.encoder == Some(*encoder_index))
                .map(|kick| (kick.start, kick.end))
                .collect(),
        );
        let values = encoder_values.remove(encoder_index).unwrap_or_default();
        if spans.is_empty() && values.is_empty() {
            continue;
        }
        encoders.push(HwCounterRow {
            label: format!("encoder {encoder_index} ({trace_id:#x})"),
            encoder_index: Some(*encoder_index),
            dispatch_index: None,
            dispatches: dispatch_info
                .values()
                .filter(|info| info.encoder == *encoder_index)
                .count(),
            gpu_time_ns: span_ns(&spans),
            foreign_overlap: foreign_fraction(&spans),
            shared: shared_of(usc_accums.encoders.get(encoder_index)),
            values,
        });
    }
    encoders.sort_by_key(|row| row.encoder_index);

    let mut dispatches = Vec::new();
    let mut kernel_spans = BTreeMap::<String, Vec<(u64, u64)>>::new();
    let mut kernel_dispatches = BTreeMap::<String, usize>::new();
    for (dispatch, values) in dispatch_values {
        let spans = dispatch_spans.get(&dispatch).cloned().unwrap_or_default();
        let kernel = kernel_key(dispatch);
        kernel_spans
            .entry(kernel.clone())
            .or_default()
            .extend(spans.iter().copied());
        *kernel_dispatches.entry(kernel.clone()).or_default() += 1;
        dispatches.push(HwCounterRow {
            label: kernel,
            encoder_index: dispatch_info.get(&dispatch).map(|info| info.encoder),
            dispatch_index: Some(dispatch),
            dispatches: 1,
            gpu_time_ns: span_ns(&spans),
            foreign_overlap: foreign_fraction(&spans),
            shared: shared_of(usc_accums.dispatches.get(&dispatch)),
            values,
        });
    }
    let usc_kernels = kernel_accums(&usc_accums);
    let mut kernels = kernel_values
        .into_iter()
        .map(|(kernel, values)| {
            let spans = merge_intervals(kernel_spans.remove(&kernel).unwrap_or_default());
            HwCounterRow {
                encoder_index: None,
                dispatch_index: None,
                dispatches: kernel_dispatches.get(&kernel).copied().unwrap_or(0),
                gpu_time_ns: span_ns(&spans),
                foreign_overlap: foreign_fraction(&spans),
                shared: shared_of(usc_kernels.get(&kernel)),
                label: kernel,
                values,
            }
        })
        .collect::<Vec<_>>();
    kernels.sort_by(|left, right| right.gpu_time_ns.total_cmp(&left.gpu_time_ns));

    if foreign_kick_count > 0 {
        warnings.push(format!(
            "{foreign_kick_count} kick(s) not from this capture ran during the counter pass; rows with foreign overlap mix their counts in"
        ));
    }
    if usc_profiles.is_empty() {
        warnings.push(
            "no per-core APS counter streams decoded: occupancy/ALU/L1 counters are missing"
                .to_owned(),
        );
    }

    Ok(HwCounterReport {
        gpu: HwGpu {
            gpu_type,
            generation,
            variant: shape.variant,
            num_cores,
            num_mgpus,
            peak_dram_gbps: shape.peak_dram_gbps,
        },
        usc_count_period_cycles: settings.count_period as u64,
        usc_streams: usc_profiles.len(),
        global_sample_period_ns,
        encoders,
        dispatches,
        kernels,
        foreign_kicks: foreign_kick_count,
        warnings,
    })
}

/// Evaluate every derived counter computable from `targets`' raw counters
/// and record the first value seen for each (target, counter).
#[cfg(target_os = "macos")]
fn derive_into<K: Ord + Clone>(
    gpu: &agxps_sys::counters::CounterGpu<'_>,
    derived: &[agxps_sys::counters::CounterIdent],
    constants: &[(&str, f64)],
    targets: &BTreeMap<K, Accum>,
    out: &mut BTreeMap<K, BTreeMap<String, f64>>,
) -> Result<()> {
    let targets = targets
        .iter()
        .filter(|(_, accum)| accum.seconds > 0.0)
        .collect::<Vec<_>>();
    if targets.is_empty() {
        return Ok(());
    }
    let raw_names = targets
        .iter()
        .flat_map(|(_, accum)| accum.raw.keys().cloned())
        .collect::<BTreeSet<_>>();
    let raw_idents = raw_names
        .iter()
        .filter_map(|name| Some((gpu.ident(name)?, name.clone())))
        .collect::<Vec<_>>();
    let available = raw_idents
        .iter()
        .map(|(ident, _)| *ident)
        .collect::<BTreeSet<_>>();
    let computable = derived
        .iter()
        .copied()
        .filter(|ident| {
            let deps = gpu.raw_dependencies(&[*ident]);
            !deps.is_empty() && deps.iter().all(|dep| available.contains(dep))
        })
        .collect::<Vec<_>>();
    if computable.is_empty() {
        return Ok(());
    }
    let raw = raw_idents
        .iter()
        .map(|(ident, name)| {
            (
                *ident,
                targets
                    .iter()
                    .map(|(_, accum)| accum.raw.get(name).copied().unwrap_or(0.0).round() as u64)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let cycles = targets
        .iter()
        .map(|(_, accum)| accum.cycles.round() as u64)
        .collect::<Vec<_>>();
    let seconds = targets
        .iter()
        .map(|(_, accum)| accum.seconds)
        .collect::<Vec<_>>();
    let results = gpu
        .compute(&raw, &cycles, &seconds, constants, &computable)
        .map_err(|error| Error::InvalidInput(error.to_string()))?;
    for (ident, series) in computable.iter().zip(results) {
        let Some(series) = series else { continue };
        let name = gpu.info(*ident).name;
        for ((key, _), value) in targets.iter().zip(series) {
            out.entry((*key).clone())
                .or_default()
                .entry(name.clone())
                .or_insert(value);
        }
    }
    Ok(())
}

/// Clique residency intervals of one USC stream, mapped to dispatches.
///
/// A work clique's `esl_index` indexes the commands the timing analyzer
/// reports for this USC (one per dispatch that ran on it, in order); the
/// command's shader-launch program address names the dispatch. Clique ends are not traced in the
/// limiter pass: a missing end is the next start on the same slot or the
/// end of the trace. Only hand-overs between cliques of the same command are
/// trusted as durations; every other end is capped at that command's median
/// measured duration on this USC, or at the next command's start.
#[cfg(target_os = "macos")]
fn usc_activity(
    profile: &agxps_sys::counters::ApsCounterProfile,
    kicks: &[Kick],
    dispatch_of_esl: &BTreeMap<(u64, u64), usize>,
) -> Vec<Activity> {
    let cliques = &profile.work_cliques;
    // Owner of a clique: (kick index, command index on this USC).
    let owners = cliques
        .iter()
        .map(|clique| {
            let kick = clique.kick_index as usize;
            let command_index = usize::try_from(clique.esl_index).ok()?;
            let command = profile.commands.get(command_index)?;
            (command.software_id == kicks.get(kick)?.software_id).then_some((kick, command_index))
        })
        .collect::<Vec<_>>();

    // A missing end is a measured duration only when the slot was handed
    // straight to another clique of the same command (steady state). A hand-
    // over to a different command may follow an idle gap, and an end that is
    // nobody's start is the end of the trace.
    let mut order = (0..cliques.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| (cliques[*index].slot, cliques[*index].start_ticks));
    let mut measured = cliques
        .iter()
        .map(|clique| !clique.missing_end)
        .collect::<Vec<_>>();
    for pair in order.windows(2) {
        let (this, next) = (&cliques[pair[0]], &cliques[pair[1]]);
        if this.slot == next.slot
            && this.end_ticks == next.start_ticks
            && owners[pair[0]].is_some()
            && owners[pair[0]] == owners[pair[1]]
        {
            measured[pair[0]] = true;
        }
    }
    let mut durations = BTreeMap::<(usize, usize), Vec<u64>>::new();
    for (index, clique) in cliques.iter().enumerate() {
        if let Some(owner) = owners[index]
            && measured[index]
            && clique.end_ticks > clique.start_ticks
        {
            durations
                .entry(owner)
                .or_default()
                .push(clique.end_ticks - clique.start_ticks);
        }
    }
    let typical = durations
        .into_iter()
        .map(|(owner, mut values)| {
            values.sort_unstable();
            (owner, values[values.len() / 2])
        })
        .collect::<BTreeMap<_, _>>();
    // Command starts per kick, to bound cliques with no measured duration
    // by the next command's start (dispatches of an encoder mostly run
    // back to back behind barriers).
    let mut starts_by_kick = BTreeMap::<u64, Vec<u64>>::new();
    for command in &profile.commands {
        starts_by_kick
            .entry(command.software_id)
            .or_default()
            .push(command.start_ticks);
    }
    for starts in starts_by_kick.values_mut() {
        starts.sort_unstable();
    }

    let mut activity = Vec::with_capacity(cliques.len());
    for (index, clique) in cliques.iter().enumerate() {
        let kick = clique.kick_index as usize;
        let Some(kick_info) = kicks.get(kick) else {
            continue;
        };
        let owner = owners[index];
        let mut end = clique.end_ticks;
        if !measured[index] {
            match owner.and_then(|owner| typical.get(&owner)) {
                Some(duration) => end = end.min(clique.start_ticks + duration),
                None => {
                    if let Some(next) = starts_by_kick
                        .get(&kick_info.software_id)
                        .and_then(|starts| starts.iter().find(|start| **start > clique.start_ticks))
                    {
                        end = end.min(*next);
                    }
                }
            }
        }
        if kick_info.end > kick_info.start {
            end = end.min(kick_info.end);
        }
        if end <= clique.start_ticks {
            continue;
        }
        let dispatch = owner.and_then(|(_, command_index)| {
            let address = profile.commands[command_index].esl_shader_address;
            dispatch_of_esl
                .get(&(kick_info.software_id >> 32, address))
                .copied()
        });
        activity.push(Activity {
            start: clique.start_ticks,
            end,
            kick,
            dispatch,
        });
    }
    activity.sort_by_key(|item| item.start);
    activity
}

/// Attribute each sample of `stream` to the kicks, encoders and dispatches
/// that were running during it.
///
/// Kicks overlapping a sample share it in proportion to their clique
/// residency inside the sample (time overlap when no clique is visible):
/// a blit or a kick idle on this core takes nothing, and the idle remainder
/// of a sample holds no events. Within a kick, dispatches share in
/// proportion to their own clique residency. Kicks of other processes take
/// their share and drop it. Time is credited per row as the part of the
/// sample the row was running (kick span for encoders, clique union for
/// dispatches).
fn attribute(
    stream: &SampleStream,
    kicks: &[Kick],
    activity: &[Activity],
    timebase_ns: f64,
    out: &mut Accums,
) {
    let mut next = 0;
    let mut active = Vec::<Activity>::new();
    for sample in 1..stream.ends.len() {
        let (start, end) = (stream.ends[sample - 1], stream.ends[sample]);
        if end <= start {
            continue;
        }
        while next < activity.len() && activity[next].start < end {
            active.push(activity[next]);
            next += 1;
        }
        active.retain(|item| item.end > start);

        let kick_overlap = kicks
            .iter()
            .enumerate()
            .filter_map(|(index, kick)| {
                let ticks = overlap(start, end, kick.start, kick.end);
                (ticks > 0).then_some((index, ticks))
            })
            .collect::<BTreeMap<_, _>>();
        if kick_overlap.is_empty() {
            continue;
        }
        let mut residency = BTreeMap::<usize, u64>::new();
        let mut by_dispatch = BTreeMap::<(usize, usize), Vec<(u64, u64)>>::new();
        for item in &active {
            if !kick_overlap.contains_key(&item.kick) {
                continue;
            }
            let ticks = overlap(start, end, item.start, item.end);
            if ticks == 0 {
                continue;
            }
            *residency.entry(item.kick).or_default() += ticks;
            if let Some(dispatch) = item.dispatch {
                by_dispatch
                    .entry((item.kick, dispatch))
                    .or_default()
                    .push((item.start.max(start), item.end.min(end)));
            }
        }
        let total_residency = residency.values().sum::<u64>();
        let shares = if total_residency > 0 {
            residency
                .iter()
                .map(|(kick, ticks)| (*kick, *ticks as f64 / total_residency as f64))
                .collect::<BTreeMap<_, _>>()
        } else {
            let total = kick_overlap.values().sum::<u64>() as f64;
            kick_overlap
                .iter()
                .map(|(kick, ticks)| (*kick, *ticks as f64 / total))
                .collect()
        };
        let duration = (end - start) as f64;
        let cycles = stream.cycles.get(sample).copied().unwrap_or(0) as f64;
        let encoders_present = shares
            .keys()
            .filter_map(|kick| kicks[*kick].encoder)
            .collect::<BTreeSet<_>>();
        let shared_encoders = encoders_present.len() > 1 || shares.len() > encoders_present.len();
        let shared_dispatches = by_dispatch.len() > 1 || shares.len() > 1;
        for (kick, share) in &shares {
            let Some(encoder) = kicks[*kick].encoder else {
                continue;
            };
            let accum = out.encoders.entry(encoder).or_default();
            accum.add_counts(stream, sample, *share, shared_encoders);
            let ticks = kick_overlap.get(kick).copied().unwrap_or(0) as f64;
            accum.add_time(cycles * ticks / duration, ticks * timebase_ns * 1e-9);
            let kick_residency = residency.get(kick).copied().unwrap_or(0);
            if kick_residency == 0 {
                continue;
            }
            for ((owner_kick, dispatch), intervals) in &by_dispatch {
                if owner_kick != kick {
                    continue;
                }
                let resident = intervals.iter().map(|(a, b)| b - a).sum::<u64>();
                let dispatch_share = share * resident as f64 / kick_residency as f64;
                let accum = out.dispatches.entry(*dispatch).or_default();
                accum.add_counts(stream, sample, dispatch_share, shared_dispatches);
                let busy = merge_intervals(intervals.clone())
                    .iter()
                    .map(|(a, b)| b - a)
                    .sum::<u64>() as f64;
                accum.add_time(cycles * busy / duration, busy * timebase_ns * 1e-9);
            }
        }
    }
}

fn merge_intervals(mut intervals: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    intervals.retain(|(start, end)| end > start);
    intervals.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(intervals.len());
    for (start, end) in intervals {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// The agxps `(generation, variant)` of the GPU a profile was recorded on:
/// the variant whose core and mGPU counts match the profile's
/// `Configuration Variables`.
#[cfg(target_os = "macos")]
pub fn agxps_gpu_for_profile(profiler_dir: &Path) -> Option<(u32, u32)> {
    let stream_data = fs::read(profiler_dir.join("streamData")).ok()?;
    let root = keyed_archive::decode(&stream_data)?;
    let config = root
        .get("APSCounterData")?
        .as_array()?
        .iter()
        .filter_map(ArchiveValue::nested)
        .find_map(|entry| entry.get("Configuration Variables").cloned())?;
    let generation = config.get("gpu_gen")?.as_u64()? as u32;
    let num_cores = config.get("num_cores")?.as_u64()?;
    let num_mgpus = config.get("num_mgpus").and_then(ArchiveValue::as_u64).unwrap_or(1);
    let api = agxps_sys::counters::counter_api().ok()?;
    api.variants(generation)
        .into_iter()
        .find(|shape| shape.num_cores == num_cores && shape.num_mgpus == num_mgpus)
        .map(|shape| (generation, shape.variant))
}

/// The APS_USC counter whose presence in the GRC list switches the parser to
/// the micro-architectural counter layout
/// (`agxps_aps_get_uarch_behaviour_from_GRC_counter_list`).
const UARCH_TRIGGER_COUNTER: &str =
    "_b08194796a2cb35a8699c8d23b129c582951d9d1941fbc8e36dbaafa02d474e7";

fn overlap(a_start: u64, a_end: u64, b_start: u64, b_end: u64) -> u64 {
    a_end.min(b_end).saturating_sub(a_start.max(b_start))
}

fn sort_stream(stream: &mut SampleStream) {
    let mut order = (0..stream.ends.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| stream.ends[*index]);
    stream.ends = order.iter().map(|index| stream.ends[*index]).collect();
    stream.cycles = order.iter().map(|index| stream.cycles[*index]).collect();
    for column in &mut stream.values {
        *column = order.iter().map(|index| column[*index]).collect();
    }
}

fn median_period(ends: &[u64]) -> f64 {
    let mut deltas = ends
        .windows(2)
        .map(|pair| pair[1].saturating_sub(pair[0]))
        .filter(|delta| *delta > 0)
        .collect::<Vec<_>>();
    if deltas.is_empty() {
        return 0.0;
    }
    deltas.sort_unstable();
    deltas[deltas.len() / 2] as f64
}

/// Split a `GPRWCNTR` blob into u64 records of `words` words each. Every
/// record starts with the `GPRWCNTR` magic.
fn gprw_records(blob: &[u8], words: usize) -> Vec<Vec<u64>> {
    const MAGIC: &[u8; 8] = b"GPRWCNTR";
    let size = words * 8;
    let mut records = Vec::new();
    let mut offset = 0;
    while offset + size <= blob.len() {
        if &blob[offset..offset + 8] != MAGIC {
            match memchr::memmem::find(&blob[offset + 1..], MAGIC) {
                Some(next) => {
                    offset += 1 + next;
                    continue;
                }
                None => break,
            }
        }
        records.push(
            blob[offset..offset + size]
                .chunks_exact(8)
                .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        );
        offset += size;
    }
    records
}

pub fn format_report(report: &HwCounterReport) -> String {
    let mut out = String::new();
    let gpu = &report.gpu;
    out.push_str(&format!(
        "GPU: {} (agxps generation {} variant {}), {} cores, {} mGPUs, peak DRAM {:.1} GB/s\n",
        gpu.gpu_type,
        gpu.generation,
        gpu.variant,
        gpu.num_cores,
        gpu.num_mgpus,
        gpu.peak_dram_gbps
    ));
    out.push_str(&format!(
        "Source: limiter pass; {} per-core streams sampled every {} core cycles, GPU-global streams every {:.1} us\n",
        report.usc_streams,
        report.usc_count_period_cycles,
        report.global_sample_period_ns / 1000.0
    ));
    for warning in &report.warnings {
        out.push_str(&format!("warning: {warning}\n"));
    }
    out.push_str("\nPer encoder (exact: samples attributed by kick):\n");
    out.push_str(&format_rows(&report.encoders));
    out.push_str("\nPer kernel (per-dispatch attribution summed by kernel):\n");
    out.push_str(&format_rows(&report.kernels));
    out.push_str("\nPer dispatch:\n");
    out.push_str(&format_rows(&report.dispatches));
    out
}

/// Fixed-width table of rows: label, dispatch count, time, contamination,
/// shared-sample fraction, then every [`METRICS`] column.
pub fn format_rows(rows: &[HwCounterRow]) -> String {
    let label_width = rows
        .iter()
        .map(|row| row_label(row).len())
        .max()
        .unwrap_or(5)
        .clamp(5, 48);
    let mut out = format!(
        "{:<label_width$} {:>5} {:>9} {:>6} {:>6}",
        "row", "disp", "gpu_us", "frgn%", "shrd%"
    );
    for spec in METRICS {
        out.push_str(&format!(" {:>13}", spec.label));
    }
    out.push('\n');
    for row in rows {
        let mut label = row_label(row);
        label.truncate(label_width);
        out.push_str(&format!(
            "{:<label_width$} {:>5} {:>9.1} {:>6.1} {:>6.1}",
            label,
            row.dispatches,
            row.gpu_time_ns / 1000.0,
            row.foreign_overlap * 100.0,
            row.shared * 100.0
        ));
        for spec in METRICS {
            match row.metric(spec) {
                Some(value) => out.push_str(&format!(" {:>13}", format_metric(spec.unit, value))),
                None => out.push_str(&format!(" {:>13}", "-")),
            }
        }
        out.push('\n');
    }
    out
}

fn row_label(row: &HwCounterRow) -> String {
    match row.dispatch_index {
        Some(dispatch) => format!("#{dispatch} {}", row.label),
        None => row.label.clone(),
    }
}

pub fn format_metric(unit: MetricUnit, value: f64) -> String {
    match unit {
        MetricUnit::Percent => format!("{value:.1}"),
        MetricUnit::GigabytesPerSecond => format!("{value:.2}"),
        MetricUnit::Bytes | MetricUnit::Count => {
            if value.abs() >= 1e6 {
                format!("{value:.4e}")
            } else {
                format!("{value:.0}")
            }
        }
    }
}
