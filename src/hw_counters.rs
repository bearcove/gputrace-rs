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
    pub encoders: Vec<HwEncoderCounters>,
    /// Kicks in the limiter pass that belong to no encoder of this capture
    /// (other processes sharing the GPU).
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

#[derive(Debug, Clone)]
pub struct HwEncoderCounters {
    /// Encoder index in capture order (`TraceId to BatchId`).
    pub encoder_index: usize,
    pub trace_id: u32,
    pub kicks: usize,
    /// Sum of this encoder's kick durations in the limiter pass.
    pub gpu_time_ns: f64,
    /// Fraction of `gpu_time_ns` during which a kick from another process
    /// was also running. Counters are GPU-wide or per-core, not per-process,
    /// so anything above a few percent makes the row unreliable.
    pub foreign_overlap: f64,
    /// agxps derived counter name -> value in agxps units.
    pub values: BTreeMap<String, f64>,
}

impl HwEncoderCounters {
    pub fn metric(&self, spec: &MetricSpec) -> Option<f64> {
        self.values
            .get(spec.name)
            .copied()
            .filter(|value| value.is_finite())
            .map(|value| spec.unit.display_value(value))
    }
}

/// A kick of the limiter pass, in one clock domain.
#[derive(Debug, Clone, Copy)]
struct Kick {
    start: u64,
    end: u64,
    encoder: Option<usize>,
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

/// Per-encoder accumulation for one counter family.
#[derive(Default, Clone)]
struct EncoderAccum {
    raw: BTreeMap<String, f64>,
    cycles: f64,
    seconds: f64,
}

pub fn report(trace: &TraceBundle) -> Result<HwCounterReport> {
    let profiler_dir = profiler::find_profiler_directory(&trace.path).ok_or_else(|| {
        Error::InvalidInput(format!(
            "no .gpuprofiler_raw directory for {}",
            trace.path.display()
        ))
    })?;
    report_for_profiler_dir(&profiler_dir)
}

#[cfg(not(target_os = "macos"))]
pub fn report_for_profiler_dir(_profiler_dir: &Path) -> Result<HwCounterReport> {
    Err(Error::Unsupported("hardware counters require macOS and Xcode"))
}

#[cfg(target_os = "macos")]
pub fn report_for_profiler_dir(profiler_dir: &Path) -> Result<HwCounterReport> {
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
    let trace_of_encoder = encoder_of_trace
        .iter()
        .map(|(trace_id, encoder)| (*encoder, *trace_id as u32))
        .collect::<BTreeMap<_, _>>();

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
    let mut firmware_kicks = Vec::<(u64, u64, u64)>::new();
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
            // Firmware records: [.., encoder id, kick id, .., start/end].
            for record in records {
                if record.len() >= 10 && record[3] == 5 {
                    let (a, b) = (record[8], record[9]);
                    firmware_kicks.push((record[4], a.min(b), a.max(b)));
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
    let global_kicks = firmware_kicks
        .iter()
        .map(|(encoder_trace, start, end)| Kick {
            start: *start,
            end: *end,
            encoder: encoder_of_trace.get(encoder_trace).copied(),
        })
        .collect::<Vec<_>>();
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

    // --- Attribute samples to encoders. ---
    let encoder_count = encoder_of_trace.values().max().map_or(0, |max| max + 1);
    let mut usc_accum = vec![EncoderAccum::default(); encoder_count];
    let mut foreign_kicks = BTreeSet::new();
    for profile in &usc_profiles {
        let kicks = profile
            .kicks
            .iter()
            .filter(|kick| !kick.missing_end && kick.end_ticks > kick.start_ticks)
            .map(|kick| {
                let encoder = encoder_of_trace.get(&(kick.software_id >> 32)).copied();
                if encoder.is_none() {
                    foreign_kicks.insert(kick.software_id);
                }
                Kick {
                    start: kick.start_ticks,
                    end: kick.end_ticks,
                    encoder,
                }
            })
            .collect::<Vec<_>>();
        let stream = SampleStream {
            ends: profile.sample_end_ticks.clone(),
            cycles: profile.sample_cycles.clone(),
            values: profile.values.clone(),
            names: profile.counter_names.clone(),
        };
        attribute(&stream, &kicks, timebase_ns, &mut usc_accum);
    }
    let usc_streams = usc_profiles.len().max(1) as f64;
    for accum in &mut usc_accum {
        // Raw counts add up across cores; cycles and time are per core.
        accum.cycles /= usc_streams;
        accum.seconds /= usc_streams;
    }
    // One family per GPU-global source: its rings are instances of the same
    // block (raw counts add up) and its own cycle column is the clock its
    // formulas expect.
    let mut global_accums = BTreeMap::<String, Vec<EncoderAccum>>::new();
    for ((source, _ring), stream) in &global_streams {
        let mut accum = vec![EncoderAccum::default(); encoder_count];
        attribute(stream, &global_kicks, timebase_ns, &mut accum);
        let family = global_accums
            .entry(source.clone())
            .or_insert_with(|| vec![EncoderAccum::default(); encoder_count]);
        for (total, part) in family.iter_mut().zip(accum) {
            for (name, value) in part.raw {
                *total.raw.entry(name).or_default() += value;
            }
            if total.seconds == 0.0 {
                total.cycles = part.cycles;
                total.seconds = part.seconds;
            }
        }
    }

    // --- Derived counters, one call per counter family. ---
    let derived = gpu.derived_counters();
    let mut values = vec![BTreeMap::<String, f64>::new(); encoder_count];
    for accum in std::iter::once(&usc_accum).chain(global_accums.values()) {
        let raw_names = accum
            .iter()
            .flat_map(|encoder| encoder.raw.keys().cloned())
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
            continue;
        }
        let raw = raw_idents
            .iter()
            .map(|(ident, name)| {
                (
                    *ident,
                    accum
                        .iter()
                        .map(|encoder| encoder.raw.get(name).copied().unwrap_or(0.0).round() as u64)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let cycles = accum
            .iter()
            .map(|encoder| encoder.cycles.round() as u64)
            .collect::<Vec<_>>();
        let seconds = accum.iter().map(|encoder| encoder.seconds).collect::<Vec<_>>();
        let results = gpu
            .compute(&raw, &cycles, &seconds, &constants, &computable)
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
        for (ident, series) in computable.iter().zip(results) {
            let Some(series) = series else { continue };
            let name = gpu.info(*ident).name;
            for (encoder, value) in series.into_iter().enumerate() {
                if accum[encoder].seconds > 0.0 {
                    values[encoder].entry(name.clone()).or_insert(value);
                }
            }
        }
    }

    // --- Per-encoder timing and contamination from the global kick list. ---
    let foreign_windows = global_kicks
        .iter()
        .filter(|kick| kick.encoder.is_none())
        .map(|kick| (kick.start, kick.end))
        .collect::<Vec<_>>();
    let mut encoders = Vec::new();
    for (encoder_index, trace_id) in &trace_of_encoder {
        let kicks = global_kicks
            .iter()
            .filter(|kick| kick.encoder == Some(*encoder_index))
            .collect::<Vec<_>>();
        if kicks.is_empty() && usc_accum[*encoder_index].seconds == 0.0 {
            continue;
        }
        let total_ticks = kicks
            .iter()
            .map(|kick| kick.end - kick.start)
            .sum::<u64>() as f64;
        let overlap_ticks = kicks
            .iter()
            .map(|kick| {
                foreign_windows
                    .iter()
                    .map(|(start, end)| overlap(kick.start, kick.end, *start, *end))
                    .sum::<u64>()
                    .min(kick.end - kick.start)
            })
            .sum::<u64>() as f64;
        encoders.push(HwEncoderCounters {
            encoder_index: *encoder_index,
            trace_id: *trace_id,
            kicks: kicks.len(),
            gpu_time_ns: total_ticks * timebase_ns,
            foreign_overlap: if total_ticks > 0.0 {
                overlap_ticks / total_ticks
            } else {
                0.0
            },
            values: std::mem::take(&mut values[*encoder_index]),
        });
    }
    let foreign_kick_count = foreign_kicks.len().max(
        global_kicks
            .iter()
            .filter(|kick| kick.encoder.is_none())
            .count(),
    );
    if foreign_kick_count > 0 {
        warnings.push(format!(
            "{foreign_kick_count} kick(s) from other processes ran on the GPU during the counter pass; rows with foreign overlap mix their counts in"
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
        foreign_kicks: foreign_kick_count,
        warnings,
    })
}

/// The APS_USC counter whose presence in the GRC list switches the parser to
/// the micro-architectural counter layout
/// (`agxps_aps_get_uarch_behaviour_from_GRC_counter_list`).
const UARCH_TRIGGER_COUNTER: &str =
    "_b08194796a2cb35a8699c8d23b129c582951d9d1941fbc8e36dbaafa02d474e7";

/// Attribute each sample of `stream` to the encoders whose kicks overlap it.
///
/// A sample overlapped by a single kick goes to that kick's encoder whole:
/// the rest of the sample was idle, so it holds no other events. A sample
/// shared by several kicks is split in proportion to overlap time. Kicks
/// of other processes (`encoder: None`) take their share and drop it.
fn attribute(stream: &SampleStream, kicks: &[Kick], timebase_ns: f64, out: &mut [EncoderAccum]) {
    for index in 1..stream.ends.len() {
        let (start, end) = (stream.ends[index - 1], stream.ends[index]);
        if end <= start {
            continue;
        }
        let overlaps = kicks
            .iter()
            .filter_map(|kick| {
                let ticks = overlap(start, end, kick.start, kick.end);
                (ticks > 0).then_some((kick.encoder, ticks))
            })
            .collect::<Vec<_>>();
        let busy = overlaps.iter().map(|(_, ticks)| *ticks).sum::<u64>();
        if busy == 0 {
            continue;
        }
        let duration = (end - start) as f64;
        for (encoder, ticks) in overlaps {
            let Some(encoder) = encoder else { continue };
            let Some(accum) = out.get_mut(encoder) else {
                continue;
            };
            let share = ticks as f64 / busy as f64;
            let active = ticks as f64 / duration;
            for (name, column) in stream.names.iter().zip(&stream.values) {
                if let Some(value) = column.get(index) {
                    *accum.raw.entry(name.clone()).or_default() += *value as f64 * share;
                }
            }
            accum.cycles += stream.cycles.get(index).copied().unwrap_or(0) as f64 * active;
            accum.seconds += ticks as f64 * timebase_ns * 1e-9;
        }
    }
}

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
    out.push('\n');
    out.push_str(&format!("{:>4} {:>10} {:>9} {:>6}", "enc", "trace_id", "gpu_us", "frgn%"));
    for spec in METRICS {
        out.push_str(&format!(" {:>13}", spec.label));
    }
    out.push('\n');
    for encoder in &report.encoders {
        out.push_str(&format!(
            "{:>4} {:>#10x} {:>9.1} {:>6.1}",
            encoder.encoder_index,
            encoder.trace_id,
            encoder.gpu_time_ns / 1000.0,
            encoder.foreign_overlap * 100.0
        ));
        for spec in METRICS {
            match encoder.metric(spec) {
                Some(value) => out.push_str(&format!(" {:>13}", format_metric(spec.unit, value))),
                None => out.push_str(&format!(" {:>13}", "-")),
            }
        }
        out.push('\n');
    }
    out
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
