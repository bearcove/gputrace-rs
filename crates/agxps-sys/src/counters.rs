//! Hardware-counter half of the `agxps` API (AGXProfilingSupport inside
//! `GTShaderProfiler.framework`).
//!
//! This is the code Xcode uses to turn raw AGX performance-counter samples
//! into named derived counters ("ALU Utilization", "AF Read Bandwidth", ...).
//! Derived-counter definitions are compiled into the framework per GPU
//! (generation, variant); there is no plist or JavaScript file for G15+
//! GPUs, so everything has to go through these entry points.
//!
//! Signatures were recovered from the framework's own callers
//! (`-[XRGPUAPSDataProcessor deriveRDECounters:...]`,
//! `-[XRGPUAPSDataProcessor loadAPSCounters:counterSet:]`,
//! `-[XRGPUAPSDataProcessor setConfig:]`); see `docs/COUNTERS_M4.md`.

use std::ffi::{CStr, CString, c_char, c_long, c_void};
use std::sync::{Mutex, OnceLock};

use crate::{AgxpsGpu, Error, Result};

/// Index into the framework's process-global counter table.
pub type CounterIdent = u64;

pub const INVALID_COUNTER: CounterIdent = u64::MAX;

/// `agxps_timeseries_scalar_t`: a tagged scalar, 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TimeseriesScalar {
    pub datatype: u32,
    pub _pad: u32,
    pub value: f64,
}

/// `agxps_timeseries_datatype_t`.
pub const TIMESERIES_F64: u32 = 0;
pub const TIMESERIES_U64: u32 = 1;

/// Name of the pseudo raw counter carrying the GPU cycles of each sample.
pub const GPU_CYCLES: &str = "GPUCycles";
/// Name of the pseudo raw counter carrying the duration of each sample in
/// seconds (an f64 series).
pub const DELTA_SECONDS: &str = "DeltaSeconds";

type FnGpuCreate = unsafe extern "C" fn(u32, u32, u32, bool) -> AgxpsGpu;
type FnGpuU64 = unsafe extern "C" fn(AgxpsGpu) -> u64;
type FnGpuF64 = unsafe extern "C" fn(AgxpsGpu) -> f64;
type FnDescriptorCreate =
    unsafe extern "C" fn(u32, u32, u64, u64, u64, u64, u64, u64) -> *mut c_void;
type FnInitialize = unsafe extern "C" fn(*mut *mut c_void, u64, *mut c_void, *mut c_void) -> bool;
type FnGetIdent = unsafe extern "C" fn(AgxpsGpu, *const c_char) -> CounterIdent;
type FnIdentStr = unsafe extern "C" fn(CounterIdent) -> *const c_char;
type FnIdentBool = unsafe extern "C" fn(CounterIdent) -> bool;
type FnIdentU64 = unsafe extern "C" fn(CounterIdent) -> u64;
type FnIdentGroup = unsafe extern "C" fn(CounterIdent, u64) -> *const c_char;
type FnRawUsed = unsafe extern "C" fn(
    AgxpsGpu,
    *const CounterIdent,
    u64,
    *mut *mut CounterIdent,
    *mut u64,
) -> bool;
type FnTsCreateNoCopy =
    unsafe extern "C" fn(u32, *const c_void, u64, unsafe extern "C" fn(*mut c_void)) -> *mut c_void;
type FnTsDestroy = unsafe extern "C" fn(*mut c_void);
type FnTsU64 = unsafe extern "C" fn(*mut c_void) -> u64;
type FnTsData = unsafe extern "C" fn(*mut c_void) -> *const c_void;
type FnTsDatatype = unsafe extern "C" fn(*mut c_void) -> u32;
type FnCompute = unsafe extern "C" fn(
    AgxpsGpu,
    *const *mut c_void,
    *const CounterIdent,
    u64,
    *const TimeseriesScalar,
    *const *const c_char,
    u64,
    *const CounterIdent,
    u64,
    *mut *mut *mut c_void,
    *mut u32,
) -> bool;
type FnParserCreate = unsafe extern "C" fn(*const ApsParserDescriptor) -> *mut c_void;
type FnParserDestroy = unsafe extern "C" fn(*mut c_void);
type FnParserParse =
    unsafe extern "C" fn(*mut c_void, *const u8, c_long, u32, *mut c_void) -> *mut c_void;
type FnPdDestroy = unsafe extern "C" fn(*mut c_void);
type FnPdCount = unsafe extern "C" fn(*mut c_void) -> u32;
type FnPdRange = unsafe extern "C" fn(*mut c_void, *mut u64, u64, u64) -> i32;
type FnPdNames = unsafe extern "C" fn(*mut c_void, *mut *const c_char, u64, u64) -> i32;
type FnPdPtrRange = unsafe extern "C" fn(*mut c_void, *mut *const u64, u64, u64) -> i32;
type FnPdNumRange = unsafe extern "C" fn(*mut c_void, *mut u64, u64, u64) -> i32;
type FnPdSystemTimestamp = unsafe extern "C" fn(*mut c_void, u64) -> u64;
type FnParseErrorString = unsafe extern "C" fn(u64) -> *const c_char;
type FnPdCount64 = unsafe extern "C" fn(*mut c_void) -> u64;
type FnPdRange32 = unsafe extern "C" fn(*mut c_void, *mut u32, u64, u64) -> i32;
type FnPdRange8 = unsafe extern "C" fn(*mut c_void, *mut u8, u64, u64) -> i32;
type FnAnalyzerCreate = unsafe extern "C" fn(u32) -> *mut c_void;
type FnAnalyzerVoid = unsafe extern "C" fn(*mut c_void);
type FnAnalyzerProcess = unsafe extern "C" fn(*mut c_void, *mut c_void);
type FnAnalyzerCount = unsafe extern "C" fn(*mut c_void, u32) -> u64;
type FnAnalyzerRange = unsafe extern "C" fn(*mut c_void, u32, *mut u64, u64, u64) -> i32;

/// `agxps_aps_descriptor_t` as `-[XRGPUAPSDataProcessor setConfig:]` fills it
/// from the profile's `APS Options`. Getting `count_period` wrong (the old
/// default of 0) makes the parser drop every counter token.
#[repr(C)]
pub struct ApsParserDescriptor {
    pub gpu: AgxpsGpu,
    /// `KickAndStateTracing.PulsePeriod`.
    pub pulse_period: u32,
    /// `SystemTimePeriod`.
    pub system_time_period: u32,
    /// `KickAndStateTracing.CountPeriod`: GPU cycles per counter sample.
    pub count_period: u32,
    pub _pad_0x14: u32,
    /// `ChunkSize` (0x1000 unless the profile says otherwise).
    pub chunk_size: u64,
    /// 1 when the GRC counter list enables the micro-architectural counter
    /// set (`agxps_aps_get_uarch_behaviour_from_GRC_counter_list`).
    pub uarch_behaviour: u32,
    pub _pad_0x24: [u8; 0x0c],
    pub field_0x30: u64,
    pub _pad_0x38: [u8; 0x20],
    pub field_0x58: u64,
    pub _pad_0x60: [u8; 0x08],
}

const _: () = assert!(std::mem::size_of::<ApsParserDescriptor>() == 0x68);

/// Settings for parsing an APS USC counter stream (`Counters_f_*.raw`).
#[derive(Debug, Clone, Copy)]
pub struct ApsParseSettings {
    pub pulse_period: u32,
    pub system_time_period: u32,
    pub count_period: u32,
    pub chunk_size: u64,
    pub uarch_behaviour: bool,
}

/// One USC's APS stream: counter samples and the kicks that ran on it.
#[derive(Debug, Clone, Default)]
pub struct ApsCounterProfile {
    pub counter_names: Vec<String>,
    /// `values[counter][sample]`: events counted during the sample.
    pub values: Vec<Vec<u64>>,
    /// End of each sample, in system (mach continuous) ticks.
    pub sample_end_ticks: Vec<u64>,
    /// USC clock cycles covered by each sample (0 for the first sample).
    pub sample_cycles: Vec<u64>,
    /// Every kick in the stream, in the parser's order (work cliques refer
    /// to kicks by index into this list).
    pub kicks: Vec<ApsKick>,
    /// Commands (dispatches) the timing analyzer found on this USC, in its
    /// order; a work clique's `esl_index` indexes this list.
    pub commands: Vec<ApsCommand>,
    pub work_cliques: Vec<ApsWorkClique>,
}

#[derive(Debug, Clone, Copy)]
pub struct ApsCommand {
    pub software_id: u64,
    /// Address of the dispatch's shader-launch program; matches the
    /// `compute-sl` entries of `Program Address Mappings`.
    pub esl_shader_address: u64,
    pub start_ticks: u64,
}

/// One clique (a batch of threadgroups) of a dispatch running on a USC.
#[derive(Debug, Clone, Copy)]
pub struct ApsWorkClique {
    pub start_ticks: u64,
    /// When the profile traces no clique ends (`missing_end`), this is the
    /// next event on the same clique slot, or the end of the trace.
    pub end_ticks: u64,
    pub kick_index: u32,
    pub esl_index: u64,
    /// Hardware clique slot; a missing end equal to the next start on the
    /// same slot is a real hand-over, otherwise the slot went idle.
    pub slot: u8,
    pub missing_end: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct ApsKick {
    pub start_ticks: u64,
    pub end_ticks: u64,
    /// `(encoder trace id << 32) | kick trace id`.
    pub software_id: u64,
    pub missing_end: bool,
}

/// The shape of one supported GPU configuration, used to pick the variant
/// that matches a profile's `Configuration Variables`.
#[derive(Debug, Clone, Copy)]
pub struct GpuShape {
    pub generation: u32,
    pub variant: u32,
    pub num_cores: u64,
    pub num_mgpus: u64,
    pub l2_cache_bytes: u64,
    pub peak_dram_gbps: f64,
}

#[derive(Debug, Clone)]
pub struct CounterInfo {
    pub ident: CounterIdent,
    pub name: String,
    pub doc: String,
    pub derived: bool,
    pub normalized: bool,
    pub relative: bool,
    pub groups: Vec<String>,
}

pub struct CounterApi {
    gpu_create: FnGpuCreate,
    gpu_num_uscs: FnGpuU64,
    gpu_num_mgpus: FnGpuU64,
    gpu_l2_size: FnGpuU64,
    gpu_peak_dram: FnGpuF64,
    descriptor_create: FnDescriptorCreate,
    initialize: FnInitialize,
    get_ident: FnGetIdent,
    get_name: FnIdentStr,
    get_doc: FnIdentStr,
    is_valid: FnIdentBool,
    is_derived: FnIdentBool,
    is_normalized: FnIdentBool,
    is_relative: FnIdentBool,
    num_groups: FnIdentU64,
    group: FnIdentGroup,
    raw_used: FnRawUsed,
    ts_create: FnTsCreateNoCopy,
    ts_destroy: FnTsDestroy,
    ts_len: FnTsU64,
    ts_data: FnTsData,
    ts_datatype: FnTsDatatype,
    compute: FnCompute,
    parser_create: FnParserCreate,
    parser_destroy: FnParserDestroy,
    parser_parse: FnParserParse,
    pd_destroy: FnPdDestroy,
    pd_kicks_num: FnPdCount,
    pd_kick_start: FnPdRange,
    pd_kick_end: FnPdRange,
    pd_kick_swid: FnPdRange,
    pd_kick_missing_end: unsafe extern "C" fn(*mut c_void, *mut u8, u64, u64) -> i32,
    pd_counter_num: FnPdCount,
    pd_counter_names: FnPdNames,
    pd_counter_values: FnPdPtrRange,
    pd_counter_values_num: FnPdNumRange,
    pd_counter_group_metadata: FnPdPtrRange,
    pd_system_ts_num: FnPdCount,
    pd_system_ts: FnPdRange,
    pd_usc_ts_num: FnPdCount,
    pd_usc_ts: FnPdRange,
    pd_sync_ts_num: FnPdCount,
    pd_sync_ts: FnPdRange,
    pd_system_timestamp: FnPdSystemTimestamp,
    parse_error_string: FnParseErrorString,
    pd_cliques_num: FnPdCount64,
    pd_clique_start: FnPdRange,
    pd_clique_end: FnPdRange,
    pd_clique_esl_id: FnPdRange,
    pd_clique_kick_id: FnPdRange32,
    pd_clique_missing_end: FnPdRange8,
    pd_clique_slot: FnPdRange8,
    analyzer_create: FnAnalyzerCreate,
    analyzer_destroy: FnAnalyzerVoid,
    analyzer_process_usc: FnAnalyzerProcess,
    analyzer_finish: FnAnalyzerVoid,
    analyzer_num_commands: FnAnalyzerCount,
    analyzer_esl_address: FnAnalyzerRange,
    analyzer_esl_start: FnAnalyzerRange,
    analyzer_kick_software_id: FnAnalyzerRange,
    /// `(generation, variant)` the process-global counter table was built for.
    initialized: Mutex<Option<(u32, u32)>>,
}

unsafe impl Send for CounterApi {}
unsafe impl Sync for CounterApi {}

static COUNTER_API: OnceLock<std::result::Result<CounterApi, String>> = OnceLock::new();

/// Load (once per process) the counter API from `GTShaderProfiler`.
pub fn counter_api() -> Result<&'static CounterApi> {
    COUNTER_API
        .get_or_init(|| CounterApi::load().map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|message| Error::Dlopen(message.clone()))
}

unsafe extern "C" fn keep_buffer(_data: *mut c_void) {}

impl CounterApi {
    fn load() -> Result<Self> {
        let handle = crate::open_framework()?;
        let sym = |name: &'static str| crate::load_sym_ptr(handle, name);
        macro_rules! s {
            ($name:literal) => {
                unsafe { std::mem::transmute_copy(&sym($name)?) }
            };
        }
        Ok(Self {
            gpu_create: s!("agxps_gpu_create"),
            gpu_num_uscs: s!("agxps_gpu_get_num_physical_uscs"),
            gpu_num_mgpus: s!("agxps_gpu_get_num_physical_mgpus"),
            gpu_l2_size: s!("agxps_gpu_get_l2_cache_size"),
            gpu_peak_dram: s!("agxps_gpu_get_peak_dram_bandwidth"),
            descriptor_create: s!("agxps_derived_counter_gpu_descriptor_create"),
            initialize: s!("agxps_initialize"),
            get_ident: s!("agxps_counter_get_ident"),
            get_name: s!("agxps_counter_get_name"),
            get_doc: s!("agxps_counter_get_doc_string"),
            is_valid: s!("agxps_counter_is_valid"),
            is_derived: s!("agxps_counter_is_derived"),
            is_normalized: s!("agxps_counter_is_normalized"),
            is_relative: s!("agxps_counter_is_relative"),
            num_groups: s!("agxps_counter_get_num_groups"),
            group: s!("agxps_counter_get_group"),
            raw_used: s!("agxps_counter_get_raw_counters_used_by_derived_counters"),
            ts_create: s!("agxps_timeseries_create_with_bytes_no_copy"),
            ts_destroy: s!("agxps_timeseries_destroy"),
            ts_len: s!("agxps_timeseries_get_length"),
            ts_data: s!("agxps_timeseries_get_data"),
            ts_datatype: s!("agxps_timeseries_get_datatype"),
            compute: s!("agxps_counter_compute_derived_counters"),
            parser_create: s!("agxps_aps_parser_create"),
            parser_destroy: s!("agxps_aps_parser_destroy"),
            parser_parse: s!("agxps_aps_parser_parse"),
            pd_destroy: s!("agxps_aps_profile_data_destroy"),
            pd_kicks_num: s!("agxps_aps_profile_data_get_kicks_num"),
            pd_kick_start: s!("agxps_aps_profile_data_get_kick_start"),
            pd_kick_end: s!("agxps_aps_profile_data_get_kick_end"),
            pd_kick_swid: s!("agxps_aps_profile_data_get_kick_software_id"),
            pd_kick_missing_end: s!("agxps_aps_profile_data_get_kick_missing_end"),
            pd_counter_num: s!("agxps_aps_profile_data_get_counter_num"),
            pd_counter_names: s!("agxps_aps_profile_data_get_counter_names"),
            pd_counter_values: s!("agxps_aps_profile_data_get_counter_values"),
            pd_counter_values_num: s!("agxps_aps_profile_data_get_counter_values_num"),
            pd_counter_group_metadata: s!("agxps_aps_profile_data_get_counter_group_metadata"),
            pd_system_ts_num: s!("agxps_aps_profile_data_get_system_timestamps_num"),
            pd_system_ts: s!("agxps_aps_profile_data_get_system_timestamps"),
            pd_usc_ts_num: s!("agxps_aps_profile_data_get_usc_timestamps_num"),
            pd_usc_ts: s!("agxps_aps_profile_data_get_usc_timestamps"),
            pd_sync_ts_num: s!("agxps_aps_profile_data_get_synchronized_timestamps_num"),
            pd_sync_ts: s!("agxps_aps_profile_data_get_synchronized_timestamps"),
            pd_system_timestamp: s!("agxps_aps_profile_data_get_system_timestamp"),
            parse_error_string: s!("agxps_aps_parse_error_type_to_string"),
            pd_cliques_num: s!("agxps_aps_profile_data_get_work_cliques_num"),
            pd_clique_start: s!("agxps_aps_profile_data_get_work_clique_start"),
            pd_clique_end: s!("agxps_aps_profile_data_get_work_clique_end"),
            pd_clique_esl_id: s!("agxps_aps_profile_data_get_work_clique_esl_id"),
            pd_clique_kick_id: s!("agxps_aps_profile_data_get_work_clique_kick_id"),
            pd_clique_missing_end: s!("agxps_aps_profile_data_get_work_clique_missing_end"),
            pd_clique_slot: s!("agxps_aps_profile_data_get_work_clique_clique_id"),
            analyzer_create: s!("agxps_aps_timing_analyzer_create"),
            analyzer_destroy: s!("agxps_aps_timing_analyzer_destroy"),
            analyzer_process_usc: s!("agxps_aps_timing_analyzer_process_usc"),
            analyzer_finish: s!("agxps_aps_timing_analyzer_finish"),
            analyzer_num_commands: s!("agxps_aps_timing_analyzer_get_num_commands"),
            analyzer_esl_address: s!("agxps_aps_timing_analyzer_get_esl_shader_address"),
            analyzer_esl_start: s!("agxps_aps_timing_analyzer_get_esl_start"),
            analyzer_kick_software_id: s!("agxps_aps_timing_analyzer_get_kick_software_id"),
            initialized: Mutex::new(None),
        })
    }

    /// Every supported variant of `generation`, with the hardware shape
    /// agxps knows for it.
    pub fn variants(&self, generation: u32) -> Vec<GpuShape> {
        (0..16)
            .filter_map(|variant| {
                let gpu = unsafe { (self.gpu_create)(generation, variant, 1, false) };
                (!gpu.is_null()).then(|| unsafe {
                    GpuShape {
                        generation,
                        variant,
                        num_cores: (self.gpu_num_uscs)(gpu),
                        num_mgpus: (self.gpu_num_mgpus)(gpu),
                        l2_cache_bytes: (self.gpu_l2_size)(gpu),
                        peak_dram_gbps: (self.gpu_peak_dram)(gpu),
                    }
                })
            })
            .collect()
    }

    /// Build the process-global counter table for one GPU and return a
    /// handle to it. The table is rebuilt when a different GPU is asked
    /// for, which invalidates idents handed out earlier.
    pub fn gpu(&self, generation: u32, variant: u32) -> Result<CounterGpu<'_>> {
        let gpu = unsafe { (self.gpu_create)(generation, variant, 1, false) };
        if gpu.is_null() {
            return Err(Error::GpuCreate {
                generation,
                variant,
                rev: 1,
            });
        }
        let mut initialized = self.initialized.lock().unwrap();
        if *initialized != Some((generation, variant)) {
            let mut descriptor =
                unsafe { (self.descriptor_create)(generation, variant, 0, 0, 0, 0, 0, 0) };
            if descriptor.is_null()
                || !unsafe {
                    (self.initialize)(
                        &mut descriptor,
                        1,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                    )
                }
            {
                return Err(Error::CounterInitialize {
                    generation,
                    variant,
                });
            }
            *initialized = Some((generation, variant));
        }
        Ok(CounterGpu {
            api: self,
            gpu,
            generation,
            variant,
        })
    }
}

/// A GPU whose counter table is loaded.
pub struct CounterGpu<'a> {
    api: &'a CounterApi,
    gpu: AgxpsGpu,
    pub generation: u32,
    pub variant: u32,
}

fn c_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }
}

impl CounterGpu<'_> {
    pub fn ident(&self, name: &str) -> Option<CounterIdent> {
        let name = CString::new(name).ok()?;
        let ident = unsafe { (self.api.get_ident)(self.gpu, name.as_ptr()) };
        (ident != INVALID_COUNTER).then_some(ident)
    }

    pub fn info(&self, ident: CounterIdent) -> CounterInfo {
        let api = self.api;
        unsafe {
            CounterInfo {
                ident,
                name: c_string((api.get_name)(ident)),
                doc: c_string((api.get_doc)(ident)),
                derived: (api.is_derived)(ident),
                normalized: (api.is_normalized)(ident),
                relative: (api.is_relative)(ident),
                groups: (0..(api.num_groups)(ident))
                    .map(|index| c_string((api.group)(ident, index)))
                    .collect(),
            }
        }
    }

    /// Every derived counter defined for this GPU.
    pub fn derived_counters(&self) -> Vec<CounterIdent> {
        let api = self.api;
        let mut out = Vec::new();
        let mut ident = 0;
        while unsafe { (api.is_valid)(ident) } {
            if unsafe { (api.is_derived)(ident) } {
                let name = unsafe { (api.get_name)(ident) };
                if !name.is_null() && unsafe { (api.get_ident)(self.gpu, name) } == ident {
                    out.push(ident);
                }
            }
            ident += 1;
        }
        out
    }

    /// Raw counters a set of derived counters reads, excluding the
    /// `GPUCycles` / `DeltaSeconds` pseudo counters (always supplied).
    pub fn raw_dependencies(&self, derived: &[CounterIdent]) -> Vec<CounterIdent> {
        if derived.is_empty() {
            return Vec::new();
        }
        let mut out: *mut CounterIdent = std::ptr::null_mut();
        let mut count = 0u64;
        let ok = unsafe {
            (self.api.raw_used)(
                self.gpu,
                derived.as_ptr(),
                derived.len() as u64,
                &mut out,
                &mut count,
            )
        };
        if !ok || out.is_null() {
            return Vec::new();
        }
        let deps = unsafe { std::slice::from_raw_parts(out, count as usize) }.to_vec();
        unsafe { free(out.cast()) };
        deps
    }

    /// Evaluate derived counters over aligned input series.
    ///
    /// `raw` holds one u64 series per raw counter ident; `gpu_cycles` and
    /// `delta_seconds` are the per-element pseudo counters. All series must
    /// have the same length. Returns one f64 series per requested derived
    /// counter (`None` when agxps produced no series for it).
    pub fn compute(
        &self,
        raw: &[(CounterIdent, Vec<u64>)],
        gpu_cycles: &[u64],
        delta_seconds: &[f64],
        constants: &[(&str, f64)],
        derived: &[CounterIdent],
    ) -> Result<Vec<Option<Vec<f64>>>> {
        let api = self.api;
        let len = gpu_cycles.len();
        if delta_seconds.len() != len || raw.iter().any(|(_, series)| series.len() != len) {
            return Err(Error::CounterCompute(
                "input series lengths differ".to_owned(),
            ));
        }
        let cycles_ident = self
            .ident(GPU_CYCLES)
            .ok_or_else(|| Error::CounterCompute("no GPUCycles counter".to_owned()))?;
        let seconds_ident = self
            .ident(DELTA_SECONDS)
            .ok_or_else(|| Error::CounterCompute("no DeltaSeconds counter".to_owned()))?;

        let mut idents = Vec::with_capacity(raw.len() + 2);
        let mut series = Vec::with_capacity(raw.len() + 2);
        unsafe {
            for (ident, values) in raw {
                idents.push(*ident);
                series.push((api.ts_create)(
                    TIMESERIES_U64,
                    values.as_ptr().cast(),
                    len as u64,
                    keep_buffer,
                ));
            }
            idents.push(cycles_ident);
            series.push((api.ts_create)(
                TIMESERIES_U64,
                gpu_cycles.as_ptr().cast(),
                len as u64,
                keep_buffer,
            ));
            idents.push(seconds_ident);
            series.push((api.ts_create)(
                TIMESERIES_F64,
                delta_seconds.as_ptr().cast(),
                len as u64,
                keep_buffer,
            ));
        }
        let names = constants
            .iter()
            .map(|(name, _)| CString::new(*name).unwrap())
            .collect::<Vec<_>>();
        let name_ptrs = names.iter().map(|name| name.as_ptr()).collect::<Vec<_>>();
        let values = constants
            .iter()
            .map(|(_, value)| TimeseriesScalar {
                datatype: TIMESERIES_F64,
                _pad: 0,
                value: *value,
            })
            .collect::<Vec<_>>();

        let mut out: *mut *mut c_void = std::ptr::null_mut();
        let mut error = 0u32;
        let ok = unsafe {
            (api.compute)(
                self.gpu,
                series.as_ptr(),
                idents.as_ptr(),
                idents.len() as u64,
                values.as_ptr(),
                name_ptrs.as_ptr(),
                values.len() as u64,
                derived.as_ptr(),
                derived.len() as u64,
                &mut out,
                &mut error,
            )
        };
        for handle in series {
            unsafe { (api.ts_destroy)(handle) };
        }
        if !ok || out.is_null() {
            return Err(Error::CounterCompute(format!(
                "agxps_counter_compute_derived_counters failed (error {error})"
            )));
        }
        let mut results = Vec::with_capacity(derived.len());
        for index in 0..derived.len() {
            let handle = unsafe { *out.add(index) };
            if handle.is_null() {
                results.push(None);
                continue;
            }
            let series = unsafe {
                let len = (api.ts_len)(handle) as usize;
                let data = (api.ts_data)(handle);
                match (api.ts_datatype)(handle) {
                    TIMESERIES_F64 => {
                        Some(std::slice::from_raw_parts(data.cast::<f64>(), len).to_vec())
                    }
                    TIMESERIES_U64 => Some(
                        std::slice::from_raw_parts(data.cast::<u64>(), len)
                            .iter()
                            .map(|value| *value as f64)
                            .collect(),
                    ),
                    _ => None,
                }
            };
            unsafe { (api.ts_destroy)(handle) };
            results.push(series);
        }
        unsafe { free(out.cast()) };
        Ok(results)
    }

    /// Parse one APS USC stream (`Counters_f_*.raw` of the limiter pass) and
    /// return its counter samples, with each sample's end time interpolated
    /// onto the system clock the same way
    /// `-[XRGPUAPSDataProcessor loadAPSCounters:counterSet:]` does.
    pub fn parse_aps_counters(
        &self,
        settings: ApsParseSettings,
        bytes: &[u8],
    ) -> Result<ApsCounterProfile> {
        let api = self.api;
        let descriptor = ApsParserDescriptor {
            gpu: self.gpu,
            pulse_period: settings.pulse_period,
            system_time_period: settings.system_time_period,
            count_period: settings.count_period,
            _pad_0x14: 0,
            chunk_size: settings.chunk_size,
            uarch_behaviour: settings.uarch_behaviour as u32,
            _pad_0x24: [0; 0x0c],
            field_0x30: u64::MAX,
            _pad_0x38: [0; 0x20],
            field_0x58: 0x32,
            _pad_0x60: [0; 0x08],
        };
        let parser = unsafe { (api.parser_create)(&descriptor) };
        if parser.is_null() {
            return Err(Error::ParserCreate);
        }
        let mut status = [0u8; 4096];
        // Profile type 1 is what `-[XRGPUAPSDataProcessor parseData:...]`
        // passes for counter streams.
        let pd = unsafe {
            (api.parser_parse)(
                parser,
                bytes.as_ptr(),
                bytes.len() as c_long,
                1,
                status.as_mut_ptr().cast(),
            )
        };
        unsafe { (api.parser_destroy)(parser) };
        let code = u64::from_le_bytes(status[..8].try_into().unwrap());
        if code != 0 || pd.is_null() {
            let message = c_string(unsafe { (api.parse_error_string)(code) });
            if !pd.is_null() {
                unsafe { (api.pd_destroy)(pd) };
            }
            return Err(Error::ParserParse { code, message });
        }
        let profile = unsafe { self.read_aps_counters(pd) };
        unsafe { (api.pd_destroy)(pd) };
        Ok(profile)
    }

    unsafe fn read_aps_counters(&self, pd: *mut c_void) -> ApsCounterProfile {
        let api = self.api;
        let range = |count: FnPdCount, get: FnPdRange| -> Vec<u64> {
            let n = unsafe { count(pd) } as usize;
            let mut out = vec![0u64; n];
            if n > 0 {
                unsafe { get(pd, out.as_mut_ptr(), 0, n as u64) };
            }
            out
        };
        let system = range(api.pd_system_ts_num, api.pd_system_ts);
        let usc = range(api.pd_usc_ts_num, api.pd_usc_ts);
        let sync = range(api.pd_sync_ts_num, api.pd_sync_ts);

        let counter_count = unsafe { (api.pd_counter_num)(pd) } as usize;
        let mut name_ptrs = vec![std::ptr::null::<c_char>(); counter_count];
        if counter_count > 0 {
            unsafe { (api.pd_counter_names)(pd, name_ptrs.as_mut_ptr(), 0, counter_count as u64) };
        }
        let counter_names = name_ptrs.into_iter().map(c_string).collect::<Vec<_>>();
        let mut values = Vec::with_capacity(counter_count);
        for index in 0..counter_count as u64 {
            let mut len = 0u64;
            let mut ptr: *const u64 = std::ptr::null();
            unsafe {
                (api.pd_counter_values_num)(pd, &mut len, index, 1);
                (api.pd_counter_values)(pd, &mut ptr, index, 1);
            }
            values.push(if ptr.is_null() {
                Vec::new()
            } else {
                unsafe { std::slice::from_raw_parts(ptr, len as usize) }.to_vec()
            });
        }
        let samples = values.first().map(Vec::len).unwrap_or(0);
        let mut metadata: *const u64 = std::ptr::null();
        unsafe { (api.pd_counter_group_metadata)(pd, &mut metadata, 0, 1) };
        let mut sample_end_ticks = Vec::with_capacity(samples);
        let mut sample_cycles = Vec::with_capacity(samples);
        if !metadata.is_null() {
            // Each sample carries (synchronization reference index, USC
            // timestamp index) as two u32 halves.
            let refs = unsafe { std::slice::from_raw_parts(metadata, samples) };
            let mut previous_usc = None;
            for packed in refs {
                let sync_index = (*packed & 0xffff_ffff) as usize;
                let usc_index = (*packed >> 32) as usize;
                sample_end_ticks.push(interpolate_system_ticks(
                    sync_index, usc_index, &system, &usc, &sync,
                ));
                let usc_now = usc.get(usc_index).copied().unwrap_or(0);
                sample_cycles
                    .push(previous_usc.map_or(0, |prev: u64| usc_now.saturating_sub(prev)));
                previous_usc = Some(usc_now);
            }
        }

        let kick_count = unsafe { (api.pd_kicks_num)(pd) } as usize;
        let mut starts = vec![0u64; kick_count];
        let mut ends = vec![0u64; kick_count];
        let mut software_ids = vec![0u64; kick_count];
        let mut missing = vec![0u8; kick_count];
        if kick_count > 0 {
            let n = kick_count as u64;
            unsafe {
                (api.pd_kick_start)(pd, starts.as_mut_ptr(), 0, n);
                (api.pd_kick_end)(pd, ends.as_mut_ptr(), 0, n);
                (api.pd_kick_swid)(pd, software_ids.as_mut_ptr(), 0, n);
                (api.pd_kick_missing_end)(pd, missing.as_mut_ptr(), 0, n);
            }
        }
        let kicks = (0..kick_count)
            .map(|index| ApsKick {
                start_ticks: unsafe { (api.pd_system_timestamp)(pd, starts[index]) },
                end_ticks: unsafe { (api.pd_system_timestamp)(pd, ends[index]) },
                software_id: software_ids[index],
                missing_end: missing[index] != 0,
            })
            .collect();

        let clique_count = unsafe { (api.pd_cliques_num)(pd) } as usize;
        let mut clique_starts = vec![0u64; clique_count];
        let mut clique_ends = vec![0u64; clique_count];
        let mut clique_esl = vec![0u64; clique_count];
        let mut clique_kick = vec![0u32; clique_count];
        let mut clique_missing = vec![0u8; clique_count];
        let mut clique_slot = vec![0u8; clique_count];
        if clique_count > 0 {
            let n = clique_count as u64;
            unsafe {
                (api.pd_clique_start)(pd, clique_starts.as_mut_ptr(), 0, n);
                (api.pd_clique_end)(pd, clique_ends.as_mut_ptr(), 0, n);
                (api.pd_clique_esl_id)(pd, clique_esl.as_mut_ptr(), 0, n);
                (api.pd_clique_kick_id)(pd, clique_kick.as_mut_ptr(), 0, n);
                (api.pd_clique_missing_end)(pd, clique_missing.as_mut_ptr(), 0, n);
                (api.pd_clique_slot)(pd, clique_slot.as_mut_ptr(), 0, n);
            }
        }
        let work_cliques = (0..clique_count)
            .map(|index| ApsWorkClique {
                start_ticks: unsafe { (api.pd_system_timestamp)(pd, clique_starts[index]) },
                end_ticks: unsafe { (api.pd_system_timestamp)(pd, clique_ends[index]) },
                kick_index: clique_kick[index],
                esl_index: clique_esl[index],
                slot: clique_slot[index],
                missing_end: clique_missing[index] != 0,
            })
            .collect();

        ApsCounterProfile {
            counter_names,
            values,
            sample_end_ticks,
            sample_cycles,
            kicks,
            commands: unsafe { self.timing_commands(pd) },
            work_cliques,
        }
    }

    /// Run the agxps timing analyzer over one USC stream and return its
    /// per-command records.
    unsafe fn timing_commands(&self, pd: *mut c_void) -> Vec<ApsCommand> {
        const KIND: u32 = 1;
        let api = self.api;
        let analyzer = unsafe { (api.analyzer_create)(KIND) };
        if analyzer.is_null() {
            return Vec::new();
        }
        unsafe {
            (api.analyzer_process_usc)(analyzer, pd);
            (api.analyzer_finish)(analyzer);
        }
        let count = unsafe { (api.analyzer_num_commands)(analyzer, KIND) } as usize;
        let mut addresses = vec![0u64; count];
        let mut starts = vec![0u64; count];
        let mut software_ids = vec![0u64; count];
        if count > 0 {
            let n = count as u64;
            unsafe {
                (api.analyzer_esl_address)(analyzer, KIND, addresses.as_mut_ptr(), 0, n);
                (api.analyzer_esl_start)(analyzer, KIND, starts.as_mut_ptr(), 0, n);
                (api.analyzer_kick_software_id)(analyzer, KIND, software_ids.as_mut_ptr(), 0, n);
            }
        }
        unsafe { (api.analyzer_destroy)(analyzer) };
        (0..count)
            .map(|index| ApsCommand {
                software_id: software_ids[index],
                esl_shader_address: addresses[index],
                start_ticks: starts[index],
            })
            .collect()
    }
}

/// Port of GTShaderProfiler's `TimestampRefToNsec`: map a USC timestamp
/// onto the system clock by interpolating between the synchronization
/// points that bracket it. Sync point `i` is `(system index, usc index)`
/// packed into a u64 (low, high halves).
fn interpolate_system_ticks(
    mut sync_index: usize,
    usc_index: usize,
    system: &[u64],
    usc: &[u64],
    sync: &[u64],
) -> u64 {
    let sys_of = |i: usize| (sync[i] & 0xffff_ffff) as usize;
    let usc_of = |i: usize| (sync[i] >> 32) as usize;
    if sync.len() < 2 {
        return usc.get(usc_index).copied().unwrap_or(0);
    }
    if sync_index + 1 >= sync.len() {
        sync_index = sync.len() - 2;
    }
    let Some(&usc_now) = usc.get(usc_index) else {
        return 0;
    };
    if usc_index < usc_of(sync_index) && sync_index > 0 {
        sync_index -= 1;
    }
    let usc_base = usc[usc_of(sync_index)];
    let mut usc_span = usc[usc_of(sync_index + 1)] as i64 - usc_base as i64;
    if usc_span == 0 && sync_index > 0 {
        usc_span = usc_base as i64 - usc[usc_of(sync_index - 1)] as i64;
    }
    let sys_base = system[sys_of(sync_index)];
    let mut sys_span = system[sys_of(sync_index + 1)].wrapping_sub(sys_base);
    if sys_span == 0 || sys_span > 0x10000 {
        sys_span = 64;
    }
    if usc_span == 0 {
        return sys_base;
    }
    let offset = (usc_now as i64 - usc_base as i64) as f64 * sys_span as f64 / usc_span as f64;
    (sys_base as f64 + offset) as u64
}

unsafe extern "C" {
    fn free(ptr: *mut c_void);
}
