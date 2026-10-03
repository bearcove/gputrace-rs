//! FFI bindings to Apple's `agxps_aps_*` sample-decoder API as bundled
//! inside Xcode's `GPUDebugger.ideplugin` (specifically
//! `GTShaderProfiler.framework`).
//!
//! The full agxps surface — ~100 functions including `kick_end`,
//! `kick_id`, `kick_time_stats_create_sampled`, synchronized/operating
//! timestamps, work_clique/hw_clique/esl_clique queries, instruction
//! traces — is exported from `GTShaderProfiler.framework` as ordinary
//! external symbols, so `dlsym` resolves them directly. No offset
//! tables, no UUID-keyed compatibility lookup.
//!
//! Compare with `agxps-noxcode-sys`, which targets the much smaller
//! 15-function subset shipped in macOS's `GPUToolsReplay.framework`
//! and resolves them by hardcoded text-section offsets per build UUID.
//! Use this crate when Xcode is installed; fall back to
//! `agxps-noxcode-sys` only when it isn't.

#![cfg(target_os = "macos")]

pub mod counters;

use std::ffi::{CStr, CString, c_char, c_int, c_long, c_uint, c_void};
use std::os::raw::c_uchar;
use std::path::Path;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("framework not loadable: {0}")]
    Dlopen(String),
    #[error("agxps_gpu_create({generation}, {variant}, {rev}) failed")]
    GpuCreate {
        generation: u32,
        variant: u32,
        rev: u32,
    },
    #[error("agxps_aps_parser_create returned NULL — descriptor invalid for this GPU")]
    ParserCreate,
    #[error("agxps_aps_parser_parse error code {code}: {message}")]
    ParserParse { code: u64, message: String },
    #[error("agxps_aps_profile_data_get_kick_* range fetch failed")]
    KickAccess,
    #[error("missing symbol in framework: {0}")]
    MissingSymbol(&'static str),
    #[error("agxps_initialize failed for gpu generation {generation} variant {variant}")]
    CounterInitialize { generation: u32, variant: u32 },
    #[error("derived counter computation failed: {0}")]
    CounterCompute(String),
}

pub type Result<T> = std::result::Result<T, Error>;

// -------- Raw C ABI types --------

pub type AgxpsGpu = *mut c_void;
pub type AgxpsApsParser = *mut c_void;
pub type AgxpsApsProfileData = *mut c_void;
pub type AgxpsStats = *mut c_void;
pub type AgxpsApsTimingAnalyzer = *mut c_void;
pub type AgxpsApsCliqueInstructionTrace = u64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AgxpsApsInstructionStats {
    pub words: [u64; 14],
}

const _: () = assert!(std::mem::size_of::<AgxpsApsInstructionStats>() == 0x70);

/// 0x68-byte descriptor passed to `agxps_aps_parser_create`. The
/// non-zero fields are values empirically known to satisfy the per-GPU
/// validators; see `docs/AGXPS_API.md` for the rules each field has to
/// obey.
#[repr(C)]
pub struct AgxpsApsDescriptor {
    pub gpu: *mut c_void, // +0x00
    pub field_0x08: u32,  // +0x08 — power of 2 in [0x10, 0x800]
    pub field_0x0c: u32,  // +0x0c — power of 2 in [0x40, 0x2000]
    pub field_0x10: u32,  // +0x10 — 0 or power of 2 in [0x80, 0x8000]
    pub field_0x14: u32,
    pub field_0x18: u64, // +0x18 — must be 0x400, 0x1000, or 0x40000
    pub _pad_0x20: [u8; 0x10],
    pub field_0x30: u64, // +0x30 — set to -1 by default
    pub _pad_0x38: [u8; 0x20],
    pub field_0x58: u64, // +0x58 — set to 0x32 by default
    pub _pad_0x60: [u8; 0x08],
}

const _: () = assert!(std::mem::size_of::<AgxpsApsDescriptor>() == 0x68);

impl AgxpsApsDescriptor {
    pub fn defaults_for(gpu: AgxpsGpu) -> Self {
        Self {
            gpu,
            field_0x08: 0x100,
            field_0x0c: 0x400,
            field_0x10: 0,
            field_0x14: 0,
            field_0x18: 0x1000,
            _pad_0x20: [0; 0x10],
            field_0x30: u64::MAX,
            _pad_0x38: [0; 0x20],
            field_0x58: 0x32,
            _pad_0x60: [0; 0x08],
        }
    }
}

/// Profile-type tag for USC sample streams (`Profiling_f_*.raw`).
pub const APS_PROFILING_TYPE_USC_SAMPLES: u32 = 0x21;

// Function-pointer types matching the agxps C ABI.
pub type FnGpuCreate =
    unsafe extern "C" fn(generation: u32, variant: u32, rev: u32, flag: bool) -> AgxpsGpu;
pub type FnGpuDestroy = unsafe extern "C" fn(AgxpsGpu);
pub type FnDescriptorCreate = unsafe extern "C" fn() -> AgxpsApsDescriptor;
pub type FnParserCreate = unsafe extern "C" fn(*const AgxpsApsDescriptor) -> AgxpsApsParser;
pub type FnParserDestroy = unsafe extern "C" fn(AgxpsApsParser);
pub type FnParserParse = unsafe extern "C" fn(
    parser: AgxpsApsParser,
    bytes: *const c_uchar,
    size: c_long,
    profile_type: u32,
    out: *mut c_void,
) -> AgxpsApsProfileData;
pub type FnProfileDataDestroy = unsafe extern "C" fn(AgxpsApsProfileData);
pub type FnGetCount = unsafe extern "C" fn(AgxpsApsProfileData) -> c_uint;
pub type FnGetCount64 = unsafe extern "C" fn(AgxpsApsProfileData) -> u64;
pub type FnGetU64Range = unsafe extern "C" fn(
    pd: AgxpsApsProfileData,
    out: *mut u64,
    start_idx: u64,
    count: u64,
) -> c_int;
pub type FnGetU32Range = unsafe extern "C" fn(
    pd: AgxpsApsProfileData,
    out: *mut u32,
    start_idx: u64,
    count: u64,
) -> c_int;
pub type FnGetU8Range = unsafe extern "C" fn(
    pd: AgxpsApsProfileData,
    out: *mut u8,
    start_idx: u64,
    count: u64,
) -> c_int;
pub type FnParseErrorString = unsafe extern "C" fn(code: u64) -> *const c_char;

/// `get_counter_names(pd, char** out, size_t start, size_t count) -> int`
/// — copies `count` `const char*` pointers into `out`, returns 1 on
/// success / 0 on oob.
pub type FnGetCounterNames = unsafe extern "C" fn(
    pd: AgxpsApsProfileData,
    out: *mut *const c_char,
    start_idx: u64,
    count: u64,
) -> c_int;

/// `agxps_load_counter_obfuscation_map(const char* path) -> int` — load
/// a two-column CSV counter-name map. Rows are `readable,obfuscated`.
/// Passing `NULL` asks the framework for
/// `com.apple.gpusw.AGXProfilingSupport`'s `RawCountersMapping.csv` resource.
/// Current Xcode builds return 0 there on macOS because that bundle/resource
/// is not present, so callers should prefer an explicit path.
pub type FnLoadObfuscationMap = unsafe extern "C" fn(path: *const c_char) -> c_int;

/// `agxps_unload_counter_obfuscation_map()`.
pub type FnUnloadObfuscationMap = unsafe extern "C" fn();

/// `agxps_counter_deobfuscate_name(const char* obfuscated) -> const char*`
/// — returns a pointer to the deobfuscated name, or the input string
/// unchanged if no mapping exists / map not loaded.
pub type FnDeobfuscateName = unsafe extern "C" fn(obfuscated: *const c_char) -> *const c_char;

/// `agxps_counter_obfuscated_name(const char* readable) -> const char*`
/// — returns a pointer to the obfuscated name, or the input string
/// unchanged if no mapping exists / map not loaded.
pub type FnObfuscatedName = unsafe extern "C" fn(readable: *const c_char) -> *const c_char;

/// `agxps_aps_kick_time_stats_create(pd, timestamp_kind, start_kind, end_kind, filter_block)`.
///
/// `timestamp_kind`: 0 = system timestamp, 1 = USC timestamp. `start_kind` /
/// `end_kind`: 0 = kick start, 1 = kick end. `filter_block` only needs a block
/// invoke pointer at offset `0x10`; the wrapper calls it synchronously as
/// `bool invoke(block)`.
pub type FnKickTimeStatsCreate = unsafe extern "C" fn(
    pd: AgxpsApsProfileData,
    timestamp_kind: c_uint,
    start_kind: c_uint,
    end_kind: c_uint,
    filter_block: *mut c_void,
) -> AgxpsStats;

pub type FnKickTimeStatsCreateSampled = unsafe extern "C" fn(
    pd: AgxpsApsProfileData,
    timestamp_kind: c_uint,
    start_kind: c_uint,
    end_kind: c_uint,
    sample_count: u64,
    sample_limit: u64,
    filter_block: *mut c_void,
) -> AgxpsStats;

pub type FnStatsDestroy = unsafe extern "C" fn(stats: AgxpsStats);
pub type FnStatsValue = unsafe extern "C" fn(stats: AgxpsStats) -> f64;
pub type FnTimestampValue =
    unsafe extern "C" fn(pd: AgxpsApsProfileData, packed_timestamp: u64) -> u64;
pub type FnInstructionTraceCount =
    unsafe extern "C" fn(pd: AgxpsApsProfileData, trace: AgxpsApsCliqueInstructionTrace) -> u64;
pub type FnInstructionTraceStats = unsafe extern "C" fn(
    gpu: AgxpsGpu,
    pd: AgxpsApsProfileData,
    trace: AgxpsApsCliqueInstructionTrace,
) -> AgxpsApsInstructionStats;
pub type FnTimingAnalyzerCreate = unsafe extern "C" fn(kind: u32) -> AgxpsApsTimingAnalyzer;
pub type FnTimingAnalyzerDestroy = unsafe extern "C" fn(analyzer: AgxpsApsTimingAnalyzer);
pub type FnTimingAnalyzerProcessUsc =
    unsafe extern "C" fn(analyzer: AgxpsApsTimingAnalyzer, pd: AgxpsApsProfileData);
pub type FnTimingAnalyzerFinish = unsafe extern "C" fn(analyzer: AgxpsApsTimingAnalyzer);
pub type FnTimingAnalyzerCount =
    unsafe extern "C" fn(analyzer: AgxpsApsTimingAnalyzer, kind: u32) -> u64;
pub type FnTimingAnalyzerU64Range = unsafe extern "C" fn(
    analyzer: AgxpsApsTimingAnalyzer,
    kind: u32,
    out: *mut u64,
    start_idx: u64,
    count: u64,
) -> c_int;

/// Resolved function-pointer table. All fields are non-null on success
/// (we treat any missing symbol as a hard load error).
pub struct AgxpsApi {
    pub gpu_create: FnGpuCreate,
    pub gpu_destroy: FnGpuDestroy,
    pub descriptor_create: FnDescriptorCreate,
    pub parser_create: FnParserCreate,
    pub parser_destroy: FnParserDestroy,
    pub parser_parse: FnParserParse,
    pub profile_data_destroy: FnProfileDataDestroy,
    pub get_kicks_num: FnGetCount,
    pub get_counter_num: FnGetCount,
    pub get_kick_start: FnGetU64Range,
    pub get_kick_end: FnGetU64Range,
    pub get_kick_software_id: FnGetU64Range,
    pub get_kick_id: FnGetU64Range,
    pub get_kick_kick_slot: FnGetU64Range,
    pub get_kick_missing_end: FnGetU8Range,
    pub get_usc_timestamps: FnGetU64Range,
    pub get_usc_timestamps_num: FnGetCount,
    pub get_synchronized_timestamps: FnGetU64Range,
    pub get_synchronized_timestamps_num: FnGetCount,
    pub get_operating_frequencies: FnGetU64Range,
    pub get_work_cliques_num: FnGetCount64,
    pub get_work_clique_start: FnGetU64Range,
    pub get_work_clique_end: FnGetU64Range,
    pub get_work_clique_esl_id: FnGetU64Range,
    pub get_work_clique_tile_id: FnGetU64Range,
    pub get_work_clique_kick_id: FnGetU32Range,
    pub get_work_clique_missing_end: FnGetU8Range,
    pub get_work_clique_clique_id: FnGetU8Range,
    pub get_work_clique_instruction_trace: FnGetU64Range,
    pub get_esl_cliques_num: FnGetCount64,
    pub get_esl_clique_start: FnGetU64Range,
    pub get_esl_clique_end: FnGetU64Range,
    pub get_esl_clique_esl_id: FnGetU64Range,
    pub get_esl_clique_kick_id: FnGetU32Range,
    pub get_esl_clique_missing_end: FnGetU8Range,
    pub get_esl_clique_clique_id: FnGetU8Range,
    pub get_esl_clique_instruction_trace: FnGetU64Range,
    pub get_counter_names: FnGetCounterNames,
    pub load_obfuscation_map: FnLoadObfuscationMap,
    pub unload_obfuscation_map: FnUnloadObfuscationMap,
    pub deobfuscate_name: FnDeobfuscateName,
    pub obfuscated_name: FnObfuscatedName,
    pub kick_time_stats_create: FnKickTimeStatsCreate,
    pub kick_time_stats_create_sampled: FnKickTimeStatsCreateSampled,
    pub stats_destroy: FnStatsDestroy,
    pub stats_min: FnStatsValue,
    pub stats_max: FnStatsValue,
    pub stats_mean: FnStatsValue,
    pub stats_median: FnStatsValue,
    pub stats_q1: FnStatsValue,
    pub stats_q3: FnStatsValue,
    pub stats_iqr: FnStatsValue,
    pub get_system_timestamp: FnTimestampValue,
    pub get_usc_timestamp: FnTimestampValue,
    pub instruction_trace_get_pc_advances_num: FnInstructionTraceCount,
    pub instruction_trace_get_init_pcs_num: FnInstructionTraceCount,
    pub instruction_trace_get_thread_execution_changes_num: FnInstructionTraceCount,
    pub instruction_trace_get_timestamp_references_num: FnInstructionTraceCount,
    pub instruction_trace_get_execution_events_num: FnInstructionTraceCount,
    pub instruction_trace_get_instruction_stats: FnInstructionTraceStats,
    pub timing_analyzer_create: FnTimingAnalyzerCreate,
    pub timing_analyzer_destroy: FnTimingAnalyzerDestroy,
    pub timing_analyzer_process_usc: FnTimingAnalyzerProcessUsc,
    pub timing_analyzer_finish: FnTimingAnalyzerFinish,
    pub timing_analyzer_get_num_commands: FnTimingAnalyzerCount,
    pub timing_analyzer_get_num_discarded_work_cliques: FnTimingAnalyzerCount,
    pub timing_analyzer_get_work_start: FnTimingAnalyzerU64Range,
    pub timing_analyzer_get_work_end: FnTimingAnalyzerU64Range,
    pub timing_analyzer_get_work_shader_address: FnTimingAnalyzerU64Range,
    pub timing_analyzer_get_esl_start: FnTimingAnalyzerU64Range,
    pub timing_analyzer_get_esl_shader_address: FnTimingAnalyzerU64Range,
    pub timing_analyzer_get_work_cliques_average_duration: FnTimingAnalyzerU64Range,
    pub timing_analyzer_get_work_cliques_min_duration: FnTimingAnalyzerU64Range,
    pub timing_analyzer_get_work_cliques_max_duration: FnTimingAnalyzerU64Range,
    pub timing_analyzer_get_work_cliques_stddev_duration: FnTimingAnalyzerU64Range,
    pub timing_analyzer_get_num_work_cliques: FnTimingAnalyzerU64Range,
    pub timing_analyzer_get_num_uscs: FnTimingAnalyzerU64Range,
    pub timing_analyzer_get_kick_software_id: FnTimingAnalyzerU64Range,
    pub parse_error_string: FnParseErrorString,
}

unsafe impl Send for AgxpsApi {}
unsafe impl Sync for AgxpsApi {}

/// Resolve `GTShaderProfiler` from the active developer dir (`xcode-select -p`,
/// which honors `DEVELOPER_DIR` and points at Xcode-beta when selected), falling
/// back to the stock `/Applications/Xcode.app`. Overridden by `AGXPS_FRAMEWORK_PATH`.
fn default_framework_path() -> String {
    const REL: &str = "PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/GTShaderProfiler.framework/GTShaderProfiler";
    std::process::Command::new("xcode-select")
        .arg("-p")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| std::path::PathBuf::from(s.trim()))
        // `xcode-select -p` → `<app>/Contents/Developer`; strip `/Developer`.
        .and_then(|dev| dev.parent().map(|p| p.to_path_buf()))
        .map(|contents| contents.join(REL).to_string_lossy().into_owned())
        .unwrap_or_else(|| format!("/Applications/Xcode.app/Contents/{REL}"))
}

fn framework_path() -> String {
    std::env::var("AGXPS_FRAMEWORK_PATH").unwrap_or_else(|_| default_framework_path())
}

/// dlopen `GTShaderProfiler` (a no-op returning the same handle when it is
/// already loaded).
pub(crate) fn open_framework() -> Result<*mut c_void> {
    let path = framework_path();
    let cpath = CString::new(path.clone()).unwrap();
    let handle = unsafe { libc::dlopen(cpath.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL) };
    if handle.is_null() {
        let err = unsafe { CStr::from_ptr(libc::dlerror()) }
            .to_string_lossy()
            .into_owned();
        return Err(Error::Dlopen(format!("{path}: {err}")));
    }
    Ok(handle)
}

pub(crate) fn load_sym_ptr(handle: *mut c_void, name: &'static str) -> Result<*mut c_void> {
    unsafe { load_sym_raw(handle, name) }.ok_or(Error::MissingSymbol(name))
}

/// dlopen `GTShaderProfiler` and dlsym every needed symbol. Returns
/// [`Error::MissingSymbol`] if any required entry-point isn't exported
/// (which would mean we're looking at an Xcode version with a renamed
/// or missing function — worth flagging early).
///
/// Set `AGXPS_FRAMEWORK_PATH` env var to override the default Xcode
/// location (e.g. for Xcode-beta.app or a custom toolchain).
pub fn load() -> Result<LoadedApi> {
    let path = framework_path();
    let handle = open_framework()?;

    let api = AgxpsApi {
        gpu_create: load_sym(handle, "agxps_gpu_create")?,
        gpu_destroy: load_sym(handle, "agxps_gpu_destroy")?,
        descriptor_create: load_sym(handle, "agxps_aps_descriptor_create")?,
        parser_create: load_sym(handle, "agxps_aps_parser_create")?,
        parser_destroy: load_sym(handle, "agxps_aps_parser_destroy")?,
        parser_parse: load_sym(handle, "agxps_aps_parser_parse")?,
        profile_data_destroy: load_sym(handle, "agxps_aps_profile_data_destroy")?,
        get_kicks_num: load_sym(handle, "agxps_aps_profile_data_get_kicks_num")?,
        get_counter_num: load_sym(handle, "agxps_aps_profile_data_get_counter_num")?,
        get_kick_start: load_sym(handle, "agxps_aps_profile_data_get_kick_start")?,
        get_kick_end: load_sym(handle, "agxps_aps_profile_data_get_kick_end")?,
        get_kick_software_id: load_sym(handle, "agxps_aps_profile_data_get_kick_software_id")?,
        get_kick_id: load_sym(handle, "agxps_aps_profile_data_get_kick_id")?,
        get_kick_kick_slot: load_sym(handle, "agxps_aps_profile_data_get_kick_kick_slot")?,
        get_kick_missing_end: load_sym(handle, "agxps_aps_profile_data_get_kick_missing_end")?,
        get_usc_timestamps: load_sym(handle, "agxps_aps_profile_data_get_usc_timestamps")?,
        get_usc_timestamps_num: load_sym(handle, "agxps_aps_profile_data_get_usc_timestamps_num")?,
        get_synchronized_timestamps: load_sym(
            handle,
            "agxps_aps_profile_data_get_synchronized_timestamps",
        )?,
        get_synchronized_timestamps_num: load_sym(
            handle,
            "agxps_aps_profile_data_get_synchronized_timestamps_num",
        )?,
        get_operating_frequencies: load_sym(
            handle,
            "agxps_aps_profile_data_get_operating_frequencies",
        )?,
        get_work_cliques_num: load_sym(handle, "agxps_aps_profile_data_get_work_cliques_num")?,
        get_work_clique_start: load_sym(handle, "agxps_aps_profile_data_get_work_clique_start")?,
        get_work_clique_end: load_sym(handle, "agxps_aps_profile_data_get_work_clique_end")?,
        get_work_clique_esl_id: load_sym(handle, "agxps_aps_profile_data_get_work_clique_esl_id")?,
        get_work_clique_tile_id: load_sym(
            handle,
            "agxps_aps_profile_data_get_work_clique_tile_id",
        )?,
        get_work_clique_kick_id: load_sym(
            handle,
            "agxps_aps_profile_data_get_work_clique_kick_id",
        )?,
        get_work_clique_missing_end: load_sym(
            handle,
            "agxps_aps_profile_data_get_work_clique_missing_end",
        )?,
        get_work_clique_clique_id: load_sym(
            handle,
            "agxps_aps_profile_data_get_work_clique_clique_id",
        )?,
        get_work_clique_instruction_trace: load_sym(
            handle,
            "agxps_aps_profile_data_get_work_clique_instruction_trace",
        )?,
        get_esl_cliques_num: load_sym(handle, "agxps_aps_profile_data_get_esl_cliques_num")?,
        get_esl_clique_start: load_sym(handle, "agxps_aps_profile_data_get_esl_clique_start")?,
        get_esl_clique_end: load_sym(handle, "agxps_aps_profile_data_get_esl_clique_end")?,
        get_esl_clique_esl_id: load_sym(handle, "agxps_aps_profile_data_get_esl_clique_esl_id")?,
        get_esl_clique_kick_id: load_sym(handle, "agxps_aps_profile_data_get_esl_clique_kick_id")?,
        get_esl_clique_missing_end: load_sym(
            handle,
            "agxps_aps_profile_data_get_esl_clique_missing_end",
        )?,
        get_esl_clique_clique_id: load_sym(
            handle,
            "agxps_aps_profile_data_get_esl_clique_clique_id",
        )?,
        get_esl_clique_instruction_trace: load_sym(
            handle,
            "agxps_aps_profile_data_get_esl_clique_instruction_trace",
        )?,
        get_counter_names: load_sym(handle, "agxps_aps_profile_data_get_counter_names")?,
        load_obfuscation_map: load_sym(handle, "agxps_load_counter_obfuscation_map")?,
        unload_obfuscation_map: load_sym_any(
            handle,
            &[
                "agxps_unload_counter_obfuscation_map",
                "_Z36agxps_unload_counter_obfuscation_mapv",
            ],
        )?,
        deobfuscate_name: load_sym(handle, "agxps_counter_deobfuscate_name")?,
        obfuscated_name: load_sym(handle, "agxps_counter_obfuscated_name")?,
        kick_time_stats_create: load_sym(handle, "agxps_aps_kick_time_stats_create")?,
        kick_time_stats_create_sampled: load_sym(
            handle,
            "agxps_aps_kick_time_stats_create_sampled",
        )?,
        stats_destroy: load_sym(handle, "agxps_stats_destroy")?,
        stats_min: load_sym(handle, "agxps_stats_min")?,
        stats_max: load_sym(handle, "agxps_stats_max")?,
        stats_mean: load_sym(handle, "agxps_stats_mean")?,
        stats_median: load_sym(handle, "agxps_stats_median")?,
        stats_q1: load_sym(handle, "agxps_stats_q1")?,
        stats_q3: load_sym(handle, "agxps_stats_q3")?,
        stats_iqr: load_sym(handle, "agxps_stats_iqr")?,
        get_system_timestamp: load_sym(handle, "agxps_aps_profile_data_get_system_timestamp")?,
        get_usc_timestamp: load_sym(handle, "agxps_aps_profile_data_get_usc_timestamp")?,
        instruction_trace_get_pc_advances_num: load_sym(
            handle,
            "agxps_aps_clique_instruction_trace_get_pc_advances_num",
        )?,
        instruction_trace_get_init_pcs_num: load_sym(
            handle,
            "agxps_aps_clique_instruction_trace_get_init_pcs_num",
        )?,
        instruction_trace_get_thread_execution_changes_num: load_sym(
            handle,
            "agxps_aps_clique_instruction_trace_get_thread_execution_changes_num",
        )?,
        instruction_trace_get_timestamp_references_num: load_sym(
            handle,
            "agxps_aps_clique_instruction_trace_get_timestamp_references_num",
        )?,
        instruction_trace_get_execution_events_num: load_sym(
            handle,
            "agxps_aps_clique_instruction_trace_get_execution_events_num",
        )?,
        instruction_trace_get_instruction_stats: load_sym(
            handle,
            "agxps_aps_clique_instruction_trace_get_instruction_stats",
        )?,
        timing_analyzer_create: load_sym(handle, "agxps_aps_timing_analyzer_create")?,
        timing_analyzer_destroy: load_sym(handle, "agxps_aps_timing_analyzer_destroy")?,
        timing_analyzer_process_usc: load_sym(handle, "agxps_aps_timing_analyzer_process_usc")?,
        timing_analyzer_finish: load_sym(handle, "agxps_aps_timing_analyzer_finish")?,
        timing_analyzer_get_num_commands: load_sym(
            handle,
            "agxps_aps_timing_analyzer_get_num_commands",
        )?,
        timing_analyzer_get_num_discarded_work_cliques: load_sym(
            handle,
            "agxps_aps_timing_analyzer_get_num_discarded_work_cliques",
        )?,
        timing_analyzer_get_work_start: load_sym(
            handle,
            "agxps_aps_timing_analyzer_get_work_start",
        )?,
        timing_analyzer_get_work_end: load_sym(handle, "agxps_aps_timing_analyzer_get_work_end")?,
        timing_analyzer_get_work_shader_address: load_sym(
            handle,
            "agxps_aps_timing_analyzer_get_work_shader_address",
        )?,
        timing_analyzer_get_esl_start: load_sym(handle, "agxps_aps_timing_analyzer_get_esl_start")?,
        timing_analyzer_get_esl_shader_address: load_sym(
            handle,
            "agxps_aps_timing_analyzer_get_esl_shader_address",
        )?,
        timing_analyzer_get_work_cliques_average_duration: load_sym(
            handle,
            "agxps_aps_timing_analyzer_get_work_cliques_average_duration",
        )?,
        timing_analyzer_get_work_cliques_min_duration: load_sym(
            handle,
            "agxps_aps_timing_analyzer_get_work_cliques_min_duration",
        )?,
        timing_analyzer_get_work_cliques_max_duration: load_sym(
            handle,
            "agxps_aps_timing_analyzer_get_work_cliques_max_duration",
        )?,
        timing_analyzer_get_work_cliques_stddev_duration: load_sym(
            handle,
            "agxps_aps_timing_analyzer_get_work_cliques_stddev_duration",
        )?,
        timing_analyzer_get_num_work_cliques: load_sym(
            handle,
            "agxps_aps_timing_analyzer_get_num_work_cliques",
        )?,
        timing_analyzer_get_num_uscs: load_sym(handle, "agxps_aps_timing_analyzer_get_num_uscs")?,
        timing_analyzer_get_kick_software_id: load_sym(
            handle,
            "agxps_aps_timing_analyzer_get_kick_software_id",
        )?,
        parse_error_string: load_sym(handle, "agxps_aps_parse_error_type_to_string")?,
    };

    // Current Xcode builds return 0 for the default path, but keep this
    // attempt for compatibility with builds that do ship a default map.
    let _ = unsafe { (api.load_obfuscation_map)(std::ptr::null()) };

    Ok(LoadedApi {
        api,
        framework_path: path,
        _keep_alive: handle,
    })
}

unsafe fn load_sym_raw(handle: *mut c_void, name: &str) -> Option<*mut c_void> {
    use std::ffi::CString;
    let cname = CString::new(name).unwrap();
    let p = unsafe { libc::dlsym(handle, cname.as_ptr()) };
    if p.is_null() { None } else { Some(p) }
}

fn load_sym<T>(handle: *mut c_void, name: &'static str) -> Result<T> {
    let p = unsafe { load_sym_raw(handle, name) }.ok_or(Error::MissingSymbol(name))?;
    Ok(unsafe { std::mem::transmute_copy(&p) })
}

fn load_sym_any<T>(handle: *mut c_void, names: &'static [&'static str]) -> Result<T> {
    for name in names {
        if let Some(p) = unsafe { load_sym_raw(handle, name) } {
            return Ok(unsafe { std::mem::transmute_copy(&p) });
        }
    }
    Err(Error::MissingSymbol(names[0]))
}

pub struct LoadedApi {
    pub api: AgxpsApi,
    pub framework_path: String,
    _keep_alive: *mut c_void,
}

unsafe impl Send for LoadedApi {}
unsafe impl Sync for LoadedApi {}

impl LoadedApi {
    /// Clear the process-global counter obfuscation maps.
    pub fn unload_counter_obfuscation_map(&self) {
        unsafe { (self.api.unload_obfuscation_map)() };
    }

    /// Load a counter obfuscation map.
    ///
    /// The file is parsed as CSV-like text with exactly two columns per
    /// row: `readable_name,obfuscated_name`. Returns the framework's
    /// boolean success value. A successful parse can still add zero
    /// useful mappings if every row is invalid for this format.
    pub fn load_counter_obfuscation_map(&self, path: Option<&Path>) -> bool {
        match path {
            Some(path) => {
                let path = path.to_string_lossy();
                let Ok(cpath) = CString::new(path.as_bytes()) else {
                    return false;
                };
                unsafe { (self.api.load_obfuscation_map)(cpath.as_ptr()) != 0 }
            }
            None => unsafe { (self.api.load_obfuscation_map)(std::ptr::null()) != 0 },
        }
    }

    /// Map an obfuscated counter name back to its readable name using
    /// the currently loaded process-global map. If no mapping exists,
    /// the framework returns the input unchanged.
    pub fn deobfuscate_counter_name(&self, obfuscated: &str) -> String {
        map_counter_name(obfuscated, self.api.deobfuscate_name)
    }

    /// Map a readable counter name to its obfuscated name using the
    /// currently loaded process-global map. If no mapping exists, the
    /// framework returns the input unchanged.
    pub fn obfuscated_counter_name(&self, readable: &str) -> String {
        map_counter_name(readable, self.api.obfuscated_name)
    }

    /// Decode a single `Profiling_f_*.raw` and return per-kick data
    /// (start, end, software_id, kick_id, kick_slot, missing_end) plus
    /// the raw `usc_timestamps` and `synchronized_timestamps` vectors.
    pub fn parse_profiling(
        &self,
        generation: u32,
        variant: u32,
        rev: u32,
        bytes: &[u8],
    ) -> Result<DecodedProfile> {
        let api = &self.api;

        let gpu = unsafe { (api.gpu_create)(generation, variant, rev, false) };
        if gpu.is_null() {
            return Err(Error::GpuCreate {
                generation,
                variant,
                rev,
            });
        }

        let descriptor = AgxpsApsDescriptor::defaults_for(gpu);
        let parser = unsafe { (api.parser_create)(&descriptor) };
        if parser.is_null() {
            return Err(Error::ParserCreate);
        }

        let mut out = vec![0u8; 4096];
        let pd = unsafe {
            (api.parser_parse)(
                parser,
                bytes.as_ptr(),
                bytes.len() as c_long,
                APS_PROFILING_TYPE_USC_SAMPLES,
                out.as_mut_ptr() as *mut c_void,
            )
        };
        let err_code = u64::from_le_bytes(out[..8].try_into().unwrap());
        if err_code != 0 {
            let msg = unsafe {
                let s = (api.parse_error_string)(err_code);
                if s.is_null() {
                    "(null)".to_owned()
                } else {
                    CStr::from_ptr(s).to_string_lossy().into_owned()
                }
            };
            return Err(Error::ParserParse {
                code: err_code,
                message: msg,
            });
        }

        let n = unsafe { (api.get_kicks_num)(pd) } as usize;
        let mut starts = vec![0u64; n];
        let mut ends = vec![0u64; n];
        let mut swids = vec![0u64; n];
        let mut kids = vec![0u64; n];
        let mut slots = vec![0u64; n];
        let mut missing = vec![0u8; n];
        if n > 0 {
            let nu = n as u64;
            let ok = unsafe {
                (api.get_kick_start)(pd, starts.as_mut_ptr(), 0, nu) != 0
                    && (api.get_kick_end)(pd, ends.as_mut_ptr(), 0, nu) != 0
                    && (api.get_kick_software_id)(pd, swids.as_mut_ptr(), 0, nu) != 0
                    && (api.get_kick_id)(pd, kids.as_mut_ptr(), 0, nu) != 0
                    && (api.get_kick_kick_slot)(pd, slots.as_mut_ptr(), 0, nu) != 0
                    && (api.get_kick_missing_end)(pd, missing.as_mut_ptr(), 0, nu) != 0
            };
            if !ok {
                return Err(Error::KickAccess);
            }
        }

        let usc_n = unsafe { (api.get_usc_timestamps_num)(pd) } as usize;
        let mut usc_timestamps = vec![0u64; usc_n];
        if usc_n > 0 {
            let _ = unsafe {
                (api.get_usc_timestamps)(pd, usc_timestamps.as_mut_ptr(), 0, usc_n as u64)
            };
        }

        let sync_n = unsafe { (api.get_synchronized_timestamps_num)(pd) } as usize;
        let mut sync_timestamps = vec![0u64; sync_n];
        if sync_n > 0 {
            let _ = unsafe {
                (api.get_synchronized_timestamps)(
                    pd,
                    sync_timestamps.as_mut_ptr(),
                    0,
                    sync_n as u64,
                )
            };
        }

        let counter_num = unsafe { (api.get_counter_num)(pd) };

        // Fetch counter names. They're const char* into framework-owned
        // string pool, so we copy to owned Strings while the framework
        // is still mapped (which is forever since we leak).
        let mut name_ptrs = vec![std::ptr::null::<c_char>(); counter_num as usize];
        let counter_names = if counter_num > 0 {
            let ok = unsafe {
                (api.get_counter_names)(pd, name_ptrs.as_mut_ptr(), 0, counter_num as u64)
            };
            if ok != 0 {
                name_ptrs
                    .iter()
                    .map(|p| {
                        if p.is_null() {
                            return String::new();
                        }
                        let deobf = unsafe { (api.deobfuscate_name)(*p) };
                        let chosen = if deobf.is_null() { *p } else { deobf };
                        unsafe { CStr::from_ptr(chosen) }
                            .to_string_lossy()
                            .into_owned()
                    })
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        // See the noxcode crate for why we leak instead of calling destroy.
        let _ = (parser, pd, gpu);

        Ok(DecodedProfile {
            kick_starts: starts,
            kick_ends: ends,
            kick_software_ids: swids,
            kick_ids: kids,
            kick_kick_slots: slots,
            kick_missing_ends: missing.into_iter().map(|b| b != 0).collect(),
            usc_timestamps,
            synchronized_timestamps: sync_timestamps,
            counter_num,
            counter_names,
        })
    }
}

fn map_counter_name(
    input: &str,
    f: unsafe extern "C" fn(*const c_char) -> *const c_char,
) -> String {
    let Ok(cinput) = CString::new(input) else {
        return input.to_owned();
    };
    let out = unsafe { f(cinput.as_ptr()) };
    if out.is_null() {
        input.to_owned()
    } else {
        unsafe { CStr::from_ptr(out) }
            .to_string_lossy()
            .into_owned()
    }
}

/// Per-kick decoded data from one `Profiling_f_*.raw` stream.
#[derive(Debug, Clone)]
pub struct DecodedProfile {
    pub kick_starts: Vec<u64>,
    pub kick_ends: Vec<u64>,
    pub kick_software_ids: Vec<u64>,
    pub kick_ids: Vec<u64>,
    pub kick_kick_slots: Vec<u64>,
    pub kick_missing_ends: Vec<bool>,
    pub usc_timestamps: Vec<u64>,
    pub synchronized_timestamps: Vec<u64>,
    pub counter_num: u32,
    /// Counter names (length = `counter_num`).
    pub counter_names: Vec<String>,
}

/// Decompose a packed timestamp/index value: `kick_start`, `kick_end`,
/// and `synchronized_timestamps` entries are stored as
/// `(time << 32) | usc_sample_index`. The `time` is in GPU profile-clock
/// ticks (~3.89 ns/tick on M4 Pro, derived empirically by mapping the
/// trace's tick span against Xcode's wall-clock total). The
/// `usc_sample_index` cross-references into the `usc_timestamps` /
/// `synchronized_timestamps` arrays.
#[inline]
pub fn unpack_time_sample(value: u64) -> (u32, u32) {
    ((value >> 32) as u32, (value & 0xffff_ffff) as u32)
}

impl DecodedProfile {
    /// Kick start time in profile-clock ticks (high 32 bits of the
    /// raw value). Use `kick_start_sample(i)` to get the sample index
    /// (low 32 bits), useful for cross-referencing usc/sync timestamps.
    #[inline]
    pub fn kick_start_time(&self, i: usize) -> u32 {
        unpack_time_sample(self.kick_starts[i]).0
    }

    /// Kick start sample index (low 32 bits of the raw value).
    #[inline]
    pub fn kick_start_sample(&self, i: usize) -> u32 {
        unpack_time_sample(self.kick_starts[i]).1
    }

    /// Kick end time in profile-clock ticks (high 32 bits of the raw
    /// value).
    #[inline]
    pub fn kick_end_time(&self, i: usize) -> u32 {
        unpack_time_sample(self.kick_ends[i]).0
    }

    /// Kick end sample index (low 32 bits of the raw value).
    #[inline]
    pub fn kick_end_sample(&self, i: usize) -> u32 {
        unpack_time_sample(self.kick_ends[i]).1
    }

    /// Kick count grouped by `software_id` high 16 bits (= kernel/clique).
    pub fn group_by_clique(&self) -> std::collections::BTreeMap<u16, usize> {
        let mut counts = std::collections::BTreeMap::new();
        for swid in &self.kick_software_ids {
            *counts.entry((*swid >> 48) as u16).or_insert(0usize) += 1;
        }
        counts
    }

    /// Sum of `kick_end_time - kick_start_time` (in profile-clock
    /// ticks) grouped by `software_id` high 16 bits. Kicks with
    /// `missing_end=true` are skipped.
    ///
    /// **Caveat:** this is *kick lifetime* (event start to event end),
    /// not active GPU compute time. A kick can be "alive" while the
    /// GPU is doing other work; the duration reported here will
    /// over-count by any wait/synchronization windows. Use this only
    /// for relative ordering or rough scaling. To match Xcode's
    /// per-pipeline cost numbers, you'll need to sum a counter (e.g.
    /// "ALU active cycles") via the un-bound vtable getters, or call
    /// `agxps_aps_kick_time_stats_create_sampled` (which takes Apple
    /// closure blocks).
    pub fn duration_by_clique(&self) -> std::collections::BTreeMap<u16, u64> {
        let mut sums = std::collections::BTreeMap::new();
        for i in 0..self.kick_starts.len() {
            if self.kick_missing_ends[i] {
                continue;
            }
            let dur = self
                .kick_end_time(i)
                .saturating_sub(self.kick_start_time(i)) as u64;
            let prefix = (self.kick_software_ids[i] >> 48) as u16;
            *sums.entry(prefix).or_insert(0u64) += dur;
        }
        sums
    }
}

mod libc {
    use std::ffi::c_char;
    use std::ffi::c_int;
    use std::ffi::c_void;

    pub const RTLD_LAZY: c_int = 1;
    pub const RTLD_LOCAL: c_int = 4;

    unsafe extern "C" {
        pub fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        pub fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
        pub fn dlerror() -> *const c_char;
    }
}
