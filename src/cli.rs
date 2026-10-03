use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use std::{io, io::Write};

use clap::{Args, Parser, Subcommand};

use crate::analysis;
use crate::analyze_usage;
use crate::apicalls;
use crate::buffer_timeline;
use crate::buffers;
use crate::clear_buffers;
use crate::commands;
use crate::correlate;
use crate::counter;
use crate::counter_export;
use crate::diff;
use crate::dump;
use crate::error::Result;
use crate::fences;
use crate::graphing;
use crate::insights;
use crate::markdown;
use crate::mtlb;
use crate::profiler;
use crate::replay_service;
use crate::report;
use crate::shaders;
use crate::synth_bench;
use crate::timeline;
use crate::timing;
use crate::trace::{RecordType, TraceBundle};
use crate::xcode_command_costs;
use crate::xcode_cost_scan;
use crate::xcode_counters;
use crate::xcode_mio;

#[derive(Debug, Parser)]
#[command(name = "gputrace")]
#[command(version)]
#[command(about = "Profile an Apple Metal .gputrace and write a GPU analysis report")]
pub struct Cli {
    #[command(subcommand)]
    command: CommandSet,
}

#[derive(Debug, Subcommand)]
enum CommandSet {
    #[command(hide = true)]
    Stats(TracePath),
    #[command(hide = true)]
    Analyze(TracePath),
    #[command(
        about = "Profile if needed, then write a Markdown report directory",
        long_about = "Profile if needed, then write a Markdown report directory.\n\nThis is the public command. It takes the original .gputrace bundle, reuses cached profiler data inside the bundle when present, otherwise profiles the trace with MTLReplayer, then writes the report."
    )]
    Report(ReportArgs),
    #[command(hide = true)]
    AnalyzeUsage(AnalyzeUsageArgs),
    #[command(hide = true)]
    ApiCalls(ApiCallsArgs),
    #[command(hide = true)]
    ClearBuffers(ClearBuffersArgs),
    #[command(hide = true)]
    Dump(DumpArgs),
    #[command(hide = true)]
    DumpRecords(DumpRecordsArgs),
    #[command(
        hide = true,
        about = "Export offline counter/profile rows from a trace bundle",
        long_about = "Export offline counter/profile rows from a trace bundle.\n\nRows combine streamData/profile dispatch timing and decoded APS counter sample rows when present. Legacy raw-counter fallback rows are only emitted when richer profiler/APS rows are unavailable. JSON includes APS-derived metric_metadata with Xcode counter graph units/groups/visibility. No Xcode Counters.csv is required; use xcode-counters for explicit CSV exports."
    )]
    ExportCounters(ExportCountersArgs),
    #[command(
        hide = true,
        about = "Decode raw APS counter data from a profiler bundle",
        long_about = "Decode raw APS counter data from a profiler bundle.\n\nReads .gpuprofiler_raw/streamData directly, decodes APSCounterData/GPRWCNTR schemas, exposes APS trace-id and program-address mappings, scans Profiling_f_* payloads for address-derived shader hits, maps raw hashes through installed AGX Metal catalogs, and runs local Apple *-derived.js formulas where the trace exposes matching raw variables. This is independent of Xcode Counters.csv."
    )]
    RawCounters(RawCountersArgs),
    #[command(hide = true)]
    RawCounterProbe(RawCounterProbeArgs),
    #[command(hide = true)]
    ProfilingAddressProbe(ProfilingAddressProbeArgs),
    #[command(alias = "perfcounters-validate", hide = true)]
    ValidateCounters(ValidateCountersArgs),
    #[command(hide = true)]
    Fences(FencesArgs),
    #[command(hide = true)]
    Mtlb(MtlbArgs),
    #[command(hide = true)]
    MtlbInventory(MtlbPathArgs),
    #[command(hide = true)]
    MtlbStats(MtlbPathArgs),
    #[command(hide = true)]
    MtlbFunctions(MtlbFunctionsArgs),
    #[command(hide = true)]
    Profiler(ProfilerArgs),
    #[command(
        hide = true,
        about = "Decode Xcode private MIO shader-profiler topology",
        long_about = "Decode Xcode private MIO shader-profiler topology.\n\nThis macOS-only command loads Xcode's private GTShaderProfiler framework and asks it to process the exported streamData. It reports the structured GPU command, encoder, and pipeline topology that Xcode derives from the regular .gpuprofiler_raw export. Cost timeline bytes are not mislabeled as Xcode Cost; unresolved cost records remain exposed as counts until their layout is fully mapped."
    )]
    XcodeMio(XcodeMioArgs),
    #[command(
        hide = true,
        about = "Report profiler bundle byte and decoder coverage",
        long_about = "Report profiler bundle byte and decoder coverage.\n\nThis is an end-user coverage/worklist report for Xcode-exported .gpuprofiler_raw bundles. It groups streamData, Profiling_f_*, Counters_f_*, Timeline_f_*, and other raw files by bytes and marks which families are decoded semantically, only decoded heuristically, or still opaque."
    )]
    ProfilerCoverage(ProfilerCoverageArgs),
    #[command(hide = true)]
    Timeline(TimelineArgs),
    #[command(hide = true)]
    Kernels(KernelsArgs),
    #[command(hide = true)]
    Encoders(EncodersArgs),
    #[command(hide = true)]
    Dependencies(DependenciesArgs),
    #[command(hide = true)]
    Shaders(ShadersArgs),
    #[command(hide = true)]
    ShaderSource(ShaderSourceArgs),
    #[command(hide = true)]
    ShaderHotspots(ShaderHotspotsArgs),
    #[command(hide = true)]
    Correlate(CorrelateArgs),
    #[command(hide = true)]
    Timing(TimingArgs),
    #[command(hide = true)]
    TimingProfiler(TimingProfilerArgs),
    #[command(hide = true)]
    CommandBuffers(CommandBuffersArgs),
    #[command(hide = true)]
    BufferAccess(BufferAccessArgs),
    #[command(hide = true)]
    Tree(TreeArgs),
    #[command(hide = true)]
    Graph(GraphArgs),
    #[command(hide = true)]
    Buffers(BuffersArgs),
    #[command(hide = true)]
    BufferTimeline(BufferTimelineArgs),
    #[command(hide = true)]
    Insights(InsightsArgs),
    #[command(hide = true)]
    Diff(DiffArgs),
    #[command(hide = true)]
    Markdown(MarkdownArgs),
    #[command(
        hide = true,
        alias = "capture-profile",
        about = "Profile an existing .gputrace headlessly via MTLReplayer.app",
        long_about = "Profile an existing .gputrace headlessly via MTLReplayer.app.\n\nDirectly invokes the Apple-shipped MTLReplayer.app CLI (no Xcode UI, no Accessibility permission required) and waits for it to write the .gpuprofiler_raw bundle. The exact invocation is:\n\n  open -W -a /System/Library/CoreServices/MTLReplayer.app \\\n    --args -CLI <trace> -collectProfilerData --all -runningInCI -verbose --output <dir>\n\nMTLReplayer.app is launched through `open` (LaunchServices) because Apple's trust cache puts a launch constraint on the binary that only LaunchServices satisfies. `-CLI` is the gating flag that puts MTLReplayer in headless mode (otherwise it idles as the GTDisplayService companion). `-collectProfilerData --all` is what drives the actual per-draw replay/profile loop."
    )]
    Profile(ProfileArgs),
    Version(VersionArgs),
    #[command(
        hide = true,
        about = "Compare pasted Xcode GPU Commands costs against AGXPS candidates",
        long_about = "Compare pasted Xcode GPU Commands costs against AGXPS candidates.\n\nPass a tab-delimited table copied from Xcode's Counters > GPU Commands > Compute Kernel view. The command groups Xcode's per-command Execution Cost by pipeline address, maps those addresses through Xcode MIO topology, then reports error metrics for the AGXPS analyzer-weighted and instruction-stats W1 candidate costs."
    )]
    XcodeCommandCosts(XcodeCommandCostsArgs),
    #[command(
        hide = true,
        about = "Scan profiler raw files for Xcode UI cost numbers",
        long_about = "Scan profiler raw files for Xcode UI cost numbers.\n\nMemory-maps every file in a .gpuprofiler_raw directory and scans in parallel for approximate little-endian f32/f64 encodings of Xcode Execution Cost percentages. Pass --table with a tab-delimited Xcode GPU Commands table to search for all copied Execution Cost values."
    )]
    XcodeCostScan(XcodeCostScanArgs),
    #[command(hide = true)]
    XcodeCounters(XcodeCountersArgs),
    #[command(
        hide = true,
        about = "Capture a synthetic workload with one dispatch per pipeline",
        long_about = "Capture a synthetic workload with one dispatch per pipeline.\n\nGenerates 16 compute kernels with distinct unrolled FMA loop counts and threadgroup grid sizes, dispatches each once under MTLCaptureManager, and writes a .gputrace at the requested path. Used to reverse-engineer per-command counter records: with one command per pipeline Xcode has nothing to aggregate, so any per-command integer in the GPU Commands tab maps to exactly one stored record.\n\nRequires METAL_CAPTURE_ENABLED=1 in the environment."
    )]
    SynthBench(SynthBenchArgs),
}

#[derive(Debug, Args)]
struct TracePath {
    trace: PathBuf,
}

#[derive(Debug, Args)]
struct ReportArgs {
    #[arg(help = "Input .gputrace bundle to profile/analyze")]
    trace: PathBuf,
    #[arg(
        short,
        long,
        help = "Directory to create/update with markdown report files; defaults to <trace>/gputrace-report"
    )]
    output: Option<PathBuf>,
    #[arg(
        long,
        hide = true,
        value_name = "DIR",
        help = "Override the .gpuprofiler_raw directory (sets GPUTRACE_PROFILER_DIR for this run). Useful when streamData lives somewhere other than next to the trace, e.g. /private/tmp/com.apple.gputools.profiling/<stem>_stream.gpuprofiler_raw or a directory produced by `MTLReplayer -CLI ... --output X`"
    )]
    profiler: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ProfileArgs {
    #[arg(help = "Input .gputrace bundle to profile")]
    trace: PathBuf,
    #[arg(
        short,
        long,
        help = "Directory MTLReplayer writes the .gpuprofiler_raw bundle into; defaults to <trace>-perfdata next to the trace"
    )]
    output: Option<PathBuf>,
    #[arg(long, help = "Capture MTLReplayer stdout to this file")]
    stdout_log: Option<PathBuf>,
    #[arg(long, help = "Capture MTLReplayer stderr to this file")]
    stderr_log: Option<PathBuf>,
    #[arg(long, default_value = "text", value_parser = ["text", "json"], help = "Output format")]
    format: String,
}

#[derive(Debug, Args)]
struct XcodeMioArgs {
    trace: PathBuf,
    #[arg(short, long, default_value = "summary")]
    format: String,
}

#[derive(Debug, Args)]
struct DiffArgs {
    left: Option<PathBuf>,
    right: Option<PathBuf>,
    #[arg(long = "left")]
    left_flag: Option<PathBuf>,
    #[arg(long = "right")]
    right_flag: Option<PathBuf>,
    #[arg(long)]
    markdown: bool,
    #[arg(long)]
    json: bool,
    #[arg(long)]
    csv: bool,
    #[arg(long)]
    by: Option<String>,
    #[arg(long)]
    show_matches: bool,
    #[arg(long)]
    show_unmatched: bool,
    #[arg(long)]
    show_occurrences: bool,
    #[arg(long)]
    explain: bool,
    #[arg(long)]
    quick: bool,
    #[arg(long)]
    by_encoder: bool,
    #[arg(long)]
    bench_dir: Option<PathBuf>,
    #[arg(long)]
    md_out: Option<PathBuf>,
    #[arg(short, long)]
    format: Option<String>,
    #[arg(long, default_value_t = 10)]
    limit: usize,
    #[arg(long)]
    min_delta_us: Option<i64>,
    #[arg(long)]
    only_encoder: Option<usize>,
    #[arg(long)]
    only_function: Option<String>,
}

#[derive(Debug, Args)]
struct BufferTimelineArgs {
    trace: PathBuf,
    #[arg(short, long, default_value = "ascii")]
    format: String,
    #[arg(short, long, default_value_t = 100)]
    width: usize,
}

#[derive(Debug, Args)]
struct ApiCallsArgs {
    trace: PathBuf,
    #[arg(short = 'k', long)]
    kernel: Option<String>,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct AnalyzeUsageArgs {
    trace: PathBuf,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct ClearBuffersArgs {
    trace: PathBuf,
    #[arg(long, default_value_t = false)]
    dry_run: bool,
    #[arg(short = 'y', long, default_value_t = false)]
    yes: bool,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct DumpArgs {
    trace: PathBuf,
    #[arg(long)]
    filter: Option<String>,
    #[arg(long, default_value_t = false)]
    buffers_only: bool,
    #[arg(long, default_value_t = false)]
    dispatch_only: bool,
    #[arg(long, default_value_t = false)]
    encoders_only: bool,
    #[arg(long)]
    command_buffer: Option<usize>,
    #[arg(long, default_value_t = false)]
    json: bool,
    #[arg(long, default_value_t = false)]
    full: bool,
}

#[derive(Debug, Args)]
struct VersionArgs {
    #[arg(long, default_value_t = false)]
    json: bool,
}

#[derive(Debug, Args)]
struct DumpRecordsArgs {
    trace: PathBuf,
    #[arg(long = "type")]
    record_type: Option<String>,
    #[arg(long)]
    contains: Option<String>,
    #[arg(long, default_value_t = 0)]
    start_index: usize,
    #[arg(long)]
    limit: Option<usize>,
    #[arg(long)]
    hex_preview: bool,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct ExportCountersArgs {
    #[arg(help = "Trace bundle or exported perfdata .gputrace path")]
    trace: PathBuf,
    #[arg(
        short,
        long,
        default_value = "csv",
        help = "Output format: csv/xcode-csv, internal-csv, json, or text"
    )]
    format: String,
}

#[derive(Debug, Args)]
struct RawCountersArgs {
    #[arg(help = "Trace bundle or exported perfdata .gputrace path")]
    trace: PathBuf,
    #[arg(
        short,
        long,
        default_value = "text",
        help = "Output format: text, json, or csv"
    )]
    format: String,
}

#[derive(Debug, Args)]
struct ProfilerCoverageArgs {
    #[arg(help = "Trace bundle or exported perfdata .gputrace path")]
    trace: PathBuf,
    #[arg(
        short,
        long,
        default_value = "text",
        help = "Output format: text or json"
    )]
    format: String,
}

#[derive(Debug, Args)]
struct RawCounterProbeArgs {
    trace: PathBuf,
    #[arg(long)]
    csv: Option<PathBuf>,
    #[arg(long)]
    metric: Option<String>,
    #[arg(long)]
    scan_files: bool,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct ProfilingAddressProbeArgs {
    trace: PathBuf,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct ValidateCountersArgs {
    trace: PathBuf,
    #[arg(long)]
    csv: Option<PathBuf>,
    #[arg(long, default_value_t = 0.5)]
    tolerance: f64,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct MtlbArgs {
    #[command(subcommand)]
    command: Option<MtlbCommand>,
    path: Option<PathBuf>,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Subcommand)]
enum MtlbCommand {
    List(MtlbPathArgs),
    Info(MtlbPathArgs),
    Inventory(MtlbPathArgs),
    Stats(MtlbPathArgs),
    Functions(MtlbFunctionsArgs),
    ExportFunctions(MtlbExportFunctionsArgs),
    Extract(MtlbExtractArgs),
}

#[derive(Debug, Args)]
struct MtlbPathArgs {
    path: PathBuf,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct MtlbFunctionsArgs {
    path: PathBuf,
    #[arg(long)]
    filter: Option<String>,
    #[arg(long, default_value_t = false)]
    used_only: bool,
    #[arg(long = "no-usage", default_value_t = false)]
    no_usage: bool,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct MtlbExportFunctionsArgs {
    path: PathBuf,
    #[arg(long)]
    filter: Option<String>,
    #[arg(long, default_value_t = false)]
    used_only: bool,
    #[arg(long = "no-usage", default_value_t = false)]
    no_usage: bool,
    #[arg(short, long, default_value = "csv")]
    format: String,
}

#[derive(Debug, Args)]
struct MtlbExtractArgs {
    path: PathBuf,
    #[arg(short, long)]
    output: Option<PathBuf>,
    #[arg(long)]
    library: Option<String>,
    #[arg(long, default_value_t = false)]
    all: bool,
    #[arg(long)]
    output_dir: Option<PathBuf>,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct FencesArgs {
    trace: PathBuf,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct XcodeCountersArgs {
    trace: PathBuf,
    #[arg(long)]
    csv: Option<PathBuf>,
    #[arg(short, long, default_value = "summary")]
    format: String,
    #[arg(long)]
    metric: Option<String>,
    #[arg(long)]
    top: Option<usize>,
}

#[derive(Debug, Args)]
struct XcodeCommandCostsArgs {
    trace: PathBuf,
    #[arg(long, value_name = "PATH")]
    table: PathBuf,
    #[arg(short, long, default_value = "summary")]
    format: String,
    #[arg(long)]
    top: Option<usize>,
}

#[derive(Debug, Args)]
struct XcodeCostScanArgs {
    #[arg(help = "Input .gpuprofiler_raw directory")]
    profiler: PathBuf,
    #[arg(long, value_name = "PATH", help = "Pasted Xcode GPU Commands table")]
    table: Option<PathBuf>,
    #[arg(
        long = "target",
        value_name = "PERCENT",
        help = "Additional Xcode UI percentage to scan for"
    )]
    targets: Vec<f64>,
    #[arg(
        long = "no-defaults",
        default_value_t = false,
        help = "Do not include built-in comparison targets"
    )]
    no_defaults: bool,
    #[arg(
        long,
        default_value_t = false,
        help = "Scan every byte offset instead of aligned words"
    )]
    unaligned: bool,
    #[arg(
        long,
        default_value_t = 3,
        help = "Maximum hits retained per target per file"
    )]
    max_hits_per_target: usize,
    #[arg(short, long, default_value = "summary")]
    format: String,
    #[arg(long)]
    top: Option<usize>,
}

#[derive(Debug, Args)]
struct SynthBenchArgs {
    #[arg(help = "Output .gputrace path to create")]
    output: PathBuf,
    #[arg(
        long,
        help = "Capture the hardware-counter oracle (ALU, DRAM copy, DRAM read, threadgroup memory, low occupancy; one kernel per encoder) instead of the FMA-ladder workload"
    )]
    counter_oracle: bool,
    #[arg(
        long,
        value_delimiter = ',',
        help = "Comma-separated FMA loop counts per kernel"
    )]
    iterations: Vec<u32>,
    #[arg(
        long = "tg-counts",
        value_delimiter = ',',
        help = "Comma-separated threadgroup counts per kernel"
    )]
    threadgroup_counts: Vec<u32>,
    #[arg(
        long,
        default_value_t = synth_bench::DEFAULT_THREADS_PER_GROUP,
        help = "Threads per threadgroup (Apple SIMD width is 32)"
    )]
    threads_per_group: u32,
}

#[derive(Debug, Args)]
struct ProfilerArgs {
    path: PathBuf,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct TimelineArgs {
    trace: PathBuf,
    #[arg(short, long, default_value = "text")]
    format: String,
    #[arg(long, default_value_t = false)]
    raw: bool,
}

#[derive(Debug, Args)]
struct TimingProfilerArgs {
    trace: PathBuf,
    #[arg(short, long, default_value_t = false)]
    verbose: bool,
    #[arg(long, default_value_t = false)]
    json: bool,
    #[arg(long, default_value_t = false)]
    agxps: bool,
}

#[derive(Debug, Args)]
struct InsightsArgs {
    trace: PathBuf,
    #[arg(long)]
    min_level: Option<String>,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct KernelsArgs {
    trace: PathBuf,
    #[arg(short, long)]
    filter: Option<String>,
    #[arg(short, long)]
    verbose: bool,
    #[arg(long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct EncodersArgs {
    trace: PathBuf,
    #[arg(short, long)]
    verbose: bool,
    #[arg(long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct DependenciesArgs {
    trace: PathBuf,
    #[arg(short, long, default_value = "dot")]
    format: String,
}

#[derive(Debug, Args)]
struct ShadersArgs {
    trace: PathBuf,
    #[arg(long = "search-path")]
    search_paths: Vec<PathBuf>,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct ShaderSourceArgs {
    trace: PathBuf,
    shader: String,
    #[arg(long = "search-path")]
    search_paths: Vec<PathBuf>,
    #[arg(long, default_value_t = 4)]
    context: usize,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct ShaderHotspotsArgs {
    trace: PathBuf,
    shader: String,
    #[arg(long = "search-path")]
    search_paths: Vec<PathBuf>,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct CorrelateArgs {
    trace: PathBuf,
    #[arg(long = "search-path")]
    search_paths: Vec<PathBuf>,
    #[arg(short, long)]
    verbose: bool,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct TimingArgs {
    trace: PathBuf,
    #[arg(short, long, default_value = "text")]
    format: String,
    #[arg(long, default_value_t = false)]
    agxps: bool,
}

#[derive(Debug, Args)]
struct CommandBuffersArgs {
    trace: PathBuf,
    #[arg(short, long)]
    detailed: bool,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct BufferAccessArgs {
    trace: PathBuf,
    #[arg(short, long)]
    verbose: bool,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct TreeArgs {
    trace: PathBuf,
    #[arg(long, default_value = "encoder")]
    group_by: String,
    #[arg(short, long, default_value = "text")]
    format: String,
}

#[derive(Debug, Args)]
struct GraphArgs {
    trace: PathBuf,
    #[arg(long, default_value = "dot")]
    format: String,
    #[arg(long = "type", default_value = "hierarchy")]
    graph_type: String,
    #[arg(long)]
    show_timing: bool,
    #[arg(long)]
    show_memory: bool,
}

#[derive(Debug, Args)]
struct BuffersArgs {
    #[command(subcommand)]
    command: BuffersCommand,
}

#[derive(Debug, Subcommand)]
enum BuffersCommand {
    List {
        trace: PathBuf,
        #[arg(short, long, default_value = "table")]
        format: String,
        #[arg(long, default_value = "size")]
        sort: String,
        #[arg(long)]
        min_size: Option<String>,
    },
    Inspect {
        trace: PathBuf,
        buffer: String,
        #[arg(long, default_value_t = 256)]
        bytes: usize,
        #[arg(long = "inspect-format", default_value = "hex")]
        inspect_format: String,
        #[arg(short, long, default_value = "text")]
        format: String,
    },
    Diff {
        left: PathBuf,
        right: PathBuf,
        #[arg(short, long, default_value = "text")]
        format: String,
    },
}

#[derive(Debug, Args)]
struct MarkdownArgs {
    #[command(subcommand)]
    command: MarkdownCommand,
}

#[derive(Debug, Subcommand)]
enum MarkdownCommand {
    Render { input: String },
    Analyze { trace: PathBuf },
    Diff { left: PathBuf, right: PathBuf },
    Buffers { trace: PathBuf },
    BuffersDiff { left: PathBuf, right: PathBuf },
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        CommandSet::Stats(args) => {
            let trace = TraceBundle::open(args.trace)?;
            println!("{}", serde_json::to_string_pretty(&trace.summary())?);
        }
        CommandSet::Analyze(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = analysis::analyze(&trace);
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        CommandSet::Report(args) => {
            ensure_report_profiler(&args)?;
            let output_dir = args
                .output
                .unwrap_or_else(|| default_report_output(&args.trace));
            let report = report::generate(
                &args.trace,
                &report::ReportOptions {
                    output_dir,
                    progress: true,
                },
            )?;
            println!(
                "Wrote {} markdown files to {} in {:.1} ms",
                report.files.len(),
                report.output_dir.display(),
                report.total_ms
            );
            if !report.failures.is_empty() {
                println!("{} sections failed:", report.failures.len());
                for failure in &report.failures {
                    println!("  - {}: {}", failure.section, failure.message);
                }
            }
        }
        CommandSet::AnalyzeUsage(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = analyze_usage::build(&trace)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", analyze_usage::format_text(&report)),
                "json" => print!("{}", analyze_usage::format_json(&report)?),
                "dot" => print!("{}", analyze_usage::format_dot(&report)),
                _ => return Err(crate::Error::Unsupported("unknown analyze-usage format")),
            }
        }
        CommandSet::ApiCalls(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = apicalls::report(&trace, args.kernel.as_deref())?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", apicalls::format_report(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown api-calls format")),
            }
        }
        CommandSet::ClearBuffers(args) => {
            let report = clear_buffers::inventory(&args.trace)?;
            if args.dry_run {
                match args.format.as_str() {
                    "text" | "table" => {
                        let mut text = clear_buffers::format_report(&report);
                        if !report.is_empty() {
                            text.push_str("\n\nDry run: no changes made\n");
                        }
                        print!("{text}");
                    }
                    "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                    _ => return Err(crate::Error::Unsupported("unknown clear-buffers format")),
                }
            } else if report.is_empty() {
                match args.format.as_str() {
                    "text" | "table" => print!("{}", clear_buffers::format_report(&report)),
                    "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                    _ => return Err(crate::Error::Unsupported("unknown clear-buffers format")),
                }
            } else {
                if !args.yes && !confirm_clear_buffers(&report)? {
                    println!("Cancelled");
                    return Ok(());
                }
                let run = clear_buffers::clear_report(&report)?;
                match args.format.as_str() {
                    "text" | "table" => {
                        println!(
                            "Zeroed {} buffer files ({})",
                            run.files_cleared,
                            clear_buffers::format_byte_size(run.bytes_cleared)
                        );
                    }
                    "json" => println!("{}", serde_json::to_string_pretty(&run)?),
                    _ => return Err(crate::Error::Unsupported("unknown clear-buffers format")),
                }
            }
        }
        CommandSet::Dump(args) => {
            let trace = TraceBundle::open(args.trace)?;
            if args.json {
                let report = apicalls::report(&trace, args.filter.as_deref())?;
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else if args.buffers_only {
                let report = buffers::analyze(&trace)?;
                print!("{}", buffers::format_table(&report));
            } else if args.dispatch_only {
                let report = apicalls::report(&trace, args.filter.as_deref())?;
                let report = apicalls::filter_call_kind_report(&report, "dispatch");
                print!("{}", apicalls::format_report(&report));
            } else if args.encoders_only {
                let report = commands::encoders(&trace)?;
                print!("{}", commands::format_encoders(&report, args.full));
            } else if let Some(command_buffer) = args.command_buffer {
                let report = apicalls::report(&trace, args.filter.as_deref())?;
                let filtered = apicalls::filter_command_buffer_report(&report, command_buffer);
                print!("{}", apicalls::format_report(&filtered));
            } else {
                let report = apicalls::report(&trace, args.filter.as_deref())?;
                print!("{}", apicalls::format_report(&report));
            }
        }
        CommandSet::DumpRecords(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let filter = dump::DumpFilter {
                record_type: args
                    .record_type
                    .as_deref()
                    .map(parse_record_type)
                    .transpose()?,
                text_contains: args.contains,
                start_index: args.start_index,
                limit: args.limit,
                include_hex_preview: args.hex_preview,
                max_preview_bytes: dump::DEFAULT_HEX_PREVIEW_BYTES,
            };
            let report = dump::parse_record_dump(&trace.capture_data()?, filter)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", dump::format_record_listing(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown dump-records format")),
            }
        }
        CommandSet::ExportCounters(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = counter_export::report(&trace)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", counter_export::format_report(&report)),
                "csv" | "xcode-csv" => print!("{}", counter_export::format_xcode_csv(&report)),
                "internal-csv" => print!("{}", counter_export::format_csv(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown export-counters format")),
            }
        }
        CommandSet::RawCounters(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = counter::raw_counters_report(&trace)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", counter::format_raw_counters_report(&report)),
                "csv" => print!("{}", counter::format_raw_counters_csv(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown raw-counters format")),
            }
        }
        CommandSet::RawCounterProbe(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = counter::probe_raw_counters(
                &trace,
                args.csv,
                args.metric.as_deref(),
                args.scan_files,
            )?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", counter::format_raw_counter_probe(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => {
                    return Err(crate::Error::Unsupported(
                        "unknown raw-counter-probe format",
                    ));
                }
            }
        }
        CommandSet::ProfilingAddressProbe(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = counter::probe_profiling_addresses(&trace)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", counter::format_profiling_address_probe(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => {
                    return Err(crate::Error::Unsupported(
                        "unknown profiling-address-probe format",
                    ));
                }
            }
        }
        CommandSet::ValidateCounters(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = xcode_counters::validate(&trace, args.csv, args.tolerance)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", xcode_counters::format_validation(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => {
                    return Err(crate::Error::Unsupported(
                        "unknown validate-counters format",
                    ));
                }
            }
            if report.mismatches > 0 {
                return Err(crate::Error::InvalidInput(format!(
                    "counter validation found {} mismatches",
                    report.mismatches
                )));
            }
        }
        CommandSet::Mtlb(args) => {
            run_mtlb_command(args)?;
        }
        CommandSet::MtlbInventory(args) => {
            let report = mtlb::inventory(args.path)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", mtlb::format_inventory_report(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown mtlb-inventory format")),
            }
        }
        CommandSet::MtlbStats(args) => {
            let report = mtlb::stats(args.path)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", mtlb::format_stats_report(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown mtlb-stats format")),
            }
        }
        CommandSet::MtlbFunctions(args) => {
            let report = mtlb::functions(
                args.path,
                &mtlb::MTLBFunctionsOptions {
                    filter: args.filter,
                    used_only: args.used_only,
                    include_usage: !args.no_usage,
                },
            )?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", mtlb::format_functions_report(&report)),
                "csv" => print!("{}", mtlb::export_functions_csv(&report)),
                "json" => print!("{}", mtlb::export_functions_json(&report)),
                _ => return Err(crate::Error::Unsupported("unknown mtlb-functions format")),
            }
        }
        CommandSet::Profiler(args) => {
            let report = profiler::report(args.path)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", profiler::format_report(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown profiler format")),
            }
        }
        CommandSet::XcodeMio(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = xcode_mio::report(&trace)?;
            match args.format.as_str() {
                "summary" | "text" | "table" => {
                    let summary = xcode_mio::summarize_report(&report);
                    print!("{}", xcode_mio::format_analysis_report(&summary));
                }
                "summary-json" => {
                    let summary = xcode_mio::summarize_report(&report);
                    println!("{}", serde_json::to_string_pretty(&summary)?);
                }
                "raw-text" => print!("{}", xcode_mio::format_report(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown xcode-mio format")),
            }
        }
        CommandSet::ProfilerCoverage(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = profiler::coverage_report(&trace)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", profiler::format_coverage_report(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => {
                    return Err(crate::Error::Unsupported(
                        "unknown profiler-coverage format",
                    ));
                }
            }
        }
        CommandSet::Fences(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = fences::report(&trace)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", fences::format_report(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown fences format")),
            }
        }
        CommandSet::Timeline(args) => {
            let trace = TraceBundle::open(args.trace)?;
            if args.raw {
                let report = timeline::raw_report(&trace)?;
                match args.format.as_str() {
                    "text" | "table" => print!("{}", timeline::format_raw_report(&report)),
                    "json" => print!("{}", timeline::export_raw_json(&report)?),
                    _ => return Err(crate::Error::Unsupported("unknown raw timeline format")),
                }
            } else {
                let report = timeline::report(&trace)?;
                match args.format.as_str() {
                    "text" | "table" => print!("{}", timeline::format_report(&report)),
                    "json" => print!("{}", timeline::export_json(&report)?),
                    "chrome" => print!("{}", timeline::format_chrome_trace_json(&report)?),
                    "perfetto" => print!("{}", timeline::format_perfetto_trace_json(&report)?),
                    _ => return Err(crate::Error::Unsupported("unknown timeline format")),
                }
            }
        }
        CommandSet::Kernels(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = commands::kernels(&trace, args.filter.as_deref())?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", commands::format_kernels(&report, args.verbose)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown kernels format")),
            }
        }
        CommandSet::Encoders(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = commands::encoders(&trace)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", commands::format_encoders(&report, args.verbose)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown encoders format")),
            }
        }
        CommandSet::Dependencies(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = commands::dependencies(&trace)?;
            match args.format.as_str() {
                "dot" => print!("{}", commands::format_dependencies_dot(&report)),
                "text" | "table" => print!("{}", commands::format_dependencies(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown dependencies format")),
            }
        }
        CommandSet::Shaders(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let search_paths = if args.search_paths.is_empty() {
                shaders::default_search_paths()
            } else {
                args.search_paths
            };
            let report = shaders::report(&trace, &search_paths)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", shaders::format_report(&report)),
                "csv" => print!("{}", shaders::format_csv(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown shaders format")),
            }
        }
        CommandSet::ShaderSource(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let search_paths = if args.search_paths.is_empty() {
                shaders::default_search_paths()
            } else {
                args.search_paths
            };
            let report = shaders::source(&trace, &args.shader, &search_paths, args.context)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", shaders::format_source(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown shader-source format")),
            }
        }
        CommandSet::ShaderHotspots(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let search_paths = if args.search_paths.is_empty() {
                shaders::default_search_paths()
            } else {
                args.search_paths
            };
            let report = shaders::hotspot_report(&trace, &args.shader, &search_paths)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", shaders::format_hotspot_report(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown shader-hotspots format")),
            }
        }
        CommandSet::Correlate(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let search_paths = if args.search_paths.is_empty() {
                shaders::default_search_paths()
            } else {
                args.search_paths
            };
            let report = correlate::report(&trace, &search_paths)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", correlate::format_report(&report, args.verbose)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown correlate format")),
            }
        }
        CommandSet::Timing(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = if args.agxps {
                let profiler_summary = profiler::stream_data_summary(&trace.path).ok();
                let xcode_mio_summary =
                    xcode_mio::agxps_analysis_report(&trace, profiler_summary.as_ref()).ok();
                timing::report_with_context(
                    &trace,
                    profiler_summary.as_ref(),
                    xcode_mio_summary.as_ref(),
                )?
            } else {
                timing::report(&trace)?
            };
            match args.format.as_str() {
                "text" | "table" => print!("{}", timing::format_report(&report)),
                "csv" => print!("{}", timing::format_csv(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown timing format")),
            }
        }
        CommandSet::TimingProfiler(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = if args.agxps {
                let profiler_summary = profiler::stream_data_summary(&trace.path).ok();
                let xcode_mio_summary =
                    xcode_mio::agxps_analysis_report(&trace, profiler_summary.as_ref()).ok();
                timing::report_with_context(
                    &trace,
                    profiler_summary.as_ref(),
                    xcode_mio_summary.as_ref(),
                )?
            } else {
                timing::report(&trace)?
            };
            if args.json {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                print!("{}", timing::format_report(&report));
                if args.verbose {
                    print!("\n=== Detailed Information ===\n");
                    println!("Data Source: {}", trace.path.display());
                    println!("Encoders with timing: {}", report.encoders.len());
                    println!("Dispatches with timing: {}", report.dispatch_count);
                    println!("Timing source: {}", report.source);
                }
            }
        }
        CommandSet::CommandBuffers(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = commands::command_buffers(&trace)?;
            match args.format.as_str() {
                "text" | "table" => print!(
                    "{}",
                    commands::format_command_buffers(&report, args.detailed)
                ),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown command-buffers format")),
            }
        }
        CommandSet::BufferAccess(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = commands::buffer_access(&trace)?;
            match args.format.as_str() {
                "text" | "table" => {
                    print!("{}", commands::format_buffer_access(&report, args.verbose))
                }
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown buffer-access format")),
            }
        }
        CommandSet::Tree(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = commands::tree(&trace, &args.group_by)?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", commands::format_tree(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown tree format")),
            }
        }
        CommandSet::Graph(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let output = graphing::generate(
                &trace,
                &args.graph_type,
                &args.format,
                args.show_timing,
                args.show_memory,
            )?;
            print!("{output}");
        }
        CommandSet::Buffers(args) => match args.command {
            BuffersCommand::List {
                trace,
                format,
                sort,
                min_size,
            } => {
                let trace = TraceBundle::open(trace)?;
                let options = buffers::BufferListOptions {
                    sort_by: Some(sort),
                    min_size: min_size.as_deref().map(buffers::parse_size).transpose()?,
                };
                let report = buffers::analyze_with_options(&trace, &options)?;
                match format.as_str() {
                    "table" | "text" => print!("{}", buffers::format_table(&report)),
                    "csv" => print!("{}", buffers::format_csv(&report)),
                    "markdown" => print!("{}", buffers::markdown_report(&report)),
                    "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                    _ => {
                        return Err(crate::Error::Unsupported("unknown buffers list format"));
                    }
                }
            }
            BuffersCommand::Inspect {
                trace,
                buffer,
                bytes,
                inspect_format,
                format,
            } => {
                let trace = TraceBundle::open(trace)?;
                let report = buffers::inspect(&trace, &buffer, bytes, &inspect_format)?;
                match format.as_str() {
                    "text" | "table" => print!("{}", buffers::format_inspection(&report)),
                    "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                    _ => {
                        return Err(crate::Error::Unsupported("unknown buffers inspect format"));
                    }
                }
            }
            BuffersCommand::Diff {
                left,
                right,
                format,
            } => {
                let left = TraceBundle::open(left)?;
                let right = TraceBundle::open(right)?;
                let report = buffers::diff(&left, &right)?;
                match format.as_str() {
                    "text" | "table" => print!("{}", buffers::format_diff(&report)),
                    "markdown" => print!("{}", buffers::markdown_diff(&report)),
                    "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                    _ => {
                        return Err(crate::Error::Unsupported("unknown buffers diff format"));
                    }
                }
            }
        },
        CommandSet::BufferTimeline(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = buffer_timeline::analyze(&trace);
            match args.format.as_str() {
                "ascii" => print!("{}", buffer_timeline::format_ascii(&report, args.width)),
                "summary" => print!("{}", buffer_timeline::format_summary(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown buffer timeline format")),
            }
        }
        CommandSet::Insights(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = insights::report(&trace, args.min_level.as_deref())?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", insights::format_report(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown insights format")),
            }
        }
        CommandSet::Diff(args) => {
            validate_diff_args(&args)?;
            let limit = args.limit;
            let min_delta_us = args.min_delta_us.unwrap_or_default();
            let only_encoder = args.only_encoder;
            let only_function = args.only_function.clone();
            let md_out = args.md_out.clone();
            let format_arg = args.format.clone();
            let markdown_flag = args.markdown;
            let json_flag = args.json;
            let csv_flag = args.csv;
            let by = args.by.clone();
            let profile_only = csv_flag || format_arg.as_deref() == Some("csv");
            let show_matches = args.show_matches;
            let show_unmatched = args.show_unmatched;
            let show_occurrences = args.show_occurrences;
            let explain = args.explain;
            let quick = args.quick;
            let by_encoder = args.by_encoder;
            let (left, right, discover_note) = resolve_diff_inputs(args)?;
            let report = diff::diff_paths_with_options(
                left,
                right,
                &diff::DiffOptions {
                    profile: diff::ProfileDiffOptions {
                        limit,
                        min_delta_us,
                        only_encoder,
                        only_function,
                    },
                    profile_only,
                },
            )?;
            if let Some(path) = md_out.as_ref() {
                let mut text = String::new();
                if let Some(note) = &discover_note {
                    text.push_str(&format!("<!-- {note} -->\n\n"));
                }
                text.push_str(&markdown::diff_report_with_limit(&report, limit));
                fs::write(path, text)?;
            }

            let format = format_arg.as_deref().unwrap_or(if csv_flag {
                "csv"
            } else if markdown_flag {
                "markdown"
            } else {
                "text"
            });
            if json_flag {
                println!("{}", serde_json::to_string_pretty(&report)?);
            } else {
                match format {
                    "markdown" | "md" => {
                        if let Some(note) = &discover_note {
                            println!("{note}\n");
                        }
                        print!("{}", markdown::diff_report_with_limit(&report, limit));
                    }
                    "text" | "table" => print!(
                        "{}",
                        diff::format_profile_text(
                            &report,
                            &diff::ProfileTextOptions {
                                by: by.as_deref(),
                                show_matches,
                                show_unmatched,
                                show_occurrences,
                                explain,
                                quick,
                                by_encoder,
                                limit,
                            },
                        )?
                    ),
                    "csv" => print!(
                        "{}",
                        diff::format_profile_csv(&report, by.as_deref(), limit)?
                    ),
                    "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                    _ => return Err(crate::Error::Unsupported("unknown diff format")),
                }
            }
        }
        CommandSet::Markdown(args) => match args.command {
            MarkdownCommand::Render { input } => {
                print!("{}", markdown::render(&input));
            }
            MarkdownCommand::Analyze { trace } => {
                let trace = TraceBundle::open(trace)?;
                let report = analysis::analyze(&trace);
                print!("{}", markdown::analysis_report(&report));
            }
            MarkdownCommand::Diff { left, right } => {
                let report = diff::diff_paths(left, right)?;
                print!("{}", markdown::diff_report(&report));
            }
            MarkdownCommand::Buffers { trace } => {
                let trace = TraceBundle::open(trace)?;
                let report = buffers::analyze(&trace)?;
                print!("{}", buffers::markdown_report(&report));
            }
            MarkdownCommand::BuffersDiff { left, right } => {
                let left = TraceBundle::open(left)?;
                let right = TraceBundle::open(right)?;
                let report = buffers::diff(&left, &right)?;
                print!("{}", buffers::markdown_diff(&report));
            }
        },
        CommandSet::Profile(args) => {
            let output_dir = args
                .output
                .unwrap_or_else(|| default_profile_output(&args.trace));
            let report = replay_service::profile(&replay_service::ProfileOptions {
                trace: args.trace.clone(),
                output_dir: output_dir.clone(),
                stdout_log: args.stdout_log,
                stderr_log: args.stderr_log,
            })?;
            match args.format.as_str() {
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => {
                    let dir_str = report
                        .gpuprofiler_raw
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|| "(none)".to_owned());
                    println!(
                        "MTLReplayer profile: trace={} output={} gpuprofiler_raw={} streamData={} elapsed_ms={:.1} open_exit={}",
                        report.trace.display(),
                        report.output_dir.display(),
                        dir_str,
                        report.has_stream_data,
                        report.elapsed_ms,
                        report.open_exit_code
                    );
                    if !report.has_stream_data {
                        return Err(crate::Error::InvalidInput(format!(
                            "MTLReplayer exited but no streamData was produced under {}",
                            report.output_dir.display()
                        )));
                    }
                }
            }
        }
        CommandSet::Version(args) => {
            let info = serde_json::json!({
                "version": env!("CARGO_PKG_VERSION"),
                "package": env!("CARGO_PKG_NAME"),
            });
            if args.json {
                println!("{}", serde_json::to_string_pretty(&info)?);
            } else {
                println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
            }
        }
        CommandSet::XcodeCommandCosts(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = xcode_command_costs::compare(&trace, args.table)?;
            match args.format.as_str() {
                "summary" | "text" | "table" => {
                    print!("{}", xcode_command_costs::format_summary(&report, args.top))
                }
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => {
                    return Err(crate::Error::Unsupported(
                        "unknown xcode-command-costs format",
                    ));
                }
            }
        }
        CommandSet::XcodeCostScan(args) => {
            let report = xcode_cost_scan::scan(&xcode_cost_scan::XcodeCostScanOptions {
                profiler_dir: args.profiler,
                table: args.table,
                targets: args.targets,
                include_defaults: !args.no_defaults,
                unaligned: args.unaligned,
                max_hits_per_target: args.max_hits_per_target,
            })?;
            match args.format.as_str() {
                "summary" | "text" | "table" => {
                    print!("{}", xcode_cost_scan::format_summary(&report, args.top))
                }
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown xcode-cost-scan format")),
            }
        }
        CommandSet::XcodeCounters(args) => {
            let trace = TraceBundle::open(args.trace)?;
            let report = xcode_counters::parse(&trace, args.csv)?;
            match args.format.as_str() {
                "summary" | "text" | "table" => print!(
                    "{}",
                    xcode_counters::format_summary(&report, args.metric.as_deref(), args.top)
                ),
                "detailed" => print!(
                    "{}",
                    xcode_counters::format_detailed(&report, args.metric.as_deref(), args.top)
                ),
                "metrics" => print!("{}", xcode_counters::format_metric_inventory(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown xcode-counters format")),
            }
        }
        CommandSet::SynthBench(args) if args.counter_oracle => {
            let kernels = synth_bench::run_counter_oracle(&args.output)?;
            print!("{}", synth_bench::format_counter_oracle_plan(&kernels));
            println!("\nWrote .gputrace to {}", args.output.display());
        }
        CommandSet::SynthBench(args) => {
            let iterations = if args.iterations.is_empty() {
                synth_bench::DEFAULT_ITERATIONS.to_vec()
            } else {
                args.iterations
            };
            let threadgroup_counts = if args.threadgroup_counts.is_empty() {
                synth_bench::DEFAULT_THREADGROUP_COUNTS.to_vec()
            } else {
                args.threadgroup_counts
            };
            let options = synth_bench::SynthBenchOptions {
                output: args.output,
                iterations,
                threadgroup_counts,
                threads_per_group: args.threads_per_group,
            };
            let plan = synth_bench::run(&options)?;
            print!("{}", synth_bench::format_plan(&plan));
            println!("\nWrote .gputrace to {}", options.output.display());
        }
    }
    Ok(())
}

fn validate_diff_args(args: &DiffArgs) -> Result<()> {
    if args.json && args.csv {
        return Err(crate::Error::InvalidInput(
            "--json and --csv are mutually exclusive".to_owned(),
        ));
    }
    if args.limit == 0 {
        return Err(crate::Error::InvalidInput("--limit must be > 0".to_owned()));
    }
    if args.min_delta_us.is_some_and(|value| value < 0) {
        return Err(crate::Error::InvalidInput(
            "--min-delta-us must be >= 0".to_owned(),
        ));
    }
    validate_diff_by(args.by.as_deref())?;
    if args.left.is_some() != args.right.is_some() {
        return Err(crate::Error::InvalidInput(
            "expected 0 or 2 positional traces, got 1".to_owned(),
        ));
    }
    if (args.left_flag.is_some()) != (args.right_flag.is_some()) {
        return Err(crate::Error::InvalidInput(
            "--left and --right must be provided together".to_owned(),
        ));
    }
    if (args.left.is_some() || args.right.is_some())
        && (args.left_flag.is_some() || args.right_flag.is_some())
    {
        return Err(crate::Error::InvalidInput(
            "positional traces cannot be combined with --left/--right".to_owned(),
        ));
    }
    if (args.left.is_some() || args.right.is_some()) && args.bench_dir.is_some() {
        return Err(crate::Error::InvalidInput(
            "positional traces cannot be combined with --bench-dir".to_owned(),
        ));
    }
    let text_only_flags = args.show_matches
        || args.show_unmatched
        || args.show_occurrences
        || args.explain
        || args.by_encoder;
    if args.json && text_only_flags {
        return Err(crate::Error::InvalidInput(
            "--json cannot be combined with text-only flags (--show-*, --explain, --by-encoder)"
                .to_owned(),
        ));
    }
    if args.csv {
        if args.quick {
            return Err(crate::Error::InvalidInput(
                "--quick cannot be combined with --csv".to_owned(),
            ));
        }
        if text_only_flags {
            return Err(crate::Error::InvalidInput(
                "--csv cannot be combined with text-only flags (--show-*, --explain, --by-encoder)"
                    .to_owned(),
            ));
        }
        if args
            .by
            .as_deref()
            .is_some_and(|value| value.trim().contains(','))
        {
            return Err(crate::Error::InvalidInput(
                "--csv requires a single --by view".to_owned(),
            ));
        }
    }
    if args.quick {
        if args.json {
            return Err(crate::Error::InvalidInput(
                "--quick cannot be combined with --json".to_owned(),
            ));
        }
        if args
            .by
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
        {
            return Err(crate::Error::InvalidInput(
                "--quick cannot be combined with --by".to_owned(),
            ));
        }
        if args.show_matches || args.show_unmatched || args.show_occurrences || args.explain {
            return Err(crate::Error::InvalidInput(
                "--quick cannot be combined with --show-matches/--show-unmatched/--show-occurrences/--explain"
                    .to_owned(),
            ));
        }
    }
    if args.by_encoder
        && args
            .by
            .as_deref()
            .is_some_and(|value| !value.trim().is_empty())
    {
        return Err(crate::Error::InvalidInput(
            "--by-encoder cannot be combined with --by".to_owned(),
        ));
    }
    Ok(())
}

fn validate_diff_by(by: Option<&str>) -> Result<()> {
    let Some(by) = by.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(());
    };
    let valid = matches!(
        by,
        "function"
            | "encoder"
            | "pipeline"
            | "timeline-windows"
            | "dispatch"
            | "unmatched"
            | "matches"
            | "occurrences"
    );
    if valid {
        Ok(())
    } else {
        Err(crate::Error::InvalidInput(format!(
            "invalid --by value {by:?}; expected one of function, encoder, pipeline, timeline-windows, dispatch, unmatched, matches, occurrences"
        )))
    }
}

fn default_profile_output(trace_path: &Path) -> PathBuf {
    embedded_profile_output(trace_path)
}

fn default_report_output(trace_path: &Path) -> PathBuf {
    if trace_path.is_dir() {
        trace_path.join("gputrace-report")
    } else {
        let parent = trace_path.parent().unwrap_or_else(|| Path::new("."));
        let stem = trace_path
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("trace");
        parent.join(format!("{stem}-report"))
    }
}

fn embedded_profile_output(trace_path: &Path) -> PathBuf {
    if trace_path.is_dir() {
        trace_path.join("gputrace-profile")
    } else {
        let parent = trace_path.parent().unwrap_or_else(|| Path::new("."));
        let stem = trace_path
            .file_stem()
            .and_then(OsStr::to_str)
            .unwrap_or("trace");
        parent.join(format!("{stem}-profile-cache"))
    }
}

fn ensure_report_profiler(args: &ReportArgs) -> Result<()> {
    if let Some(profiler) = &args.profiler {
        set_profiler_dir(profiler);
        return Ok(());
    }

    if let Some(existing) = profiler::find_profiler_directory(&args.trace)
        && existing.join("streamData").is_file()
    {
        set_profiler_dir(&existing);
        return Ok(());
    }

    let output_dir = embedded_profile_output(&args.trace);
    println!(
        "No profiler data found; profiling now into {}",
        output_dir.display()
    );
    let profile = replay_service::profile(&replay_service::ProfileOptions {
        trace: args.trace.clone(),
        output_dir: output_dir.clone(),
        stdout_log: None,
        stderr_log: None,
    })?;
    let Some(raw_dir) = profile.gpuprofiler_raw else {
        return Err(crate::Error::InvalidInput(format!(
            "MTLReplayer exited but no .gpuprofiler_raw directory was produced under {}",
            output_dir.display()
        )));
    };
    if !profile.has_stream_data {
        return Err(crate::Error::InvalidInput(format!(
            "MTLReplayer exited but no streamData was produced under {}",
            raw_dir.display()
        )));
    }
    println!(
        "Profiled trace in {:.1} ms; using {}",
        profile.elapsed_ms,
        raw_dir.display()
    );
    set_profiler_dir(&raw_dir);
    Ok(())
}

fn set_profiler_dir(path: &Path) {
    // SAFETY: single-threaded CLI startup; downstream code reads this env var via
    // `find_profiler_directory` to resolve the chosen profiler directory.
    unsafe {
        std::env::set_var("GPUTRACE_PROFILER_DIR", path);
    }
}

fn resolve_diff_inputs(args: DiffArgs) -> Result<(PathBuf, PathBuf, Option<String>)> {
    if args.left.is_some() && args.left_flag.is_some() {
        return Err(crate::Error::InvalidInput(
            "positional traces cannot be combined with --left/--right".to_owned(),
        ));
    }
    if args.right.is_some() && args.right_flag.is_some() {
        return Err(crate::Error::InvalidInput(
            "positional traces cannot be combined with --left/--right".to_owned(),
        ));
    }
    let explicit_left = args.left_flag.or(args.left);
    let explicit_right = args.right_flag.or(args.right);
    if explicit_left.is_some() != explicit_right.is_some() {
        return Err(crate::Error::InvalidInput(
            "diff requires both left and right trace paths".to_owned(),
        ));
    }
    if let (Some(left), Some(right)) = (explicit_left, explicit_right) {
        let note = args
            .bench_dir
            .as_ref()
            .map(|_| "--bench-dir ignored because --left/--right were provided".to_owned());
        return Ok((left, right, note));
    }
    let bench_dir = args.bench_dir.ok_or_else(|| {
        crate::Error::InvalidInput(
            "diff requires <left> <right>, --left/--right, or --bench-dir".to_owned(),
        )
    })?;
    let pair = discover_bench_pair(&bench_dir)?;
    let note = format!(
        "auto-pair: stem={} left={} right={}",
        pair.stem,
        pair.left
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("-"),
        pair.right
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("-")
    );
    let note = if pair.left_raw.is_some()
        || pair.right_raw.is_some()
        || pair.left_csv.is_some()
        || pair.right_csv.is_some()
    {
        format!(
            "{note} siblings(left_raw={} right_raw={} left_csv={} right_csv={})",
            option_file_name(pair.left_raw.as_deref()),
            option_file_name(pair.right_raw.as_deref()),
            option_file_name(pair.left_csv.as_deref()),
            option_file_name(pair.right_csv.as_deref())
        )
    } else {
        note
    };
    Ok((pair.left, pair.right, Some(note)))
}

fn option_file_name(path: Option<&Path>) -> String {
    path.and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("-")
        .to_owned()
}

#[derive(Debug, Clone)]
struct BenchPair {
    stem: String,
    left: PathBuf,
    right: PathBuf,
    left_raw: Option<PathBuf>,
    right_raw: Option<PathBuf>,
    left_csv: Option<PathBuf>,
    right_csv: Option<PathBuf>,
    left_mtime: SystemTime,
    right_mtime: SystemTime,
}

#[derive(Debug, Clone)]
struct BenchCandidate {
    path: PathBuf,
    mtime: SystemTime,
}

#[derive(Default)]
struct BenchGroup {
    go_perf: Vec<BenchCandidate>,
    py_perf: Vec<BenchCandidate>,
    go_raw: Vec<BenchCandidate>,
    py_raw: Vec<BenchCandidate>,
}

fn discover_bench_pair(dir: &Path) -> Result<BenchPair> {
    let mut groups = std::collections::BTreeMap::<String, BenchGroup>::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some((stem, side, kind)) = classify_bench_trace_name(&name) else {
            continue;
        };
        let candidate = BenchCandidate {
            path: entry.path(),
            mtime: entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH),
        };
        let group = groups.entry(stem).or_default();
        match (side, kind) {
            ("go", "perf") => group.go_perf.push(candidate),
            ("py", "perf") => group.py_perf.push(candidate),
            ("go", "raw") => group.go_raw.push(candidate),
            ("py", "raw") => group.py_raw.push(candidate),
            _ => {}
        }
    }

    let mut pairs = groups
        .into_iter()
        .filter_map(|(stem, group)| select_bench_pair(dir, stem, group))
        .collect::<Vec<_>>();
    pairs.sort_by(|left, right| {
        let left_time = left.left_mtime.max(left.right_mtime);
        let right_time = right.left_mtime.max(right.right_mtime);
        right_time
            .cmp(&left_time)
            .then_with(|| left.stem.cmp(&right.stem))
    });
    pairs.into_iter().next().ok_or_else(|| {
        crate::Error::InvalidInput(format!(
            "no Go/Python trace pair found in {}",
            dir.display()
        ))
    })
}

fn classify_bench_trace_name(name: &str) -> Option<(String, &'static str, &'static str)> {
    if !name.ends_with(".gputrace") {
        return None;
    }
    let perf = name.ends_with("-perfdata.gputrace");
    if let Some(stem) = classify_bench_side(name, "_Go") {
        return Some((stem, "go", if perf { "perf" } else { "raw" }));
    }
    if let Some(stem) = classify_bench_side(name, "_Python") {
        return Some((stem, "py", if perf { "perf" } else { "raw" }));
    }
    None
}

fn classify_bench_side(name: &str, marker: &str) -> Option<String> {
    let marker_index = name.rfind(marker)?;
    let after_marker = &name[marker_index + marker.len()..];
    if after_marker.contains('_') {
        Some(name[..marker_index].to_owned())
    } else {
        None
    }
}

fn select_bench_pair(dir: &Path, stem: String, group: BenchGroup) -> Option<BenchPair> {
    let go_perf = newest_bench_candidate(&group.go_perf);
    let py_perf = newest_bench_candidate(&group.py_perf);
    let go_raw = newest_bench_candidate(&group.go_raw);
    let py_raw = newest_bench_candidate(&group.py_raw);

    let mut pair = BenchPair {
        stem,
        left: PathBuf::new(),
        right: PathBuf::new(),
        left_raw: go_raw.map(|candidate| candidate.path.clone()),
        right_raw: py_raw.map(|candidate| candidate.path.clone()),
        left_csv: None,
        right_csv: None,
        left_mtime: SystemTime::UNIX_EPOCH,
        right_mtime: SystemTime::UNIX_EPOCH,
    };

    if let (Some(left), Some(right)) = (go_perf, py_perf) {
        pair.left = left.path.clone();
        pair.right = right.path.clone();
        pair.left_mtime = left.mtime;
        pair.right_mtime = right.mtime;
    } else if let (Some(left), Some(right)) = (go_raw, py_raw) {
        pair.left = left.path.clone();
        pair.right = right.path.clone();
        pair.left_mtime = left.mtime;
        pair.right_mtime = right.mtime;
    } else {
        return None;
    }
    pair.left_csv = find_sibling_counter_csv(dir, &pair.left);
    pair.right_csv = find_sibling_counter_csv(dir, &pair.right);
    Some(pair)
}

fn find_sibling_counter_csv(dir: &Path, trace_path: &Path) -> Option<PathBuf> {
    let base = trace_path.file_name()?.to_str()?;
    let stem = base
        .strip_suffix("-perfdata.gputrace")
        .or_else(|| base.strip_suffix(".gputrace"))?;
    let csv = dir.join(format!("{stem}_counters.csv"));
    csv.exists().then_some(csv)
}

fn newest_bench_candidate(candidates: &[BenchCandidate]) -> Option<&BenchCandidate> {
    candidates.iter().max_by_key(|candidate| candidate.mtime)
}

fn run_mtlb_command(args: MtlbArgs) -> Result<()> {
    match args.command {
        Some(MtlbCommand::List(args)) | Some(MtlbCommand::Inventory(args)) => {
            print_mtlb_inventory(args.path, &args.format, "mtlb inventory")
        }
        Some(MtlbCommand::Info(args)) => print_mtlb_info(args.path, &args.format),
        Some(MtlbCommand::Stats(args)) => print_mtlb_stats(args.path, &args.format, "mtlb stats"),
        Some(MtlbCommand::Functions(args)) => print_mtlb_functions(
            args.path,
            args.filter,
            args.used_only,
            args.no_usage,
            &args.format,
            "mtlb functions",
        ),
        Some(MtlbCommand::ExportFunctions(args)) => print_mtlb_functions(
            args.path,
            args.filter,
            args.used_only,
            args.no_usage,
            &args.format,
            "mtlb export-functions",
        ),
        Some(MtlbCommand::Extract(args)) => {
            let report = mtlb::extract(
                args.path,
                &mtlb::MTLBExtractOptions {
                    output: args.output,
                    library: args.library,
                    all: args.all,
                    output_dir: args.output_dir,
                },
            )?;
            match args.format.as_str() {
                "text" | "table" => print!("{}", mtlb::format_extract_report(&report)),
                "json" => println!("{}", serde_json::to_string_pretty(&report)?),
                _ => return Err(crate::Error::Unsupported("unknown mtlb extract format")),
            }
            Ok(())
        }
        None => {
            let path = args.path.ok_or_else(|| {
                crate::Error::InvalidInput("mtlb requires a path or subcommand".to_owned())
            })?;
            print_mtlb_info(path, &args.format)
        }
    }
}

fn print_mtlb_info(path: PathBuf, format: &str) -> Result<()> {
    let metadata = std::fs::metadata(&path).map_err(crate::Error::Io)?;
    if metadata.is_dir() {
        let report = mtlb::scan_bundle(path)?;
        match format {
            "text" | "table" => print!("{}", mtlb::format_bundle_report(&report)),
            "json" => println!("{}", serde_json::to_string_pretty(&report)?),
            _ => return Err(crate::Error::Unsupported("unknown mtlb info format")),
        }
    } else {
        let report = mtlb::inspect_file(path)?;
        match format {
            "text" | "table" => print!("{}", mtlb::format_file_report(&report)),
            "json" => println!("{}", serde_json::to_string_pretty(&report)?),
            _ => return Err(crate::Error::Unsupported("unknown mtlb info format")),
        }
    }
    Ok(())
}

fn print_mtlb_inventory(path: PathBuf, format: &str, command_name: &'static str) -> Result<()> {
    let report = mtlb::inventory(path)?;
    match format {
        "text" | "table" => print!("{}", mtlb::format_inventory_report(&report)),
        "json" => println!("{}", serde_json::to_string_pretty(&report)?),
        _ => {
            return Err(crate::Error::Unsupported(match command_name {
                "mtlb inventory" => "unknown mtlb inventory format",
                _ => "unknown mtlb list format",
            }));
        }
    }
    Ok(())
}

fn print_mtlb_stats(path: PathBuf, format: &str, command_name: &'static str) -> Result<()> {
    let report = mtlb::stats(path)?;
    match format {
        "text" | "table" => print!("{}", mtlb::format_stats_report(&report)),
        "json" => println!("{}", serde_json::to_string_pretty(&report)?),
        _ => {
            return Err(crate::Error::Unsupported(match command_name {
                "mtlb stats" => "unknown mtlb stats format",
                _ => "unknown mtlb-stats format",
            }));
        }
    }
    Ok(())
}

fn print_mtlb_functions(
    path: PathBuf,
    filter: Option<String>,
    used_only: bool,
    no_usage: bool,
    format: &str,
    command_name: &'static str,
) -> Result<()> {
    let report = mtlb::functions(
        path,
        &mtlb::MTLBFunctionsOptions {
            filter,
            used_only,
            include_usage: !no_usage,
        },
    )?;
    match format {
        "text" | "table" => print!("{}", mtlb::format_functions_report(&report)),
        "csv" => print!("{}", mtlb::export_functions_csv(&report)),
        "json" => print!("{}", mtlb::export_functions_json(&report)),
        _ => {
            return Err(crate::Error::Unsupported(match command_name {
                "mtlb export-functions" => "unknown mtlb export-functions format",
                _ => "unknown mtlb functions format",
            }));
        }
    }
    Ok(())
}

fn confirm_clear_buffers(report: &clear_buffers::ClearBuffersReport) -> Result<bool> {
    print!(
        "{}\n\nZero out all buffer files? [y/N]: ",
        clear_buffers::format_report(report)
    );
    io::stdout().flush()?;

    let mut response = String::new();
    io::stdin().read_line(&mut response)?;
    let response = response.trim().to_ascii_lowercase();
    Ok(matches!(response.as_str(), "y" | "yes"))
}

fn parse_record_type(value: &str) -> Result<RecordType> {
    match value {
        "C" => Ok(RecordType::C),
        "C@3ul@3ul" | "C3ul" => Ok(RecordType::C3ul),
        "CS" => Ok(RecordType::CS),
        "CSuwuw" => Ok(RecordType::CSuwuw),
        "Ct" => Ok(RecordType::Ct),
        "Ctt" => Ok(RecordType::Ctt),
        "CtU" => Ok(RecordType::CtU),
        "Ctulul" => Ok(RecordType::Ctulul),
        "CU" => Ok(RecordType::CU),
        "Cui" => Ok(RecordType::Cui),
        "Cul" => Ok(RecordType::Cul),
        "Culul" => Ok(RecordType::Culul),
        "Cut" => Ok(RecordType::Cut),
        "CUt" => Ok(RecordType::CUt),
        "CUUU" => Ok(RecordType::CUUU),
        "Cuw" => Ok(RecordType::Cuw),
        "Ci" => Ok(RecordType::Ci),
        "CiulSl" => Ok(RecordType::CiulSl),
        "Ciulul" => Ok(RecordType::Ciulul),
        "Unknown" | "unknown" => Ok(RecordType::Unknown),
        _ => Err(crate::Error::InvalidInput(format!(
            "unknown record type: {value}"
        ))),
    }
}
