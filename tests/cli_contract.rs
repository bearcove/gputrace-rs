use clap::{CommandFactory, Parser};
use gputrace_rs::cli::Cli;

fn render_help(path: &[&str]) -> String {
    let mut command = Cli::command();
    for segment in path {
        command = command
            .find_subcommand_mut(segment)
            .unwrap_or_else(|| panic!("missing subcommand {segment}"))
            .clone();
    }

    let mut buffer = Vec::new();
    command.write_long_help(&mut buffer).unwrap();
    String::from_utf8(buffer).unwrap()
}

#[test]
fn top_level_help_lists_only_report() {
    let help = render_help(&[]);

    assert!(
        help.contains("report"),
        "top-level help should mention report\n{help}"
    );
    // `report` is the one public workflow; internal subcommands stay hidden.
    for internal in [
        "dump-records",
        "raw-counters",
        "xcode-counters",
        "profiler-coverage",
        "clear-buffers",
    ] {
        assert!(
            !help.contains(internal),
            "top-level help should not list internal {internal}\n{help}"
        );
    }
}

#[test]
fn report_help_mentions_markdown_output_directory() {
    let help = render_help(&["report"]);

    assert!(help.contains("report"));
    assert!(help.contains("<TRACE>"));
    assert!(help.contains("--output <OUTPUT>"));
    assert!(help.contains("Markdown"));
}

#[test]
fn analyze_usage_help_keeps_expected_formats() {
    let help = render_help(&["analyze-usage"]);

    assert!(help.contains("analyze-usage"));
    assert!(help.contains("<TRACE>"));
    assert!(help.contains("--format <FORMAT>"));
    assert!(help.contains("[default: text]"));
}

#[test]
fn clear_buffers_help_keeps_current_safety_flags() {
    let help = render_help(&["clear-buffers"]);

    assert!(help.contains("clear-buffers"));
    assert!(help.contains("<TRACE>"));
    assert!(help.contains("--dry-run"));
    assert!(help.contains("--yes"));
    assert!(help.contains("--format <FORMAT>"));
}

#[test]
fn timeline_help_mentions_raw_and_export_formats() {
    let help = render_help(&["timeline"]);

    assert!(help.contains("timeline"));
    assert!(help.contains("<TRACE>"));
    assert!(help.contains("--raw"));
    assert!(help.contains("--format <FORMAT>"));
}

#[test]
fn export_counters_help_mentions_trace_and_formats() {
    let help = render_help(&["export-counters"]);

    assert!(help.contains("export-counters"));
    assert!(help.contains("<TRACE>"));
    assert!(help.contains("--format <FORMAT>"));
    assert!(help.contains("[default: csv]"));
    assert!(help.contains("APS counter sample rows"));
    assert!(help.contains("metric_metadata"));
    assert!(help.contains("xcode-counters"));
}

#[test]
fn raw_counters_help_mentions_trace_and_formats() {
    let help = render_help(&["raw-counters"]);

    assert!(help.contains("raw-counters"));
    assert!(help.contains("<TRACE>"));
    assert!(help.contains("--format <FORMAT>"));
    assert!(help.contains("[default: text]"));
    assert!(help.contains("APSCounterData"));
    assert!(help.contains("derived.js"));
    assert!(help.contains("independent of Xcode Counters.csv"));
}

#[test]
fn profiler_coverage_help_mentions_byte_coverage() {
    let help = render_help(&["profiler-coverage"]);

    assert!(help.contains("profiler-coverage"));
    assert!(help.contains("<TRACE>"));
    assert!(help.contains("--format <FORMAT>"));
    assert!(help.contains("[default: text]"));
    assert!(help.contains("Profiling_f_*"));
    assert!(help.contains("opaque"));
}

#[test]
fn validate_counters_help_mentions_csv_and_tolerance() {
    let help = render_help(&["validate-counters"]);

    assert!(help.contains("<TRACE>"));
    assert!(help.contains("--csv <CSV>"));
    assert!(help.contains("--tolerance <TOLERANCE>"));
}

#[test]
fn perfcounters_validate_alias_parses_to_validate_counters() {
    let cli = Cli::try_parse_from([
        "gputrace",
        "perfcounters-validate",
        "trace.gputrace",
        "--csv",
        "Counters.csv",
    ]);
    assert!(cli.is_ok());
}

#[test]
fn raw_counters_json_parses() {
    let cli = Cli::try_parse_from([
        "gputrace",
        "raw-counters",
        "trace.gputrace",
        "--format",
        "json",
    ]);
    assert!(cli.is_ok());
}

#[test]
fn dump_help_matches_go_surface() {
    let help = render_help(&["dump"]);

    assert!(help.contains("<TRACE>"));
    assert!(help.contains("--filter <FILTER>"));
    assert!(help.contains("--buffers-only"));
    assert!(help.contains("--dispatch-only"));
    assert!(help.contains("--encoders-only"));
    assert!(help.contains("--command-buffer <COMMAND_BUFFER>"));
    assert!(help.contains("--json"));
    assert!(help.contains("--full"));
}

#[test]
fn mtlb_help_lists_go_subcommands() {
    let help = render_help(&["mtlb"]);

    for subcommand in [
        "list",
        "info",
        "inventory",
        "stats",
        "functions",
        "export-functions",
        "extract",
    ] {
        assert!(
            help.contains(subcommand),
            "mtlb help should mention {subcommand}\n{help}"
        );
    }
}

#[test]
fn mtlb_extract_help_mentions_selection_and_output_flags() {
    let help = render_help(&["mtlb", "extract"]);

    assert!(help.contains("<PATH>"));
    assert!(help.contains("--output <OUTPUT>"));
    assert!(help.contains("--library <LIBRARY>"));
    assert!(help.contains("--all"));
    assert!(help.contains("--output-dir <OUTPUT_DIR>"));
    assert!(help.contains("--format <FORMAT>"));
}

#[test]
fn mtlb_export_functions_defaults_to_csv() {
    let help = render_help(&["mtlb", "export-functions"]);

    assert!(help.contains("[default: csv]"));
}

#[test]
fn timing_profiler_alias_parses() {
    let cli = Cli::try_parse_from(["gputrace", "timing-profiler", "trace.gputrace", "--json"]);
    assert!(cli.is_ok());
}

#[test]
fn xcode_counters_help_mentions_metric_and_top() {
    let help = render_help(&["xcode-counters"]);
    assert!(help.contains("<TRACE>"));
    assert!(help.contains("--metric <METRIC>"));
    assert!(help.contains("--top <TOP>"));
    assert!(help.contains("[default: summary]"));
}

#[test]
fn profile_help_mentions_capture_profile_contract() {
    let help = render_help(&["profile"]);
    assert!(help.contains("<TRACE>"));
    assert!(help.contains("--output <OUTPUT>"));
    assert!(help.contains("--stdout-log <STDOUT_LOG>"));
    assert!(help.contains("--stderr-log <STDERR_LOG>"));
    assert!(help.contains("[default: text]"));
    assert!(help.contains(".gpuprofiler_raw"));
}

#[test]
fn shaders_help_mentions_format_and_search_path() {
    let help = render_help(&["shaders"]);
    assert!(help.contains("<TRACE>"));
    assert!(help.contains("--format <FORMAT>"));
    assert!(help.contains("--search-path <SEARCH_PATHS>"));
}

#[test]
fn shader_hotspots_help_mentions_shader_and_format() {
    let help = render_help(&["shader-hotspots"]);
    assert!(help.contains("<TRACE>"));
    assert!(help.contains("<SHADER>"));
    assert!(help.contains("--format <FORMAT>"));
    assert!(help.contains("--search-path <SEARCH_PATHS>"));
}

#[test]
fn fences_help_keeps_format_flag() {
    let help = render_help(&["fences"]);

    assert!(help.contains("fences"));
    assert!(help.contains("<TRACE>"));
    assert!(help.contains("--format <FORMAT>"));
}

#[test]
fn api_calls_help_keeps_kernel_filter() {
    let help = render_help(&["api-calls"]);

    assert!(help.contains("api-calls"));
    assert!(help.contains("<TRACE>"));
    assert!(help.contains("--kernel <KERNEL>"));
    assert!(help.contains("--format <FORMAT>"));
}

#[test]
fn dump_records_help_keeps_filters() {
    let help = render_help(&["dump-records"]);

    for flag in [
        "--type <RECORD_TYPE>",
        "--contains <CONTAINS>",
        "--start-index <START_INDEX>",
        "--limit <LIMIT>",
        "--hex-preview",
    ] {
        assert!(
            help.contains(flag),
            "dump-records help missing {flag}\n{help}"
        );
    }
}

#[test]
fn diff_help_keeps_markdown_and_format_outputs() {
    let help = render_help(&["diff"]);

    assert!(help.contains("[LEFT]"));
    assert!(help.contains("[RIGHT]"));
    assert!(help.contains("--markdown"));
    assert!(help.contains("--json"));
    assert!(help.contains("--md-out <MD_OUT>"));
    assert!(help.contains("--format <FORMAT>"));
    assert!(help.contains("--limit <LIMIT>"));
    assert!(help.contains("--min-delta-us <MIN_DELTA_US>"));
    assert!(help.contains("--only-encoder <ONLY_ENCODER>"));
    assert!(help.contains("--only-function <ONLY_FUNCTION>"));
}

#[test]
fn buffers_help_lists_subcommands() {
    let help = render_help(&["buffers"]);

    for subcommand in ["list", "inspect", "diff"] {
        assert!(
            help.contains(subcommand),
            "buffers help should mention {subcommand}\n{help}"
        );
    }
}

#[test]
fn buffers_list_help_keeps_sorting_contract() {
    let help = render_help(&["buffers", "list"]);

    for flag in [
        "--format <FORMAT>",
        "--sort <SORT>",
        "--min-size <MIN_SIZE>",
    ] {
        assert!(
            help.contains(flag),
            "buffers list help missing {flag}\n{help}"
        );
    }
}

#[test]
fn buffers_inspect_help_keeps_content_flags() {
    let help = render_help(&["buffers", "inspect"]);

    for flag in [
        "--bytes <BYTES>",
        "--inspect-format <INSPECT_FORMAT>",
        "--format <FORMAT>",
    ] {
        assert!(
            help.contains(flag),
            "buffers inspect help missing {flag}\n{help}"
        );
    }
}

#[test]
fn important_top_level_commands_parse_their_existing_contracts() {
    for argv in [
        vec!["gputrace", "analyze-usage", "trace.gputrace"],
        vec![
            "gputrace",
            "analyze-usage",
            "trace.gputrace",
            "--format",
            "dot",
        ],
        vec!["gputrace", "clear-buffers", "--dry-run", "trace.gputrace"],
        vec![
            "gputrace",
            "clear-buffers",
            "--yes",
            "--format",
            "json",
            "trace.gputrace",
        ],
        vec![
            "gputrace",
            "timeline",
            "trace.gputrace",
            "--format",
            "chrome",
        ],
        vec![
            "gputrace",
            "timeline",
            "trace.gputrace",
            "--raw",
            "--format",
            "json",
        ],
        vec!["gputrace", "fences", "trace.gputrace", "--format", "json"],
        vec!["gputrace", "profiler", "trace.gputrace", "--format", "json"],
        vec![
            "gputrace",
            "profile",
            "trace.gputrace",
            "--output",
            "trace-perfdata.gputrace",
        ],
        vec![
            "gputrace",
            "diff",
            "left.gputrace",
            "right.gputrace",
            "--format",
            "markdown",
            "--md-out",
            "diff.md",
            "--limit",
            "5",
            "--min-delta-us",
            "10",
            "--only-encoder",
            "2",
            "--only-function",
            "gemm",
        ],
        vec![
            "gputrace",
            "mtlb-inventory",
            "trace.gputrace",
            "--format",
            "json",
        ],
        vec![
            "gputrace",
            "mtlb-stats",
            "trace.gputrace",
            "--format",
            "json",
        ],
        vec![
            "gputrace",
            "mtlb-functions",
            "trace.gputrace",
            "--used-only",
            "--format",
            "csv",
        ],
        vec![
            "gputrace",
            "api-calls",
            "trace.gputrace",
            "--kernel",
            "foo",
            "--format",
            "json",
        ],
        vec![
            "gputrace",
            "dump-records",
            "trace.gputrace",
            "--type",
            "Ct",
            "--contains",
            "kernel",
            "--start-index",
            "3",
            "--limit",
            "10",
            "--hex-preview",
            "--format",
            "json",
        ],
        vec![
            "gputrace",
            "buffers",
            "list",
            "trace.gputrace",
            "--format",
            "csv",
            "--sort",
            "name",
            "--min-size",
            "1MB",
        ],
        vec![
            "gputrace",
            "buffers",
            "inspect",
            "trace.gputrace",
            "buffer-1",
            "--bytes",
            "64",
            "--inspect-format",
            "float16",
            "--format",
            "json",
        ],
        vec![
            "gputrace",
            "buffers",
            "diff",
            "left.gputrace",
            "right.gputrace",
            "--format",
            "markdown",
        ],
    ] {
        Cli::try_parse_from(argv).unwrap();
    }
}

#[test]
fn clap_command_tree_is_consistent() {
    Cli::command().debug_assert();
}
