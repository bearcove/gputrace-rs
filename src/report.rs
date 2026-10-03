use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::analysis;
use crate::apicalls;
use crate::buffers;
use crate::commands;
use crate::counter;
use crate::error::Result;
use crate::hw_counters;
use crate::insights;
use crate::markdown;
use crate::profiler;
use crate::shaders;
use crate::timeline;
use crate::timing;
use crate::trace::TraceBundle;
use crate::xcode_mio;

#[derive(Debug, Clone)]
pub struct ReportOptions {
    pub output_dir: PathBuf,
    pub progress: bool,
}

#[derive(Debug, Clone)]
pub struct GeneratedReport {
    pub output_dir: PathBuf,
    pub files: Vec<PathBuf>,
    pub failures: Vec<ReportFailure>,
    pub section_timings: Vec<ReportSectionTiming>,
    pub total_ms: f64,
}

#[derive(Debug, Clone)]
pub struct ReportFailure {
    pub section: String,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct ReportSectionTiming {
    pub section: String,
    pub ms: f64,
}

struct ReportWriter {
    output_dir: PathBuf,
    progress: bool,
    files: Vec<PathBuf>,
    failures: Vec<ReportFailure>,
    section_timings: Vec<ReportSectionTiming>,
}

pub fn generate(trace_path: &Path, options: &ReportOptions) -> Result<GeneratedReport> {
    let total_start = Instant::now();
    fs::create_dir_all(&options.output_dir)?;
    let trace = TraceBundle::open(trace_path)?;
    let mut writer = ReportWriter {
        output_dir: options.output_dir.clone(),
        progress: options.progress,
        files: Vec::new(),
        failures: Vec::new(),
        section_timings: Vec::new(),
    };

    let profiler_summary_start = writer.start_section("shared profiler streamData");
    let profiler_summary = profiler::stream_data_summary(&trace.path).ok();
    match profiler::missing_profile_families(&trace.path) {
        Some(missing) if !missing.is_empty() => writer.record_failure(
            "shared profiler streamData",
            &format!(
                "MTLReplayer wrote no {} files: shader costs and counters are missing, not zero. \
                 It drops them silently when a capture holds too much GPU work (seen at ~140 ms \
                 over ~3,000 dispatches); capture a smaller window. Delete gputrace-profile to retry.",
                missing.join(" / ")
            ),
            profiler_summary_start,
        ),
        _ => writer.record_timing("shared profiler streamData", profiler_summary_start),
    }

    let raw_counters_start = writer.start_section("shared raw counters");
    let raw_counters = counter::raw_counters_report(&trace).ok();
    writer.record_timing("shared raw counters", raw_counters_start);

    let raw_probe_start = writer.start_section("shared raw counter probe");
    let raw_probe = counter::probe_raw_counters(&trace, None, None, false).ok();
    writer.record_timing("shared raw counter probe", raw_probe_start);

    let hw_start = writer.start_section("shared hardware counters");
    let hw_counter_report = hw_counters::cached_report_result(&trace.path);
    writer.record_timing("shared hardware counters", hw_start);

    let mut xcode_mio_summary = None;
    let xcode_start = writer.start_section("xcode mio");
    match xcode_mio::report_with_profiler_summary(&trace, profiler_summary.as_ref()) {
        Ok(report) => {
            let summary = xcode_mio::summarize_report(&report);
            xcode_mio_summary = Some(summary.clone());
            writer.write_section(
                "xcode-mio.md",
                "Xcode MIO",
                "xcode mio",
                xcode_mio::format_analysis_report(&summary),
                xcode_start,
            )?;
        }
        Err(error) => {
            writer.write_failure(
                "xcode-mio.md",
                "xcode mio",
                "Xcode MIO",
                &error.to_string(),
                xcode_start,
            )?;
        }
    }

    let timing_start = writer.start_section("timing");
    let timing_report = match timing::report_with_context(
        &trace,
        profiler_summary.as_ref(),
        xcode_mio_summary.as_ref(),
    ) {
        Ok(report) => {
            writer.write_section(
                "timing.md",
                "Timing",
                "timing",
                timing::format_report(&report),
                timing_start,
            )?;
            Some(report)
        }
        Err(error) => {
            writer.write_failure(
                "timing.md",
                "timing",
                "Timing",
                &error.to_string(),
                timing_start,
            )?;
            None
        }
    };

    let shader_start = writer.start_section("shaders");
    let shader_report = match shaders::report_with_context(
        &trace,
        &shaders::default_search_paths(),
        profiler_summary.as_ref(),
        raw_counters
            .as_ref()
            .and_then(|report| report.profiling_address_summary.as_ref()),
        hw_counter_report.as_deref().ok(),
    ) {
        Ok(report) => {
            writer.write_section(
                "shaders.md",
                "Shaders",
                "shaders",
                shaders::format_report(&report),
                shader_start,
            )?;
            Some(report)
        }
        Err(error) => {
            writer.write_failure(
                "shaders.md",
                "shaders",
                "Shaders",
                &error.to_string(),
                shader_start,
            )?;
            None
        }
    };

    let analysis_start = writer.start_section("analysis");
    let analysis =
        analysis::analyze_with_context(&trace, xcode_mio_summary.clone(), timing_report.clone());
    writer.write_markdown(
        "analysis.md",
        markdown::analysis_report(&analysis),
        "analysis",
        analysis_start,
    )?;

    writer.write_result("insights.md", "Insights", "insights", || {
        insights::report_with_context(
            &trace,
            None,
            xcode_mio_summary.clone(),
            timing_report.clone(),
            profiler_summary.clone(),
            shader_report.clone(),
        )
        .map(|report| insights::format_report(&report))
    })?;
    writer.write_result("profiler.md", "Profiler", "profiler", || {
        profiler::report_with_stream_data_summary(&trace.path, profiler_summary.clone())
            .map(|report| profiler::format_report(&report))
    })?;
    writer.write_result(
        "profiler-coverage.md",
        "Profiler Coverage",
        "profiler coverage",
        || {
            profiler::coverage_report_with_decoded(
                &trace,
                raw_counters.as_ref(),
                raw_probe.as_ref(),
            )
            .map(|report| profiler::format_coverage_report(&report))
        },
    )?;
    let raw_counters_format_start = writer.start_section("raw counters");
    if let Some(raw_counters) = &raw_counters {
        writer.write_section(
            "raw-counters.md",
            "Raw Counters",
            "raw counters",
            counter::format_raw_counters_report(raw_counters),
            raw_counters_format_start,
        )?;
    } else {
        writer.write_failure(
            "raw-counters.md",
            "raw counters",
            "Raw Counters",
            "raw counter report unavailable",
            raw_counters_format_start,
        )?;
    }
    let counters_start = writer.start_section("counters");
    match &hw_counter_report {
        Ok(report) => writer.write_markdown(
            "counters.md",
            hw_counters::format_markdown(report),
            "counters",
            counters_start,
        )?,
        Err(error) => writer.write_failure(
            "counters.md",
            "counters",
            "Counters",
            &format!("no limiter-pass hardware counters: {error}"),
            counters_start,
        )?,
    }
    writer.write_optional_result("buffers.md", "Buffers", "buffers", || {
        buffers::analyze(&trace).map(|report| buffers::markdown_report(&report))
    })?;
    writer.write_optional_result("timeline.md", "Timeline", "timeline", || {
        timeline::report_with_profiler_summary(&trace, profiler_summary.as_ref())
            .map(|report| timeline::format_report(&report))
    })?;
    writer.write_optional_result("api-calls.md", "API Calls", "api calls", || {
        apicalls::report(&trace, None).map(|report| apicalls::format_report(&report))
    })?;
    writer.write_optional_result("encoders.md", "Encoders", "encoders", || {
        commands::encoders_with_profiler_summary(&trace, profiler_summary.as_ref())
            .map(|report| commands::format_encoders(&report, true))
    })?;
    writer.write_optional_result(
        "command-buffers.md",
        "Command Buffers",
        "command buffers",
        || {
            commands::command_buffers(&trace)
                .map(|report| commands::format_command_buffers(&report, true))
        },
    )?;
    writer.write_optional_result("dependencies.md", "Dependencies", "dependencies", || {
        commands::dependencies(&trace).map(|report| commands::format_dependencies(&report))
    })?;

    let total_ms = elapsed_ms(total_start);
    writer.write_index(
        trace_path,
        &trace,
        IndexSources {
            analysis: &analysis,
            xcode_mio: xcode_mio_summary.as_ref(),
            raw_counters: raw_counters.as_ref(),
            hw_counters: hw_counter_report.as_deref().ok(),
        },
        total_ms,
    )?;

    Ok(GeneratedReport {
        output_dir: writer.output_dir,
        files: writer.files,
        failures: writer.failures,
        section_timings: writer.section_timings,
        total_ms,
    })
}

/// Reports whose headline numbers go into index.md.
struct IndexSources<'a> {
    analysis: &'a analysis::AnalysisReport,
    xcode_mio: Option<&'a xcode_mio::XcodeMioAnalysisReport>,
    raw_counters: Option<&'a counter::RawCountersReport>,
    hw_counters: Option<&'a hw_counters::HwCounterReport>,
}

impl ReportWriter {
    fn start_section(&self, section: &str) -> Instant {
        if self.progress {
            eprintln!("report: {section}...");
        }
        Instant::now()
    }

    fn write_result<F>(
        &mut self,
        file_name: &str,
        title: &str,
        section: &str,
        build: F,
    ) -> Result<()>
    where
        F: FnOnce() -> Result<String>,
    {
        let start = self.start_section(section);
        match build() {
            Ok(body) => self.write_section(file_name, title, section, body, start),
            Err(error) => self.write_failure(file_name, section, title, &error.to_string(), start),
        }
    }

    fn write_optional_result<F>(
        &mut self,
        file_name: &str,
        title: &str,
        section: &str,
        build: F,
    ) -> Result<()>
    where
        F: FnOnce() -> Result<String>,
    {
        let start = self.start_section(section);
        match build() {
            Ok(body) => self.write_section(file_name, title, section, body, start),
            Err(error) => {
                self.record_failure(section, &error.to_string(), start);
                Ok(())
            }
        }
    }

    fn write_section(
        &mut self,
        file_name: &str,
        title: &str,
        section: &str,
        body: String,
        start: Instant,
    ) -> Result<()> {
        let markdown = plain_section(title, &body);
        self.write_markdown(file_name, markdown, section, start)
    }

    fn write_failure(
        &mut self,
        file_name: &str,
        section: &str,
        title: &str,
        message: &str,
        start: Instant,
    ) -> Result<()> {
        self.failures.push(ReportFailure {
            section: section.to_owned(),
            message: message.to_owned(),
        });
        let body = format!("# {title}\n\nThis section failed:\n\n```text\n{message}\n```\n");
        self.write_markdown_with_status(file_name, body, section, start, "failed")
    }

    fn record_failure(&mut self, section: &str, message: &str, start: Instant) {
        self.failures.push(ReportFailure {
            section: section.to_owned(),
            message: message.to_owned(),
        });
        self.record_elapsed(section, start, "failed");
    }

    fn record_timing(&mut self, section: &str, start: Instant) {
        self.record_elapsed(section, start, "done");
    }

    fn record_elapsed(&mut self, section: &str, start: Instant, status: &str) {
        let ms = elapsed_ms(start);
        self.section_timings.push(ReportSectionTiming {
            section: section.to_owned(),
            ms,
        });
        if self.progress {
            eprintln!("report: {section} {status} in {ms:.1} ms");
        }
    }

    fn write_markdown(
        &mut self,
        file_name: &str,
        markdown: String,
        section: &str,
        start: Instant,
    ) -> Result<()> {
        self.write_markdown_with_status(file_name, markdown, section, start, "wrote")
    }

    fn write_markdown_with_status(
        &mut self,
        file_name: &str,
        markdown: String,
        section: &str,
        start: Instant,
        status: &str,
    ) -> Result<()> {
        let path = self.output_dir.join(file_name);
        fs::write(&path, markdown)?;
        self.files.push(path);
        self.record_elapsed(section, start, status);
        Ok(())
    }

    fn write_index(
        &mut self,
        trace_path: &Path,
        trace: &TraceBundle,
        sources: IndexSources<'_>,
        total_ms: f64,
    ) -> Result<()> {
        let IndexSources {
            analysis,
            xcode_mio,
            raw_counters,
            hw_counters,
        } = sources;
        let mut out = String::new();
        out.push_str("# gputrace Report\n\n");
        out.push_str(&format!("- Trace: `{}`\n", trace_path.display()));
        out.push_str(&format!("- Bundle: `{}`\n", trace.path.display()));
        out.push_str(&format!("- Total report time: `{total_ms:.1} ms`\n"));
        if let Some(report) = xcode_mio {
            out.push_str(&format!(
                "- Xcode MIO: `{}` commands, `{}` pipelines, `{}` cost records, `{:.3} ms` GPU time\n",
                report.gpu_command_count,
                report.pipeline_state_count,
                report.cost_record_count,
                report.gpu_time_ns as f64 / 1_000_000.0,
            ));
            if let Some(top) = report.top_pipelines.first() {
                out.push_str(&format!(
                    "- Top Xcode MIO pipeline: `{}` (`{}` commands, `{:.1}%` command share)\n",
                    top.function_name.as_deref().unwrap_or("<unknown function>"),
                    top.command_count,
                    top.command_percent,
                ));
            }
        } else {
            out.push_str(&format!(
                "- Dispatches: `{}`\n- Kernels: `{}`\n",
                analysis.dispatch_count, analysis.kernel_count,
            ));
        }
        if let Some(report) = hw_counters {
            out.push_str(&format!(
                "- Hardware counters ([counters](counters.md)): `{}`, `{}` encoders, `{}` dispatches attributed{}\n",
                report.gpu.gpu_type,
                report.encoders.len(),
                report.dispatches.len(),
                if report.foreign_kicks > 0 {
                    format!(
                        ", `{}` kicks not from this capture during the counter pass",
                        report.foreign_kicks
                    )
                } else {
                    String::new()
                },
            ));
        }
        out.push_str(&format!(
            "- Buffer inventory: `{}` files / `{}` bytes\n- Unused resources: `{}` entries / `{}` logical bytes\n",
            analysis.buffer_inventory_count,
            analysis.buffer_inventory_bytes,
            analysis.unused_resource_count,
            analysis.unused_resource_bytes,
        ));
        out.push('\n');

        out.push_str("## Files\n\n");
        for file in &self.files {
            let name = file
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<unknown>");
            out.push_str(&format!("- [{}]({name})\n", title_from_file_name(name)));
        }

        if !self.failures.is_empty() {
            out.push_str("\n## Failed Sections\n\n");
            for failure in &self.failures {
                out.push_str(&format!("- `{}`: {}\n", failure.section, failure.message));
            }
        }

        out.push_str("\n## Section Timings\n\n");
        for timing in &self.section_timings {
            out.push_str(&format!("- `{}`: `{:.1} ms`\n", timing.section, timing.ms));
        }

        if let Some(raw_counters) = raw_counters
            && !raw_counters.timings.is_empty()
        {
            out.push_str("\n## Raw Counter Decode Timings\n\n");
            for timing in &raw_counters.timings {
                out.push_str(&format!("- `{}`: `{:.1} ms`\n", timing.stage, timing.ms));
            }
        }

        let path = self.output_dir.join("index.md");
        fs::write(&path, out)?;
        self.files.insert(0, path);
        Ok(())
    }
}

fn plain_section(title: &str, body: &str) -> String {
    if body.trim_start().starts_with('#') {
        return body.to_owned();
    }
    format!("# {title}\n\n```text\n{body}\n```\n")
}

fn title_from_file_name(file_name: &str) -> String {
    file_name
        .trim_end_matches(".md")
        .split('-')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}
