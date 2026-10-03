use serde::Serialize;
use std::collections::BTreeMap;

use crate::analysis;
use crate::error::{Error, Result};
use crate::hw_counters::{self, HwCounterRow};
use crate::profiler;
use crate::shaders;
use crate::timing;
use crate::trace::TraceBundle;
use crate::xcode_mio;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InsightSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InsightType {
    Bottleneck,
    Optimization,
    AntiPattern,
    Info,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerformanceInsight {
    pub insight_type: InsightType,
    pub severity: InsightSeverity,
    pub shader_name: Option<String>,
    pub title: String,
    pub description: String,
    pub recommendations: Vec<String>,
    pub impact: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InsightsReport {
    pub synthetic: bool,
    pub total_gpu_time_ms: f64,
    pub top_bottlenecks: Vec<String>,
    pub critical_count: usize,
    pub high_count: usize,
    pub medium_count: usize,
    pub low_count: usize,
    pub info_count: usize,
    pub insights: Vec<PerformanceInsight>,
}

pub fn report(trace: &TraceBundle, min_level: Option<&str>) -> Result<InsightsReport> {
    let xcode_mio_report = xcode_mio::analysis_report(trace).ok();
    report_with_xcode_mio(trace, min_level, xcode_mio_report)
}

pub fn report_with_xcode_mio(
    trace: &TraceBundle,
    min_level: Option<&str>,
    xcode_mio_report: Option<xcode_mio::XcodeMioAnalysisReport>,
) -> Result<InsightsReport> {
    report_with_context(trace, min_level, xcode_mio_report, None, None, None)
}

pub fn report_with_context(
    trace: &TraceBundle,
    min_level: Option<&str>,
    xcode_mio_report: Option<xcode_mio::XcodeMioAnalysisReport>,
    precomputed_timing: Option<timing::TimingReport>,
    precomputed_profiler_summary: Option<profiler::ProfilerStreamDataSummary>,
    precomputed_shader_report: Option<shaders::ShaderReport>,
) -> Result<InsightsReport> {
    let profiler_summary =
        precomputed_profiler_summary.or_else(|| profiler::stream_data_summary(&trace.path).ok());
    let timing = if let Some(report) = precomputed_timing {
        report
    } else {
        timing::report_with_profiler_summary(trace, profiler_summary.as_ref())?
    };
    let analysis =
        analysis::analyze_with_context(trace, xcode_mio_report.clone(), Some(timing.clone()));
    let shader_report = precomputed_shader_report.or_else(|| {
        shaders::report_with_profiler_summary(
            trace,
            &shaders::default_search_paths(),
            profiler_summary.as_ref(),
        )
        .ok()
    });
    let dispatch_time_exceeds_total = timing.kernels.iter().fold(0u64, |sum, kernel| {
        sum.saturating_add(kernel.synthetic_duration_ns)
    }) > timing.total_duration_ns;
    let time_label = if timing.synthetic {
        "synthetic GPU time"
    } else if dispatch_time_exceeds_total {
        "summed profiler dispatch time"
    } else {
        "GPU time"
    };
    let time_title = if dispatch_time_exceeds_total {
        "profiler dispatch time"
    } else {
        "GPU time"
    };
    let mut insights = Vec::new();

    if let Some(top_kernel) = timing.kernels.first() {
        if top_kernel.percent_of_total > 50.0 {
            insights.push(PerformanceInsight {
                insight_type: InsightType::Bottleneck,
                severity: InsightSeverity::Critical,
                shader_name: Some(top_kernel.name.clone()),
                title: format!("{} dominates {time_title}", top_kernel.name),
                description: format!(
                    "{} accounts for {:.1}% of {} across {} dispatches.",
                    top_kernel.name,
                    top_kernel.percent_of_total,
                    time_label,
                    top_kernel.dispatch_count
                ),
                recommendations: vec![
                    "Profile this shader path first.".to_owned(),
                    "Reduce work per dispatch or fuse adjacent passes only if total work drops."
                        .to_owned(),
                    "Check bound buffers and source for obvious bandwidth-heavy loops.".to_owned(),
                ],
                impact: Some("Dominates end-to-end GPU execution time.".to_owned()),
            });
        } else if top_kernel.percent_of_total > 30.0 {
            insights.push(PerformanceInsight {
                insight_type: InsightType::Bottleneck,
                severity: InsightSeverity::High,
                shader_name: Some(top_kernel.name.clone()),
                title: format!("{} is a major {time_title} bottleneck", top_kernel.name),
                description: format!(
                    "{} accounts for {:.1}% of {} across {} dispatches.",
                    top_kernel.name,
                    top_kernel.percent_of_total,
                    time_label,
                    top_kernel.dispatch_count
                ),
                recommendations: vec![
                    "Focus optimization work on this shader before lower-rank kernels.".to_owned(),
                    "Inspect source attribution with `shader-source` and `correlate`.".to_owned(),
                ],
                impact: Some("Large contributor to GPU runtime.".to_owned()),
            });
        }
    }

    if let Some(report) = &xcode_mio_report
        && let Some(top_pipeline) = report.top_pipelines.first()
        && top_pipeline.command_percent > 25.0
    {
        let name = top_pipeline
            .function_name
            .clone()
            .unwrap_or_else(|| "<unknown function>".to_owned());
        insights.push(PerformanceInsight {
            insight_type: InsightType::Info,
            severity: InsightSeverity::Low,
            shader_name: Some(name.clone()),
            title: format!("{name} dominates Xcode MIO command topology"),
            description: format!(
                "Xcode's private MIO backend reports {name} as {} of {} GPU commands ({:.1}%), with {} executable shader-binary references.",
                top_pipeline.command_count,
                report.gpu_command_count,
                top_pipeline.command_percent,
                top_pipeline.executable_shader_binary_reference_count,
            ),
            recommendations: vec![
                "Use `xcode-mio --format summary-json` for the Xcode-derived topology fields."
                    .to_owned(),
                "Treat this as structure/topology until private GUI Cost fields are decoded."
                    .to_owned(),
            ],
            impact: Some("Shows where Xcode attributes repeated GPU command structure.".to_owned()),
        });
    }

    if profiler_summary.is_none()
        && let Some(shader_report) = &shader_report
    {
        if let Some(top_shader) = shader_report
            .shaders
            .iter()
            .find(|shader| shader.weighted_percent_of_total.is_some())
        {
            let weight_percent = top_shader.weighted_percent_of_total.unwrap_or_default();
            if weight_percent > 50.0 {
                insights.push(PerformanceInsight {
                        insight_type: InsightType::Bottleneck,
                        severity: InsightSeverity::High,
                        shader_name: Some(top_shader.name.clone()),
                        title: format!("{} dominates Xcode-weighted cost", top_shader.name),
                        description: format!(
                            "{} accounts for {:.1}% of the CSV-weighted shader cost model.",
                            top_shader.name, weight_percent
                        ),
                        recommendations: vec![
                            "Treat this as the primary optimization target when CSV counters are available."
                                .to_owned(),
                            "Use `shader-hotspots` to inspect the most expensive source lines under the weighted model."
                                .to_owned(),
                        ],
                        impact: Some(
                            "Dominates the fallback cost model derived from Xcode counters."
                                .to_owned(),
                        ),
                    });
            }
        }

        for shader in &shader_report.shaders {
            if let Some(occupancy) = shader.occupancy_percent
                && occupancy < 30.0
            {
                insights.push(PerformanceInsight {
                        insight_type: InsightType::Optimization,
                        severity: if occupancy < 15.0 {
                            InsightSeverity::High
                        } else {
                            InsightSeverity::Medium
                        },
                        shader_name: Some(shader.name.clone()),
                        title: format!("{} shows low kernel occupancy", shader.name),
                        description: format!(
                            "{} averages {:.1}% kernel occupancy from imported Xcode counters.",
                            shader.name, occupancy
                        ),
                        recommendations: vec![
                            "Treat threadgroup size and register pressure as likely occupancy levers."
                                .to_owned(),
                            "Validate with `xcode-counters` or `validate-counters` if you have adjacent CSV exports."
                                .to_owned(),
                        ],
                        impact: Some(
                            "Low occupancy can leave GPU execution resources underutilized."
                                .to_owned(),
                        ),
                    });
            }

            if let Some(alu) = shader.alu_utilization_percent
                && alu > 70.0
            {
                insights.push(PerformanceInsight {
                    insight_type: InsightType::Info,
                    severity: InsightSeverity::Medium,
                    shader_name: Some(shader.name.clone()),
                    title: format!("{} is ALU-heavy", shader.name),
                    description: format!(
                        "{} shows {:.1}% ALU utilization from imported Xcode counters.",
                        shader.name, alu
                    ),
                    recommendations: vec![
                        "Look for approximation opportunities and repeated math on hot lines."
                            .to_owned(),
                        "Use `shader-hotspots` to inspect compute-heavy source regions.".to_owned(),
                    ],
                    impact: Some(
                        "Suggests arithmetic pressure even when raw profiler data is unavailable."
                            .to_owned(),
                    ),
                });
            }

            if let Some(dev_bw) = shader.device_memory_bandwidth_gbps
                && dev_bw >= 10.0
            {
                insights.push(PerformanceInsight {
                        insight_type: InsightType::Optimization,
                        severity: InsightSeverity::Medium,
                        shader_name: Some(shader.name.clone()),
                        title: format!("{} is bandwidth-heavy", shader.name),
                        description: format!(
                            "{} shows {:.2} GB/s device-memory bandwidth from imported Xcode counters.",
                            shader.name, dev_bw
                        ),
                        recommendations: vec![
                            "Inspect access locality, reuse, and buffer layout.".to_owned(),
                            "Cross-check with `buffer-access` and `shader-hotspots`.".to_owned(),
                        ],
                        impact: Some(
                            "Points to memory pressure without needing streamData or raw counter parsing."
                                .to_owned(),
                        ),
                    });
            }

            if let Some(gpu_read_bw) = shader.gpu_read_bandwidth_gbps
                && gpu_read_bw >= 8.0
            {
                insights.push(PerformanceInsight {
                        insight_type: InsightType::Optimization,
                        severity: InsightSeverity::Medium,
                        shader_name: Some(shader.name.clone()),
                        title: format!("{} has heavy GPU read bandwidth", shader.name),
                        description: format!(
                            "{} reports {:.2} GB/s GPU read bandwidth from imported Xcode counters.",
                            shader.name, gpu_read_bw
                        ),
                        recommendations: vec![
                            "Inspect buffer reuse and read amplification before adding more parallelism."
                                .to_owned(),
                            "Check whether neighboring dispatches can share or fuse reads."
                                .to_owned(),
                        ],
                        impact: Some(
                            "Points to read-side memory pressure in the fallback counter path."
                                .to_owned(),
                        ),
                    });
            }

            if let Some(l1_miss_rate) = shader.buffer_l1_miss_rate_percent
                && l1_miss_rate >= 10.0
            {
                insights.push(PerformanceInsight {
                        insight_type: InsightType::Optimization,
                        severity: InsightSeverity::Medium,
                        shader_name: Some(shader.name.clone()),
                        title: format!("{} shows Buffer L1 miss pressure", shader.name),
                        description: format!(
                            "{} reports {:.1}% Buffer L1 miss rate from imported Xcode counters.",
                            shader.name, l1_miss_rate
                        ),
                        recommendations: vec![
                            "Inspect access stride, data layout, and per-thread working-set size."
                                .to_owned(),
                            "Use `shader-hotspots` and `buffer-access` together to localize the miss-heavy path."
                                .to_owned(),
                        ],
                        impact: Some(
                            "Suggests cache-unfriendly accesses are contributing to the weighted shader cost."
                                .to_owned(),
                        ),
                    });
            }

            if let Some(miss_rate) = shader
                .metric_source
                .eq("xcode-weighted")
                .then_some(shader.last_level_cache_percent)
                .flatten()
                && miss_rate >= 5.0
            {
                insights.push(PerformanceInsight {
                        insight_type: InsightType::Optimization,
                        severity: InsightSeverity::Low,
                        shader_name: Some(shader.name.clone()),
                        title: format!("{} shows notable cache pressure", shader.name),
                        description: format!(
                            "{} reports {:.1}% cache-pressure signal alongside Xcode-weighted cost.",
                            shader.name, miss_rate
                        ),
                        recommendations: vec![
                            "Inspect memory access stride and reuse around the hottest lines."
                                .to_owned(),
                            "Cross-check with `buffer-access` if cache pressure aligns with shared or oversized buffers."
                                .to_owned(),
                        ],
                        impact: Some(
                            "Suggests memory locality is contributing to the weighted shader cost."
                                .to_owned(),
                        ),
                    });
            }
        }
    }

    for kernel in &timing.kernels {
        let avg_duration_us =
            kernel.synthetic_duration_ns as f64 / kernel.dispatch_count.max(1) as f64 / 1_000.0;
        if kernel.dispatch_count > 100 && avg_duration_us < 50.0 && kernel.percent_of_total > 5.0 {
            insights.push(PerformanceInsight {
                insight_type: InsightType::Optimization,
                severity: InsightSeverity::High,
                shader_name: Some(kernel.name.clone()),
                title: format!("{} shows dispatch overhead pressure", kernel.name),
                description: format!(
                    "{} runs {} times with only {:.1} us average {}.",
                    kernel.name,
                    kernel.dispatch_count,
                    avg_duration_us,
                    if timing.synthetic {
                        "synthetic duration"
                    } else {
                        "duration"
                    }
                ),
                recommendations: vec![
                    "Batch small dispatches when semantics allow.".to_owned(),
                    "Consider kernel fusion if the intermediate state is cheap to keep on-GPU."
                        .to_owned(),
                    "Check whether encoder transitions are forcing extra micro-dispatches."
                        .to_owned(),
                ],
                impact: Some(
                    "Likely wasting CPU submission overhead on tiny dispatches.".to_owned(),
                ),
            });
        }
    }

    if let Some(summary) = &profiler_summary {
        let mut pipeline_stats_by_name = BTreeMap::new();
        for pipeline in &summary.pipelines {
            if let (Some(name), Some(stats)) = (&pipeline.function_name, &pipeline.stats) {
                pipeline_stats_by_name
                    .entry(name.clone())
                    .or_insert_with(|| stats.clone());
            }
        }
        // Per-kernel hardware counters from the limiter pass.
        let hw_report = hw_counters::cached_report(&trace.path);
        let hw_by_name = hw_report
            .as_deref()
            .map(|report| {
                report
                    .kernels
                    .iter()
                    .map(|row| (row.label.clone(), row.clone()))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        let peak_dram_gbps = hw_report.as_deref().map(|report| report.gpu.peak_dram_gbps);

        let mut sample_totals = BTreeMap::<String, (usize, u64, usize, f64)>::new();
        let mut total_samples = 0usize;
        let mut total_duration_us = 0u64;
        for dispatch in &summary.dispatches {
            let name = dispatch
                .function_name
                .clone()
                .unwrap_or_else(|| format!("pipeline_{}", dispatch.pipeline_index));
            let entry = sample_totals.entry(name).or_default();
            entry.0 += dispatch.sample_count;
            entry.1 += dispatch.duration_us;
            entry.2 += 1;
            entry.3 += dispatch.sampling_density;
            total_samples += dispatch.sample_count;
            total_duration_us += dispatch.duration_us;
        }

        for (name, (samples, duration_us, dispatch_count, density_sum)) in sample_totals {
            if samples == 0 || total_samples == 0 || total_duration_us == 0 {
                continue;
            }
            let sample_pct = samples as f64 * 100.0 / total_samples as f64;
            let time_pct = duration_us as f64 * 100.0 / total_duration_us as f64;
            let delta = sample_pct - time_pct;
            let avg_density = density_sum / dispatch_count.max(1) as f64;

            if delta > 10.0 {
                insights.push(PerformanceInsight {
                    insight_type: InsightType::Bottleneck,
                    severity: InsightSeverity::High,
                    shader_name: Some(name.clone()),
                    title: format!("{name} is sample-heavy relative to its runtime"),
                    description: format!(
                        "{name} accounts for {:.1}% of correlated profiler samples but only {:.1}% of GPU time, with {:.3} samples/us on average.",
                        sample_pct, time_pct, avg_density
                    ),
                    recommendations: vec![
                        "Inspect this shader for sustained ALU or bandwidth pressure.".to_owned(),
                        "Compare sample-heavy kernels against source using `shader-source` and `correlate`.".to_owned(),
                    ],
                    impact: Some(
                        "Suggests higher GPU utilization than duration alone would imply.".to_owned(),
                    ),
                });
            }
        }

        for kernel in &timing.kernels {
            let hw = hw_by_name.get(&kernel.name);
            if let Some(row) = hw {
                hw_counter_insights(&kernel.name, row, peak_dram_gbps, &mut insights);
            }

            let Some(stats) = pipeline_stats_by_name.get(&kernel.name) else {
                continue;
            };

            if stats.spilled_bytes >= 256 {
                insights.push(PerformanceInsight {
                    insight_type: InsightType::Optimization,
                    severity: if stats.spilled_bytes >= 1024 {
                        InsightSeverity::High
                    } else {
                        InsightSeverity::Medium
                    },
                    shader_name: Some(kernel.name.clone()),
                    title: format!("{} shows register spilling", kernel.name),
                    description: format!(
                        "{} spills {} bytes and allocates {} registers, which is consistent with register-pressure losses.",
                        kernel.name, stats.spilled_bytes, stats.temporary_register_count
                    ),
                    recommendations: vec![
                        "Reduce live temporary values or split overly wide kernels.".to_owned(),
                        "Check whether threadgroup size can be reduced without hurting occupancy.".to_owned(),
                        "Use `shader-source` and `correlate` to inspect the hottest code path.".to_owned(),
                    ],
                    impact: Some(
                        "Spilling can push hot values to memory and raise both latency and bandwidth pressure."
                            .to_owned(),
                    ),
                });
            }

            if stats.temporary_register_count >= 96 && stats.spilled_bytes == 0 {
                insights.push(PerformanceInsight {
                    insight_type: InsightType::Info,
                    severity: InsightSeverity::Medium,
                    shader_name: Some(kernel.name.clone()),
                    title: format!("{} uses a large register footprint", kernel.name),
                    description: format!(
                        "{} allocates {} registers even without reported spills.",
                        kernel.name, stats.temporary_register_count
                    ),
                    recommendations: vec![
                        "Treat threadgroup size and in-kernel temporary arrays as likely occupancy levers.".to_owned(),
                        "Watch for occupancy losses when comparing variants of this shader.".to_owned(),
                    ],
                    impact: Some(
                        "High register pressure can cap occupancy before spills become visible."
                            .to_owned(),
                    ),
                });
            }

            if stats.threadgroup_memory >= 16 * 1024 {
                insights.push(PerformanceInsight {
                    insight_type: InsightType::Optimization,
                    severity: InsightSeverity::Medium,
                    shader_name: Some(kernel.name.clone()),
                    title: format!("{} uses substantial threadgroup memory", kernel.name),
                    description: format!(
                        "{} reserves {} bytes of threadgroup memory.",
                        kernel.name, stats.threadgroup_memory
                    ),
                    recommendations: vec![
                        "Trim shared scratch usage or split phases if occupancy is lower than expected.".to_owned(),
                        "Compare threadgroup memory against register pressure before increasing group size.".to_owned(),
                    ],
                    impact: Some(
                        "Large threadgroup allocations can reduce active wave occupancy.".to_owned(),
                    ),
                });
            }
        }
    }

    if analysis.shared_buffer_count > 0
        && analysis.shared_buffer_count * 2 >= analysis.buffer_count.max(1)
    {
        insights.push(PerformanceInsight {
            insight_type: InsightType::AntiPattern,
            severity: InsightSeverity::Medium,
            shader_name: None,
            title: "Buffer sharing is widespread across encoders".to_owned(),
            description: format!(
                "{} of {} attributed buffers are touched by multiple encoders.",
                analysis.shared_buffer_count, analysis.buffer_count
            ),
            recommendations: vec![
                "Inspect high-traffic shared buffers with `buffer-access`.".to_owned(),
                "Reduce cross-encoder sharing where it causes ordering constraints.".to_owned(),
                "Check whether some shared buffers can be split by phase or lifetime.".to_owned(),
            ],
            impact: Some(
                "Can increase synchronization pressure and dependency complexity.".to_owned(),
            ),
        });
    }

    if analysis.single_use_buffer_count > 0
        && analysis.single_use_buffer_count * 2 >= analysis.buffer_count.max(1)
    {
        insights.push(PerformanceInsight {
            insight_type: InsightType::Optimization,
            severity: InsightSeverity::Medium,
            shader_name: None,
            title: "Trace contains many single-use buffers".to_owned(),
            description: format!(
                "{} of {} buffers appear in exactly one attributed dispatch.",
                analysis.single_use_buffer_count, analysis.buffer_count
            ),
            recommendations: vec![
                "Review whether short-lived temporaries can be pooled or reused.".to_owned(),
                "Inspect bundle inventory to find large one-shot allocations.".to_owned(),
            ],
            impact: Some("Suggests allocation churn or poor buffer reuse.".to_owned()),
        });
    }

    if analysis.long_lived_buffer_count > 0 {
        insights.push(PerformanceInsight {
            insight_type: InsightType::Info,
            severity: InsightSeverity::Low,
            shader_name: None,
            title: "Long-lived buffers span much of the trace".to_owned(),
            description: format!(
                "{} buffers live for more than 3x the average dispatch lifetime.",
                analysis.long_lived_buffer_count
            ),
            recommendations: vec![
                "Inspect `buffer-timeline` for persistent allocations with low reuse density."
                    .to_owned(),
                "Separate hot persistent buffers from cold archival buffers if possible."
                    .to_owned(),
            ],
            impact: Some("May indicate memory residency pressure.".to_owned()),
        });
    }

    if analysis.buffer_inventory_aliases > 0 {
        insights.push(PerformanceInsight {
            insight_type: InsightType::Info,
            severity: InsightSeverity::Low,
            shader_name: None,
            title: "Bundle contains aliased backing buffers".to_owned(),
            description: format!(
                "The trace bundle exposes {} backing-buffer aliases.",
                analysis.buffer_inventory_aliases
            ),
            recommendations: vec![
                "Inspect `buffers diff` and `buffer-access` before assuming unique resources."
                    .to_owned(),
                "Treat alias-heavy traces carefully when reasoning about memory footprint."
                    .to_owned(),
            ],
            impact: Some("Can obscure true resource identity during analysis.".to_owned()),
        });
    }

    if analysis.kernel_count > 50 {
        insights.push(PerformanceInsight {
            insight_type: InsightType::Info,
            severity: InsightSeverity::Low,
            shader_name: None,
            title: "Trace uses many unique kernels".to_owned(),
            description: format!(
                "The trace attributes work to {} distinct kernels.",
                analysis.kernel_count
            ),
            recommendations: vec![
                "Check whether kernel specialization has exploded the shader set.".to_owned(),
                "Group related kernels with `tree --group-by pipeline`.".to_owned(),
            ],
            impact: Some(
                "May increase shader management and optimization surface area.".to_owned(),
            ),
        });
    }

    if let Some(top_kernel) = timing.kernels.first()
        && top_kernel.percent_of_total > 70.0
    {
        insights.push(PerformanceInsight {
            insight_type: InsightType::Info,
            severity: InsightSeverity::Info,
            shader_name: Some(top_kernel.name.clone()),
            title: "Optimization focus is unusually clear".to_owned(),
            description: format!(
                "{} alone accounts for {:.1}% of {}.",
                top_kernel.name, top_kernel.percent_of_total, time_label
            ),
            recommendations: vec![
                "Spend optimization time on this shader before broad trace-wide cleanup."
                    .to_owned(),
            ],
            impact: Some("A single hotspot likely dominates returns.".to_owned()),
        });
    }

    insights.sort_by_key(|insight| insight.severity);

    let min_severity = match min_level {
        Some(level) => Some(parse_severity(level)?),
        None => None,
    };
    if let Some(min_severity) = min_severity {
        insights.retain(|insight| insight.severity <= min_severity);
    }

    let critical_count = insights
        .iter()
        .filter(|insight| insight.severity == InsightSeverity::Critical)
        .count();
    let high_count = insights
        .iter()
        .filter(|insight| insight.severity == InsightSeverity::High)
        .count();
    let medium_count = insights
        .iter()
        .filter(|insight| insight.severity == InsightSeverity::Medium)
        .count();
    let low_count = insights
        .iter()
        .filter(|insight| insight.severity == InsightSeverity::Low)
        .count();
    let info_count = insights
        .iter()
        .filter(|insight| insight.severity == InsightSeverity::Info)
        .count();

    Ok(InsightsReport {
        synthetic: timing.synthetic,
        total_gpu_time_ms: timing.total_duration_ns as f64 / 1_000_000.0,
        top_bottlenecks: timing
            .kernels
            .iter()
            .take(5)
            .map(|kernel| kernel.name.clone())
            .collect(),
        critical_count,
        high_count,
        medium_count,
        low_count,
        info_count,
        insights,
    })
}

/// Compute-relevant agxps limiters: each is the busy fraction of one unit,
/// relative to that unit's peak. `Compute Shader Launch Limiter` is left out:
/// it reads 82-100 % on every full-grid oracle kernel whatever bounds it
/// (docs/COUNTERS_M4.md).
const COMPUTE_LIMITERS: &[(&str, &str)] = &[
    ("ALU Limiter", "ALU"),
    ("F32 Limiter", "F32 ALU"),
    ("F16 Limiter", "F16 ALU"),
    ("IC Limiter", "integer/complex ALU"),
    ("MXU Limiter", "matrix unit"),
    ("Instruction Issue Limiter", "instruction issue"),
    ("Instruction Dispatch Limiter", "instruction dispatch"),
    ("Control Flow Limiter", "control flow"),
    ("Address Generation Limiter", "address generation"),
    ("Buffer Load Limiter", "buffer loads"),
    ("Buffer Store Limiter", "buffer stores"),
    ("Threadgroup Load Limiter", "threadgroup memory loads"),
    ("Threadgroup Store Limiter", "threadgroup memory stores"),
    ("Threadgroup Atomic Limiter", "threadgroup atomics"),
    ("L1 Cache Limiter", "L1 cache"),
    ("L2 Cache Limiter", "L2 cache"),
    ("MMU Limiter", "MMU (address translation)"),
];

/// Kernels shorter than this (summed over their dispatches) carry too few
/// counter samples for a verdict.
const HW_INSIGHT_MIN_GPU_NS: f64 = 20_000.0;

fn hw_counter_insights(
    kernel: &str,
    row: &HwCounterRow,
    peak_dram_gbps: Option<f64>,
    insights: &mut Vec<PerformanceInsight>,
) {
    if row.gpu_time_ns < HW_INSIGHT_MIN_GPU_NS {
        return;
    }
    let caveat = if row.foreign_overlap > 0.2 {
        format!(
            " Another process ran on the GPU for {:.0}% of this kernel's time, so these counts include its work.",
            row.foreign_overlap * 100.0
        )
    } else {
        String::new()
    };
    let top_limiter = COMPUTE_LIMITERS
        .iter()
        .filter_map(|(name, unit)| Some((*name, *unit, row.percent(name)?)))
        .max_by(|left, right| left.2.total_cmp(&right.2));
    let dram = row.dram_gbps();
    let occupancy = row.percent("Compute Occupancy");

    if let Some((name, unit, value)) = top_limiter
        && value >= 60.0
    {
        insights.push(PerformanceInsight {
            insight_type: InsightType::Bottleneck,
            severity: if value >= 80.0 {
                InsightSeverity::High
            } else {
                InsightSeverity::Medium
            },
            shader_name: Some(kernel.to_owned()),
            title: format!("{kernel} is limited by {unit}"),
            description: format!(
                "{kernel}: `{name}` is {value:.0}% (busy fraction of that unit's peak, limiter-pass hardware counters).{caveat}"
            ),
            recommendations: vec![
                format!("Reduce the work this kernel puts on {unit}, or trade it for a less busy unit."),
                "See counters.md for the kernel's full counter row.".to_owned(),
            ],
            impact: Some(format!(
                "{unit} saturates first; other optimizations will not help until it is relieved."
            )),
        });
    }

    if let (Some(dram), Some(peak)) = (dram, peak_dram_gbps)
        && peak > 0.0
        && dram >= 0.6 * peak
    {
        insights.push(PerformanceInsight {
            insight_type: InsightType::Bottleneck,
            severity: if dram >= 0.8 * peak {
                InsightSeverity::High
            } else {
                InsightSeverity::Medium
            },
            shader_name: Some(kernel.to_owned()),
            title: format!("{kernel} runs near DRAM bandwidth peak"),
            description: format!(
                "{kernel} moves {dram:.1} GB/s to and from DRAM, {:.0}% of the {peak:.0} GB/s peak.{caveat}",
                dram / peak * 100.0
            ),
            recommendations: vec![
                "Cut DRAM traffic: smaller data types, fusion with neighbouring kernels, or better reuse in cache.".to_owned(),
            ],
            impact: Some("Memory-bound: arithmetic changes will not speed it up.".to_owned()),
        });
    }

    let busiest = top_limiter.map(|(_, _, value)| value).unwrap_or(0.0);
    let dram_fraction = match (dram, peak_dram_gbps) {
        (Some(dram), Some(peak)) if peak > 0.0 => dram / peak,
        _ => 0.0,
    };
    if let Some(occupancy) = occupancy
        && occupancy < 25.0
        && busiest < 40.0
        && dram_fraction < 0.4
    {
        insights.push(PerformanceInsight {
            insight_type: InsightType::Optimization,
            severity: if occupancy < 10.0 {
                InsightSeverity::High
            } else {
                InsightSeverity::Medium
            },
            shader_name: Some(kernel.to_owned()),
            title: format!("{kernel} is latency-bound at low occupancy"),
            description: format!(
                "{kernel} averages {occupancy:.1}% compute occupancy while no unit is above {busiest:.0}% busy and DRAM is at {:.0}% of peak.{caveat}",
                dram_fraction * 100.0
            ),
            recommendations: vec![
                "Launch more threads (larger grid or split-K) if the problem is too small to fill the GPU.".to_owned(),
                "Otherwise check register and threadgroup-memory use, which cap resident SIMD groups.".to_owned(),
            ],
            impact: Some("The GPU is mostly idle waiting on latency.".to_owned()),
        });
    }
}

pub fn format_report(report: &InsightsReport) -> String {
    let mut out = String::new();
    out.push_str("=== GPU Performance Insights ===\n\n");
    if report.synthetic {
        out.push_str("Synthetic timing and trace attribution only.\n");
    } else {
        out.push_str("Profiler-backed timing with trace attribution.\n");
    }
    out.push_str(&format!(
        "Total GPU Time: {:.2} ms\n",
        report.total_gpu_time_ms
    ));
    out.push_str(&format!("Insights Found: {}\n", report.insights.len()));
    out.push_str(&format!(
        "  Critical: {}, High: {}, Medium: {}, Low: {}, Info: {}\n\n",
        report.critical_count,
        report.high_count,
        report.medium_count,
        report.low_count,
        report.info_count
    ));
    if !report.top_bottlenecks.is_empty() {
        out.push_str("Top Bottlenecks:\n");
        for (index, name) in report.top_bottlenecks.iter().enumerate() {
            out.push_str(&format!("  {}. {}\n", index + 1, name));
        }
        out.push('\n');
    }
    out.push_str("=== Detailed Insights ===\n\n");
    for (index, insight) in report.insights.iter().enumerate() {
        out.push_str(&format!(
            "[{}] [{}] {}\n",
            index + 1,
            severity_label(insight.severity),
            insight.title
        ));
        if let Some(shader_name) = &insight.shader_name {
            out.push_str(&format!("    Shader: {shader_name}\n"));
        }
        out.push_str(&format!(
            "    Type: {}\n\n",
            insight_type_label(insight.insight_type)
        ));
        out.push_str(&format!("    {}\n\n", insight.description));
        if let Some(impact) = &insight.impact {
            out.push_str(&format!("    Impact: {impact}\n\n"));
        }
        if !insight.recommendations.is_empty() {
            out.push_str("    Recommendations:\n");
            for recommendation in &insight.recommendations {
                out.push_str(&format!("      - {recommendation}\n"));
            }
            out.push('\n');
        }
        out.push_str(
            "    ----------------------------------------------------------------------\n\n",
        );
    }
    out
}

fn parse_severity(value: &str) -> Result<InsightSeverity> {
    match value.to_ascii_lowercase().as_str() {
        "critical" => Ok(InsightSeverity::Critical),
        "high" => Ok(InsightSeverity::High),
        "medium" => Ok(InsightSeverity::Medium),
        "low" => Ok(InsightSeverity::Low),
        "info" => Ok(InsightSeverity::Info),
        _ => Err(Error::InvalidInput(format!(
            "unknown severity level: {value} (expected critical, high, medium, low, or info)"
        ))),
    }
}

fn severity_label(value: InsightSeverity) -> &'static str {
    match value {
        InsightSeverity::Critical => "CRITICAL",
        InsightSeverity::High => "HIGH",
        InsightSeverity::Medium => "MEDIUM",
        InsightSeverity::Low => "LOW",
        InsightSeverity::Info => "INFO",
    }
}

fn insight_type_label(value: InsightType) -> &'static str {
    match value {
        InsightType::Bottleneck => "BOTTLENECK",
        InsightType::Optimization => "OPTIMIZATION",
        InsightType::AntiPattern => "ANTI-PATTERN",
        InsightType::Info => "INFO",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_severity_names() {
        assert_eq!(
            parse_severity("critical").expect("critical should parse"),
            InsightSeverity::Critical
        );
        assert_eq!(
            parse_severity("INFO").expect("info should parse"),
            InsightSeverity::Info
        );
        assert!(parse_severity("urgent").is_err());
    }

    #[test]
    fn formats_report() {
        let report = InsightsReport {
            synthetic: true,
            total_gpu_time_ms: 1.25,
            top_bottlenecks: vec!["kernel".into()],
            critical_count: 1,
            high_count: 0,
            medium_count: 0,
            low_count: 0,
            info_count: 0,
            insights: vec![PerformanceInsight {
                insight_type: InsightType::Bottleneck,
                severity: InsightSeverity::Critical,
                shader_name: Some("kernel".into()),
                title: "kernel dominates GPU time".into(),
                description: "kernel accounts for most time".into(),
                recommendations: vec!["profile it".into()],
                impact: Some("dominates runtime".into()),
            }],
        };
        let output = format_report(&report);
        assert!(output.contains("GPU Performance Insights"));
        assert!(output.contains("[CRITICAL]"));
        assert!(output.contains("kernel dominates GPU time"));
    }
}
