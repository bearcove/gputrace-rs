use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use rayon::prelude::*;
use serde::Serialize;
use walkdir::WalkDir;

use crate::error::{Error, Result};
use crate::xcode_command_costs;

const DEFAULT_PERCENT_TARGETS: &[f64] = &[
    69.98, 9.35, 9.13, 4.33, 2.54, 1.45, 1.35, 0.66, 0.44, 0.39, 0.30, 0.03, 0.02, 0.01, 4.680,
    4.672, 0.726, 0.724, 0.270, 0.269, 0.228, 0.225, 0.173, 0.005,
];

#[derive(Debug, Clone)]
pub struct XcodeCostScanOptions {
    pub profiler_dir: PathBuf,
    pub table: Option<PathBuf>,
    pub targets: Vec<f64>,
    pub include_defaults: bool,
    pub unaligned: bool,
    pub max_hits_per_target: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct XcodeCostScanReport {
    pub profiler_dir: PathBuf,
    pub file_count: usize,
    pub byte_count: u64,
    pub target_count: usize,
    pub variant_count: usize,
    pub matches: Vec<XcodeCostScanMatch>,
}

#[derive(Debug, Clone, Serialize)]
pub struct XcodeCostScanMatch {
    pub file: PathBuf,
    pub offset: u64,
    pub encoding: &'static str,
    pub target_label: String,
    pub target_value: f64,
    pub matched_value: f64,
    pub delta: f64,
}

#[derive(Debug, Clone)]
struct TargetVariant {
    label: String,
    value: f64,
    epsilon: f64,
}

pub fn scan(options: &XcodeCostScanOptions) -> Result<XcodeCostScanReport> {
    if !options.profiler_dir.is_dir() {
        return Err(Error::NotDirectory(options.profiler_dir.clone()));
    }

    let files = profiler_files(&options.profiler_dir)?;
    let byte_count = files.iter().map(|file| file.size).sum::<u64>();
    let targets = collect_targets(options)?;
    let variants = target_variants(&targets);

    let matches = files
        .par_iter()
        .map(|file| scan_file(file, &variants, options))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    Ok(XcodeCostScanReport {
        profiler_dir: options.profiler_dir.clone(),
        file_count: files.len(),
        byte_count,
        target_count: targets.len(),
        variant_count: variants.len(),
        matches,
    })
}

pub fn format_summary(report: &XcodeCostScanReport, top: Option<usize>) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Xcode cost binary scan\nprofiler_dir={}\nfiles={} bytes={} targets={} variants={} matches={}\n\n",
        report.profiler_dir.display(),
        report.file_count,
        report.byte_count,
        report.target_count,
        report.variant_count,
        report.matches.len(),
    ));

    let mut by_target = BTreeMap::<&str, Vec<&XcodeCostScanMatch>>::new();
    for hit in &report.matches {
        by_target.entry(&hit.target_label).or_default().push(hit);
    }

    let limit = top.unwrap_or(80);
    for (target, hits) in by_target.into_iter().take(limit) {
        let first = hits[0];
        out.push_str(&format!(
            "{:<28} hits={:<5} first={}@0x{:x} {} value={:.9} delta={:.3e}\n",
            target,
            hits.len(),
            first.file.display(),
            first.offset,
            first.encoding,
            first.matched_value,
            first.delta,
        ));
    }

    if report.matches.is_empty() {
        out.push_str("No approximate f32/f64 matches found for the selected targets.\n");
    }
    out
}

fn collect_targets(options: &XcodeCostScanOptions) -> Result<Vec<f64>> {
    let mut targets = Vec::new();
    if options.include_defaults || (options.targets.is_empty() && options.table.is_none()) {
        targets.extend(DEFAULT_PERCENT_TARGETS);
    }
    targets.extend(options.targets.iter().copied());
    if let Some(table) = &options.table {
        let table = xcode_command_costs::parse_table(table)?;
        targets.extend(
            table
                .rows
                .iter()
                .filter_map(|row| row.execution_cost_percent)
                .filter(|value| value.is_finite()),
        );
    }
    targets.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    targets.dedup_by(|left, right| (*left - *right).abs() < 0.000_000_1);
    Ok(targets)
}

fn target_variants(targets: &[f64]) -> Vec<TargetVariant> {
    let mut variants = Vec::with_capacity(targets.len() * 2);
    for target in targets {
        variants.push(TargetVariant {
            label: format!("{target:.6}%"),
            value: *target,
            epsilon: ((*target).abs() * 0.0002).max(0.0005),
        });
        variants.push(TargetVariant {
            label: format!("{target:.6}%/100"),
            value: *target / 100.0,
            epsilon: ((*target / 100.0).abs() * 0.0002).max(0.000005),
        });
    }
    variants.sort_by(|left, right| {
        left.value
            .partial_cmp(&right.value)
            .unwrap_or(Ordering::Equal)
    });
    variants
}

fn scan_file(
    file: &ProfilerFile,
    variants: &[TargetVariant],
    options: &XcodeCostScanOptions,
) -> Result<Vec<XcodeCostScanMatch>> {
    let fd = File::open(&file.path)?;
    let mmap = unsafe { Mmap::map(&fd)? };
    let mut matches = Vec::new();
    let mut counts = BTreeMap::<String, usize>::new();

    scan_f32(
        &mmap,
        file,
        variants,
        if options.unaligned { 1 } else { 4 },
        options.max_hits_per_target,
        &mut counts,
        &mut matches,
    );
    scan_f64(
        &mmap,
        file,
        variants,
        if options.unaligned { 1 } else { 8 },
        options.max_hits_per_target,
        &mut counts,
        &mut matches,
    );

    Ok(matches)
}

fn scan_f32(
    bytes: &[u8],
    file: &ProfilerFile,
    variants: &[TargetVariant],
    step: usize,
    max_hits: usize,
    counts: &mut BTreeMap<String, usize>,
    matches: &mut Vec<XcodeCostScanMatch>,
) {
    let mut offset = 0usize;
    while offset + 4 <= bytes.len() {
        let value = f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as f64;
        scan_value(
            file, offset, "f32le", value, variants, max_hits, counts, matches,
        );
        offset += step;
    }
}

fn scan_f64(
    bytes: &[u8],
    file: &ProfilerFile,
    variants: &[TargetVariant],
    step: usize,
    max_hits: usize,
    counts: &mut BTreeMap<String, usize>,
    matches: &mut Vec<XcodeCostScanMatch>,
) {
    let mut offset = 0usize;
    while offset + 8 <= bytes.len() {
        let value = f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        scan_value(
            file, offset, "f64le", value, variants, max_hits, counts, matches,
        );
        offset += step;
    }
}

fn scan_value(
    file: &ProfilerFile,
    offset: usize,
    encoding: &'static str,
    value: f64,
    variants: &[TargetVariant],
    max_hits: usize,
    counts: &mut BTreeMap<String, usize>,
    matches: &mut Vec<XcodeCostScanMatch>,
) {
    if !value.is_finite() {
        return;
    }
    let first = lower_bound(variants, value - 0.02);
    for variant in &variants[first..] {
        if variant.value > value + 0.02 {
            break;
        }
        let delta = value - variant.value;
        if delta.abs() <= variant.epsilon {
            let count = counts.entry(variant.label.clone()).or_default();
            *count += 1;
            if *count <= max_hits {
                matches.push(XcodeCostScanMatch {
                    file: file.path.clone(),
                    offset: offset as u64,
                    encoding,
                    target_label: variant.label.clone(),
                    target_value: variant.value,
                    matched_value: value,
                    delta,
                });
            }
        }
    }
}

fn lower_bound(variants: &[TargetVariant], value: f64) -> usize {
    let mut left = 0usize;
    let mut right = variants.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if variants[mid].value < value {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

#[derive(Debug, Clone)]
struct ProfilerFile {
    path: PathBuf,
    size: u64,
}

fn profiler_files(root: &Path) -> Result<Vec<ProfilerFile>> {
    let mut files = Vec::new();
    for entry in WalkDir::new(root).follow_links(false) {
        let entry = entry.map_err(|error| Error::InvalidInput(error.to_string()))?;
        if entry.file_type().is_file() {
            let metadata = entry
                .metadata()
                .map_err(|error| Error::InvalidInput(error.to_string()))?;
            files.push(ProfilerFile {
                path: entry.path().to_path_buf(),
                size: metadata.len(),
            });
        }
    }
    files.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(files)
}
