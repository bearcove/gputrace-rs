//! Decode a single `Profiling_f_*.raw` via Xcode's `GTShaderProfiler`
//! framework and print per-clique kick counts, durations, and shares.
//!
//! Usage:
//! ```
//! cargo run -p agxps-sys --example decode_profiling -- /path/to/Profiling_f_0.raw
//! ```
//!
//! Defaults: M4 Pro (`gen=16, variant=3, rev=1`). Override via
//! `AGXPS_GEN`, `AGXPS_VARIANT`, `AGXPS_REV`. Override the framework
//! location via `AGXPS_FRAMEWORK_PATH`.

use std::env;
use std::fs;
use std::process;

fn main() {
    let path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: decode_profiling <Profiling_f_N.raw>");
        process::exit(2);
    });

    let generation: u32 = env::var("AGXPS_GEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let variant: u32 = env::var("AGXPS_VARIANT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let rev: u32 = env::var("AGXPS_REV")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {}: {e}", path);
            process::exit(1);
        }
    };
    println!("loaded {} bytes from {}", bytes.len(), path);

    let loaded = match agxps_sys::load() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("load: {e}");
            process::exit(1);
        }
    };
    println!("framework: {}", loaded.framework_path);

    let decoded = match loaded.parse_profiling(generation, variant, rev, &bytes) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("parse_profiling(gen={generation}, variant={variant}, rev={rev}): {e}");
            process::exit(1);
        }
    };

    println!("\ndecoded:");
    println!("  kicks:                    {}", decoded.kick_starts.len());
    println!(
        "  usc_timestamps:           {}",
        decoded.usc_timestamps.len()
    );
    println!(
        "  synchronized_timestamps:  {}",
        decoded.synchronized_timestamps.len()
    );
    println!("  counters:                 {}", decoded.counter_num);

    let kick_groups = decoded.group_by_clique();
    let dur_groups = decoded.duration_by_clique();
    let total_kicks: usize = kick_groups.values().sum();
    let total_dur: u64 = dur_groups.values().sum();

    println!("\ngrouped by software-id high16 (= kernel/clique):");
    println!("  prefix    kicks  share-by-kicks       duration       share-by-duration");
    for (prefix, kicks) in &kick_groups {
        let dur = dur_groups.get(prefix).copied().unwrap_or(0);
        let kshare = if total_kicks == 0 {
            0.0
        } else {
            100.0 * *kicks as f64 / total_kicks as f64
        };
        let dshare = if total_dur == 0 {
            0.0
        } else {
            100.0 * dur as f64 / total_dur as f64
        };
        println!("  0x{prefix:04x}  {kicks:>5}  {kshare:>13.4}%  {dur:>16}  {dshare:>16.4}%",);
    }

    // The trace's full tick span vs Xcode's wall gives us the
    // profile-clock period. Empirically ~3.89 ns/tick (≈257 MHz) on
    // M4 Pro.
    let min_start_t = (0..decoded.kick_starts.len())
        .map(|i| decoded.kick_start_time(i))
        .min()
        .unwrap_or(0);
    let max_end_t = (0..decoded.kick_ends.len())
        .map(|i| decoded.kick_end_time(i))
        .max()
        .unwrap_or(0);
    let span_ticks = max_end_t.saturating_sub(min_start_t);
    println!(
        "\ntick span (this bank): {min_start_t} .. {max_end_t}  ({span_ticks} ticks ≈ {:.2} µs at 3.89 ns/tick)",
        span_ticks as f64 * 3.89 / 1000.0,
    );

    println!("\nfirst 16 kicks (high32=time-ticks, low32=usc_sample_idx):");
    for i in 0..decoded.kick_starts.len().min(16) {
        let st = decoded.kick_start_time(i);
        let ss = decoded.kick_start_sample(i);
        let et = decoded.kick_end_time(i);
        let es = decoded.kick_end_sample(i);
        let dur_ticks = et.saturating_sub(st);
        let prefix = (decoded.kick_software_ids[i] >> 48) as u16;
        let sub = (decoded.kick_software_ids[i] & 0xffff) as u16;
        println!(
            "  [{i:>3}] start=({st:>6}t,{ss:>5}s)  end=({et:>6}t,{es:>5}s)  lifetime={dur_ticks:>6}t  prefix=0x{prefix:04x}  sub=0x{sub:04x}",
        );
    }

    if !decoded.synchronized_timestamps.is_empty() {
        let first = decoded.synchronized_timestamps.first().unwrap();
        let last = decoded.synchronized_timestamps.last().unwrap();
        let (ft, fs) = agxps_sys::unpack_time_sample(*first);
        let (lt, ls) = agxps_sys::unpack_time_sample(*last);
        println!(
            "\nsynchronized_timestamps: count={}  first=(time={ft}, sample={fs})  last=(time={lt}, sample={ls})",
            decoded.synchronized_timestamps.len(),
        );
    }

    println!("\ncounters: ({} total)", decoded.counter_num);
    println!(
        "  (Profiling_f_* streams carry counter names only; counter samples live in the \
         limiter pass's Counters_f_* streams, see agxps_sys::counters)",
    );
    for (idx, name) in decoded.counter_names.iter().enumerate() {
        let short: String = name.chars().take(16).collect();
        println!("  [{idx:>2}] {short}…");
    }
}
