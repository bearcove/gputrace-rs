# Decoding Profiling_f / Counters_f via Apple's `agxps` C API

`Profiling_f_*.raw` carries the per-USC-stream sample data Xcode uses to
compute kernel cost percentages. The on-disk format isn't documented but
**is decodable via Apple's `agxps` C API** — exported by
`GPUToolsReplay.framework` (part of macOS, not Xcode) and reachable
either by linking MTLReplayer.app's process or by dlopen.

## Where the API lives

| Build | Path | Symbols |
|---|---|---|
| **macOS (host or VM)** | `/System/Library/PrivateFrameworks/GPUToolsReplay.framework/GPUToolsReplay` (in dyld_shared_cache) | 15 `agxps_aps_*` + ~358 `agxps_gpu_*` / `agxps_timeseries_*` |
| **Xcode (host only)** | `/Applications/Xcode.app/Contents/PlugIns/GPUDebugger.ideplugin/Contents/Frameworks/GTShaderProfiler.framework/Versions/A/GTShaderProfiler` (real on-disk binary) | 374 `agxps_*` (richer API including `agxps_aps_kick_time_stats_create_sampled` and friends) |

The macOS-on-VM build has a leaner API surface but is enough to decode
the sample stream and read per-kick timestamps, software IDs, and
counter values.

## Verified end-to-end working call chain (macOS 26.4 VM, no GPU)

Done via `lldb` driving `MTLReplayer` to which `GPUToolsReplay` is
already linked. Successfully decoded `Profiling_f_0.raw` from the
`gputrace-sample` trace into 48 kicks.

```
gpu = agxps_gpu_create(/*gen=*/16, /*variant=*/3, /*rev=*/1, /*flag=*/false)

descriptor = agxps_aps_descriptor_create()           // 0x68-byte struct
*(uintptr_t*)(descriptor + 0x00) = (uintptr_t)gpu    // gpu pointer
*(u32*)(descriptor + 0x08) = 0x100                   // see "validation" below
*(u32*)(descriptor + 0x0c) = 0x400
*(u32*)(descriptor + 0x10) = 0                       // 0 is OK
*(u64*)(descriptor + 0x18) = 0x1000                  // already set by descriptor_create

parser = agxps_aps_parser_create(descriptor)         // returns NULL if dispatch lookup fails

uint8_t out[4096] = {0};
profile_data = agxps_aps_parser_parse(parser,
                                      file_bytes, file_size,
                                      /*type=*/0x21, out)
// out[0] = 0 on success; non-zero = error code (use agxps_aps_parse_error_type_to_string)

n = agxps_aps_profile_data_get_kicks_num(profile_data)
agxps_aps_profile_data_get_kick_start       (profile_data, &starts[0],  start_idx, count)
agxps_aps_profile_data_get_kick_software_id (profile_data, &swids[0],   start_idx, count)
agxps_aps_profile_data_get_usc_timestamps   (profile_data, &tses[0],    start_idx, count)
```

## What we got: 48 kicks for a 10-dispatch trace on M4 Pro

The `gputrace-sample` workload was 5×`light_add` + 5×`heavy_alu`
interleaved, ~10 dispatches. One `Profiling_f` file (one of 20 USC
streams) decoded into 48 kicks. Software IDs cluster into 4 distinct
high-32-bit groups (likely 4 distinct kernels / pipeline-state hashes),
each with multiple sequence numbers spread across 3 sub-channels:

```
kick[ 0]  start=1739461754897    swid=0x3761d72234921602
kick[ 1]  start=3058016714783    swid=0x3761d72334921604
kick[ 2]  start=4294967296045    swid=0x3761d72434921606
kick[ 3]  start=2121713844245    swid=0x33920b3634921607   ← rare group
kick[ 4]  start=3079491551263    swid=0x3761d72234921601
...
kick[37]  start=314288526855510  swid=0x19e4518f3492160b
kick[38]  start=327478371421671  swid=0xe85… (bottom 5 are this group)
...
```

Bit layout we suspect for `swid`:
- bits 0..15: kick sequence number
- bits 16..31: trace/process tag (`0x3492` constant on this run)
- bits 32..47: USC-channel sub-index (3 values per kernel-group)
- bits 48..63: kernel-group hash (one per pipeline / clique type)

So the four high16 prefixes (`0x3761`, `0x3392`, `0x19e4`, `0xe85…`)
correspond to four distinct compute pipelines. We haven't verified
this against the streamData metadata yet — that's a documented
follow-up.

## The dispatch tree

`agxps_aps_parser_create` does an internal red-black-tree lookup
keyed on `(gen, variant, rev_with_aps_fallback)` to find a per-GPU
parser implementation. The tree lives in `__DATA.__bss` of
`GPUToolsReplay`:

| Slide | Tree address |
|---|---|
| `framework_base + 0x4DF0B578` | global root pointer |

In our VM run with `GPUToolsReplay` slid to `0x25742d000`, the tree
root was at `0x2a5406578`. **It's populated by C++ static
constructors that run during `dyld` load**, contrary to my earlier
assumption that something else triggered it. 53 entries on this
build — one per supported (gen, variant, rev) triple. Each tree
node is 0x38 bytes:

```
+0x00 left*
+0x08 right*
+0x10 parent*  (low bit = red/black flag)
+0x18 padding/color
+0x20 u64 (gen | (variant << 32))
+0x28 u32 rev_with_aps_fallback  + u32 padding
+0x30 PAC-signed function pointer  → per-GPU parser_create
```

The first node we examined had `(gen=15, variant=4, rev=1)` and a
parser-impl pointer `0x25747c03c`.

## Descriptor layout (96 bytes total, validated by per-GPU parser_create)

`agxps_aps_descriptor_create()` zeroes a 0x68-byte struct, then writes:

```
+0x00 = NULL                     (gpu pointer; you must populate)
+0x18 = 0x1000                   (acceptable: 0x400, 0x1000, 0x40000)
+0x30 = -1
+0x58 = 0x32
```

Per-GPU `parser_create` validates:

| Field | Required value |
|---|---|
| `desc[0x08]` (u32) | power of 2 in `[0x10, 0x800]` |
| `desc[0x0c]` (u32) | power of 2 in `[0x40, 0x2000]` |
| `desc[0x10]` (u32) | `0` OR power of 2 in `[0x80, 0x8000]` |
| `desc[0x18]` (u64) | exactly one of `{0x400, 0x1000, 0x40000}` |

`{0x100, 0x400, 0, 0x1000}` works. These are likely sample-period /
ring-buffer sizing knobs; default values that aren't outlandish for
a typical Xcode capture.

## GPU table

`agxps_aps_gpu_find_supported_revision(gen, variant, rev_in, *rev_out)`
exhaustively probed: 26 supported (gen, variant) pairs across gens
15..20, all rev=1.

```
gen=15 variant=4..7
gen=16 variant=3..6
gen=17 variant=2..7
gen=18 variant=1..4
gen=19 variant=2..7
gen=20 variant=2..3
```

**The variant is the GPU's size, not its marketing name.** The derived
counter GPU descriptors (`agxps_derived_counter_gpu_descriptor_create`)
give each variant's shape; for gen 16: variant 3 = 6 cores, 4 = 10 cores
(M4, `G16G`, 1 mGPU), 5 = 20 cores in 2 mGPUs (M4 Pro, `G16S`), 6 = 40
cores. gputrace picks the variant whose core and mGPU counts match the
`Configuration Variables` MTLReplayer logs into the profile
(`num_cores`, `num_mgpus`; `hw_counters::agxps_gpu_for_profile`). An
earlier version of this page said "M4 Pro is variant 3" because variant 3
decoded `Profiling_f_0.raw` without tile errors; the timing decode is
tolerant of the wrong size, but derived counters are not (their formulas
use the core and L2-bank counts), so the matched variant is used for both.

`agxps_gpu_create(gen, variant, rev, false)` allocates a 0x28-byte
heap struct with `[gen u32][variant u32][rev u32][rev_with_aps_fallback
u32][16 zero bytes]`. The VM build automatically writes
`rev_with_aps_fallback` after `find_supported_revision` runs, even
if rev_in == rev_out — different from the host build which only
writes when they differ.

## profile_type

The 4th argument to `parser_parse` is `0x21` — an enum value
(`eAPSProfilingType`). Confirmed working empirically; haven't
enumerated other values.

## Error reporting

`parser_parse` writes a u64 error code into `out[0]` on failure
(returns non-NULL `profile_data` even on error — only `out[0]` tells
you the parse succeeded). Decode with
`agxps_aps_parse_error_type_to_string(code) -> const char *`. Useful
codes seen so far:

| code | message |
|---|---|
| 0 | (success) |
| 2 | `Encountered tile start while a tile was still active on the same channel` (= wrong GPU gen/variant for the data) |

## profile_data getters (VM build, 9 functions)

All `profile_data_get_*_by_index` functions take `(profile_data, out_buf,
start_idx, count)` and return `int` (1 = success, 0 = bounds
violation). Output buffer is filled with `count` u64 values.

```c
unsigned int agxps_aps_profile_data_get_kicks_num     (void *pd);
int          agxps_aps_profile_data_get_kick_start    (void *pd, uint64_t *out, size_t start, size_t count);
int          agxps_aps_profile_data_get_kick_software_id(void *pd, uint64_t *out, size_t start, size_t count);
int          agxps_aps_profile_data_get_usc_timestamps(void *pd, uint64_t *out, size_t start, size_t count);
unsigned int agxps_aps_profile_data_get_counter_num   (void *pd);
// ...get_counter_index, get_counter_values_by_index, get_counter_group_metadata_by_index,
//    get_counter_values_num_by_index — virtual dispatch through profile_data internals,
//    haven't worked out their full sigs yet.
```

For the qa-decode trace's Profiling_f_0:
- `get_kicks_num` = 48
- `get_counter_num` = 12
- `usc_timestamps` vector length = 115 394 (per-USC sample timestamps)

### Each `Profiling_f_N.raw` is one USC bank's view

The `.gpuprofiler_raw` directory contains one `Profiling_f_N.raw` per
USC bank (20 of them in our trace). Each file decodes to its own kick
list — kick counts vary across files (48, 35, 44, 36, 27, 39, 21, …)
because dispatches fan out to different cores/banks. To get the full
GPU-side view of a trace, the consumer has to decode every file and
union/aggregate.

### Kick `software_id` → pipeline (= dispatch source)

Empirical mapping from a 5× light_add + 5× heavy_alu sample, decoded
across all 20 banks:

| swid high16 | per-bank kicks | inferred mapping                          |
| ----------- | -------------- | ----------------------------------------- |
| `0x0e84`    | always 5       | light_add (5 dispatches × 1 kick/bank)    |
| `0x3392`    | always 1       | driver-inserted setup/cleanup kick        |
| `0x19e4`    | 3..18 / bank   | heavy_alu fanout                          |
| `0x3761`    | 12..24 / bank  | heavy_alu fanout                          |

`0x0e84` is identifiable on its own: 5 kicks per bank, every bank,
totalling exactly the user's 5 light_add dispatches. heavy_alu kicks
are spread across multiple high16 prefixes per bank; the prefix is
*not* uniquely a pipeline ID — it likely encodes a (pipeline,
shader-core-cluster) tuple. The low 48 bits of swid look like a
combination of dispatch sub-ID + a per-kick monotonic counter
(visible as `0x...160X` suffixes that march by 1 per kick within a
group).

Bottom line: `swid >> 48` is a *coarse* clique-key that's stable for
small dispatches (light_add) but ambiguous for big fanned-out ones
(heavy_alu). Fully resolving "which dispatch did this kick come from"
needs another field — likely the streamData side of the bridge
(`pipeline_id` or kernel address).

### `usc_timestamps` encoding is unclear

The vector probed at 115 394 entries on Profiling_f_0. Raw values
are tiny (0..29 M) compared to `kick_starts` (1.7 T..463 T) and start
with 8 zeros, then a stride-of-256 pattern (`0xff, 0x1ff, 0x2ff, ...,
0x6ff, 0x700`). This is *not* a parallel sequence of GPU ticks. Best
guess: packed indices, sample-counts, or per-kick-relative offsets.
Decoding requires either the vtable for `get_counter_values_by_index`
(which probably exposes the same data with proper structure) or more
RE on the host's richer
`agxps_aps_kick_time_stats_create_sampled` build.

### `kick_start`, `kick_end`, `synchronized_timestamps` are packed

The u64 values returned by `agxps_aps_profile_data_get_kick_start`,
`get_kick_end`, and `get_synchronized_timestamps` are NOT plain
timestamps — they're packed as:

```
high 32 bits = profile-clock tick count
low  32 bits = usc_sample_index
```

Cross-check from `synchronized_timestamps`:

| index | raw value           | high32 (time) | low32 (sample) |
| ----- | ------------------- | ------------- | -------------- |
| 0     | `0x0000000e_00000000` | 14          | 0              |
| 1     | `0x00000024_00000001` | 36          | 1              |
| 5373  | `0x0001c2b3_000014fd` | 115 379     | 5 373          |

The low32 of `synchronized_timestamps[i]` is exactly `i`, confirming
it's the sample index. For `kick_start` / `kick_end`, the low32 is
the index into the `usc_timestamps` array of the sample that bracketed
the kick boundary.

**Profile-clock period (M4 Pro):** ~3.89 ns/tick, derived empirically
by mapping the trace's tick span (108 285 ticks for the qa-decode
sample) against Xcode's reported wall total of 421.08 µs:

```
108 285 ticks × 3.89 ns/tick ≈ 421.23 µs   (vs Xcode 421.08, 0.04% off)
```

That clock corresponds to ~257 MHz, consistent with an Apple GPU PMU
sampler clock.

### Counters in USC profile data are metadata-only

`agxps_aps_profile_data_get_counter_num` returns 12 on our trace —
but every counter's values vector is empty. Both value APIs report
length 0:

- `get_counter_values_by_index` / `get_counter_values_num_by_index`
- `get_counter_values` / `get_counter_values_num` after resolving the
  counter's ident with `agxps_counter_get_ident`

Conclusion: USC profile data (parsed with `profile_type=0x21` from
`Profiling_f_*.raw`) ships counter *metadata* (12 named counter
slots) but no actual counter values. Hardware counter samples for the
exported bundle are not exposed through this profile-data object.

### `Counters_f_*.raw` is not Xcode's RDE counter stream

The file name is misleading. In the sample bundle, the `streamData`
metadata labels `Counters_f_*.raw` as `Source=APS_USC` and
`SourceIndex=5`, the same USC source family as `Profiling_f_*.raw`.
The RDE/BMPR hardware-counter data is instead embedded in
`streamData` `APSCounterData` entries as `ShaderProfilerData` blobs
whose payload starts with `GPRWCNTR`.

That matches Xcode's private importer behavior:

```text
cargo run -p agxps-sys --example probe_rde_importer -- \
  /tmp/gputrace-sample/sample-perfdata/sample.gpuprofiler_raw/Counters_f_0.raw

AGXPCTR2 occurrences: 0
CTRSAMPL occurrences: 0
GPRWCNTR occurrences: 0
direct RDE-record-shaped chains (static scan): 0
_parseAGXBlock(full file): 0
parseRDEBuffer(candidate slices): 0 successful guesses
```

Running that across all 20 `Counters_f_*.raw` files gives the same
result. `agxps_aps_parser_parse` also rejects those files with
"tile start while a tile was still active" for every tested
`profile_type` (0x10, 0x20, 0x21, 0x22, 0x40, 0x100), but the more
important finding is that `XRGPUATRCImporter` does not recognize them
as AGX `CTRSAMPL` blocks or direct RDE buffers either.

So the useful hardware-counter route is the one `gputrace raw-counters`
already follows: decode `streamData`'s `APSCounterData`/`GPRWCNTR`
payloads. That gives pass/source/ring and encoder-sample aggregates.
It does **not** by itself give per-dispatch/per-pipeline cost. In the
sample trace there is only one encoder sample row for the whole
encoder, so it cannot distinguish the five `light_add` dispatches from
the five `heavy_alu` dispatches.

`raw-counters` now prints the timestamp alignment explicitly. On the
sample, all `GPRWCNTR` records are before the profiler dispatch
windows:

```text
profiler dispatches=10 ticks=16161378792923-16161378801666 span=8743 (364.292 us)
raw counter records=18096 ticks=16161292897771-16161372070097 span=79172326 (3298.847 ms)
direct_overlap_records=0 before_dispatch=18096 after_dispatch=0
raw_end_to_dispatch_start_gap=6722826 (280.118 ms)
```

The diagnostic shift `raw_start_to_dispatch_start` maps 58 raw records
onto all 10 dispatches, but it is not a validated join. Evaluating AGX
derived counters under that shift yields fragment/texture-style metrics
(`Fragment Generator Primitive Utilization`, `Texture Filtering
Utilization`, `Average Overdraw`) rather than Xcode's compute command
metrics. The AGX G14G catalog keys for Xcode's compute table
(`CSInvocation`, `CSALUInstructions`, `CSALUF32Percent`) are defined in
the system counter plists, but their raw hashes are not present in this
`GPRWCNTR` schema. Treat shifted dispatch rows as RE diagnostics only,
not as counter-derived execution cost.

### Counter names are obfuscated; map needs RE

Counter names returned by `get_counter_names` are 64-char uppercase
hex strings (e.g. `79E88035C9BC883D403F17831B8C9264E643C6B76E9B3C1451B49B0F672C32BF`),
not human-readable. They look SHA-256-shaped, but the simple
`SHA256(name)` route is not confirmed: that example does not match
any obvious `GPUCounterGraph.plist` display/vendor/unit/description
string under basic normalization. The framework exports a
deobfuscation API:

| Symbol | Purpose |
| ------ | ------- |
| `agxps_load_counter_obfuscation_map(const char *path)` | load mapping (pass NULL for default) |
| `agxps_unload_counter_obfuscation_map()` | clear map (exported as C++ mangled `_Z36agxps_unload_counter_obfuscation_mapv` for `dlsym`; `nm` shows the Mach-O leading underscore too) |
| `agxps_counter_deobfuscate_name(const char *hash)` | hash → readable |
| `agxps_counter_obfuscated_name(const char *name)` | readable → hash |

The de/obfuscation functions are only map lookups, not hash
calculators. `agxps_load_counter_obfuscation_map(path)` accepts a
CSV-like text map with exactly two columns per row:

```csv
readable counter name,64-hex obfuscated counter name
```

Then `agxps_counter_deobfuscate_name(hash)` returns the readable name
and `agxps_counter_obfuscated_name(name)` returns the hash. Rows with
anything other than two fields are skipped with the internal warning
`Skipping invalid raw counter mapping`.

Calling `load_counter_obfuscation_map(NULL)` asks
`[NSBundle bundleWithIdentifier:@"com.apple.gpusw.AGXProfilingSupport"]`
for `pathForResource:@"RawCountersMapping" ofType:@"csv"`, then reads
the result as UTF-8 text. It returns 0 on the tested Xcode build
because that bundle/resource is not present on macOS. Nearby binary
strings also name `/Apple/Internal/Library/AGX/`,
`AGXRawCounterMapping.csv`, `RawCountersMapping.csv`, and
`AGXCounterMapping.csv`, but those files do not ship in this Xcode
bundle.

Passing the bundled `GPUCounterGraph.plist` returns success but does
not create useful entries; that plist is the *display-name* graph, not
the agxps obfuscation map.

### Why `(kick_end_time - kick_start_time)` is NOT compute time

Even with the packing fixed, summing `(end_time - start_time)` per
kick gives ~10× too much for small kernels (e.g. light_add yields
9.3 % of summed lifetime vs Xcode's 0.10 % of compute). That's
because the kick *lifetime* (event start → event end) includes wait
windows, cross-kick synchronization, and any overlap with parallel
kicks on other cores.

`agxps_aps_kick_time_stats_create_sampled` is callable from Rust with
a real Objective-C block filter, but it is also lifetime-shaped. It
does not expose Xcode's ALU/compute-cost metric by itself.

### Useful paths: timing analyzer ESL address + clique duration

The high-value path is `agxps_aps_timing_analyzer_*`, not
`Counters_f_*.raw`:

1. Parse every `Profiling_f_*.raw` as USC samples (`profile_type =
   0x21`).
2. Feed each parsed profile into
   `agxps_aps_timing_analyzer_process_usc` + `finish`.
3. Fetch command records with:
   - `get_work_start`
   - `get_work_end`
   - `get_work_shader_address`
   - `get_esl_shader_address`
   - `get_work_cliques_average_duration`
   - `get_num_work_cliques`
   - `get_kick_software_id`
4. Use `get_esl_shader_address`, not `get_work_shader_address`, to
   bridge into MIO. The ESL address exactly matches
   `shaderBinaryInfo.raw2` for executable shader-binary references
   (`raw5 == 6 && raw6 == 28`).
5. For a fast command-level cost proxy, aggregate
   `num_work_cliques * work_cliques_average_duration` per matched ESL
   shader address.

In the sample trace, this command-level aggregate produces:

```text
Pipelines by AGXPS timing-analyzer clique duration:
  99.974% weighted=   112335150 avg_sum=  17555881 rec_cliques=    640 heavy_alu
   0.026% weighted=       28672 avg_sum=     28672 rec_cliques=      5 light_add
```

This route is much cheaper than walking every work clique because it
uses the timing analyzer's per-command aggregates directly. It is not
yet proven to be Xcode's exact "compute cost" denominator, but it is
the strongest current candidate for a fast, per-pipeline signal.

Xcode ground truth from the same trace's **Counters → GPU Commands →
Compute Kernel** table is per-dispatch `Execution Cost`:

```text
light#0   0.020%   heavy#0  18.379%
light#1   0.011%   heavy#1  16.476%
light#2   0.028%   heavy#2  21.367%
light#3   0.030%   heavy#3  21.376%
light#4   0.006%   heavy#4  22.306%

pipeline totals:
heavy_alu  99.904%
light_add   0.095%
```

The matching **Pipeline Statistics** inspector rows show that Xcode's
displayed command `Execution Cost` is derived from function execution
time divided by the profile's total GPU time:

```text
total GPU time: 417.25 us

light_add dispatches:
  #8   82 ns
  #20  46 ns
  #32 118 ns
  #44 124 ns
  #56  27 ns
  total 397 ns = 0.095% of total GPU time

heavy_alu dispatches:
  #14 76.69 us
  #26 68.75 us
  #38 89.16 us
  #50 89.20 us
  #62 93.07 us
  total 416.85 us = 99.904% of total GPU time
```

Sorting MIO executable ESL shader addresses ascending gives the same
interleaved dispatch order as Xcode's GPU Commands table:

```text
0x...b8000 light#0
0x...b80c0 heavy#0
0x...b8180 light#1
0x...b8240 heavy#1
...
0x...b8600 light#4
0x...b86c0 heavy#4
```

The current `xcode-mio --format raw-text` report prints this as
`AGXPS timing rows by ESL shader address`. On the sample, analyzer
weighted duration is close for the heavy rows but undercounts the tiny
light rows by roughly an order of magnitude. The `w1` instruction-stat
join catches more light work in aggregate, but is noisier per dispatch
and over-attributes `light#0`/`light#4`. Do not treat either metric as
exact Xcode `Execution Cost` yet.

Non-synthetic sanity check:
`/Users/amos/bearcove/bee/target/gputrace-captures/qa-decode-ar-legacy.gputrace`
paired with
`qa-decode-ar-legacy-perfdata/qa-decode-ar-legacy.gpuprofiler_raw`
decodes 908 GPU commands and 15 pipeline states. Our
analyzer-weighted output for that trace was:

```text
tq6_1s_matvec_prerot_qa                    76.867%
tq6_1s_matmul_prerot_tile_wide             17.462%
tq1s_attention_sequence_gqa_roped_cache     1.408%
tq1s_fill_range_attention_mask              1.243%
tq6_fused_rmsnorm_activation_quantize        0.677%
```

The same Xcode values can be copied directly from **Counters → GPU
Commands → Compute Kernel**. Pasting those cells to `/tmp/ha.txt`
gave 908 command rows, matching the MIO command count exactly, and
the copied `Execution Cost` column sums to 99.993%. The helper command
for this validation is:

```sh
GPUTRACE_PROFILER_DIR=/path/to/qa-decode-ar-legacy.gpuprofiler_raw \
  cargo run -- xcode-command-costs \
    /Users/amos/bearcove/bee/target/gputrace-captures/qa-decode-ar-legacy.gputrace \
    --table /tmp/ha.txt
```

The pasted Xcode pipeline totals are:

```text
tq6_1s_matvec_prerot_qa                    69.999%
tq6_1s_matmul_prerot_tile_wide              9.483%
tq1s_attention_sequence_gqa_roped_cache     9.182%
tq6_fused_rmsnorm_activation_quantize        4.365%
tq6_activation_quantize                      2.407%
tq6_silu_mul_activation_quantize             1.366%
tq1s_argmax_rows                             1.312%
tq1s_add_residual_rms_norm_heads_to          0.656%
tq1s_rms_norm_qk_pair_q_rope_to              0.444%
tq1s_quantize_and_store_k_tq8_v_tq4          0.406%
tq1s_add_inplace                             0.305%
tq1s_rms_norm_heads_to                       0.022%
tq1s_rotate_hidden                           0.019%
tq6_1s_rows                                  0.017%
tq1s_fill_range_attention_mask               0.010%
```

Comparison against the full pasted table:

```text
metric     MAE       RMSE      max error  top-5 MAE
analyzer   2.215pp   3.590pp   7.979pp    5.657pp
w1         1.903pp   3.282pp   7.894pp    4.879pp
cmd-count  5.360pp   8.452pp  26.830pp   10.164pp
```

So `w1` is slightly better than analyzer-weighted duration on this
real workload, while analyzer-weighted duration was better on the
synthetic heavy/light aggregate. Neither metric is the Xcode shader
cost denominator. The current output treats them as candidate metrics,
not as ground truth.

### Xcode's private command-time path

Disassembly of
`GPUToolsAdvancedUI.GTProfilingTimelineDataSource.shaderProfilerCostPercentage`
shows the exact UI path Xcode uses for shader/command cost:

1. Get `shaderProfilerResults()` from the timeline data-source proxy.
2. Read `result.timingInfo.time` as the denominator.
3. For a command, call
   `gpuCommandForFunctionIndex:subCommandIndex:`.
4. Read the returned command's `timingInfo.computeTime` for compute
   shaders.
5. Display `computeTime / result.timingInfo.time * 100`.

The Swift ABI-visible wrapper passes `forDraw` as `UInt32` and
`subCommandIndex` as `Int32`.

The standalone `xcode-mio` processor can obtain a
`GTMioShaderProfilerResult`, and that object responds to
`gpuCommandForFunctionIndex:subCommandIndex:`. On the sample trace,
probing likely draw IDs produced:

```text
processor.shader_result:
  class=GTMioShaderProfilerResult
  gpuCommand scan attempts=138
  non_nil=13
  first_class=GTMioShaderProfilerGPUCommand
  timing_info=0
  nonzero=0

direct.shader_result:
  class=GTAGX2ShaderProfilerResult
  gpuCommand scan attempts=138
  non_nil=0
```

The reason `timing_info` is zero is visible in
`-[GTMioShaderProfilerGPUCommand timingInfo]`: it does not return a
stored timing object. It calls
`costForLevel:levelIdentifier:scope:scopeIdentifier:cost:` with
`scope=4`, then scales the returned cost by `globalGPUTime`. Our
standalone `requestCostTimeline`/`costTimeline` path currently returns
the right topology (`drawCount`, `pipelineStateCount`,
`globalGPUTime`) but zero cost records/cost values. In other words,
the remaining exact-Xcode blocker is not the cost formula; it is
populating the same private cost timeline data that Xcode's UI-backed
object has.

This makes the private-framework path useful as an oracle and RE aid,
but too fragile to be the product path. The stable path should keep
using native profile decoding plus Xcode-copied fixtures to measure
error bounds.

`GTMioNonOverlappingCounters` is also not the missing public Compute
Kernel table. For this raw-directory profile, its per-command rows only
carry the internal `PredicatedALUPercentage=100` value; the display
columns Xcode shows (`Execution Cost`, `Kernel Invocations`, `Kernel
ALU Instructions`, `Kernel ALU Float Instructions`) are not populated
through that object.

The Xcode **Pipeline Statistics** inspector is a much stronger lead
than the counter panes. For selected dispatches it shows **Total
Function Execution Time** and per-dispatch times. Manual samples from
the same `qa-decode` trace:

```text
pipeline      function                         Xcode cost  Pipeline Stats time  implied denom
0x72a65dc00   tq6_1s_matmul_prerot_tile_wide      9.483%          1.48449 ms       15.654 ms
0x72a63ad80   tq1s_attention_sequence...          9.182%          1.47 ms          16.010 ms
0x72a639c00   tq6_fused_rmsnorm_activation...     4.365%          548.13 us        12.557 ms
0x72a65df80   tq1s_argmax_rows                    1.312%          124.45 us         9.486 ms
0x72a639880   tq6_1s_rows                         0.017%          2.87 us          16.882 ms
0x72a639500   tq1s_fill_range_attention_mask      0.010%          789.01 ns         7.890 ms
```

For `0x72a65dc00`, Xcode's left-tree dispatch costs are exactly time
shares over a ~15.51 ms denominator:

```text
#1529 795.66 us / 15.51 ms = 5.13%
#3065 688.83 us / 15.51 ms = 4.44%
```

`0x72a63ad80` also lines up: 1.47 ms / 9.182% = ~16.01 ms. This
explains the largest AGXPS miss: Xcode's table is not using the
timing-trace `w1` or analyzer-weighted counters for that kernel; it is
using a Pipeline Statistics function-time model.

The inconsistent denominators on `0x72a639c00`, `0x72a65df80`, and
`0x72a639500` are likely shader-variant/selection effects. In the
inspector, some dispatches under the same pipeline show `0 ns` while
the copied GPU Commands table has nonzero `Execution Cost` rows for
them. The selected `[Just-In-Time]` statistics node is therefore not
always the complete per-pipeline cost source. The next RE target is to
extract **all** Pipeline Statistics shader-variant function-time rows
from Xcode's model, then sum them by pipeline address.

The older, heavier route is still useful for RE: join work cliques to
timing-analyzer command spans by system-time overlap, then call
`agxps_aps_clique_instruction_trace_get_instruction_stats`.

In the sample trace, `xcode-mio` now filters analyzer records to ESL
addresses present in MIO's executable shader-binary references and
aggregates `instruction_stats.words[1]`:

```text
Pipelines by AGXPS timing-trace instruction stats:
   98.91% w1=  1308837836 analyzer_weighted=   112335150 events=  9539685 cliques= 410372 cmds= 100 heavy_alu
    1.09% w1=    14479123 analyzer_weighted=       28672 events=    43087 cliques=   1359 cmds=   5 light_add
```

That is the first route that gives a real per-pipeline split from the
raw exported profile directory. Internal profiler/runtime shader rows
also appear in the timing-analyzer output (`0x100013a0000`,
`0x100013a0140`, etc.), but they are excluded because they are not
present in MIO's user-pipeline executable shader-binary references.

Observed `AgxpsApsInstructionStats` layout is 14 `u64` words. On this
fixture, only `words[0]` and `words[1]` are materially nonzero. We use
`words[1]` as the cost weight because it gives the most Xcode-like
heavy/light split; the exact semantic name of that field still needs
RE.

### Dead end: ESL clique instruction traces

The Xcode framework also exports `esl_clique_*` profile-data getters:

- `agxps_aps_profile_data_get_esl_cliques_num`
- `agxps_aps_profile_data_get_esl_clique_start`
- `agxps_aps_profile_data_get_esl_clique_end`
- `agxps_aps_profile_data_get_esl_clique_esl_id`
- `agxps_aps_profile_data_get_esl_clique_kick_id`
- `agxps_aps_profile_data_get_esl_clique_clique_id`
- `agxps_aps_profile_data_get_esl_clique_instruction_trace`

Those names looked promising, but on the sample trace their
instruction traces return zero `words[1]` after the same timestamp join
that works for work cliques. They are not the missing cost source.

`agxps_aps_profile_data_get_work_clique_esl_id` is also not the MIO
shader-address bridge. It returns small IDs such as `0xa` and `0xb`,
and those IDs are shared across heavy/light rows. The robust bridge is
still the timing analyzer's `get_esl_shader_address`.

## Open questions / next steps

1. **Name `AgxpsApsInstructionStats.words[0/1]`.** The timing-trace
   join works, but the exact semantic labels for the two nonzero
   fields are still unknown.

2. **Reduce integrated runtime.** The current `xcode-mio` integration
   still walks all work cliques and calls the private stats getter for
   each matched clique to populate the `w1` section. The analyzer
   weighted-duration section does not need that pass, so the obvious
   next performance step is to make the expensive instruction-stat
   join opt-in.

3. **`get_counter_*` virtual dispatch.** The counter-value getters
   call into a vtable on the parser. Need to either decode the vtable
   layout or just call them blindly with reasonable signatures.

4. **profile_type semantics.** `0x21` works; haven't tried other
   values. `eAPSProfilingType` enum likely has options for
   per-kick, per-instruction, etc.

## Tooling used

- **macOS VM with SIP off** + `lldb-rpc-server` exposed via MCP:
  drove the entire experiment by attaching to MTLReplayer pre-main,
  reading the populated dispatch tree, calling functions via
  `expression`, populating descriptors, single-stepping into the
  per-GPU parser.
- The Apple GPU is paravirtualized in the VM, so we can't *generate*
  perfdata in the VM — but we don't need to. The decoder runs fine
  without a real GPU. Copied a real `Profiling_f_0.raw` from the
  host (where `gputrace profile` had generated it via MTLReplayer
  on actual hardware) and decoded it successfully.

## Implications for gputrace-rs

The Xcode-rich path now uses the exported agxps symbols from
`GTShaderProfiler.framework` directly:

1. `dlopen` GTShaderProfiler via `agxps-sys`.
2. Parse every `Profiling_f_*.raw`.
3. Run the timing analyzer.
4. Match timing-analyzer ESL shader addresses to MIO
   `shaderBinaryInfo.raw2`.
5. Aggregate timing-analyzer weighted clique duration per pipeline.
6. Optionally aggregate work-clique instruction stats per pipeline for
   the slower `w1` RE metric.

`timing.md` now includes an `AGXPS analyzer-weighted pipeline cost`
section when a precomputed Xcode-MIO summary is available, and the
standalone `timing` command can request the same slow path with
`--agxps`. Without AGXPS, `timing` continues to print streamData's
dispatch-cadence timing and explicitly labels it as not real GPU cost.

The no-Xcode fallback still cannot do this through
`GPUToolsReplay.framework` without extra symbol-resolution work: the
same agxps text exists there, but the needed symbols are not exported
in the dyld trie.
