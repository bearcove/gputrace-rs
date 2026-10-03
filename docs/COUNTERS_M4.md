# Hardware counters on M3/M4-family GPUs

`counters.md`, the `Occ %` / `ALU %` / `LLC %` / `Dev BW` columns of
`shaders.md`, the counter insights in `insights.md` and the limiter tracks of
the timeline all come from one place: `src/hw_counters.rs`. This page says
where its numbers come from, how they are joined to encoders, dispatches and
kernels, and what has been checked against kernels whose answer is known.

Verified on an M4 (`G16G`, 10 cores, ahri) and an M4 Pro (`G16S`, 20 cores in
2 mGPUs, scooter), Xcode 27 / macOS 27.

## What was wrong before

- **Wrong counter catalog.** `counters.md` evaluated the JavaScript derived
  counter scripts that ship in the AGX driver bundles. Those exist only for
  G13/G14 (M1/M2); on a G16 capture the M2 `G14G` script was chosen, so the
  names (`Frag Ticks Count`, ...) and formulas belonged to another GPU. The
  scripts are now only used for their own generation
  (`counter::choose_agx_derived_script`).
- **Byte scanning.** "Occupancy" was the median of every f32 in
  `Profiling_f_*.raw` that fell in (0, 1] (a constant ~0.59 on every
  encoder). "Limiters" were floats scanned out of `Counters_f_*.raw`, read as
  if `Counters_f_N` held one named counter (a table ported from the Go
  tool). `Counters_f_N` is the APS stream of shader core N. All of this is
  deleted.
- **No per-dispatch join.** The "Derived Counter Sample Data" passes record
  one counter set per kick, holding only the counts since the last periodic
  sample (the final ~10 us of the kick): per-kick totals came out at ~1/30 of
  the truth on the oracles. Their raw records also never overlapped the
  dispatch tick windows. These passes are not used.
- Re-running the replay with `-batchIdFilter` gives graphics counters only;
  it is not a route to compute per-dispatch counters.

## Where the numbers come from

MTLReplayer's profile (`gputrace-profile/*.gpuprofiler_raw`) contains several
counter passes in `streamData` → `APSCounterData`. Only the **limiter pass**
(the entry with a `Limiter Counter List Map`) is time-sampled on every unit,
so only it is used:

| stream | where | sampling | clock |
|---|---|---|---|
| `APS_USC`, one per shader core | `Counters_f_<core>.raw` (`APSTraceDataFile`) | every `CountPeriod` core cycles (16384 on M4) | continuous ticks |
| `RDE_0`, `BMPR_RDE_0` (GPU-global), one per ring | `GPRWCNTR` records in `ShaderProfilerData`, type 6 | ~10 us | absolute ticks |
| `Firmware` | `GPRWCNTR` records, type 5: one per finished kick | per kick | absolute ticks |

Each sample is a delta since the previous sample of the same stream.

Decoding is Xcode's own: `agxps` in `GTShaderProfiler.framework`
(`crates/agxps-sys/src/counters.rs`).

- **GPU.** The agxps variant is chosen by matching `num_cores` / `num_mgpus`
  from the profile's `Configuration Variables` against the variants'
  derived-counter descriptors (gen 16: v3 = 6 cores, v4 = 10 cores = M4,
  v5 = 20 cores / 2 mGPUs = M4 Pro, v6 = 40 cores).
- **USC streams** are parsed with `agxps_aps_parse` using the profile's
  `APS Options` (`PulsePeriod`, `SystemTimePeriod`, `CountPeriod`,
  `ChunkSize`) and the uarch layout flag
  (`agxps_aps_get_uarch_behaviour_from_GRC_counter_list`). The parsed profile
  gives per-sample counter values and system times, the kicks (software id =
  `encoder trace id << 32 | kick id`), the timing analyzer's commands (kick,
  shader-launch program address, start) and the work cliques (start, end,
  kick, running shader-launch index, slot).
- **Derived counters** are evaluated by
  `agxps_counter_compute_derived_counters` on the summed raw counts of a row,
  with `GPUCycles` / `DeltaSeconds` and the constants `NUM_CORES`,
  `NUM_L2_BANKS` (2 × `num_gps`), `NUM_AGCS`, `NUM_GPS`, `NSEC_PER_SEC`,
  `TIME_SCALE` (the profile timebase, 125/3 ns per tick),
  `OMU_EVAL_WINDOW_DEFAULT`. Every derived counter whose raw inputs are all
  present is computed; `counters.md` shows the subset in `METRICS`.

## The join

1. **Encoders.** `TraceId to BatchId` maps a kick's encoder trace id to the
   encoder index of the capture. Kicks with no mapping are *foreign*:
   replayer-internal work or other processes.
2. **Dispatches.** `Program Address Mappings` entries of type `compute-sl`
   give (`encID`, shader-launch program address) → `drawCallIndex`; type
   `compute` gives the kernel's program address for that dispatch.
3. **Clocks.** USC streams run on continuous time, `RDE`/`Firmware` on
   absolute time. The offset is the median difference between the same
   software id's kick start in both.
4. **Cliques to dispatches (per USC).** A work clique names its dispatch by
   `esl_index`, a running index of the shader-launch programs that core has
   seen (not the analyzer's command index: the analyzer drops commands and
   foreign kicks interleave). Within a kick, each analyzer command claims the
   unclaimed clique group whose first clique starts closest to the command
   start (`match_commands`); the command's program address then names the
   dispatch via step 2.
5. **Clique ends.** The limiter pass does not trace clique ends. A missing
   end is trusted as a duration only when the slot is handed straight to
   another clique of the same command; any other end is capped at that
   command's median measured duration on the core, or at the next other
   command's start in the kick.
6. **Samples to rows.** A sample is shared by the kicks that overlap it in
   proportion to their clique residency inside the sample (plain time overlap
   when no clique is visible); foreign kicks take their share and drop it.
   Within a kick, the share is split between that kick's mapped dispatches by
   their residency. Samples a kick covers with no clique go to the dispatch
   whose span (first start to last end) covers them, else to the kick's most
   recently started dispatch, which is then also credited that time. A sample
   may start up to 64 ticks before a kick or clique and still be charged to
   it (`GPUTRACE_HW_GUARD_TICKS`).
7. **Units.** Raw counts add across cores; cycles and seconds are divided by
   the number of USC streams. GPU-global sources form one family each; their
   rings are instances of the same block, so counts add and the lowest ring
   gives the time.
8. **Kernels** are dispatch accumulations summed by kernel name (from the
   profiler's dispatch table), or by kernel program address when unnamed.

Per-encoder rows are exact up to the foreign-kick caveat: every sample of the
encoder's kicks is the encoder's. Per-dispatch rows are exact when a
dispatch has its core to itself; where dispatches of one encoder overlap on a
core they are a residency split (`shrd%` in `counters.md`).

`frgn%` is the fraction of a row's time during which a foreign kick ran.
Counters are per unit, not per process: those counts are mixed in. Profile on
a quiet GPU.

## Oracle kernels

`gputrace synth-bench --counter-oracle /abs/path/oracle.gputrace` captures five
single-kernel encoders whose counts are known from the source, then one
encoder (`oracle_mixed`) running all five again on their own buffers, to check
per-dispatch attribution against the single-kernel truth. Buffers are filled
with random data (zero pages compress and under-count DRAM).

| kernel | grid | known quantities |
|---|---|---|
| `oracle_alu` | 160 × 1024 | 8 FMA chains × 4096: 32768 F32 FMA per thread, one 4 B store |
| `oracle_copy` | 4096 × 1024 | 64 MiB read, 64 MiB written, one float4 per thread |
| `oracle_read` | 256 × 1024 | 64 MiB read (16 float4 per thread), 1 MiB written, 67 F32 adds per thread |
| `oracle_tgmem` | 160 × 1024 | 4096 dependent 4 B threadgroup loads per thread |
| `oracle_lowocc` | 80 × 32 | one 32-thread SIMD group per threadgroup holding 32 KiB of threadgroup memory |

M4 (ahri), one capture replayed 4 times, per-encoder rows (`gputrace
hw-counters`):

| quantity | expected | measured (4 replays) |
|---|---|---|
| copy: DRAM bytes read | 67,108,864 | 6.7128e7–6.7132e7 (+0.03 %) |
| copy: DRAM bytes written | 67,108,864 | 6.5587e7–6.5835e7 (−1.9 … −2.3 %) |
| copy: threads | 4,194,304 | 4,194,304 (3 replays), 4.1663e6 (1) |
| copy: DRAM read + write bandwidth | | 93–96 GB/s of 119.5 peak |
| read: DRAM bytes read | 67,108,864 | 6.7122e7–6.7124e7 (+0.02 %) |
| read: DRAM bytes written | 1,048,576 | 0.49–0.60 M |
| read: threads | 262,144 | 262,144 |
| read: ALU F32 instructions | 17,563,648 | 1.7564e7 |
| alu: ALU F32 instructions | 5.3687e9 FMA + 1.1e6 epilogue adds | 5.3667e9–5.3711e9 |
| alu: threads | 163,840 | 151,552 (3 replays), 160,768 (1): 92.5–98 %, first encoder of the replay |
| alu: F32 Utilization | high | 87.4–87.5 % |
| tgmem: threads | 163,840 | 163,840 |
| tgmem: ALU F32 instructions | 671,744,000 | 6.7174e8 |
| tgmem: threadgroup load bandwidth | 654 GB/s (2.68 GB in 4.10 ms) | 441–443 GB/s (0.68×) |
| alu, tgmem, lowocc: DRAM | ~0 | 0.3–0.5 GB/s of writes (see unknowns) |
| lowocc: threads | 2,560 | 2,560 |
| lowocc: ALU F32 instructions | 83,886,080 | 8.3885e7–8.3889e7 |
| lowocc: Compute Occupancy | two 32 KiB threadgroups (one SIMD group each) fit a core: 2 of 96 SIMD groups | 2.1 % (`Compute Simdgroups Inflight Per Shader Core` 1.97, `L1 Threadgroup Bytes Occupancy` 63.9 KB) |
| alu, tgmem: Compute Occupancy | 3 × 32 SIMD groups of 1024-thread groups | 91 % (87.9 of 96) |
| copy, read: Compute Occupancy | | 33 %, with `Occupancy Manager Target` 35 % (100 % on the other oracles): the occupancy manager caps DRAM-streaming kernels |
| alu: F32 Limiter | F32-bound | 99 % |
| `oracle_mixed`: threads, F32 instructions | sums of the five | exact (4,786,688; 6.1444e9) |

Per dispatch inside `oracle_mixed`, against the known values (same four
replays; the five dispatches run without barriers and overlap):

| dispatch | DRAM bytes | ALU F32 instructions | threads |
|---|---|---|---|
| alu | – | −0.1 … +0.0 % | exact |
| copy | read +0.5 … +3.6 %, written +1.2 … +2.3 % (vs. the single-kernel encoder) | 2–6 M leaked in from neighbours (should be 0) | +0.2 … +0.3 % |
| read | read −0.6 … −3.8 % | −0.3 … +14 % | −1.8 … −3.2 % |
| tgmem | – | −0.4 … +0.4 % | −0.9 … −3.7 % |
| lowocc | – | −3.1 … +0.2 % | −17 … 0 % (2136–2560 of 2560) |

Long dispatches are good to a few percent; short ones next to long ones lose
or gain samples at their edges.

M4 Pro (scooter, earlier clean run, same oracles): copy read 67.13 MB and
threads exact; read 67.1 MB, F32 instructions 1.756e7 exact, ~188 GB/s DRAM;
lowocc occupancy 2.1 %; tgmem threadgroup load bandwidth 785 GB/s, 0.68× the
expected bytes again.

## What is still unknown

- **Threadgroup-memory bandwidth scale.** Both GPUs report 0.68× the bytes the
  oracle loads. The counter or its formula counts something narrower than 4 B
  per thread load; the absolute value is not trustworthy, ratios between
  kernels probably are.
- **Thread shortfall on the first encoder.** `Compute Threads Launched` of the
  first encoder in a replay is 92.5–98 % of the grid while its instruction
  counts are complete. Not explained; a larger guard band (up to 16384 ticks)
  does not recover it.
- **ALU Utilization.** 55 % on a pure F32 FMA kernel whose F32 Utilization is
  87 % and F32 Limiter 99 %. Its definition (which pipes, which peak) is not
  known; prefer the per-type utilizations and limiters.
- **Launch limiter.** `Compute Shader Launch Limiter` reads 82–100 % on every
  full-grid oracle (ALU-, DRAM- and threadgroup-bound alike) while `Compute
  Shader Launch Utilization` is under 3 %: it rises whenever the cores are
  full and launches wait. Insights ignore it. `Shader Core Limiter` equals
  `Instruction Issue Limiter` on every oracle.
- **No threadgroup-memory limiter.** The limiter pass on M4 does not carry
  the raw counters of `Threadgroup Load Limiter`; the threadgroup oracle's
  highest limiter is `Instruction Issue Limiter` (70 %).
- **DRAM write floor.** Kernels that write almost nothing show 0.3–0.5 GB/s of
  DRAM writes (e.g. 1.9 MB for the ALU oracle's 0.66 MB of stores). Writes
  also land late (−2 % for copy, half the read kernel's 1 MiB): the fabric
  counters see write-backs, not stores. Small write volumes are not
  meaningful.
- **Not oracle-checked:** the L1/L2/MMU limiters, L2 and buffer L1
  bandwidths, the instruction issue limiter, F16 counters. They
  are agxps' formulas on correctly attributed raw counts, but no kernel with a
  known answer has pinned them.
- **Concurrent dispatches.** Per-dispatch values for dispatches that overlap
  on the same cores are a residency split, not a measurement; `shrd%` says how
  much of a row that is.
- **RDE ring semantics.** Rings of one RDE source are summed as instances of
  the same block. This matches the DRAM oracles; it has not been checked for
  the texture/RDE counters that compute kernels do not exercise.
