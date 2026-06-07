# tracepoints

A library for sequence alignment compression and reconstruction using tracepoints.

## Overview

`tracepoints` converts a CIGAR string into a sparse set of tracepoints and reconstructs an equal-or-better CIGAR on demand. Use this library when you want to store or send alignments compactly: only the tracepoints are kept, and the full alignment is recovered by re-aligning the short segments between them.

### What are tracepoints?

Rather than storing every alignment operation in a full CIGAR string, tracepoints record a sparse set of coordinate pairs along the alignment path. Each pair of consecutive tracepoints defines a short subalignment interval whose CIGAR can be reconstructed on-demand by re-aligning the corresponding sequence segments with [WFA](https://github.com/smarco/WFA2-lib).

This library implements **adaptive tracepoints**: instead of segmenting at fixed intervals, it segments based on local alignment complexity, creating larger segments in conserved regions and smaller ones in divergent regions. Reconstruction from adaptive tracepoints guarantees identical or improved alignment scores, never worse.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
tracepoints = { git = "https://github.com/AndreaGuarracino/tracepoints" }
```

Then simply build your project:

```shell
cargo build --release
```

## Usage 

See the examples:
- [`dual_gap_affine.rs`](examples/dual_gap_affine.rs) for dual gap-affine distance usage;
- [`edit_distance.rs`](examples/edit_distance.rs) for edit distance usage.

Run it with `cargo run --example dual_gap_affine` or `cargo run --example edit_distance`.

## Features

- **Tracepoint types**:
  - **Standard**: `(a_len, b_len)` pairs for each segment
  - **FastGA**: Fixed-spacing tracepoints compatible with the [FastGA](https://github.com/thegenemyers/FASTGA) aligner
- **Complexity metrics**: `EditDistance` (count of mismatches + indels) and `DiagonalDistance` (max diagonal shift within a segment)
- **CIGAR reconstruction**: Conversion from tracepoints back to CIGAR strings using [WFA](https://github.com/smarco/WFA2-lib) alignment
- **Distance modes**: Support for edit distance, gap-affine, and dual gap-affine penalties

## How It Works

### CIGAR to Tracepoints Conversion

The library segments a CIGAR string into tracepoints where each segment contains at most `max_diff` differences (mismatches or indels):

- Match operations ('=' and 'M') don't count as differences
- Mismatch operations ('X') can be split across segments if needed
- Indels ('I', 'D') are kept intact within a single segment when possible
- Long indels exceeding `max_diff` become their own segments

### Tracepoints to CIGAR Reconstruction

For each tracepoint pair, the library performs the alignment of the corresponding sequence segments using [WFA](https://github.com/smarco/WFA2-lib) alignment:
- Pure insertions (a_len > 0, b_len = 0) are directly converted to 'I' operations
- Pure deletions (a_len = 0, b_len > 0) are directly converted to 'D' operations
- Mixed segments are realigned using the [WFA](https://github.com/smarco/WFA2-lib) algorithm

## Related repositories

- **[tpa](https://github.com/AndreaGuarracino/tpa)**: the TracePoint Alignment (TPA) binary format library for efficient storage and random access of sequence alignments with tracepoints.
- **[cigzip](https://github.com/AndreaGuarracino/cigzip)**: the command-line tool for alignment encoding (CIGAR → tracepoints), compression (PAF → TPA), decompression (TPA → PAF), and decoding (tracepoints → CIGAR).

## History

Inspired by Gene Myers' tracepoint concept: [Recording Alignments with Trace Points](https://dazzlerblog.wordpress.com/2015/11/05/trace-points/).

## License

MIT
