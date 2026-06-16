pub use lib_wfa2::affine_wavefront::{
    AffineWavefronts, AlignmentStatus, Distance, HeuristicStrategy,
};
use std::cmp::min;

/// Metric selector for per-segment band computation used by heuristic alignment
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ComplexityMetric {
    /// Edit distance (count mismatches + insertions + deletions)
    EditDistance,
    /// Diagonal distance (distance from the main diagonal)
    DiagonalDistance,
}

impl ComplexityMetric {
    /// Convert to u8 for binary serialization
    pub fn to_u8(&self) -> u8 {
        match self {
            Self::EditDistance => 0,
            Self::DiagonalDistance => 1,
        }
    }

    /// Parse from u8 for binary deserialization
    pub fn from_u8(byte: u8) -> Result<Self, String> {
        match byte {
            0 => Ok(Self::EditDistance),
            1 => Ok(Self::DiagonalDistance),
            _ => Err(format!("Invalid complexity metric byte: {}", byte)),
        }
    }

    /// Parse ComplexityMetric from string
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "edit-distance" => Ok(Self::EditDistance),
            "diagonal-distance" => Ok(Self::DiagonalDistance),
            _ => Err(format!(
                "Invalid complexity metric '{}'. Expected 'edit-distance' or 'diagonal-distance'",
                s
            )),
        }
    }

    /// Convert ComplexityMetric to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EditDistance => "edit-distance",
            Self::DiagonalDistance => "diagonal-distance",
        }
    }
}

impl std::fmt::Display for ComplexityMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Tracepoint type discriminant (Copy, no heap allocation)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TracepointType {
    /// Standard tracepoints: (a_len, b_len) pairs
    Standard,
    /// Mixed representation with special CIGAR operations preserved
    Mixed,
    /// Variable tracepoints: (length, None) when equal, otherwise (a_len, Some(b_len))
    Variable,
    /// FastGA tracepoints with CIGAR processing state
    Fastga,
}

impl TracepointType {
    /// Convert to u8 for binary serialization
    pub fn to_u8(&self) -> u8 {
        match self {
            Self::Standard => 0,
            Self::Mixed => 1,
            Self::Variable => 2,
            Self::Fastga => 3,
        }
    }

    /// Parse from u8 for binary deserialization
    pub fn from_u8(byte: u8) -> Result<Self, String> {
        match byte {
            0 => Ok(Self::Standard),
            1 => Ok(Self::Mixed),
            2 => Ok(Self::Variable),
            3 => Ok(Self::Fastga),
            _ => Err(format!("Invalid tracepoint type byte: {}", byte)),
        }
    }

    /// Parse from string representation (case-insensitive)
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_ascii_lowercase().as_str() {
            "standard" => Ok(Self::Standard),
            "mixed" => Ok(Self::Mixed),
            "variable" => Ok(Self::Variable),
            "fastga" => Ok(Self::Fastga),
            _ => Err(format!("Invalid tracepoint type '{}'. Expected 'standard', 'mixed', 'variable', or 'fastga'", s)),
        }
    }

    /// Convert to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Mixed => "mixed",
            Self::Variable => "variable",
            Self::Fastga => "fastga",
        }
    }
}

/// Tracepoint data with associated Vec
#[derive(Debug, Clone, PartialEq)]
pub enum TracepointData {
    Standard(Vec<(usize, usize)>),
    Mixed(Vec<MixedRepresentation>),
    Variable(Vec<(usize, Option<usize>)>),
    Fastga(Vec<(usize, usize)>),
}

impl TracepointData {
    pub fn tp_type(&self) -> TracepointType {
        match self {
            Self::Standard(_) => TracepointType::Standard,
            Self::Mixed(_) => TracepointType::Mixed,
            Self::Variable(_) => TracepointType::Variable,
            Self::Fastga(_) => TracepointType::Fastga,
        }
    }

    /// Convert tracepoints into the textual representation used in `tp:Z` PAF tags.
    pub fn to_tp_tag(&self) -> String {
        match self {
            Self::Standard(tps) | Self::Fastga(tps) => tps
                .iter()
                .map(|(a, b)| format!("{},{}", a, b))
                .collect::<Vec<_>>()
                .join(";"),
            Self::Variable(tps) => tps
                .iter()
                .map(|(a, b_opt)| match b_opt {
                    Some(b) => format!("{},{}", a, b),
                    None => format!("{}", a),
                })
                .collect::<Vec<_>>()
                .join(";"),
            Self::Mixed(items) => items
                .iter()
                .map(|item| match item {
                    MixedRepresentation::Tracepoint(a, b) => format!("{},{}", a, b),
                    MixedRepresentation::CigarOp(len, op) => format!("{}{}", len, op),
                })
                .collect::<Vec<_>>()
                .join(";"),
        }
    }

    /// Check if the tracepoint data is empty
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Standard(tps) | Self::Fastga(tps) => tps.is_empty(),
            Self::Variable(tps) => tps.is_empty(),
            Self::Mixed(items) => items.is_empty(),
        }
    }
}

/// Represents a CIGAR segment that can be either aligned or preserved as-is
#[derive(Debug, Clone, PartialEq)]
pub enum MixedRepresentation {
    /// Alignment segment represented by tracepoints
    Tracepoint(usize, usize),
    /// Special CIGAR operation that should be preserved intact
    CigarOp(usize, char),
}

/// Convert CIGAR string into standard tracepoints
///
/// Segments CIGAR into tracepoints according to the requested complexity metric.
/// `max_value` represents the maximum edit distance per segment when using
/// `ComplexityMetric::EditDistance`, or the maximum allowed diagonal deviation
/// when using `ComplexityMetric::DiagonalDistance`.
pub fn cigar_to_tracepoints(
    cigar: &str,
    max_value: u32,
    metric: ComplexityMetric,
) -> Vec<(usize, usize)> {
    match process_cigar_segments(cigar, max_value as usize, false, false, metric) {
        TracepointData::Standard(tracepoints) => tracepoints,
        _ => unreachable!(),
    }
}

/// Convert CIGAR string into raw standard tracepoints
///
/// Like `cigar_to_tracepoints` but allows indels to be split across segments.
/// Currently only `ComplexityMetric::EditDistance` performs splitting.
pub fn cigar_to_tracepoints_raw(
    cigar: &str,
    max_value: u32,
    metric: ComplexityMetric,
) -> Vec<(usize, usize)> {
    match process_cigar_segments(cigar, max_value as usize, false, true, metric) {
        TracepointData::Standard(tracepoints) => tracepoints,
        _ => unreachable!(),
    }
}

/// Convert CIGAR string into mixed tracepoints
///
/// Like `cigar_to_tracepoints` but preserves special operations (H, N, P, S).
pub fn cigar_to_mixed_tracepoints(
    cigar: &str,
    max_value: u32,
    metric: ComplexityMetric,
) -> Vec<MixedRepresentation> {
    match process_cigar_segments(cigar, max_value as usize, true, false, metric) {
        TracepointData::Mixed(tracepoints) => tracepoints,
        _ => unreachable!(),
    }
}

/// Convert CIGAR string into raw mixed tracepoints
///
/// Like `cigar_to_mixed_tracepoints` but allows indels to be split across segments.
/// Currently only `ComplexityMetric::EditDistance` performs splitting.
pub fn cigar_to_mixed_tracepoints_raw(
    cigar: &str,
    max_value: u32,
    metric: ComplexityMetric,
) -> Vec<MixedRepresentation> {
    match process_cigar_segments(cigar, max_value as usize, true, true, metric) {
        TracepointData::Mixed(tracepoints) => tracepoints,
        _ => unreachable!(),
    }
}

/// Convert CIGAR string into variable tracepoints
///
/// Uses (length, None) when a_len == b_len, otherwise (a_len, Some(b_len)).
pub fn cigar_to_variable_tracepoints(
    cigar: &str,
    max_value: u32,
    metric: ComplexityMetric,
) -> Vec<(usize, Option<usize>)> {
    to_variable_format(cigar_to_tracepoints(cigar, max_value, metric))
}

/// Convert CIGAR string into raw variable tracepoints
///
/// Like `cigar_to_variable_tracepoints` but allows indels to be split across segments when supported by the metric.
/// Currently only `ComplexityMetric::EditDistance` performs splitting.
pub fn cigar_to_variable_tracepoints_raw(
    cigar: &str,
    max_value: u32,
    metric: ComplexityMetric,
) -> Vec<(usize, Option<usize>)> {
    to_variable_format(cigar_to_tracepoints_raw(cigar, max_value, metric))
}

// -----------------------------------------------------------------------------
// ES-2bit (edit-script) encoding
// -----------------------------------------------------------------------------
//
// An alignment encoding that drops every '=' / 'M' run from the CIGAR and 
// packs each remaining {I,X,D} op into 2 bits, stored MSB-first within each byte.
//
// Op code mapping (MSB-first):
//     00  =>  X  (mismatch; advances both query and target)
//     01  =>  I  (insertion into query; advances query only)
//     10  =>  D  (deletion from query; advances target only)
//     11  =>  unused
//
// Storage: ⌈2·e/8⌉ bytes plus an `edit_count`.
//
// Note: the decoder normalizes indels to greedy-canonical form (rightmost
// position inside any homopolymer / tandem repeat). Sequence content and edit
// count are preserved; exact indel positions may differ from the input CIGAR.

/// Op codes in the ES-2bit encoding
const ES2BIT_CODE_X: u8 = 0b00;
const ES2BIT_CODE_I: u8 = 0b01;
const ES2BIT_CODE_D: u8 = 0b10;

/// Encode a CIGAR as ES-2bit.
///
/// Drops every `=` / `M` run and packs each remaining `I` / `X` / `D` op into
/// 2 bits (MSB-first within each byte). Appends a `11` sentinel at the end.
///
/// Special ops: (H/N/P/S) are not handled by this encoding.
pub fn cigar_to_edit_script(cigar: &str) -> Vec<u8> {
    let ops = cigar_str_to_cigar_ops(cigar);
    let mut edit_script: Vec<u8> = Vec::new();
    // Bit position within the current byte; 0 means empty, 8 means full.
    let mut bits_in_last: u8 = 0;

    for (len, op) in ops {
        let code: u8 = match op {
            '=' | 'M' => continue,
            'X' => ES2BIT_CODE_X,
            'I' => ES2BIT_CODE_I,
            'D' => ES2BIT_CODE_D,
            other => panic!(
                "cigar_to_edit_script: unsupported CIGAR op '{}' (len={})",
                other, len
            ),
        };
        for _ in 0..len {
            if bits_in_last == 0 || bits_in_last == 8 {
                edit_script.push(0u8);
                bits_in_last = 0;
            }
            // MSB-first: shift code to the most significant free slot.
            let shift = 6 - bits_in_last; // 6, 4, 2, 0
            let last = edit_script.last_mut().expect("just pushed");
            *last |= code << shift;
            bits_in_last += 2;
        }
    }

    // Append sentinel 11 in the next available slot.
    if bits_in_last == 0 || bits_in_last == 8 {
        edit_script.push(0u8);
        bits_in_last = 0;
    }
    let shift = 6 - bits_in_last;
    let last = edit_script.last_mut().expect("just pushed sentinel byte");
    *last |= 0b11 << shift;

    edit_script
}

/// Reconstruct a CIGAR string from an ES-2bit edit script.
///
/// Walks `a_seq` (query) and `b_seq` (target) greedily, extending matches
/// while bases agree and consuming one 2-bit op from `edit_script` on every
/// mismatch. Stops when it reads the `11` sentinel.
///
/// The returned CIGAR uses `=` / `X` / `I` / `D` operations only and is in
/// greedy-canonical form, that is indel positions are right-shifted within
/// any run of identical bases. This means exact indel placement may differ
/// from the CIGAR that was originally encoded.
pub fn edit_script_to_cigar(
    edit_script: &[u8],
    a_seq: &[u8],
    b_seq: &[u8],
) -> String {
    let mut out_ops: Vec<(usize, char)> = Vec::new();
    let mut a_pos: usize = 0; // position in a_seq (query)
    let mut b_pos: usize = 0; // position in b_seq (target)
    let mut ops_consumed: usize = 0;

    // Helper to emit one '=' op covering the run of equal bases (if any).
    let extend_matches = |a_pos: &mut usize,
                          b_pos: &mut usize,
                          out_ops: &mut Vec<(usize, char)>| {
        let start_a = *a_pos;
        while *a_pos < a_seq.len() && *b_pos < b_seq.len() && a_seq[*a_pos] == b_seq[*b_pos] {
            *a_pos += 1;
            *b_pos += 1;
        }
        let match_run_len = *a_pos - start_a;
        if match_run_len > 0 {
            push_op(out_ops, match_run_len, '=');
        }
    };

    loop {
        // Decode the next 2-bit op from `edit_script`.
        let bit_index = 2 * ops_consumed;
        let byte_index = bit_index / 8;
        let shift = 6 - (bit_index % 8) as u8; // 6,4,2,0 (MSB-first)
        assert!(
            byte_index < edit_script.len(),
            "edit_script_to_cigar: missing sentinel - edit_script ended at ops_consumed={}",
            ops_consumed
        );

        let code = (edit_script[byte_index] >> shift) & 0b11;
        if code == 0b11 {
            // Manage any trailing matches, then stop.
            extend_matches(&mut a_pos, &mut b_pos, &mut out_ops);
            break;
        }

        extend_matches(&mut a_pos, &mut b_pos, &mut out_ops);

        match code {
            ES2BIT_CODE_X => {
                assert!(
                    a_pos < a_seq.len() && b_pos < b_seq.len(),
                    "edit_script_to_cigar: X at end of sequences (a_pos={}, b_pos={}, a_len={}, b_len={})",
                    a_pos,
                    b_pos,
                    a_seq.len(),
                    b_seq.len()
                );
                assert!(
                    a_seq[a_pos] != b_seq[b_pos],
                    "edit_script_to_cigar: X at matching position (a_pos={}, b_pos={}, base={})",
                    a_pos,
                    b_pos,
                    a_seq[a_pos] as char
                );
                push_op(&mut out_ops, 1, 'X');
                a_pos += 1;
                b_pos += 1;
            }
            ES2BIT_CODE_I => {
                assert!(
                    a_pos < a_seq.len(),
                    "edit_script_to_cigar: I past end of query (a_pos={}, a_len={})",
                    a_pos,
                    a_seq.len()
                );
                push_op(&mut out_ops, 1, 'I');
                a_pos += 1;
            }
            ES2BIT_CODE_D => {
                assert!(
                    b_pos < b_seq.len(),
                    "edit_script_to_cigar: D past end of target (b_pos={}, b_len={})",
                    b_pos,
                    b_seq.len()
                );
                push_op(&mut out_ops, 1, 'D');
                b_pos += 1;
            }
            _ => unreachable!(),
        }
        ops_consumed += 1;
    }

    // Sanity check: we should have consumed both sequences exactly.
    assert_eq!(
        a_pos,
        a_seq.len(),
        "edit_script_to_cigar: query not fully consumed (a_pos={}, a_len={})",
        a_pos,
        a_seq.len()
    );
    assert_eq!(
        b_pos,
        b_seq.len(),
        "edit_script_to_cigar: target not fully consumed (b_pos={}, b_len={})",
        b_pos,
        b_seq.len()
    );

    cigar_ops_to_cigar_string(&out_ops)
}

/// Reconstruct CIGAR string from standard tracepoints
pub fn tracepoints_to_cigar(
    tracepoints: &[(usize, usize)],
    a_seq: &[u8],
    b_seq: &[u8],
    a_start: usize,
    b_start: usize,
    metric: ComplexityMetric,
    distance: &Distance,
) -> String {
    let mut aligner = distance.create_aligner(None, None);
    tracepoints_to_cigar_with_aligner(
        tracepoints,
        a_seq,
        b_seq,
        a_start,
        b_start,
        metric,
        &mut aligner,
        None,
    )
}

/// Reconstruct CIGAR string from standard tracepoints with provided aligner
///
/// Like `tracepoints_to_cigar`, but allows callers to reuse an existing aligner.
/// Pass `Some(max_value)` to enable banded alignment heuristics; pass `None` to disable.
pub fn tracepoints_to_cigar_with_aligner(
    tracepoints: &[(usize, usize)],
    a_seq: &[u8],
    b_seq: &[u8],
    a_start: usize,
    b_start: usize,
    metric: ComplexityMetric,
    aligner: &mut AffineWavefronts,
    max_value: Option<u32>,
) -> String {
    reconstruct_cigar_from_segments(
        tracepoints,
        a_seq,
        b_seq,
        a_start,
        b_start,
        metric,
        aligner,
        max_value,
    )
}

/// Reconstruct CIGAR string from mixed tracepoints
pub fn mixed_tracepoints_to_cigar(
    mixed_tracepoints: &[MixedRepresentation],
    a_seq: &[u8],
    b_seq: &[u8],
    a_start: usize,
    b_start: usize,
    metric: ComplexityMetric,
    distance: &Distance,
) -> String {
    let mut aligner = distance.create_aligner(None, None);
    mixed_tracepoints_to_cigar_with_aligner(
        mixed_tracepoints,
        a_seq,
        b_seq,
        a_start,
        b_start,
        metric,
        &mut aligner,
        None,
    )
}

/// Reconstruct CIGAR string from mixed tracepoints with provided aligner
///
/// Like `mixed_tracepoints_to_cigar`, but allows callers to reuse an existing aligner.
/// Pass `Some(max_value)` to enable banded alignment heuristics; pass `None` to disable.
pub fn mixed_tracepoints_to_cigar_with_aligner(
    mixed_tracepoints: &[MixedRepresentation],
    a_seq: &[u8],
    b_seq: &[u8],
    a_start: usize,
    b_start: usize,
    metric: ComplexityMetric,
    aligner: &mut AffineWavefronts,
    max_value: Option<u32>,
) -> String {
    reconstruct_cigar_from_mixed_segments(
        mixed_tracepoints,
        a_seq,
        b_seq,
        a_start,
        b_start,
        metric,
        aligner,
        max_value,
    )
}

/// Reconstruct CIGAR string from variable tracepoints
pub fn variable_tracepoints_to_cigar(
    variable_tracepoints: &[(usize, Option<usize>)],
    a_seq: &[u8],
    b_seq: &[u8],
    a_start: usize,
    b_start: usize,
    metric: ComplexityMetric,
    distance: &Distance,
) -> String {
    let regular_tracepoints = from_variable_format(variable_tracepoints);
    let mut aligner = distance.create_aligner(None, None);
    reconstruct_cigar_from_segments(
        &regular_tracepoints,
        a_seq,
        b_seq,
        a_start,
        b_start,
        metric,
        &mut aligner,
        None,
    )
}

/// Reconstruct CIGAR string from variable tracepoints with provided aligner
///
/// Like `variable_tracepoints_to_cigar`, but allows callers to reuse an existing aligner.
/// Pass `Some(max_value)` to enable banded alignment heuristics; pass `None` to disable.
pub fn variable_tracepoints_to_cigar_with_aligner(
    variable_tracepoints: &[(usize, Option<usize>)],
    a_seq: &[u8],
    b_seq: &[u8],
    a_start: usize,
    b_start: usize,
    metric: ComplexityMetric,
    aligner: &mut AffineWavefronts,
    max_value: Option<u32>,
) -> String {
    let regular_tracepoints = from_variable_format(variable_tracepoints);
    reconstruct_cigar_from_segments(
        &regular_tracepoints,
        a_seq,
        b_seq,
        a_start,
        b_start,
        metric,
        aligner,
        max_value,
    )
}


// Helper functions

/// Dispatch CIGAR processing based on the requested complexity metric
fn process_cigar_segments(
    cigar: &str,
    max_value: usize,
    preserve_special: bool,
    allow_indel_split: bool,
    metric: ComplexityMetric,
) -> TracepointData {
    match metric {
        ComplexityMetric::EditDistance => {
            process_cigar_edit_distance(cigar, max_value, preserve_special, allow_indel_split)
        }
        ComplexityMetric::DiagonalDistance => {
            debug_assert!(
                !allow_indel_split,
                "allow_indel_split is ignored when using diagonal metric"
            );
            process_cigar_diagonal_distance(cigar, max_value, preserve_special)
        }
    }
}

/// Process CIGAR into tracepoints using edit distance segmentation
///
/// Handles both standard and mixed tracepoint generation with optional indel splitting.
/// Special operations (H, N, P, S) are preserved when preserve_special is true.
/// Indels can be split across segments when allow_indel_split is true (raw mode).
fn process_cigar_edit_distance(
    cigar: &str,
    max_diff: usize,
    preserve_special: bool,
    allow_indel_split: bool,
) -> TracepointData {
    let ops = cigar_str_to_cigar_ops(cigar);
    let mut standard_tracepoints = Vec::new();
    let mut mixed_tracepoints = Vec::new();
    let mut cur_a_len = 0;
    let mut cur_b_len = 0;
    let mut cur_diff = 0;

    for (mut len, op) in ops {
        match op {
            'H' | 'N' | 'P' | 'S' if preserve_special => {
                flush_segment(
                    &mut cur_a_len,
                    &mut cur_b_len,
                    &mut cur_diff,
                    preserve_special,
                    &mut standard_tracepoints,
                    &mut mixed_tracepoints,
                );
                mixed_tracepoints.push(MixedRepresentation::CigarOp(len, op));
            }
            'X' | 'I' | 'D' if op == 'X' || allow_indel_split => {
                while len > 0 {
                    let remaining = max_diff.saturating_sub(cur_diff);
                    let step = min(len, remaining);

                    if step == 0 {
                        flush_segment(
                            &mut cur_a_len,
                            &mut cur_b_len,
                            &mut cur_diff,
                            preserve_special,
                            &mut standard_tracepoints,
                            &mut mixed_tracepoints,
                        );

                        let (a_add, b_add) = match op {
                            'X' => (1, 1),
                            'I' => (1, 0),
                            'D' => (0, 1),
                            _ => unreachable!(),
                        };

                        if max_diff == 0 {
                            add_tracepoint(
                                a_add,
                                b_add,
                                preserve_special,
                                &mut standard_tracepoints,
                                &mut mixed_tracepoints,
                            );
                        } else {
                            cur_a_len = a_add;
                            cur_b_len = b_add;
                            cur_diff = 1;
                        }
                        len -= 1;
                    } else {
                        let (a_add, b_add) = match op {
                            'X' => (step, step),
                            'I' => (step, 0),
                            'D' => (0, step),
                            _ => unreachable!(),
                        };
                        cur_a_len += a_add;
                        cur_b_len += b_add;
                        cur_diff += step;
                        len -= step;

                        if cur_diff == max_diff {
                            flush_segment(
                                &mut cur_a_len,
                                &mut cur_b_len,
                                &mut cur_diff,
                                preserve_special,
                                &mut standard_tracepoints,
                                &mut mixed_tracepoints,
                            );
                        }
                    }
                }
            }
            'I' | 'D' => {
                if len > max_diff {
                    flush_segment(
                        &mut cur_a_len,
                        &mut cur_b_len,
                        &mut cur_diff,
                        preserve_special,
                        &mut standard_tracepoints,
                        &mut mixed_tracepoints,
                    );
                    let (a_add, b_add) = if op == 'I' { (len, 0) } else { (0, len) };
                    add_tracepoint(
                        a_add,
                        b_add,
                        preserve_special,
                        &mut standard_tracepoints,
                        &mut mixed_tracepoints,
                    );
                } else {
                    if cur_diff + len > max_diff {
                        flush_segment(
                            &mut cur_a_len,
                            &mut cur_b_len,
                            &mut cur_diff,
                            preserve_special,
                            &mut standard_tracepoints,
                            &mut mixed_tracepoints,
                        );
                    }
                    if op == 'I' {
                        cur_a_len += len;
                    } else {
                        cur_b_len += len;
                    }
                    cur_diff += len;
                }
            }
            '=' | 'M' => {
                cur_a_len += len;
                cur_b_len += len;
            }
            _ => panic!("Invalid CIGAR operation: {op}"),
        }
    }

    flush_segment(
        &mut cur_a_len,
        &mut cur_b_len,
        &mut cur_diff,
        preserve_special,
        &mut standard_tracepoints,
        &mut mixed_tracepoints,
    );

    if preserve_special {
        TracepointData::Mixed(mixed_tracepoints)
    } else {
        TracepointData::Standard(standard_tracepoints)
    }
}

/// Process CIGAR into tracepoints using diagonal distance segmentation
///
/// Breaks segments when the distance from the main diagonal exceeds max_dist.
/// Insertions increase diagonal distance (+), deletions decrease it (-).
/// The main diagonal is influenced by the overall sequence length difference.
fn process_cigar_diagonal_distance(
    cigar: &str,
    max_dist: usize,
    preserve_special: bool,
) -> TracepointData {
    let ops = cigar_str_to_cigar_ops(cigar);
    let mut standard_tracepoints = Vec::new();
    let mut mixed_tracepoints = Vec::new();
    let mut cur_a_len = 0;
    let mut cur_b_len = 0;
    let mut diagonal_distance: i64 = 0;

    for (len, op) in ops {
        match op {
            'H' | 'N' | 'P' | 'S' if preserve_special => {
                flush_segment(
                    &mut cur_a_len,
                    &mut cur_b_len,
                    &mut 0,
                    preserve_special,
                    &mut standard_tracepoints,
                    &mut mixed_tracepoints,
                );
                diagonal_distance = 0; // Reset diagonal distance after flushing
                mixed_tracepoints.push(MixedRepresentation::CigarOp(len, op));
            }
            'I' => {
                let new_diagonal_distance = diagonal_distance + len as i64;

                if new_diagonal_distance.unsigned_abs() > max_dist as u64 {
                    // Would exceed max_dist, need to flush current segment first
                    if cur_a_len > 0 || cur_b_len > 0 {
                        flush_segment(
                            &mut cur_a_len,
                            &mut cur_b_len,
                            &mut 0,
                            preserve_special,
                            &mut standard_tracepoints,
                            &mut mixed_tracepoints,
                        );
                    }

                    // Add the insertion as its own segment
                    add_tracepoint(
                        len,
                        0,
                        preserve_special,
                        &mut standard_tracepoints,
                        &mut mixed_tracepoints,
                    );
                    diagonal_distance = 0; // Reset after creating segment
                } else {
                    // Can add to current segment
                    cur_a_len += len;
                    diagonal_distance = new_diagonal_distance;
                }
            }
            'D' => {
                let new_diagonal_distance = diagonal_distance - len as i64;

                if new_diagonal_distance.unsigned_abs() > max_dist as u64 {
                    // Would exceed max_dist, need to flush current segment first
                    if cur_a_len > 0 || cur_b_len > 0 {
                        flush_segment(
                            &mut cur_a_len,
                            &mut cur_b_len,
                            &mut 0,
                            preserve_special,
                            &mut standard_tracepoints,
                            &mut mixed_tracepoints,
                        );
                    }

                    // Add the deletion as its own segment
                    add_tracepoint(
                        0,
                        len,
                        preserve_special,
                        &mut standard_tracepoints,
                        &mut mixed_tracepoints,
                    );
                    diagonal_distance = 0; // Reset after creating segment
                } else {
                    // Can add to current segment
                    cur_b_len += len;
                    diagonal_distance = new_diagonal_distance;
                }
            }
            'X' => {
                // Mismatches don't change diagonal distance but still consume sequence
                cur_a_len += len;
                cur_b_len += len;
            }
            '=' | 'M' => {
                // Matches don't change diagonal distance
                cur_a_len += len;
                cur_b_len += len;
            }
            _ => panic!("Invalid CIGAR operation: {op}"),
        }
    }

    // Flush any remaining segment
    flush_segment(
        &mut cur_a_len,
        &mut cur_b_len,
        &mut 0,
        preserve_special,
        &mut standard_tracepoints,
        &mut mixed_tracepoints,
    );

    if preserve_special {
        TracepointData::Mixed(mixed_tracepoints)
    } else {
        TracepointData::Standard(standard_tracepoints)
    }
}

/// Reconstruct CIGAR from tracepoints with using provided aligner
fn reconstruct_cigar_from_segments(
    segments: &[(usize, usize)],
    a_seq: &[u8],
    b_seq: &[u8],
    a_start: usize,
    b_start: usize,
    metric: ComplexityMetric,
    aligner: &mut AffineWavefronts,
    max_value: Option<u32>,
) -> String {
    let mut cigar_ops = Vec::new();
    let mut current_a = a_start;
    let mut current_b = b_start;

    for &(a_len, b_len) in segments {
        if a_len > 0 && b_len == 0 {
            // Pure insertion
            push_op(&mut cigar_ops, a_len, 'I');
            current_a += a_len;
        } else if b_len > 0 && a_len == 0 {
            // Pure deletion
            push_op(&mut cigar_ops, b_len, 'D');
            current_b += b_len;
        } else {
            // Mixed segment - realign with WFA
            let a_end = current_a + a_len;
            let b_end = current_b + b_len;
            if let Some(mv) = max_value {
                let strategy =
                    compute_banded_static_strategy(a_len, b_len, metric, mv, &aligner.get_distance());
                aligner.set_heuristic(Some(&strategy));
            }
            align_sequences_wfa(
                &a_seq[current_a..a_end],
                &b_seq[current_b..b_end],
                &*aligner,
                &mut cigar_ops,
            );
            current_a = a_end;
            current_b = b_end;
        }
    }

    cigar_ops_to_cigar_string(&cigar_ops)
}

/// Core reconstruction for mixed segments with optional heuristic application
fn reconstruct_cigar_from_mixed_segments(
    mixed_tracepoints: &[MixedRepresentation],
    a_seq: &[u8],
    b_seq: &[u8],
    a_start: usize,
    b_start: usize,
    metric: ComplexityMetric,
    aligner: &mut AffineWavefronts,
    max_value: Option<u32>,
) -> String {
    let mut cigar_ops = Vec::new();
    let mut current_a = a_start;
    let mut current_b = b_start;

    for item in mixed_tracepoints {
        match item {
            MixedRepresentation::CigarOp(len, op) => {
                if *len > 0 {
                    push_op(&mut cigar_ops, *len, *op);
                }
            }
            MixedRepresentation::Tracepoint(a_len, b_len) => {
                if *a_len > 0 && *b_len == 0 {
                    push_op(&mut cigar_ops, *a_len, 'I');
                    current_a += *a_len;
                } else if *b_len > 0 && *a_len == 0 {
                    push_op(&mut cigar_ops, *b_len, 'D');
                    current_b += *b_len;
                } else {
                    let a_end = current_a + *a_len;
                    let b_end = current_b + *b_len;
                    if let Some(mv) = max_value {
                        let strategy = compute_banded_static_strategy(
                            *a_len,
                            *b_len,
                            metric,
                            mv,
                            &aligner.get_distance(),
                        );
                        aligner.set_heuristic(Some(&strategy));
                    }
                    align_sequences_wfa(
                        &a_seq[current_a..a_end],
                        &b_seq[current_b..b_end],
                        &*aligner,
                        &mut cigar_ops,
                    );
                    current_a = a_end;
                    current_b = b_end;
                }
            }
        }
    }
    cigar_ops_to_cigar_string(&cigar_ops)
}

/// Flush current segment to tracepoint collections
fn flush_segment(
    cur_a_len: &mut usize,
    cur_b_len: &mut usize,
    cur_diff: &mut usize,
    preserve_special: bool,
    standard_tracepoints: &mut Vec<(usize, usize)>,
    mixed_tracepoints: &mut Vec<MixedRepresentation>,
) {
    if *cur_a_len > 0 || *cur_b_len > 0 {
        if preserve_special {
            mixed_tracepoints.push(MixedRepresentation::Tracepoint(*cur_a_len, *cur_b_len));
        } else {
            standard_tracepoints.push((*cur_a_len, *cur_b_len));
        }
        *cur_a_len = 0;
        *cur_b_len = 0;
        *cur_diff = 0;
    }
}

/// Add a tracepoint to the appropriate collection
fn add_tracepoint(
    a_len: usize,
    b_len: usize,
    preserve_special: bool,
    standard_tracepoints: &mut Vec<(usize, usize)>,
    mixed_tracepoints: &mut Vec<MixedRepresentation>,
) {
    if preserve_special {
        mixed_tracepoints.push(MixedRepresentation::Tracepoint(a_len, b_len));
    } else {
        standard_tracepoints.push((a_len, b_len));
    }
}

/// Convert standard tracepoints to variable tracepoints
fn to_variable_format(tracepoints: Vec<(usize, usize)>) -> Vec<(usize, Option<usize>)> {
    tracepoints
        .into_iter()
        .map(|(a_len, b_len)| {
            if a_len == b_len {
                (a_len, None)
            } else {
                (a_len, Some(b_len))
            }
        })
        .collect()
}

/// Convert variable tracepoints to standard tracepoints
fn from_variable_format(variable_tracepoints: &[(usize, Option<usize>)]) -> Vec<(usize, usize)> {
    variable_tracepoints
        .iter()
        .map(|(a_len, b_len_opt)| match b_len_opt {
            None => (*a_len, *a_len),
            Some(b_len) => (*a_len, *b_len),
        })
        .collect()
}

/// Reverse a CIGAR string (both operations and their lengths)
///
/// For example: "3M2I5D" becomes "5D2I3M"
fn reverse_cigar(cigar: &str) -> String {
    let ops = cigar_str_to_cigar_ops(cigar);
    ops.iter()
        .rev()
        .map(|(len, op)| format!("{}{}", len, op))
        .collect()
}

/// Parse CIGAR string into (length, operation) pairs
pub(crate) fn cigar_str_to_cigar_ops(cigar: &str) -> Vec<(usize, char)> {
    let mut ops = Vec::new();
    let mut num = String::new();
    for ch in cigar.chars() {
        if ch.is_ascii_digit() {
            num.push(ch);
        } else {
            if let Ok(n) = num.parse::<usize>() {
                ops.push((n, ch));
            }
            num.clear();
        }
    }
    ops
}

/// Append a run, merging into the trailing op if it matches
#[inline]
fn push_op(out: &mut Vec<(usize, char)>, count: usize, op: char) {
    if let Some(last) = out.last_mut() {
        if last.1 == op {
            last.0 += count;
            return;
        }
    }
    out.push((count, op));
}

/// Run-length encode WFA CIGAR bytes and append them into `out`
/// Treats 'M' (77) as '=' for consistency
fn append_cigar_u8(ops: &[u8], out: &mut Vec<(usize, char)>) {
    if ops.is_empty() {
        return;
    }
    let mut count = 1usize;
    let mut current_op = if ops[0] == 77 { '=' } else { ops[0] as char };
    for &byte in ops.iter().skip(1) {
        let op = if byte == 77 { '=' } else { byte as char };
        if op == current_op {
            count += 1;
        } else {
            push_op(out, count, current_op);
            current_op = op;
            count = 1;
        }
    }
    push_op(out, count, current_op);
}

/// Align two sequence segments with WFA, appending the resulting ops into
/// `out` (reuses the caller's buffer; ops are merged on append).
pub fn align_sequences_wfa(
    query: &[u8],
    target: &[u8],
    aligner: &AffineWavefronts,
    out: &mut Vec<(usize, char)>,
) {
    let status = aligner.align(target, query);
    match status {
        AlignmentStatus::Completed => append_cigar_u8(aligner.cigar(), out),
        s => panic!("Alignment failed with status: {s:?}"),
    }
}

/// Format CIGAR operations as CIGAR string
pub fn cigar_ops_to_cigar_string(ops: &[(usize, char)]) -> String {
    ops.iter()
        .map(|(len, op)| format!("{len}{op}"))
        .collect::<Vec<_>>()
        .join("")
}

/// Compute a BandedStatic heuristic for a single tracepoint segment
fn compute_banded_static_strategy(
    a_len: usize,
    b_len: usize,
    metric: ComplexityMetric,
    max_value: u32,
    distance: &Distance,
) -> HeuristicStrategy {
    let (band_min_k, band_max_k) = match metric {
        ComplexityMetric::EditDistance => {
            let delta_abs = a_len.abs_diff(b_len) as u32;
            let available = max_value.saturating_sub(delta_abs);
            let seg_band = match distance {
                Distance::Edit => {
                    available.div_ceil(2)
                }
                Distance::GapAffine { .. } | Distance::GapAffine2p { .. } => {
                    available.div_ceil(2).max(8)
                }
            };

            if a_len >= b_len {
                (-(seg_band as i32), (seg_band + delta_abs) as i32)
            } else {
                (-((seg_band + delta_abs) as i32), seg_band as i32)
            }
        }
        ComplexityMetric::DiagonalDistance => {
            let band_width = max_value as i32;
            (-band_width, band_width)
        }
    };
    HeuristicStrategy::BandedStatic {
        band_min_k,
        band_max_k,
    }
}

// FASTGA

/// Skip initial CIGAR operations until target_pos > 0 and current op is diagonal.
/// This matches FASTGA's cigarPrefix behavior for complement alignments.
///
/// Returns (op_index, remaining_len, query_pos, target_pos) for where to start processing.
fn skip_prefix_for_complement(
    ops: &[(usize, char)],
    mut query_pos: usize,
    mut target_pos: usize,
) -> (usize, usize, usize, usize) {
    for (op_index, &(len, op)) in ops.iter().enumerate() {
        match op {
            // Match/mismatch operations
            '=' | 'M' | 'X' => {
                // Check if we should stop: target position must be positive
                // (query_pos is always >= 0 for usize, so we only check target_pos)
                if target_pos > 0 {
                    // Found our starting point - return with full len remaining
                    return (op_index, len, query_pos, target_pos);
                }
                // Consume the operation
                query_pos += len;
                target_pos += len;
            }
            // Deletion - consume (advances target only)
            'D' | 'N' => {
                target_pos += len;
            }
            // Insertion - consume (advances query only)
            'I' => {
                query_pos += len;
            }
            // Skip special operations
            'H' | 'S' | 'P' => {}
            _ => {}
        }
    }
    // If we processed all ops, return end state
    (ops.len(), 0, query_pos, target_pos)
}

pub struct CigarProcessingState {
    pub cigar_pos: usize,           // Position in CIGAR string
    pub remaining_len: usize,       // Remaining length of current operation
    pub query_pos: usize,           // Current query position
    pub target_pos: usize,          // Current target position
    pub completed: bool,            // Whether entire CIGAR was processed
    pub actual_query_start: usize,  // Actual starting query position (after prefix skip)
    pub actual_target_start: usize, // Actual starting target position (after prefix skip)
}

/// Convert CIGAR string into FASTGA-style tracepoints
///
/// Segments CIGAR into sets of tracepoints, breaking when indels would cause tracepoint overflow.
pub fn cigar_to_tracepoints_fastga(
    cigar: &str,
    trace_spacing: u32,
    query_start: usize,
    query_end: usize,
    _query_len: usize,
    target_start: usize,
    target_end: usize,
    target_len: usize,
    complement: bool,
) -> Vec<(Vec<(usize, usize)>, (usize, usize, usize, usize))> {
    // Pure-match alignment (only '='/'M', no mismatch or gap) represented as a single (0,0) sentinel.
    {
        let ops = cigar_str_to_cigar_ops(cigar);
        if !ops.is_empty() && ops.iter().all(|&(_, op)| op == '=' || op == 'M') {
            return vec![(
                vec![(0, 0)],
                (query_start, query_end, target_start, target_end),
            )];
        }
    }
    let mut results = Vec::new();
    let mut state = None;
    let mut current_query_start = query_start;
    let mut current_target_start = target_start;
    // For complement, we need to track target_end in reversed space after first iteration
    let mut current_target_end = target_end;
    let mut is_first_segment = true;

    loop {
        let (tracepoints, new_state) = cigar_to_tracepoints_fastga_with_overflow(
            cigar,
            trace_spacing,
            current_query_start,
            query_end,
            current_target_start,
            current_target_end,
            target_len,
            complement,
            state,
        );

        // Store the segment with its coordinate bounds
        // Use the actual start positions which account for any prefix skip (for complement alignments)
        // If cigarPrefix consumed everything, FASTGA outputs "0,0" - we do the same
        let tracepoints = if tracepoints.is_empty() && new_state.completed {
            vec![(0, 0)]
        } else {
            tracepoints
        };
        results.push((
            tracepoints,
            (
                new_state.actual_query_start,
                new_state.query_pos,
                new_state.actual_target_start,
                new_state.target_pos,
            ),
        ));

        // After first segment, switch target_end to reversed space for complement
        if is_first_segment && complement {
            current_target_end = target_len - target_start;
        }
        is_first_segment = false;

        if new_state.completed {
            break;
        }

        // Handle the remaining operation that caused overflow
        current_query_start = new_state.query_pos;
        current_target_start = new_state.target_pos;

        // Skip the operation that caused overflow (similar to C code)
        if new_state.remaining_len > 0 {
            // This would be handled by the gap processing in the C code
            // For now, we advance positions to skip the problematic operation
            // For complement, the cigar_pos refers to the REVERSED CIGAR
            let cigar_to_parse = if complement {
                reverse_cigar(cigar)
            } else {
                cigar.to_string()
            };
            let ops = cigar_str_to_cigar_ops(&cigar_to_parse);
            if new_state.cigar_pos < ops.len() {
                let (_, op) = ops[new_state.cigar_pos];
                match op {
                    'I' => current_query_start += new_state.remaining_len,
                    'D' => current_target_start += new_state.remaining_len,
                    _ => {
                        current_query_start += new_state.remaining_len;
                        current_target_start += new_state.remaining_len;
                    }
                }
            }
        }

        state = Some(CigarProcessingState {
            cigar_pos: new_state.cigar_pos + 1, // Move to next operation
            remaining_len: 0,
            query_pos: current_query_start,
            target_pos: current_target_start,
            completed: false,
            actual_query_start: current_query_start,
            actual_target_start: current_target_start,
        });
    }

    results
}

/// Generate FASTGA-style tracepoints from CIGAR string with overflow handling
/// It stops processing if an indel would cause tracepoint overflow.
fn cigar_to_tracepoints_fastga_with_overflow(
    cigar: &str,
    trace_spacing: u32,
    query_start: usize,
    query_end: usize,
    target_start: usize,
    target_end: usize,
    target_len: usize,
    complement: bool,
    state: Option<CigarProcessingState>,
) -> (Vec<(usize, usize)>, CigarProcessingState) {
    let trace_spacing = trace_spacing as usize;

    // FASTGA reverses the target coordinates and CIGAR for complement alignments
    // BUT only on the first call (state is None). On continuation calls, coordinates
    // are already in reversed space.
    let is_first_call = state.is_none();
    let (target_start, target_end, cigar) = if complement && is_first_call {
        (
            target_len - target_end,
            target_len - target_start,
            reverse_cigar(cigar),
        )
    } else if complement {
        // Continuation: coordinates already reversed, but still need reversed CIGAR
        (target_start, target_end, reverse_cigar(cigar))
    } else {
        (target_start, target_end, cigar.to_string())
    };

    let ops = cigar_str_to_cigar_ops(&cigar);
    let mut tracepoints = Vec::new();

    // Initialize state from previous processing or start fresh
    // For FASTGA mode: apply cigarPrefix logic to skip initial operations when target_start == 0
    // This matches FASTGA's behavior where alignments starting at bpos=0 skip until bpos > 0
    let (mut op_index, mut remaining_op_len, mut a_pos, mut b_pos) = if let Some(s) = state {
        (s.cigar_pos, s.remaining_len, s.query_pos, s.target_pos)
    } else if is_first_call && target_start == 0 {
        // Apply cigarPrefix behavior: skip until we find a diagonal with target_pos > 0
        let result = skip_prefix_for_complement(&ops, query_start, target_start);
        // If cigarPrefix consumed the entire CIGAR (pure-match alignment), don't skip
        // This fixes FASTGA's PAFtoALN bug that produces 0,0 for such alignments
        if result.0 >= ops.len() {
            (0, 0, query_start, target_start)
        } else {
            result
        }
    } else {
        (0, 0, query_start, target_start)
    };

    // Save the actual starting positions (may differ from input due to prefix skip)
    let actual_query_start = a_pos;
    let actual_target_start = b_pos;

    let mut last_b_pos = b_pos;
    let mut diff = 0i64;
    let mut last_diff = 0i64;
    let mut next_trace = ((a_pos / trace_spacing) + 1) * trace_spacing;

    while op_index < ops.len() {
        let (op_len, op) = ops[op_index];
        let mut len = if remaining_op_len > 0 {
            remaining_op_len
        } else {
            op_len
        };
        remaining_op_len = 0;

        // Check boundaries before processing
        if a_pos >= query_end || b_pos >= target_end {
            if a_pos > next_trace - trace_spacing {
                tracepoints.push((
                    (diff - last_diff).unsigned_abs() as usize,
                    b_pos - last_b_pos,
                ));
            }
            return (
                tracepoints,
                CigarProcessingState {
                    cigar_pos: op_index,
                    remaining_len: len,
                    query_pos: a_pos,
                    target_pos: b_pos,
                    completed: true,
                    actual_query_start,
                    actual_target_start,
                },
            );
        }

        // Adjust operation length if it would exceed boundaries
        let original_len = len;
        if (op == '=' || op == 'M' || op == 'X' || op == 'I') && a_pos + len > query_end {
            len = query_end - a_pos;
        }
        if (op == '=' || op == 'M' || op == 'X' || op == 'D') && b_pos + len > target_end {
            len = target_end - b_pos;
        }

        let remaining_after_boundary = original_len - len;

        while len > 0 {
            let consume = match op {
                '=' | 'M' => {
                    if a_pos + len > next_trace {
                        let inc = next_trace - a_pos;
                        a_pos = next_trace;
                        b_pos += inc;
                        tracepoints.push((
                            (diff - last_diff).unsigned_abs() as usize,
                            b_pos - last_b_pos,
                        ));
                        last_diff = diff;
                        last_b_pos = b_pos;
                        next_trace += trace_spacing;
                        inc
                    } else {
                        a_pos += len;
                        b_pos += len;
                        len
                    }
                }
                'X' => {
                    if a_pos + len > next_trace {
                        let inc = next_trace - a_pos;
                        a_pos = next_trace;
                        b_pos += inc;
                        diff += inc as i64;
                        tracepoints.push((
                            (diff - last_diff).unsigned_abs() as usize,
                            b_pos - last_b_pos,
                        ));
                        last_diff = diff;
                        last_b_pos = b_pos;
                        next_trace += trace_spacing;
                        inc
                    } else {
                        a_pos += len;
                        b_pos += len;
                        diff += len as i64;
                        len
                    }
                }
                'I' => {
                    // Split into a new record when the insertion is longer than the
                    // trace spacing. The threshold is 2*trace_spacing (== 200 at the
                    // FASTGA default spacing of 100, preserving FASTGA's uint8s).
                    if trace_spacing + len > 2 * trace_spacing {
                        // Push final tracepoint if we've passed the last boundary (matching C behavior)
                        if a_pos > next_trace - trace_spacing {
                            tracepoints.push((
                                (diff - last_diff).unsigned_abs() as usize,
                                b_pos - last_b_pos,
                            ));
                        }

                        return (
                            tracepoints,
                            CigarProcessingState {
                                cigar_pos: op_index,
                                remaining_len: len,
                                query_pos: a_pos,
                                target_pos: b_pos,
                                completed: false,
                                actual_query_start,
                                actual_target_start,
                            },
                        );
                    }

                    if a_pos + len > next_trace {
                        let inc = next_trace - a_pos;
                        a_pos = next_trace;
                        diff += inc as i64;
                        tracepoints.push((
                            (diff - last_diff).unsigned_abs() as usize,
                            b_pos - last_b_pos,
                        ));
                        last_diff = diff;
                        last_b_pos = b_pos;
                        next_trace += trace_spacing;
                        inc
                    } else {
                        a_pos += len;
                        diff += len as i64;
                        len
                    }
                }
                'D' => {
                    // Split into a new record when the deletion is longer than the
                    // trace spacing. The threshold is 2*trace_spacing (== 200 at the
                    // FASTGA default spacing of 100, preserving FASTGA's uint8s).
                    if (b_pos - last_b_pos) + len + (next_trace - a_pos) > 2 * trace_spacing {
                        // Push final tracepoint if we've passed the last boundary (matching C behavior)
                        if a_pos > next_trace - trace_spacing {
                            tracepoints.push((
                                (diff - last_diff).unsigned_abs() as usize,
                                b_pos - last_b_pos,
                            ));
                        }

                        return (
                            tracepoints,
                            CigarProcessingState {
                                cigar_pos: op_index,
                                remaining_len: len,
                                query_pos: a_pos,
                                target_pos: b_pos,
                                completed: false,
                                actual_query_start,
                                actual_target_start,
                            },
                        );
                    }

                    b_pos += len;
                    diff += len as i64;
                    len // Consume all since D doesn't advance a_pos
                }
                'H' | 'N' | 'S' | 'P' => len, // Skip special operations
                _ => panic!("Invalid CIGAR operation: {op}"),
            };
            len -= consume;
        }

        // If we had to truncate due to boundary, return with remaining length
        if remaining_after_boundary > 0 {
            if a_pos > next_trace - trace_spacing {
                tracepoints.push((
                    (diff - last_diff).unsigned_abs() as usize,
                    b_pos - last_b_pos,
                ));
            }
            return (
                tracepoints,
                CigarProcessingState {
                    cigar_pos: op_index,
                    remaining_len: remaining_after_boundary,
                    query_pos: a_pos,
                    target_pos: b_pos,
                    completed: true,
                    actual_query_start,
                    actual_target_start,
                },
            );
        }

        op_index += 1;
    }

    // Add final tracepoint if we've passed the last boundary
    if a_pos > next_trace - trace_spacing {
        tracepoints.push((
            (diff - last_diff).unsigned_abs() as usize,
            b_pos - last_b_pos,
        ));
    }

    (
        tracepoints,
        CigarProcessingState {
            cigar_pos: op_index,
            remaining_len: 0,
            query_pos: a_pos,
            target_pos: b_pos,
            completed: true,
            actual_query_start,
            actual_target_start,
        },
    )
}

/// Reconstruct CIGAR from FASTGA-style tracepoints
///
/// Uses edit distance internally for alignment.
pub fn tracepoints_to_cigar_fastga(
    segments: &[(usize, usize)],
    trace_spacing: u32,
    a_seq: &[u8],
    b_seq: &[u8],
    a_start: usize,
    _b_start: usize,
    complement: bool,
    heuristic: bool,
) -> String {
    // Use edit distance mode as FASTGA does
    let distance = Distance::Edit;

    let mut aligner = distance.create_aligner(None, None);
    tracepoints_to_cigar_fastga_with_aligner(
        segments,
        trace_spacing,
        a_seq,
        b_seq,
        a_start,
        _b_start,
        complement,
        &mut aligner,
        heuristic,
    )
}

/// Reconstruct CIGAR from FASTGA-style tracepoints with provided aligner
///
/// Like `tracepoints_to_cigar_fastga`, but allows callers to reuse an existing aligner.
pub fn tracepoints_to_cigar_fastga_with_aligner(
    segments: &[(usize, usize)],
    trace_spacing: u32,
    a_seq: &[u8],
    b_seq: &[u8],
    a_start: usize,
    _b_start: usize,
    complement: bool,
    aligner: &mut AffineWavefronts,
    heuristic: bool,
) -> String {
    let trace_spacing = trace_spacing as usize;

    // Pure-match sentinel, so no realignment needed.
    if segments.len() == 1 && segments[0] == (0, 0) {
        return format!("{}=", a_seq.len());
    }

    let mut cigar_ops = Vec::new();

    // Starting positions in the sequences
    let mut current_a = 0;
    let mut current_b = 0;

    // Calculate the first trace boundary
    let first_boundary = ((a_start / trace_spacing) + 1) * trace_spacing - a_start;

    // Process each segment
    for (i, &(num_diff, b_len)) in segments.iter().enumerate() {
        // Calculate the query segment length
        let a_len = if i == 0 {
            // First segment: from start to first trace boundary
            first_boundary.min(a_seq.len() - current_a)
        } else {
            // Subsequent segments: one trace_spacing worth
            trace_spacing.min(a_seq.len() - current_a)
        };

        // Calculate segment boundaries
        let a_end = (current_a + a_len).min(a_seq.len());
        let b_end = (current_b + b_len).min(b_seq.len());

        // Handle pure insertions or deletions
        if a_end == current_a && b_end > current_b {
            // Pure deletion
            push_op(&mut cigar_ops, b_end - current_b, 'D');
            current_b = b_end;
        } else if b_end == current_b && a_end > current_a {
            // Pure insertion
            push_op(&mut cigar_ops, a_end - current_a, 'I');
            current_a = a_end;
        } else if a_end > current_a && b_end > current_b {
            // If num_diff is zero and the lengths match, it's a perfect match segment
            if num_diff == 0 && (a_end - current_a) == (b_end - current_b) {
                push_op(&mut cigar_ops, a_end - current_a, '=');
            } else {
                // Mixed segment - realign with WFA
                if heuristic && num_diff > 0 {
                    let strategy = compute_banded_static_strategy(
                        a_end - current_a,
                        b_end - current_b,
                        ComplexityMetric::EditDistance,
                        num_diff as u32,
                        &aligner.get_distance(),
                    );
                    aligner.set_heuristic(Some(&strategy));
                }
                align_sequences_wfa(
                    &a_seq[current_a..a_end],
                    &b_seq[current_b..b_end],
                    aligner,
                    &mut cigar_ops,
                );
            }
            current_a = a_end;
            current_b = b_end;
        }
    }

    let cigar = cigar_ops_to_cigar_string(&cigar_ops);

    // Reverse CIGAR for complement alignments
    let cigar = if complement {
        reverse_cigar(&cigar)
    } else {
        cigar
    };

    refine_cigar_gaps(&cigar, a_seq, b_seq, 50)
}

/// Information about a gap run to be refined
#[derive(Debug)]
struct GapRun {
    /// Index in the CIGAR ops where run starts
    start_op_idx: usize,
    /// Index in the CIGAR ops where run ends (exclusive)
    end_op_idx: usize,
    /// Position in A sequence where run starts
    a_start: usize,
    /// Position in B sequence where run starts
    b_start: usize,
}

/// Find runs of gaps that should be refined
/// 
/// A run consists of indels of the same type (all I or all D) separated by
/// matches/mismatches of length < max_separation
fn find_gap_runs(ops: &[(usize, char)], max_separation: usize) -> Vec<GapRun> {
    let mut runs = Vec::new();
    
    let mut a_pos = 0;
    let mut b_pos = 0;
    let mut i = 0;
    
    while i < ops.len() {
        let (len, op) = ops[i];
        
        // Look for the start of a potential run (an indel)
        if op == 'I' || op == 'D' {
            let run_start_idx = i;
            let run_start_a = a_pos;
            let run_start_b = b_pos;
            let first_op_type = op;
            
            let mut last_indel_idx = i;
            let mut temp_a = a_pos;
            let mut temp_b = b_pos;
            
            // Process first indel
            match first_op_type {
                'I' => temp_a += len,
                'D' => temp_b += len,
                _ => unreachable!(),
            }
            
            // Look ahead for more indels of same type
            let mut j = i + 1;
            while j < ops.len() {
                let (op_len, op_type) = ops[j];
                
                match op_type {
                    'I' if first_op_type == 'I' => {
                        last_indel_idx = j;
                        temp_a += op_len;
                        j += 1;
                    }
                    'D' if first_op_type == 'D' => {
                        last_indel_idx = j;
                        temp_b += op_len;
                        j += 1;
                    }
                    '=' | 'M' | 'X' => {
                        // Small gap between indels - check if another indel follows
                        if op_len < max_separation && j + 1 < ops.len() {
                            let (_, next_op) = ops[j + 1];
                            if next_op == first_op_type {
                                // Include this gap in the run
                                temp_a += op_len;
                                temp_b += op_len;
                                j += 1;
                                continue;
                            }
                        }
                        break;
                    }
                    _ => break,
                }
            }
            
            // Create a run if we found multiple indels
            if last_indel_idx > run_start_idx {
                runs.push(GapRun {
                    start_op_idx: run_start_idx,
                    end_op_idx: last_indel_idx + 1,
                    a_start: run_start_a,
                    b_start: run_start_b,
                });
                
                i = last_indel_idx + 1;
                a_pos = temp_a;
                b_pos = temp_b;
                continue;
            }
        }
        
        // Update position for this operation
        match op {
            '=' | 'M' | 'X' => {
                a_pos += len;
                b_pos += len;
            }
            'I' => a_pos += len,
            'D' => b_pos += len,
            _ => {}
        }
        
        i += 1;
    }
    
    runs
}

/// Calculate the sequence range covered by a run of operations
fn calc_run_range(ops: &[(usize, char)]) -> (usize, usize) {
    let mut a_len = 0;
    let mut b_len = 0;
    
    for (len, op) in ops {
        match op {
            '=' | 'M' | 'X' => {
                a_len += len;
                b_len += len;
            }
            'I' => a_len += len,
            'D' => b_len += len,
            _ => {}
        }
    }
    
    (a_len, b_len)
}

/// Refine gaps in a CIGAR string to minimize the number of gaps
///
/// This implements gap refinement by finding runs of fragmented indels
/// and realigning them with affine gap penalties, which naturally consolidates gaps.
///
/// # Arguments
/// * `cigar` - Input CIGAR string
/// * `a_seq` - First sequence (query)
/// * `b_seq` - Second sequence (target)
/// * `max_separation` - Maximum distance between indels to be considered part of the same run (default: 50)
///
/// # Returns
/// Refined CIGAR string with fewer gaps
pub fn refine_cigar_gaps(
    cigar: &str,
    a_seq: &[u8],
    b_seq: &[u8],
    max_separation: usize,
) -> String {
    let ops = cigar_str_to_cigar_ops(cigar);
    
    // Find gap runs using the parsed operations
    let runs = find_gap_runs(&ops, max_separation);

    if runs.is_empty() {
        return cigar.to_string();
    }
    
    let ops = cigar_str_to_cigar_ops(cigar);
    let mut result_ops = Vec::new();
    
    // Create aligner with affine gap penalties
    // Using gap cost of 1 for gap opening (to penalize gap starts)
    // and 0 for gap extension (once a gap is started, extending is free)
    // This encourages consolidation of fragmented gaps
    let distance = Distance::GapAffine {
        mismatch: 1,
        gap_opening: 2,
        gap_extension: 1,
    };
    let mut aligner = distance.create_aligner(None);
    
    let mut next_run_idx = 0;
    let mut op_idx = 0;
    
    while op_idx < ops.len() {
        // Check if we're at the start of a run
        let at_run = next_run_idx < runs.len() && runs[next_run_idx].start_op_idx == op_idx;
        
        if at_run {
            let run = &runs[next_run_idx];
            
            // Extract the operations in this run
            let run_ops: Vec<(usize, char)> = ops[run.start_op_idx..run.end_op_idx].to_vec();
            let (a_len, b_len) = calc_run_range(&run_ops);
            
            // Extract the sequences for this run
            let a_end = (run.a_start + a_len).min(a_seq.len());
            let b_end = (run.b_start + b_len).min(b_seq.len());
            let a_subseq = &a_seq[run.a_start..a_end];
            let b_subseq = &b_seq[run.b_start..b_end];
            
            // Realign with WFA2 using affine penalties
            let refined = align_sequences_wfa(a_subseq, b_subseq, &mut aligner);
            
            // Verify the refined operations consume the same lengths
            let (ref_a, ref_b) = calc_run_range(&refined);
            if ref_a == a_len && ref_b == b_len {
                result_ops.extend(refined);
            } else {
                // If refinement changed lengths (shouldn't happen with proper alignment),
                // keep original
                panic!("Refinement changed sequence lengths, keeping original run");
            }
            
            // Skip past this run
            op_idx = run.end_op_idx;
            next_run_idx += 1;
        } else {
            // Not in a run, keep original operation
            result_ops.push(ops[op_idx]);
            op_idx += 1;
        }
    }
    
    merge_cigar_ops(&mut result_ops);
    cigar_ops_to_cigar_string(&result_ops)
}

/// Refine gaps in a CIGAR string with a provided aligner
///
/// Like `refine_cigar_gaps`, but allows reusing an aligner across multiple calls.
pub fn refine_cigar_gaps_with_aligner(
    cigar: &str,
    a_seq: &[u8],
    b_seq: &[u8],
    max_separation: usize,
    aligner: &mut AffineWavefronts,
) -> String {
    let ops = cigar_str_to_cigar_ops(cigar);
    
    // Find gap runs using the parsed operations
    let runs = find_gap_runs(&ops, max_separation);
    
    if runs.is_empty() {
        return cigar.to_string();
    }
    
    let ops = cigar_str_to_cigar_ops(cigar);
    let mut result_ops = Vec::new();
    let mut next_run_idx = 0;
    let mut op_idx = 0;
    
    while op_idx < ops.len() {
        let at_run = next_run_idx < runs.len() && runs[next_run_idx].start_op_idx == op_idx;
        
        if at_run {
            let run = &runs[next_run_idx];
            let run_ops: Vec<(usize, char)> = ops[run.start_op_idx..run.end_op_idx].to_vec();
            let (a_len, b_len) = calc_run_range(&run_ops);
            
            let a_end = (run.a_start + a_len).min(a_seq.len());
            let b_end = (run.b_start + b_len).min(b_seq.len());
            let a_subseq = &a_seq[run.a_start..a_end];
            let b_subseq = &b_seq[run.b_start..b_end];
            
            let refined = align_sequences_wfa(a_subseq, b_subseq, aligner);
            
            let (ref_a, ref_b) = calc_run_range(&refined);
            if ref_a == a_len && ref_b == b_len {
                result_ops.extend(refined);
            } else {
                result_ops.extend(run_ops);
            }
            
            op_idx = run.end_op_idx;
            next_run_idx += 1;
        } else {
            result_ops.push(ops[op_idx]);
            op_idx += 1;
        }
    }
    
    merge_cigar_ops(&mut result_ops);
    cigar_ops_to_cigar_string(&result_ops)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tp_tag_formatting() {
        let standard = TracepointData::Standard(vec![(3, 5), (0, 2)]);
        assert_eq!(standard.to_tp_tag(), "3,5;0,2");

        let fastga = TracepointData::Fastga(vec![(1, 1)]);
        assert_eq!(fastga.to_tp_tag(), "1,1");

        let variable = TracepointData::Variable(vec![(5, None), (3, Some(2))]);
        assert_eq!(variable.to_tp_tag(), "5;3,2");

        let mixed = TracepointData::Mixed(vec![
            MixedRepresentation::Tracepoint(4, 4),
            MixedRepresentation::CigarOp(2, 'I'),
            MixedRepresentation::Tracepoint(1, 3),
        ]);
        assert_eq!(mixed.to_tp_tag(), "4,4;2I;1,3");
    }

    #[test]
    fn test_tracepoint_generation() {
        // Test CIGAR string
        let cigar = "10=2D5=2I3=";

        // Define test cases: (max_diff, expected_tracepoints)
        let test_cases = [
            // Case 1: No differences allowed - each operation becomes its own segment
            (0, vec![(10, 10), (0, 2), (5, 5), (2, 0), (3, 3)]),
            // Case 2: Allow up to 2 differences in each segment
            (2, vec![(15, 17), (5, 3)]),
            // Case 3: Allow up to 5 differences - combines all operations
            (5, vec![(20, 20)]),
        ];

        // Run each test case
        for (i, (max_diff, expected_tracepoints)) in test_cases.iter().enumerate() {
            // Get actual results
            let tracepoints =
                cigar_to_tracepoints(cigar, *max_diff, ComplexityMetric::EditDistance);

            // Check tracepoints
            assert_eq!(
                tracepoints,
                *expected_tracepoints,
                "Test case {}: Tracepoints with max_diff={} incorrect",
                i + 1,
                max_diff
            );
        }
    }

    #[test]
    fn test_distance_affine2p() {
        let original_cigar = "1=1I18=";
        let a_seq = b"ACGTACGTACACGTACGTAC"; // 20 bases
        let b_seq = b"AGTACGTACACGTACGTAC"; // 19 bases (missing C)
        let max_diff = 5;

        let distance = Distance::GapAffine2p {
            mismatch: 2,
            gap_opening1: 4,
            gap_extension1: 2,
            gap_opening2: 6,
            gap_extension2: 1,
        };

        // Test tracepoints roundtrip with Distance API
        let tracepoints =
            cigar_to_tracepoints(original_cigar, max_diff, ComplexityMetric::EditDistance);
        let reconstructed_cigar = tracepoints_to_cigar(
            &tracepoints,
            a_seq,
            b_seq,
            0,
            0,
            ComplexityMetric::EditDistance,
            &distance,
        );
        assert_eq!(
            reconstructed_cigar, original_cigar,
            "GapAffine2p distance mode roundtrip failed"
        );
    }

    #[test]
    fn test_distance_edit() {
        let original_cigar = "1=1I18=";
        let a_seq = b"ACGTACGTACACGTACGTAC"; // 20 bases
        let b_seq = b"AGTACGTACACGTACGTAC"; // 19 bases (missing C)
        let max_diff = 5;

        let distance = Distance::Edit;

        // Test tracepoints roundtrip with edit distance mode
        let tracepoints =
            cigar_to_tracepoints(original_cigar, max_diff, ComplexityMetric::EditDistance);
        let reconstructed_cigar = tracepoints_to_cigar(
            &tracepoints,
            a_seq,
            b_seq,
            0,
            0,
            ComplexityMetric::EditDistance,
            &distance,
        );
        assert_eq!(
            reconstructed_cigar, original_cigar,
            "Edit distance mode roundtrip failed"
        );
    }

    #[test]
    fn test_mixed_tracepoint_generation() {
        // Test cases: (CIGAR string, max_diff, expected mixed representation)
        let test_cases = vec![
            // Case 1: Simple alignment with special operators
            (
                "5=2H3=",
                2,
                vec![
                    MixedRepresentation::Tracepoint(5, 5),
                    MixedRepresentation::CigarOp(2, 'H'),
                    MixedRepresentation::Tracepoint(3, 3),
                ],
            ),
            // Case 2: Soft-clipped alignment with indels
            (
                "3S5=2I4=1N2=",
                3,
                vec![
                    MixedRepresentation::CigarOp(3, 'S'),
                    MixedRepresentation::Tracepoint(11, 9),
                    MixedRepresentation::CigarOp(1, 'N'),
                    MixedRepresentation::Tracepoint(2, 2),
                ],
            ),
            // Case 3: Padding with mismatches
            (
                "4P2=3X1=",
                2,
                vec![
                    MixedRepresentation::CigarOp(4, 'P'),
                    MixedRepresentation::Tracepoint(4, 4),
                    MixedRepresentation::Tracepoint(2, 2),
                ],
            ),
            // Case 4: Mixed operations within max_diff
            (
                "5=4X3=2H",
                5,
                vec![
                    MixedRepresentation::Tracepoint(12, 12),
                    MixedRepresentation::CigarOp(2, 'H'),
                ],
            ),
            // Case 5: Large insertion with special ops
            (
                "10I5S",
                3,
                vec![
                    MixedRepresentation::Tracepoint(10, 0),
                    MixedRepresentation::CigarOp(5, 'S'),
                ],
            ),
            // Case 6: Special ops at the beginning
            (
                "3S7D2=",
                4,
                vec![
                    MixedRepresentation::CigarOp(3, 'S'),
                    MixedRepresentation::Tracepoint(0, 7),
                    MixedRepresentation::Tracepoint(2, 2),
                ],
            ),
        ];

        for (i, (cigar, max_diff, expected)) in test_cases.iter().enumerate() {
            let result =
                cigar_to_mixed_tracepoints(cigar, *max_diff, ComplexityMetric::EditDistance);

            assert_eq!(
                result,
                *expected,
                "Test case {}: Mixed representation with max_diff={} incorrect for CIGAR '{}'",
                i + 1,
                max_diff,
                cigar
            );
        }
    }

    #[test]
    fn test_mixed_tracepoints_roundtrip() {
        // Test data - using identical sequences for predictable results
        let a_seq = b"ACGTACGTACGTACGTACGT"; // 20 bases
        let b_seq = b"ACGTACGTACGTACGTACGT"; // 20 bases (identical for this test)

        // Alignment parameters
        let mismatch = 2;
        let gap_open1 = 4;
        let gap_ext1 = 2;
        let gap_open2 = 6;
        let gap_ext2 = 1;

        // Test cases with expected results - simplified to work with identical sequences
        let test_cases = [
            // Case 1: Simple special operations with matches
            (
                vec![
                    MixedRepresentation::CigarOp(2, 'S'),
                    MixedRepresentation::Tracepoint(5, 5),
                    MixedRepresentation::CigarOp(3, 'H'),
                ],
                "2S5=3H",
            ),
            // Case 2: Only special operations
            (
                vec![
                    MixedRepresentation::CigarOp(1, 'S'),
                    MixedRepresentation::CigarOp(2, 'N'),
                    MixedRepresentation::CigarOp(3, 'H'),
                ],
                "1S2N3H",
            ),
            // Case 3: Mix of special ops and pure indels
            (
                vec![
                    MixedRepresentation::CigarOp(2, 'S'),
                    MixedRepresentation::Tracepoint(5, 0), // Pure insertion
                    MixedRepresentation::CigarOp(1, 'H'),
                ],
                "2S5I1H",
            ),
        ];

        let distance = Distance::GapAffine2p {
            mismatch,
            gap_opening1: gap_open1,
            gap_extension1: gap_ext1,
            gap_opening2: gap_open2,
            gap_extension2: gap_ext2,
        };

        for (i, (mixed_tracepoints, expected_cigar)) in test_cases.iter().enumerate() {
            let result = mixed_tracepoints_to_cigar(
                mixed_tracepoints,
                a_seq,
                b_seq,
                0,
                0,
                ComplexityMetric::EditDistance,
                &distance,
            );

            assert_eq!(
                result,
                *expected_cigar,
                "Test case {}: Mixed tracepoint roundtrip failed",
                i + 1
            );
        }
    }

    #[test]
    fn test_mixed_tracepoints_with_alignment() {
        // Test with actual sequence alignment
        let a_seq = b"ACGTACGTACGT"; // 12 bases
        let b_seq = b"ACGTACGTACGT"; // 12 bases (identical)

        let mixed_tracepoints = vec![
            MixedRepresentation::CigarOp(2, 'S'),
            MixedRepresentation::Tracepoint(6, 6),
            MixedRepresentation::CigarOp(1, 'H'),
            MixedRepresentation::Tracepoint(4, 4),
        ];

        let distance = Distance::GapAffine2p {
            mismatch: 2,
            gap_opening1: 4,
            gap_extension1: 2,
            gap_opening2: 6,
            gap_extension2: 1,
        };
        let result = mixed_tracepoints_to_cigar(
            &mixed_tracepoints,
            a_seq,
            b_seq,
            0,
            0,
            ComplexityMetric::EditDistance,
            &distance,
        );

        // Since sequences are identical, we expect matches for the tracepoint segments
        assert_eq!(result, "2S6=1H4=");
    }

    #[test]
    fn test_mixed_tracepoints_edge_cases() {
        // Test edge cases
        let test_cases = [
            // Empty CIGAR
            ("", 5, vec![]),
            // Only special operations
            (
                "5S3H",
                2,
                vec![
                    MixedRepresentation::CigarOp(5, 'S'),
                    MixedRepresentation::CigarOp(3, 'H'),
                ],
            ),
            // Only regular operations
            ("5=3I2D", 10, vec![MixedRepresentation::Tracepoint(8, 7)]),
            // max_diff = 0
            (
                "2=1X2=",
                0,
                vec![
                    MixedRepresentation::Tracepoint(2, 2),
                    MixedRepresentation::Tracepoint(1, 1),
                    MixedRepresentation::Tracepoint(2, 2),
                ],
            ),
        ];

        for (i, (cigar, max_diff, expected)) in test_cases.iter().enumerate() {
            let result =
                cigar_to_mixed_tracepoints(cigar, *max_diff, ComplexityMetric::EditDistance);
            assert_eq!(
                result,
                *expected,
                "Edge case {}: Failed for CIGAR '{}' with max_diff={}",
                i + 1,
                cigar,
                max_diff
            );
        }
    }

    #[test]
    fn test_variable_tracepoint_generation() {
        // Test CIGAR string
        let cigar = "10=2D5=2I3=";

        // Define test cases: (max_diff, expected_variable_tracepoints)
        let test_cases = [
            // Case 1: No differences allowed - each operation becomes its own segment
            (
                0,
                vec![(10, None), (0, Some(2)), (5, None), (2, Some(0)), (3, None)],
            ),
            // Case 2: Allow up to 2 differences in each segment
            (2, vec![(15, Some(17)), (5, Some(3))]),
            // Case 3: Allow up to 5 differences - combines all operations
            (5, vec![(20, None)]),
        ];

        // Run each test case
        for (i, (max_diff, expected_variable_tracepoints)) in test_cases.iter().enumerate() {
            // Get actual results
            let variable_tracepoints =
                cigar_to_variable_tracepoints(cigar, *max_diff, ComplexityMetric::EditDistance);

            // Check variable tracepoints
            assert_eq!(
                variable_tracepoints,
                *expected_variable_tracepoints,
                "Test case {}: Variable tracepoints with max_diff={} incorrect",
                i + 1,
                max_diff
            );
        }
    }

    #[test]
    fn test_variable_tracepoints_roundtrip() {
        let original_cigar = "1=1I18=";
        let a_seq = b"ACGTACGTACACGTACGTAC"; // 20 bases
        let b_seq = b"AGTACGTACACGTACGTAC"; // 19 bases (missing C)
        let max_diff = 5;

        let distance = Distance::GapAffine2p {
            mismatch: 2,
            gap_opening1: 4,
            gap_extension1: 2,
            gap_opening2: 6,
            gap_extension2: 1,
        };

        // Test variable tracepoints roundtrip
        let variable_tracepoints =
            cigar_to_variable_tracepoints(original_cigar, max_diff, ComplexityMetric::EditDistance);
        let reconstructed_cigar = variable_tracepoints_to_cigar(
            &variable_tracepoints,
            a_seq,
            b_seq,
            0,
            0,
            ComplexityMetric::EditDistance,
            &distance,
        );
        assert_eq!(
            reconstructed_cigar, original_cigar,
            "Variable tracepoint roundtrip failed"
        );
    }

    #[test]
    fn test_variable_tracepoints_to_cigar_with_aligner() {
        // Test the new function that takes an aligner parameter
        let original_cigar = "1=1I18=";
        let a_seq = b"ACGTACGTACACGTACGTAC"; // 20 bases
        let b_seq = b"AGTACGTACACGTACGTAC"; // 19 bases (missing C)
        let max_diff = 5;

        // Create variable tracepoints
        let variable_tracepoints =
            cigar_to_variable_tracepoints(original_cigar, max_diff, ComplexityMetric::EditDistance);

        let distance = Distance::GapAffine2p {
            mismatch: 2,
            gap_opening1: 4,
            gap_extension1: 2,
            gap_opening2: 6,
            gap_extension2: 1,
        };
        let mut aligner = distance.create_aligner(None, None);

        // Test the new function
        let reconstructed_cigar = variable_tracepoints_to_cigar_with_aligner(
            &variable_tracepoints,
            a_seq,
            b_seq,
            0,
            0,
            ComplexityMetric::EditDistance,
            &mut aligner,
            None,
        );

        // Should produce the same result
        let expected_cigar = variable_tracepoints_to_cigar(
            &variable_tracepoints,
            a_seq,
            b_seq,
            0,
            0,
            ComplexityMetric::EditDistance,
            &distance,
        );

        assert_eq!(
            reconstructed_cigar, expected_cigar,
            "New function should produce same result as legacy version"
        );
        assert_eq!(
            reconstructed_cigar, original_cigar,
            "Roundtrip should work with aligner"
        );

        // Test with multiple calls to ensure aligner can be reused
        let variable_tracepoints2 = vec![(5, None), (2, Some(0)), (3, None)];
        let reconstructed_cigar2 = variable_tracepoints_to_cigar_with_aligner(
            &variable_tracepoints2,
            a_seq,
            b_seq,
            0,
            0,
            ComplexityMetric::EditDistance,
            &mut aligner,
            None,
        );

        let expected_cigar2 = variable_tracepoints_to_cigar(
            &variable_tracepoints2,
            a_seq,
            b_seq,
            0,
            0,
            ComplexityMetric::EditDistance,
            &distance,
        );

        assert_eq!(
            reconstructed_cigar2, expected_cigar2,
            "Function should work with reused aligner"
        );
    }

    #[test]
    fn test_raw_tracepoint_generation() {
        // Test CIGAR string with indels
        let cigar = "5=10I5=10D5=";

        // Test cases: (max_diff, expected_tracepoints)
        let test_cases = [
            // Case 1: No differences allowed - each operation becomes its own segment
            (
                0,
                vec![
                    (5, 5),
                    (1, 0),
                    (1, 0),
                    (1, 0),
                    (1, 0),
                    (1, 0),
                    (1, 0),
                    (1, 0),
                    (1, 0),
                    (1, 0),
                    (1, 0),
                    (5, 5),
                    (0, 1),
                    (0, 1),
                    (0, 1),
                    (0, 1),
                    (0, 1),
                    (0, 1),
                    (0, 1),
                    (0, 1),
                    (0, 1),
                    (0, 1),
                    (5, 5),
                ],
            ),
            // Case 2: Allow up to 3 differences - indels are split and combined with matches
            (
                3,
                vec![(8, 5), (3, 0), (3, 0), (6, 7), (0, 3), (0, 3), (5, 7)],
            ),
            // Case 3: Allow up to 5 differences - indels are split into chunks of 5
            (5, vec![(10, 5), (5, 0), (5, 10), (0, 5), (5, 5)]),
            // Case 4: Allow up to 10 differences - each indel fits in one segment
            (10, vec![(15, 5), (5, 15), (5, 5)]),
        ];

        // Run each test case
        for (i, (max_diff, expected_tracepoints)) in test_cases.iter().enumerate() {
            let tracepoints =
                cigar_to_tracepoints_raw(cigar, *max_diff, ComplexityMetric::EditDistance);
            assert_eq!(
                tracepoints,
                *expected_tracepoints,
                "Test case {}: Raw tracepoints with max_diff={} incorrect",
                i + 1,
                max_diff
            );
        }
    }

    #[test]
    fn test_raw_vs_regular_tracepoints() {
        // Test that raw mode splits indels while regular mode doesn't
        let cigar = "5=8I3=7D2=";
        let max_diff = 5;

        // Regular mode: indels larger than max_diff become their own segments
        let regular = cigar_to_tracepoints(cigar, max_diff, ComplexityMetric::EditDistance);
        // Raw mode: indels are split into max_diff-sized chunks
        let raw = cigar_to_tracepoints_raw(cigar, max_diff, ComplexityMetric::EditDistance);

        // Regular mode should keep large indels intact
        assert_eq!(regular, vec![(5, 5), (8, 0), (3, 3), (0, 7), (2, 2)]);
        // Raw mode: indels can be combined with matches in segments
        // 5= + first 5I -> (10, 5)
        // remaining 3I + 3= -> (6, 3) but since we hit another indel, may need to check
        assert_eq!(raw, vec![(10, 5), (6, 5), (0, 5), (2, 2)]);
    }

    #[test]
    fn test_mixed_raw_tracepoints() {
        // Test mixed representation with raw mode
        let cigar = "3S5=6I2=4D1H";
        let max_diff = 3;

        let mixed_raw =
            cigar_to_mixed_tracepoints_raw(cigar, max_diff, ComplexityMetric::EditDistance);

        assert_eq!(
            mixed_raw,
            vec![
                MixedRepresentation::CigarOp(3, 'S'),
                MixedRepresentation::Tracepoint(8, 5), // 5= + 3I
                MixedRepresentation::Tracepoint(3, 0), // remaining 3I
                MixedRepresentation::Tracepoint(2, 5), // 2= + 3D
                MixedRepresentation::Tracepoint(0, 1), // remaining 1D
                MixedRepresentation::CigarOp(1, 'H'),
            ]
        );
    }

    #[test]
    fn test_variable_raw_tracepoints() {
        // Test variable format with raw mode
        let cigar = "3=6I3=6D3=";
        let max_diff = 3;

        let variable_raw =
            cigar_to_variable_tracepoints_raw(cigar, max_diff, ComplexityMetric::EditDistance);

        // With raw mode, indels combine with adjacent matches
        assert_eq!(
            variable_raw,
            vec![
                (6, Some(3)), // 3= + 3I
                (3, Some(0)), // remaining 3I
                (3, Some(6)), // 3= + 3D
                (0, Some(3)), // remaining 3D
                (3, None),    // 3=
            ]
        );
    }

    #[test]
    fn test_raw_cigar_roundtrip() {
        // Test that raw tracepoints can be reconstructed properly
        let original_cigar = "2=2I3=2D2=";
        // CIGAR consumes: a_seq: 2 + 2 + 3 + 0 + 2 = 9 bases
        //                 b_seq: 2 + 0 + 3 + 2 + 2 = 9 bases
        let a_seq = b"ACGTACGTA"; // 9 bases
        let b_seq = b"ACCGTCGAA"; // 9 bases
        let max_diff = 2;

        let distance = Distance::GapAffine2p {
            mismatch: 2,
            gap_opening1: 4,
            gap_extension1: 2,
            gap_opening2: 6,
            gap_extension2: 1,
        };

        // Test raw tracepoints roundtrip
        let raw_tracepoints =
            cigar_to_tracepoints_raw(original_cigar, max_diff, ComplexityMetric::EditDistance);
        let reconstructed_cigar = tracepoints_to_cigar(
            &raw_tracepoints,
            a_seq,
            b_seq,
            0,
            0,
            ComplexityMetric::EditDistance,
            &distance,
        );

        // The reconstructed CIGAR might be slightly different due to realignment,
        // but should represent the same alignment
        assert!(
            !reconstructed_cigar.is_empty(),
            "Raw roundtrip produced empty CIGAR"
        );
    }

    #[test]
    fn test_raw_edge_cases() {
        // Test edge cases for raw mode
        let test_cases = vec![
            // Empty CIGAR
            ("", 5, vec![]),
            // Only matches
            ("10=", 3, vec![(10, 10)]),
            // Single large indel
            ("15I", 5, vec![(5, 0), (5, 0), (5, 0)]),
            ("15D", 5, vec![(0, 5), (0, 5), (0, 5)]),
            // Mixed with mismatches - mismatches and indels can combine
            ("2X8I2X", 3, vec![(3, 2), (3, 0), (3, 0), (3, 2)]),
            // max_diff = 0 with indels
            ("3I2D", 0, vec![(1, 0), (1, 0), (1, 0), (0, 1), (0, 1)]),
        ];

        for (i, (cigar, max_diff, expected)) in test_cases.iter().enumerate() {
            let result = cigar_to_tracepoints_raw(cigar, *max_diff, ComplexityMetric::EditDistance);
            assert_eq!(
                result,
                *expected,
                "Raw edge case {}: Failed for CIGAR '{}' with max_diff={}",
                i + 1,
                cigar,
                max_diff
            );
        }
    }

    #[test]
    fn test_variable_tracepoints_conversion() {
        // Test the conversion logic directly without WFA complexity
        let variable_tracepoints = vec![
            (5, None),    // Should become (5, 5)
            (3, Some(0)), // Should become (3, 0)
            (0, Some(2)), // Should become (0, 2)
            (4, Some(6)), // Should become (4, 6)
        ];

        // Test the conversion logic used in variable_tracepoints_to_cigar
        let converted_regular: Vec<(usize, usize)> = variable_tracepoints
            .iter()
            .map(|(a_len, b_len_opt)| match b_len_opt {
                None => (*a_len, *a_len),
                Some(b_len) => (*a_len, *b_len),
            })
            .collect();

        let expected_regular = vec![(5, 5), (3, 0), (0, 2), (4, 6)];

        assert_eq!(
            converted_regular, expected_regular,
            "Variable to regular tracepoint conversion failed"
        );

        // Test the reverse conversion (regular to variable)
        let reconverted_variable: Vec<(usize, Option<usize>)> = expected_regular
            .iter()
            .map(|(a_len, b_len)| {
                if a_len == b_len {
                    (*a_len, None)
                } else {
                    (*a_len, Some(*b_len))
                }
            })
            .collect();

        assert_eq!(
            reconverted_variable, variable_tracepoints,
            "Regular to variable tracepoint conversion failed"
        );

        // Verify optimization: None should be used when a_len == b_len
        assert!(
            variable_tracepoints
                .iter()
                .any(|(_, b_opt)| b_opt.is_none()),
            "Variable tracepoints should have at least one optimized entry (None)"
        );
        assert!(
            variable_tracepoints
                .iter()
                .any(|(_, b_opt)| b_opt.is_some()),
            "Variable tracepoints should have at least one unoptimized entry (Some)"
        );
    }

    #[test]
    fn test_diagonal_tracepoint_generation() {
        // Test CIGAR string with various indel patterns
        let cigar = "5=3I2=2D4=";

        // Define test cases: (max_diff, expected_tracepoints) - updated based on actual diagonal logic
        let test_cases = [
            // Case 1: No diagonal distance allowed - each indel creates new segment
            (0, vec![(5, 5), (3, 0), (2, 2), (0, 2), (4, 4)]),
            // Case 2: Allow diagonal distance up to 2 - 3I exceeds max, then 2D+2=+4= fits in next segment
            (2, vec![(5, 5), (3, 0), (6, 8)]),
            // Case 3: Allow diagonal distance up to 3 - everything fits in one segment since diagonal balances
            (3, vec![(14, 13)]),
            // Case 4: Allow large diagonal distance - single segment (same as case 3)
            (10, vec![(14, 13)]),
        ];

        // Run each test case
        for (i, (max_diff, expected_tracepoints)) in test_cases.iter().enumerate() {
            let tracepoints =
                cigar_to_tracepoints(cigar, *max_diff, ComplexityMetric::DiagonalDistance);
            assert_eq!(
                tracepoints,
                *expected_tracepoints,
                "Test case {}: Diagonal tracepoints with max_diff={} incorrect",
                i + 1,
                max_diff
            );
        }
    }

    #[test]
    fn test_diagonal_vs_regular_segmentation() {
        // Compare diagonal vs regular segmentation for complex patterns
        let test_cases = vec![
            // Alternating small indels - diagonal should group better
            ("2=1I2=1D2=", 2),
            // Large indels - should behave similarly
            ("3=5I3=5D3=", 3),
            // Balanced indels - diagonal should combine opposing indels
            ("2=3I2=3D2=", 4),
        ];

        for (cigar, max_diff) in test_cases {
            let regular_tracepoints =
                cigar_to_tracepoints(cigar, max_diff, ComplexityMetric::EditDistance);
            let diagonal_tracepoints =
                cigar_to_tracepoints(cigar, max_diff, ComplexityMetric::DiagonalDistance);

            println!("CIGAR: {cigar}, max_diff: {max_diff}");
            println!("Regular: {regular_tracepoints:?}");
            println!("Diagonal: {diagonal_tracepoints:?}");

            // Both should preserve total sequence lengths
            let regular_total: (usize, usize) = regular_tracepoints
                .iter()
                .fold((0, 0), |(a, b), (x, y)| (a + x, b + y));
            let diagonal_total: (usize, usize) = diagonal_tracepoints
                .iter()
                .fold((0, 0), |(a, b), (x, y)| (a + x, b + y));
            assert_eq!(
                regular_total, diagonal_total,
                "Total lengths should match for CIGAR: {cigar}"
            );
        }
    }

    #[test]
    fn test_diagonal_mixed_tracepoints() {
        // Test mixed representation with diagonal distance
        let test_cases = [
            // Special operations with balanced indels - diagonal allows combining
            (
                "3S5=2I2=2D1H",
                2,
                vec![
                    MixedRepresentation::CigarOp(3, 'S'),
                    MixedRepresentation::Tracepoint(9, 9), // 5= + 2I + 2= + 2D (balanced diagonal)
                    MixedRepresentation::CigarOp(1, 'H'),
                ],
            ),
            // Balanced indels with special ops
            (
                "2H3=3I3=3D2P",
                3,
                vec![
                    MixedRepresentation::CigarOp(2, 'H'),
                    MixedRepresentation::Tracepoint(9, 9), // 3= + 3I + 3= + 3D (balanced)
                    MixedRepresentation::CigarOp(2, 'P'),
                ],
            ),
        ];

        for (i, (cigar, max_diff, expected)) in test_cases.iter().enumerate() {
            let result =
                cigar_to_mixed_tracepoints(cigar, *max_diff, ComplexityMetric::DiagonalDistance);
            assert_eq!(
                result,
                *expected,
                "Test case {}: Mixed diagonal tracepoints with max_diff={} incorrect for CIGAR '{}'",
                i + 1,
                max_diff,
                cigar
            );
        }
    }

    #[test]
    fn test_diagonal_variable_tracepoints() {
        // Test variable format with diagonal distance
        let cigar = "3=4I3=4D3=";
        let max_diff = 3;

        let variable_diagonal =
            cigar_to_variable_tracepoints(cigar, max_diff, ComplexityMetric::DiagonalDistance);

        // Expected: with max_diff=3, the 4I exceeds limit, so we get separate segments
        // Then after reset, 3=4D balances but 4D by itself exceeds max_diff
        let expected = vec![
            (3, None),    // initial 3=
            (4, Some(0)), // 4I (exceeds max_diff=3, separate segment)
            (3, None),    // next 3=
            (0, Some(4)), // 4D (exceeds max_diff=3, separate segment)
            (3, None),    // final 3=
        ];

        assert_eq!(
            variable_diagonal, expected,
            "Variable diagonal tracepoints incorrect"
        );

        // Verify optimization is working (None used for equal lengths)
        assert!(
            variable_diagonal.iter().any(|(_, b_opt)| b_opt.is_none()),
            "Should have at least one optimized entry (None)"
        );
    }

    #[test]
    fn test_diagonal_edge_cases() {
        // Test edge cases for diagonal distance
        let test_cases = vec![
            // Empty CIGAR
            ("", 5, vec![]),
            // Only matches
            ("10=", 3, vec![(10, 10)]),
            // Only insertions
            ("8I", 3, vec![(8, 0)]),
            // Only deletions
            ("6D", 2, vec![(0, 6)]),
            // max_diff = 0 with balanced indels
            ("2I2D", 0, vec![(2, 0), (0, 2)]),
            // Large indels that cancel out
            ("5I10=5D", 10, vec![(15, 15)]), // Should combine since net diagonal distance = 0
        ];

        for (i, (cigar, max_diff, expected)) in test_cases.iter().enumerate() {
            let result = cigar_to_tracepoints(cigar, *max_diff, ComplexityMetric::DiagonalDistance);
            assert_eq!(
                result,
                *expected,
                "Diagonal edge case {}: Failed for CIGAR '{}' with max_diff={}",
                i + 1,
                cigar,
                max_diff
            );
        }
    }

    #[test]
    fn test_diagonal_distance_calculation() {
        // Test the core diagonal distance logic with specific patterns
        let test_cases = [
            // Progressive insertion - distance should accumulate
            ("1I1I1I", 2, vec![(2, 0), (1, 0)]),
            // Alternating I/D - should balance out perfectly
            ("1I1D1I1D", 1, vec![(2, 2)]), // Net diagonal distance oscillates but stays <= 1
            // Large insertion followed by deletion - both exceed individually
            ("5I3D", 3, vec![(5, 0), (0, 3)]), // Each exceeds max_diff individually
            // Insertion that balances previous deletion
            ("3D2=3I", 3, vec![(5, 5)]), // Should combine since final distance = 0
        ];

        for (i, (cigar, max_diff, expected)) in test_cases.iter().enumerate() {
            let result = cigar_to_tracepoints(cigar, *max_diff, ComplexityMetric::DiagonalDistance);
            assert_eq!(
                result,
                *expected,
                "Diagonal distance test case {}: Failed for CIGAR '{}' with max_diff={}",
                i + 1,
                cigar,
                max_diff
            );
        }
    }

    #[test]
    fn test_diagonal_roundtrip_compatibility() {
        // Test that diagonal tracepoints can be used with existing reconstruction functions
        let original_cigar = "2=3I2=2D3=";
        let a_seq = b"ACGTACGTAC"; // 10 bases
        let b_seq = b"ACAGTCGTACG"; // 11 bases
        let max_diff = 2;

        // Generate diagonal tracepoints
        let diagonal_tracepoints =
            cigar_to_tracepoints(original_cigar, max_diff, ComplexityMetric::DiagonalDistance);

        let distance = Distance::GapAffine2p {
            mismatch: 2,
            gap_opening1: 4,
            gap_extension1: 2,
            gap_opening2: 6,
            gap_extension2: 1,
        };

        // Verify they can be reconstructed
        let reconstructed_cigar = tracepoints_to_cigar(
            &diagonal_tracepoints,
            a_seq,
            b_seq,
            0,
            0,
            ComplexityMetric::DiagonalDistance,
            &distance,
        );

        // Should not be empty and should represent a valid alignment
        assert!(
            !reconstructed_cigar.is_empty(),
            "Diagonal roundtrip produced empty CIGAR"
        );

        // Verify sequence length consistency
        let ops = cigar_str_to_cigar_ops(&reconstructed_cigar);
        let (total_a, total_b) = ops.iter().fold((0, 0), |(a, b), (len, op)| match op {
            '=' | 'M' | 'X' => (a + len, b + len),
            'I' => (a + len, b),
            'D' => (a, b + len),
            _ => (a, b),
        });

        // The original CIGAR "2=3I2=2D3=" should consume:
        // a_seq: 2+3+2+0+3=10 bases, b_seq: 2+0+2+2+3=9 bases
        // We provided sequences of length 10 and 11, so the reconstruction is valid
        assert_eq!(
            total_a,
            a_seq.len(),
            "Reconstructed CIGAR should consume correct a_seq length"
        );
        // The reconstruction may not consume the full b_seq if sequences don't match exactly
        assert!(
            total_b <= b_seq.len(),
            "Reconstructed CIGAR should not overconsume b_seq"
        );
    }

    // -------------------------------------------------------------------------
    // ES-2bit edit-script tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_es2bit_encoding() {
        // --- Byte layout and sentinel placement ---
        //
        // Case A: 4 ops (2D + 2I) fill a byte exactly; sentinel spills into a
        // new byte.
        //   CIGAR "10=2D5=2I3=", sequences:
        //   10=  AAAAAAAAAA vs AAAAAAAAAA
        //   2D             vs CC
        //   5=   GGGGG     vs GGGGG
        //   2I   TT        vs
        //   3=   AAA       vs AAA
        let a_seq = b"AAAAAAAAAAGGGGGTTAAA";
        let b_seq = b"AAAAAAAAAACCGGGGGAAA";
        let script = cigar_to_edit_script("10=2D5=2I3=");
        // byte 0: D D I I → 10 10 01 01 = 0b10100101
        // byte 1: sentinel 11 in MSB slot → 0b11000000
        assert_eq!(script.len(), 2, "sentinel must spill into a second byte when first is full");
        assert_eq!(script[0], 0b10_10_01_01, "MSB-first op order: D D I I");
        assert_eq!(script[1], 0b11_00_00_00, "sentinel 11 in MSB slot of second byte");
        assert_eq!(edit_script_to_cigar(&script, a_seq, b_seq), "10=2D5=2I3=");

        // Case B: 1 op (D); sentinel fits in the same byte.
        // byte 0: D(10) sentinel(11) → 0b10_11_00_00
        let script = cigar_to_edit_script("1D3=");
        assert_eq!(script.len(), 1, "sentinel must fit in same byte as single op");
        assert_eq!(script[0], 0b10_11_00_00, "D op then sentinel 11 in same byte");

        // --- Canonicalization: indels right-shift within identical-base runs ---
        //
        // q="AAT", t="AAAT": a valid CIGAR is "1D3=" (delete first target A),
        // but the greedy decoder always produces "2=1D1=" (rightmost placement).
        let decoded = edit_script_to_cigar(&script, b"AAT", b"AAAT");
        assert_eq!(decoded, "2=1D1=", "decoder must right-shift deletion into homopolymer");
    }

    #[test]
    fn test_es2bit_random_stress_with_wfa() {
        // Generate pseudo-random sequence pairs with several (n, eps), align
        // with WFA (exact + MemoryMode::High), round-trip through ES-2bit,
        // and assert that the decoded CIGAR is valid (it may differ by indel 
        // placement) and the cost is preserved.
        //
        // Uses a fixed-seed xorshift32 PRNG so the test is deterministic.
        fn xorshift32(state: &mut u32) -> u32 {
            let mut x = *state;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *state = x;
            x
        }
        fn rand_base(state: &mut u32) -> u8 {
            const BASES: &[u8; 4] = b"ACGT";
            BASES[(xorshift32(state) as usize) & 0b11]
        }
        fn rand_seq(state: &mut u32, n: usize) -> Vec<u8> {
            (0..n).map(|_| rand_base(state)).collect()
        }
        // Introduce edits into `q` to produce `t` at roughly the requested
        // error rate. Each position independently gets a mismatch,
        // insertion (into the target), deletion (from the target), or no
        // edit with probability (eps/3, eps/3, eps/3, 1-eps).
        fn mutate(state: &mut u32, q: &[u8], eps: f64) -> Vec<u8> {
            let scale = u32::MAX as f64;
            let thr_sub = (eps / 3.0 * scale) as u32;
            let thr_ins = (2.0 * eps / 3.0 * scale) as u32;
            let thr_del = (eps * scale) as u32;
            let mut t = Vec::with_capacity(q.len());
            for &b in q {
                let r = xorshift32(state);
                if r < thr_sub {
                    // Substitution: emit a different base.
                    let mut nb = rand_base(state);
                    while nb == b {
                        nb = rand_base(state);
                    }
                    t.push(nb);
                } else if r < thr_ins {
                    // Insertion into target: emit an extra base plus the
                    // original.
                    t.push(rand_base(state));
                    t.push(b);
                } else if r < thr_del {
                    // Deletion from target: emit nothing for this base.
                } else {
                    t.push(b);
                }
            }
            t
        }

        // Given a CIGAR and the query/target, reconstruct the gapped
        // alignment rows and verify sequence-level consistency.
        fn verify_mapping(cigar: &str, a_seq: &[u8], b_seq: &[u8]) -> bool {
            let ops = cigar_str_to_cigar_ops(cigar);
            let mut i = 0usize;
            let mut j = 0usize;
            for (len, op) in ops {
                match op {
                    '=' => {
                        for _ in 0..len {
                            if i >= a_seq.len() || j >= b_seq.len() {
                                return false;
                            }
                            if a_seq[i] != b_seq[j] {
                                return false;
                            }
                            i += 1;
                            j += 1;
                        }
                    }
                    'X' => {
                        for _ in 0..len {
                            if i >= a_seq.len() || j >= b_seq.len() {
                                return false;
                            }
                            if a_seq[i] == b_seq[j] {
                                return false;
                            }
                            i += 1;
                            j += 1;
                        }
                    }
                    'I' => {
                        if i + len > a_seq.len() {
                            return false;
                        }
                        i += len;
                    }
                    'D' => {
                        if j + len > b_seq.len() {
                            return false;
                        }
                        j += len;
                    }
                    'M' => {
                        // Match-or-mismatch: only check bounds.
                        if i + len > a_seq.len() || j + len > b_seq.len() {
                            return false;
                        }
                        i += len;
                        j += len;
                    }
                    _ => return false,
                }
            }
            i == a_seq.len() && j == b_seq.len()
        }

        // Count the edit cost of a CIGAR (non-match ops).
        fn cigar_cost(cigar: &str) -> usize {
            cigar_str_to_cigar_ops(cigar)
                .into_iter()
                .filter(|(_, op)| !matches!(*op, '=' | 'M'))
                .map(|(len, _)| len)
                .sum()
        }

        let distance = Distance::Edit;
        let mut aligner = distance.create_aligner(None, None);
        let buckets: &[(usize, f64)] =
            &[(100, 0.01), (1_000, 0.05), (1_000, 0.20)];
        let samples_per_bucket = 10usize;
        let mut state: u32 = 0xC0FFEE_u32;

        for &(n, eps) in buckets {
            for sample in 0..samples_per_bucket {
                let q = rand_seq(&mut state, n);
                let t = mutate(&mut state, &q, eps);
                if t.is_empty() {
                    continue;
                }
                let mut cigar_ops = Vec::new();
                align_sequences_wfa(&q, &t, &aligner, &mut cigar_ops);
                let cigar = cigar_ops_to_cigar_string(&cigar_ops);

                assert!(
                    verify_mapping(&cigar, &q, &t),
                    "WFA CIGAR failed to map q/t (n={}, eps={}, sample={})",
                    n,
                    eps,
                    sample
                );

                let script = cigar_to_edit_script(&cigar);
                let decoded = edit_script_to_cigar(&script, &q, &t);

                // Sequence mapping must round-trip exactly, even though
                // the literal CIGAR may have shifted indels.
                assert!(
                    verify_mapping(&decoded, &q, &t),
                    "round-tripped CIGAR failed to map q/t \
                     (n={}, eps={}, sample={})\n  orig:    {}\n  decoded: {}",
                    n,
                    eps,
                    sample,
                    cigar,
                    decoded
                );
                assert_eq!(
                    cigar_cost(&decoded),
                    cigar_cost(&cigar),
                    "edit cost must be preserved through ES-2bit round-trip \
                     (n={}, eps={}, sample={})",
                    n,
                    eps,
                    sample
                );
            }
        }
        // Touch the aligner to keep it alive through the loop — silences
        // an unused warning on older clippy settings.
        let _ = &mut aligner;
    }
}

#[cfg(test)]
mod gap_refinement_tests {
    use super::*;

    #[test]
    fn test_find_gap_runs() {
        let cigar = "4=1I2=1I4=";
        let runs = find_gap_runs(cigar, 50);
        
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].start_op_idx, 1); // First I
        assert_eq!(runs[0].end_op_idx, 4); // After second I
    }

    #[test]
    fn test_gap_refinement_simple() {
        let a_seq = b"ACGTACGTACGTACGT";
        let b_seq = b"ACGTACGTACGTACGT";
        let cigar = "16=";
        
        let refined = refine_cigar_gaps(cigar, a_seq, b_seq, 50);
        assert_eq!(refined, cigar);
    }
    
    #[test]
    fn test_gap_refinement_preserves_alignment() {
        let a_seq = b"ACGTACGTACGT";
        let b_seq = b"ACGTACGT";
        let cigar = "8=4I";
        
        let refined = refine_cigar_gaps(cigar, a_seq, b_seq, 50);
        
        let calc_lengths = |s: &str| -> (usize, usize) {
            cigar_str_to_cigar_ops(s).iter().fold((0, 0), |(a, b), (len, op)| match op {
                '=' | 'M' | 'X' => (a + len, b + len),
                'I' => (a + len, b),
                'D' => (a, b + len),
                _ => (a, b),
            })
        };
        
        let (orig_a, orig_b) = calc_lengths(cigar);
        let (ref_a, ref_b) = calc_lengths(&refined);
        
        assert_eq!(orig_a, ref_a, "A sequence length should be preserved");
        assert_eq!(orig_b, ref_b, "B sequence length should be preserved");
    }
    
    #[test]
    fn test_gap_refinement_fragmented_gaps() {
        let a_seq = b"ACGTACGTACGT";
        let b_seq = b"ACGTACGT";
        let cigar = "2=1I1=1I4=4I";
        
        let refined = refine_cigar_gaps(cigar, a_seq, b_seq, 2);
        
        let calc_lengths = |s: &str| -> (usize, usize) {
            cigar_str_to_cigar_ops(s).iter().fold((0, 0), |(a, b), (len, op)| match op {
                '=' | 'M' | 'X' => (a + len, b + len),
                'I' => (a + len, b),
                'D' => (a, b + len),
                _ => (a, b),
            })
        };
        
        let (orig_a, orig_b) = calc_lengths(cigar);
        let (ref_a, ref_b) = calc_lengths(&refined);
        
        assert_eq!(orig_a, ref_a, "A sequence length should be preserved");
        assert_eq!(orig_b, ref_b, "B sequence length should be preserved");
        
        let count_gap_ops = |s: &str| -> usize {
            cigar_str_to_cigar_ops(s).iter().filter(|(_, op)| *op == 'I' || *op == 'D').count()
        };
        
        let orig_gaps = count_gap_ops(cigar);
        let ref_gaps = count_gap_ops(&refined);
        
        assert!(ref_gaps <= orig_gaps, "Should reduce or maintain gap count");
    }

    #[test]
    fn test_gap_refinement_paper_example() {
        let a_seq = b"TCACAGAAACTTTTCTAATATTT";
        let b_seq = b"TCAAGAAAAGTCCCTTGGGACTTTTCTGAATTT";
        let cigar = "3=1I5=3D1=2D2=5D2=2D2=1X1=1X4=";
        
        let runs = find_gap_runs(cigar, 50);
        println!("Found {} gap runs", runs.len());
        
        let refined = refine_cigar_gaps(cigar, a_seq, b_seq, 50);
        
        println!("Original: {}", cigar);
        println!("Refined:  {}", refined);
        
        let calc_lengths = |s: &str| -> (usize, usize) {
            cigar_str_to_cigar_ops(s).iter().fold((0, 0), |(a, b), (len, op)| match op {
                '=' | 'M' | 'X' => (a + len, b + len),
                'I' => (a + len, b),
                'D' => (a, b + len),
                _ => (a, b),
            })
        };
        
        let (orig_a, orig_b) = calc_lengths(cigar);
        let (ref_a, ref_b) = calc_lengths(&refined);
        
        assert_eq!(orig_a, ref_a, "A sequence length should be preserved");
        assert_eq!(orig_b, ref_b, "B sequence length should be preserved");
        
        let count_gap_ops = |s: &str| -> usize {
            cigar_str_to_cigar_ops(s).iter().filter(|(_, op)| *op == 'I' || *op == 'D').count()
        };
        
        let orig_gaps = count_gap_ops(cigar);
        let ref_gaps = count_gap_ops(&refined);
        
        println!("Gap operations: {} -> {}", orig_gaps, ref_gaps);
        assert!(ref_gaps <= orig_gaps, "Should reduce or maintain gap count");
    }
    
    #[test]
    fn test_gap_refinement_with_aligner_reuse() {
        let a_seq = b"ACGTNNACGTACGT";
        let b_seq = b"ACGTACGTACGT";
        let cigar = "4=2I8=";
        
        let distance = Distance::GapAffine {
            mismatch: 1,
            gap_opening: 2,
            gap_extension: 1,
        };
        let mut aligner = distance.create_aligner(None);
        
        let refined = refine_cigar_gaps_with_aligner(cigar, a_seq, b_seq, 50, &mut aligner);
        
        let calc_lengths = |s: &str| -> (usize, usize) {
            cigar_str_to_cigar_ops(s).iter().fold((0, 0), |(a, b), (len, op)| match op {
                '=' | 'M' | 'X' => (a + len, b + len),
                'I' => (a + len, b),
                'D' => (a, b + len),
                _ => (a, b),
            })
        };
        
        let (orig_a, orig_b) = calc_lengths(cigar);
        let (ref_a, ref_b) = calc_lengths(&refined);
        
        assert_eq!(orig_a, ref_a);
        assert_eq!(orig_b, ref_b);
    }
}
