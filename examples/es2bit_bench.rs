//! Small benchmark comparing ES-2bit against FL-TP / EB-TP / DB-TP.
//!
//! Usage:
//!     cargo run --release --example es2bit_bench -- <paf_file> <fasta_file> [max_records]
//!
//! Both inputs must be uncompressed.
//!
//! For each PAF record: encode the CIGAR four ways, measure storage
//! (theoretical bits, disk bytes, gzip bytes) and encode/decode time,
//! round-trip-check, and write:
//!   - per-record TSV at `<paf>.es2bit.tsv`;
//!   - aggregated summary to stdout.
//!
//! Theoretical bits per edit (from the paper's Table 1):
//!   ES-2bit:  e × 2                   → always 2.0 bits/edit
//!   FL-TP:    num_tps × ⌈log₂(l + e)⌉ → bits to encode (edit_distance, target_advance) per TP
//!   EB-TP:    num_tps × ⌈log₂(n × e)⌉ → bits for (query_advance, target_advance) per TP
//!   DB-TP:    num_tps × ⌈log₂(n × b)⌉ → same, but range depends on band b, not e

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process;
use std::time::Instant;

use flate2::write::GzEncoder;
use flate2::Compression;
use tracepoints::{
    cigar_to_edit_script, cigar_to_tracepoints, edit_script_to_cigar,
    tracepoints_to_cigar, tracepoints_to_cigar_fastga, ComplexityMetric, Distance,
};

const FLTP_SPACING: u32 = 100;
const EBTP_DELTA: u32 = 32;
const DBTP_BAND: u32 = 32;

const CIGAR: usize = 0;
const ES2BIT: usize = 1;
const FLTP: usize = 2;
const EBTP: usize = 3;
const DBTP: usize = 4;
const N_METHODS: usize = 5;
const METHOD_NAMES: [&str; N_METHODS] = ["cigar", "es2bit", "fltp", "ebtp", "dbtp"];

// =============================================================================
// FASTA / PAF helpers
// =============================================================================

fn load_fasta(path: &Path) -> HashMap<String, Vec<u8>> {
    let reader = BufReader::new(
        File::open(path).unwrap_or_else(|e| panic!("cannot open {:?}: {}", path, e)),
    );
    let mut seqs: HashMap<String, Vec<u8>> = HashMap::new();
    let mut cur_name: Option<String> = None;
    let mut cur_seq: Vec<u8> = Vec::new();
    for line in reader.lines() {
        let line = line.expect("FASTA read error");
        if line.is_empty() || line.starts_with(';') { continue; }
        if let Some(hdr) = line.strip_prefix('>') {
            if let Some(name) = cur_name.take() {
                seqs.insert(name, std::mem::take(&mut cur_seq));
            }
            cur_name = Some(
                hdr.split_ascii_whitespace().next().expect("empty FASTA header").to_string(),
            );
        } else {
            cur_seq.extend(line.bytes().map(|b| b.to_ascii_uppercase()));
        }
    }
    if let Some(name) = cur_name { seqs.insert(name, cur_seq); }
    seqs
}

fn revcomp(seq: &[u8]) -> Vec<u8> {
    seq.iter().rev().map(|&b| match b {
        b'A' => b'T', b'C' => b'G', b'G' => b'C', b'T' => b'A', _ => b'N',
    }).collect()
}

struct PafRecord<'a> {
    qname: &'a str, qstart: usize, qend: usize, strand: char,
    tname: &'a str, tstart: usize, tend: usize, cigar: &'a str,
}

fn parse_paf_line(line: &str) -> Option<PafRecord<'_>> {
    let mut it = line.split('\t');
    let qname = it.next()?;
    let _qlen: usize = it.next()?.parse().ok()?;
    let qstart: usize = it.next()?.parse().ok()?;
    let qend: usize = it.next()?.parse().ok()?;
    let strand = it.next()?.chars().next()?;
    let tname = it.next()?;
    let _tlen: usize = it.next()?.parse().ok()?;
    let tstart: usize = it.next()?.parse().ok()?;
    let tend: usize = it.next()?.parse().ok()?;
    let _matches: usize = it.next()?.parse().ok()?;
    let _aln_len: usize = it.next()?.parse().ok()?;
    let _mapq: usize = it.next()?.parse().ok()?;
    let cigar = it.find_map(|tag| tag.strip_prefix("cg:Z:"))?;
    Some(PafRecord { qname, qstart, qend, strand, tname, tstart, tend, cigar })
}

// =============================================================================
// Small helpers
// =============================================================================

fn write_leb128(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 { out.push(((v as u8) & 0x7F) | 0x80); v >>= 7; }
    out.push(v as u8);
}

fn encode_tp_record(out: &mut Vec<u8>, tps: &[(usize, usize)]) -> usize {
    let start = out.len();
    write_leb128(out, tps.len() as u64);
    for &(a, b) in tps { write_leb128(out, a as u64); write_leb128(out, b as u64); }
    out.len() - start
}

fn gzip_len(data: &[u8]) -> usize {
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data).expect("gzip write failed");
    enc.finish().expect("gzip finalize failed").len()
}

fn ceil_log2(x: usize) -> usize {
    if x <= 1 { 1 } else { (usize::BITS - (x - 1).leading_zeros()) as usize }
}

fn cigar_edit_count(cigar: &str) -> usize {
    let mut total = 0usize;
    let mut num = String::new();
    for ch in cigar.chars() {
        if ch.is_ascii_digit() { num.push(ch); }
        else {
            let n: usize = num.parse().unwrap_or(0);
            num.clear();
            if !matches!(ch, '=' | 'M') { total += n; }
        }
    }
    total
}

fn verify_cigar_mapping(cigar: &str, a_seq: &[u8], b_seq: &[u8]) -> bool {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut num = String::new();
    for ch in cigar.chars() {
        if ch.is_ascii_digit() { num.push(ch); continue; }
        let n: usize = num.parse().unwrap_or(0);
        num.clear();
        match ch {
            '=' => { for _ in 0..n { if i >= a_seq.len() || j >= b_seq.len() || a_seq[i] != b_seq[j] { return false; } i += 1; j += 1; } }
            'X' => { for _ in 0..n { if i >= a_seq.len() || j >= b_seq.len() || a_seq[i] == b_seq[j] { return false; } i += 1; j += 1; } }
            'I' => { if i + n > a_seq.len() { return false; } i += n; }
            'D' => { if j + n > b_seq.len() { return false; } j += n; }
            'M' => { if i + n > a_seq.len() || j + n > b_seq.len() { return false; } i += n; j += n; }
            _ => return false,
        }
    }
    i == a_seq.len() && j == b_seq.len()
}

/// Simple FL-TP encoder: emit (edits, target_advance) every `l` query bases.
fn cigar_to_fltp_simple(cigar: &str, l: usize) -> Vec<(usize, usize)> {
    let mut tps = Vec::new();
    let mut a_pos = 0usize;
    let mut edits = 0usize;
    let mut b_adv = 0usize;
    let mut next = l;
    let mut num_buf = String::new();
    for ch in cigar.chars() {
        if ch.is_ascii_digit() { num_buf.push(ch); continue; }
        let op_len: usize = num_buf.parse().unwrap_or(0);
        num_buf.clear();
        match ch {
            '=' | 'M' | 'X' => {
                let is_edit = ch == 'X';
                let mut rem = op_len;
                while rem > 0 {
                    let c = rem.min(next - a_pos);
                    a_pos += c; b_adv += c;
                    if is_edit { edits += c; }
                    rem -= c;
                    if a_pos == next { tps.push((edits, b_adv)); edits = 0; b_adv = 0; next += l; }
                }
            }
            'I' => {
                let mut rem = op_len;
                while rem > 0 {
                    let c = rem.min(next - a_pos);
                    a_pos += c; edits += c; rem -= c;
                    if a_pos == next { tps.push((edits, b_adv)); edits = 0; b_adv = 0; next += l; }
                }
            }
            'D' => { b_adv += op_len; edits += op_len; }
            _ => {}
        }
    }
    if a_pos > next - l || edits > 0 || b_adv > 0 { tps.push((edits, b_adv)); }
    tps
}

// =============================================================================
// Main
// =============================================================================

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("usage: es2bit_bench <paf_file> <fasta_file> [max_records]");
        process::exit(2);
    }
    let paf_path = Path::new(&args[1]);
    let fasta_path = Path::new(&args[2]);
    let max_records: usize = if args.len() == 4 {
        args[3].parse().unwrap_or_else(|_| panic!("bad max_records: {:?}", args[3]))
    } else { usize::MAX };

    eprintln!("[es2bit_bench] loading FASTA: {:?}", fasta_path);
    let fasta = load_fasta(fasta_path);
    eprintln!("[es2bit_bench] loaded {} sequences", fasta.len());

    let paf_reader = BufReader::new(
        File::open(paf_path).unwrap_or_else(|e| panic!("cannot open PAF {:?}: {}", paf_path, e)),
    );
    let tsv_path = paf_path.with_extension("es2bit.tsv");
    let mut tsv = BufWriter::new(
        File::create(&tsv_path).unwrap_or_else(|e| panic!("cannot create {:?}: {}", tsv_path, e)),
    );
    writeln!(tsv, "qname\ttname\tstrand\tn\tm\te\tmethod\ttheo_bits\tdisk_bytes\tnum_tps\tencode_ns\tdecode_ns\tcorrect\tidentical").unwrap();

    // Accumulators: [cigar, es2bit, fltp, ebtp, dbtp]
    let mut records = 0usize;
    let mut sum_n = 0u64;
    let mut sum_edits = 0u64;
    let mut total_theo_bits = [0u64; N_METHODS];
    let mut total_disk = [0u64; N_METHODS];
    let mut total_enc_ns = [0u64; N_METHODS];
    let mut total_dec_ns = [0u64; N_METHODS];
    let mut cigar_identical = [0u64; N_METHODS]; // decoded CIGAR == input CIGAR
    let mut gz_buf: [Vec<u8>; N_METHODS] = Default::default();

    let mut line_no = 0usize;
    let mut skipped = 0usize;

    for line in paf_reader.lines() {
        if records >= max_records { break; }
        line_no += 1;
        let line = line.expect("PAF read error");
        if line.is_empty() || line.starts_with('#') { continue; }
        let rec = match parse_paf_line(&line) {
            Some(r) => r,
            None => { skipped += 1; continue; }
        };

        let qseq = fasta.get(rec.qname).unwrap_or_else(|| panic!("FASTA missing {}", rec.qname));
        let tseq = fasta.get(rec.tname).unwrap_or_else(|| panic!("FASTA missing {}", rec.tname));
        let a_seq: &[u8] = &qseq[rec.qstart..rec.qend];
        let b_vec = if rec.strand == '-' {
            revcomp(&tseq[rec.tstart..rec.tend])
        } else {
            tseq[rec.tstart..rec.tend].to_vec()
        };
        let b_seq: &[u8] = &b_vec;

        let n = a_seq.len();
        let m = b_seq.len();
        let e = cigar_edit_count(rec.cigar);

        assert!(verify_cigar_mapping(rec.cigar, a_seq, b_seq),
            "[line {}] source CIGAR mismatch ({}→{} n={} m={})", line_no, rec.qname, rec.tname, n, m);

        records += 1;
        sum_n += n as u64;
        sum_edits += e as u64;

        let mut emit = |idx: usize, theo: u64, disk: u64, ntps: usize, enc_ns: u64, dec_ns: u64, ok: bool, identical: bool| {
            total_theo_bits[idx] += theo;
            total_disk[idx] += disk;
            total_enc_ns[idx] += enc_ns;
            total_dec_ns[idx] += dec_ns;
            if identical { cigar_identical[idx] += 1; }
            writeln!(tsv, "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                rec.qname, rec.tname, rec.strand, n, m, e,
                METHOD_NAMES[idx], theo, disk, ntps, enc_ns, dec_ns, ok, identical).unwrap();
        };

        // --- CIGAR (raw baseline, no encoding/decoding) ---
        let cigar_bytes = rec.cigar.len() as u64;
        let cigar_bits = cigar_bytes * 8;
        gz_buf[CIGAR].extend_from_slice(rec.cigar.as_bytes());
        // No encode/decode for CIGAR — it IS the input. identical = N/A, report true.
        emit(CIGAR, cigar_bits, cigar_bytes, 0, 0, 0, true, true);

        // Helper: encode → decode → verify mapping → check identity.
        let mut check = |idx: usize, dec: &str, enc_ns: u64, dec_ns: u64, theo: u64, disk: u64, ntps: usize| {
            let ok = verify_cigar_mapping(dec, a_seq, b_seq);
            assert!(ok, "{} round-trip failed at line {}", METHOD_NAMES[idx], line_no);
            let identical = dec == rec.cigar;
            emit(idx, theo, disk, ntps, enc_ns, dec_ns, ok, identical);
        };

        // --- ES-2bit ---
        let t0 = Instant::now();
        let es = cigar_to_edit_script(rec.cigar);
        let enc_ns = t0.elapsed().as_nanos() as u64;
        let disk = es.len() as u64;
        gz_buf[ES2BIT].extend_from_slice(&es);
        let t0 = Instant::now();
        let dec = edit_script_to_cigar(&es, a_seq, b_seq);
        let dec_ns = t0.elapsed().as_nanos() as u64;
        check(ES2BIT, &dec, enc_ns, dec_ns, (e as u64) * 2, disk, e);

        // --- FL-TP ---
        let t0 = Instant::now();
        let tps = cigar_to_fltp_simple(rec.cigar, FLTP_SPACING as usize);
        let enc_ns = t0.elapsed().as_nanos() as u64;
        let disk = encode_tp_record(&mut gz_buf[FLTP], &tps) as u64;
        let theo = (tps.len() as u64) * ceil_log2(FLTP_SPACING as usize + e) as u64;
        let t0 = Instant::now();
        let dec = tracepoints_to_cigar_fastga(&tps, FLTP_SPACING, a_seq, b_seq, 0, 0, false, false);
        let dec_ns = t0.elapsed().as_nanos() as u64;
        check(FLTP, &dec, enc_ns, dec_ns, theo, disk, tps.len());

        // --- EB-TP ---
        let t0 = Instant::now();
        let tps = cigar_to_tracepoints(rec.cigar, EBTP_DELTA, ComplexityMetric::EditDistance);
        let enc_ns = t0.elapsed().as_nanos() as u64;
        let disk = encode_tp_record(&mut gz_buf[EBTP], &tps) as u64;
        let theo = (tps.len() as u64) * ceil_log2(n.max(1) * e.max(1)) as u64;
        let t0 = Instant::now();
        let dec = tracepoints_to_cigar(&tps, a_seq, b_seq, 0, 0, ComplexityMetric::EditDistance, &Distance::Edit);
        let dec_ns = t0.elapsed().as_nanos() as u64;
        check(EBTP, &dec, enc_ns, dec_ns, theo, disk, tps.len());

        // --- DB-TP ---
        let t0 = Instant::now();
        let tps = cigar_to_tracepoints(rec.cigar, DBTP_BAND, ComplexityMetric::DiagonalDistance);
        let enc_ns = t0.elapsed().as_nanos() as u64;
        let disk = encode_tp_record(&mut gz_buf[DBTP], &tps) as u64;
        let theo = (tps.len() as u64) * ceil_log2(n.max(1) * DBTP_BAND as usize) as u64;
        let t0 = Instant::now();
        let dec = tracepoints_to_cigar(&tps, a_seq, b_seq, 0, 0, ComplexityMetric::DiagonalDistance, &Distance::Edit);
        let dec_ns = t0.elapsed().as_nanos() as u64;
        check(DBTP, &dec, enc_ns, dec_ns, theo, disk, tps.len());
    }

    tsv.flush().unwrap();
    if records == 0 { eprintln!("[es2bit_bench] no records"); process::exit(1); }
    if skipped > 0 { eprintln!("[es2bit_bench] skipped {} malformed lines", skipped); }

    // --- Summary ---
    let gz: [u64; N_METHODS] = std::array::from_fn(|i| gzip_len(&gz_buf[i]) as u64);
    let se = sum_edits as f64;
    let pe = |v: u64| -> f64 { if se == 0.0 { 0.0 } else { v as f64 / se } };

    // Header
    print!("file\trecords\tmean_n\tmean_e");
    for m in METHOD_NAMES { print!("\t{m}_theo_bits_per_e"); }
    for m in METHOD_NAMES { print!("\t{m}_disk_bits_per_e"); }
    for m in METHOD_NAMES { print!("\t{m}_disk_B_per_e"); }
    for m in METHOD_NAMES { print!("\t{m}_gz_B_per_e"); }
    for m in METHOD_NAMES { print!("\t{m}_enc_ns_per_e"); }
    for m in METHOD_NAMES { print!("\t{m}_dec_ns_per_e"); }
    for m in METHOD_NAMES { print!("\t{m}_total_disk_B"); }
    for m in METHOD_NAMES { print!("\t{m}_total_gz_B"); }
    for m in METHOD_NAMES { print!("\t{m}_total_enc_ms"); }
    for m in METHOD_NAMES { print!("\t{m}_total_dec_ms"); }
    for m in METHOD_NAMES { print!("\t{m}_cigar_identical"); }
    println!();

    // Values
    print!("{}\t{}\t{:.1}\t{:.1}", paf_path.display(), records,
        sum_n as f64 / records as f64, se / records as f64);
    for i in 0..N_METHODS { print!("\t{:.3}", pe(total_theo_bits[i])); }
    for i in 0..N_METHODS { print!("\t{:.3}", pe(total_disk[i] * 8)); }
    for i in 0..N_METHODS { print!("\t{:.4}", pe(total_disk[i])); }
    for i in 0..N_METHODS { print!("\t{:.4}", pe(gz[i])); }
    for i in 0..N_METHODS { print!("\t{:.1}", pe(total_enc_ns[i])); }
    for i in 0..N_METHODS { print!("\t{:.1}", pe(total_dec_ns[i])); }
    for i in 0..N_METHODS { print!("\t{}", total_disk[i]); }
    for i in 0..N_METHODS { print!("\t{}", gz[i]); }
    for i in 0..N_METHODS { print!("\t{:.1}", total_enc_ns[i] as f64 / 1e6); }
    for i in 0..N_METHODS { print!("\t{:.1}", total_dec_ns[i] as f64 / 1e6); }
    for i in 0..N_METHODS { print!("\t{}/{}", cigar_identical[i], records); }
    println!();

    eprintln!("[es2bit_bench] per-record TSV written to {:?}", tsv_path);
}
