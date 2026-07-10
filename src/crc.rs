use crate::linalg::{berlekamp_massey, mat_mul_gf2, BitMatrix};
use crate::proto::{extract_varying, ProtocolStructure};
use crate::{from_txt, positionwise_entropy, BitkitError, Bitstream};
use rayon::prelude::*;

/// RankResult - result of the windowed rank analysis.
/// `rank` : the row rank of the windowed matrix
/// `width`: the width of the windowed matrix (going from index=0 to index=width - 1)
/// `diff` : the difference between the width of the window and the rank of the matrix. diff=0 means
///          full rank, diff>0 signals probable CRC bit(s) entering the window
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct RankResult {
    rank: usize,
    width: usize,
    diff: usize,
}
/// Find the CRC in the Bitstreams, if present, and the location of the CRC bits in the protocol.
/// Assumptions: Bitstreams are aligned correctly, each one is the same length, **the bitstreams are
/// noiseless** (will implement improvements to loosen that requirement later) and there are enough
/// Bitstreams to reveal the CRC. That exact requirement is tricky to define precisely without
/// knowing how many data bits are in the stream, but if there are fewer samples than data bits in
/// the stream we won't be able to find CRC because we won't have enough degrees of freedom to
/// detect the drop in rank. We'll return an error if we happen to detect that the number of
/// samples is too low, but a lack of an error is not a guarantee that there are enough Bitstreams.
/// That said, if there are at least as many Bitstreams as there are varying bits in the protocol,
/// that should be enough (although it's better to have more for a safe cushion).
pub fn find_crc(bitstrs: &[Bitstream]) -> Result<BitMatrix, BitkitError> {
    let ps = ProtocolStructure::infer_structure(&positionwise_entropy(bitstrs));
    let varying_bitstrs: Vec<Bitstream> = bitstrs
        .iter()
        .map(|bs| extract_varying(bs, &ps).and_then(Bitstream::new))
        .collect::<Result<Vec<_>, _>>()?;
    find_crc_from_varying(varying_bitstrs)
}
/// Do the actual work to find the CRC. Expects a slice of Bitstreams composed of only the varying
/// bits from the protocol.
pub fn find_crc_from_varying(varying_bitstrs: Vec<Bitstream>) -> Result<BitMatrix, BitkitError> {
    // XORing to remove any affine element (ex if the CRC was XOR'd by a constant)
    let mut bitmat = BitMatrix::new(&varying_bitstrs).unwrap();
    for ii in 1..bitmat.num_rows() {
        for jj in 0..bitmat.num_cols() {
            bitmat[ii][jj] ^= bitmat[0][jj];
        }
    }
    // zero out this row - since it was xor'd with everything else it's no longer contributing to
    // the rowspace
    for jj in 0..bitmat.num_cols() {
        bitmat[0][jj] = 0;
    }
    let base_rank = bitmat.mat_rank();
    if base_rank <= varying_bitstrs.len() {
        let error_msg: String = format!(
            "Matrix rank {} is too low to detect CRC with linear algebra methods.\
                More bitstream samples needed",
            base_rank
        );
        return Err(BitkitError::MiscellaneousError(error_msg));
    }
    // For now, we're doing an exhaustive search, fully aware that this is dumb, but at least it's
    // threaded. We don't want to miss it if it's in weird place.
    let mut rank_drop: Vec<RankResult> = (1..=bitmat.num_cols())
        .into_par_iter()
        .map(|width| {
            let rank = bitmat.window(0, width).unwrap().mat_rank();
            RankResult {
                rank,
                width,
                diff: width - rank,
            }
        })
        .filter(|res| res.diff > 0)
        .collect();
    rank_drop.sort_by_key(|r| r.width);
    let mut prev = rank_drop[0];
    // Check for contiguous CRC bits
    for entry in &rank_drop[1..] {
        if entry.width != prev.width + 1 || entry.rank != prev.rank {
            return Err(BitkitError::MiscellaneousError(
                "Candidate CRC fields are NOT contiguous. Either something unexpected is going on\
                (weird data) or the CRC is interleaved or something. More investigation needed."
                    .to_string(),
            ));
        }
        prev = *entry;
    }
    // let ns = bitmat.clone().nullspace();
    // assert!(mat_mul_gf2(&bitmat, &ns)?.is_zero());

    Ok(bitmat.nullspace())
} // find_crc
pub fn test_all() {
    let bitstrs = from_txt("/home/emily/work_area/bitkit-rust/test_bits_interlaken.txt").unwrap();
    let res = find_crc(&bitstrs).unwrap();
    let k = res.num_rows() - res.num_cols();
    let null_vecs = res.row_window(k).transpose();
    println!("{res}\n-----\n{null_vecs}");
    for ii in 0..null_vecs.num_rows() {
        let polynomial = berlekamp_massey(&null_vecs[ii]);
        println!("{:?}", polynomial);
        // assert_eq!(polynomial, vec![1, 1, 0, 1]);  // 3-bit GSM
        assert_eq!(polynomial, vec![1, 1, 0, 0, 1]); // 3-bit GSM
    }
}
