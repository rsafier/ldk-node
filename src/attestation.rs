// This file is Copyright its original authors, visible in version control history.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option.

//! Read-only channel attestation export.
//!
//! Regulated Lightning custodians need point-in-time cryptographic proofs of
//! channel state for compliance reporting. [`Node::export_channel_attestation`]
//! surfaces the minimum data an external auditor needs to independently verify
//! (against on-chain-visible funding pubkeys) that the counterparty acknowledged
//! this exact channel state at snapshot time.
//!
//! The bundle is assembled from:
//! - [`ChannelMonitor::unsafe_get_latest_holder_commitment_txn`] — signed commit tx
//! - Witness parsing — counterparty ECDSA signature, both funding pubkeys
//! - [`ChannelMonitor::channel_keys_id`] + [`SignerProvider::derive_channel_signer`]
//!   — identify which of the two funding pubkeys is the holder's
//!
//! No secret material is exposed. The counterparty's signature is cryptographically
//! attributable to them regardless of who holds it (that's the entire point of
//! Lightning's commitment scheme).
//!
//! See the upstream RFC filed at `lightningdevkit/ldk-server`: "include
//! commitment-bundle data on channel-state events".

use std::sync::Arc;

use bitcoin::blockdata::transaction::Transaction;
use bitcoin::blockdata::witness::Witness;
use bitcoin::hashes::Hash;
use bitcoin::secp256k1::ecdsa::Signature;
use bitcoin::secp256k1::{PublicKey, Secp256k1};

use lightning::ln::types::ChannelId;
use lightning::sign::SignerProvider;

use crate::error::Error;
use crate::types::{ChainMonitor, ChannelManager};
use crate::KeysManager;
use crate::logger::Logger;

/// A cryptographically-verifiable snapshot of channel state suitable for external attestation.
///
/// The verifier re-computes the BIP-143 sighash over `holder_commitment_tx` using the
/// 2-of-2 funding redeem script reconstructed from `holder_funding_pubkey` +
/// `counterparty_funding_pubkey`, and verifies `counterparty_signature` against
/// `counterparty_funding_pubkey`. That binds the counterparty to this exact state.
#[derive(Clone, Debug)]
pub struct ChannelAttestation {
	/// ID of the channel this attestation covers.
	pub channel_id: ChannelId,
	/// Funding outpoint (the 2-of-2 multisig output on-chain).
	pub funding_outpoint: bitcoin::OutPoint,
	/// Channel capacity in satoshis.
	pub capacity_sats: u64,
	/// Counterparty node identity pubkey (not the funding pubkey).
	pub counterparty_node_id: PublicKey,
	/// Holder's funding pubkey — one side of the 2-of-2.
	pub holder_funding_pubkey: PublicKey,
	/// Counterparty's funding pubkey — the other side of the 2-of-2.
	pub counterparty_funding_pubkey: PublicKey,
	/// The unsigned commitment transaction the holder would broadcast on force-close
	/// (wire-format, no witnesses on the funding input).
	pub holder_commitment_tx: Transaction,
	/// Counterparty's ECDSA signature over `holder_commitment_tx` (BIP-143 sighash,
	/// SIGHASH_ALL, DER-encoded).
	pub counterparty_signature: Signature,
	/// Holder balance in msat at this commitment (approximation — see ldk-node
	/// `ChannelDetails::outbound_capacity_msat` semantics).
	pub holder_balance_msat: u64,
	/// Counterparty balance in msat.
	pub counterparty_balance_msat: u64,
	/// Pending HTLCs attached to this commitment.
	pub pending_htlcs: Vec<AttestationHtlc>,
	/// Commitment height (monotonically increasing). May be 0 if not recoverable
	/// without additional state.
	pub commitment_number: u64,
	/// True if we opened the channel.
	pub is_outbound: bool,
}

/// A single pending HTLC entry on `ChannelAttestation`.
#[derive(Clone, Debug)]
pub struct AttestationHtlc {
	/// True if we offered (sent) this HTLC; false if we received it.
	pub offered: bool,
	/// HTLC amount in millisatoshis.
	pub amount_msat: u64,
	/// SHA-256 payment hash committed to by this HTLC.
	pub payment_hash: [u8; 32],
	/// Absolute block height at which the HTLC expires.
	pub cltv_expiry: u32,
}

/// Extract `ChannelAttestation` for a single channel. Used by `Node::export_channel_attestation`.
pub(crate) fn export_channel_attestation(
	channel_manager: &Arc<ChannelManager>, chain_monitor: &Arc<ChainMonitor>,
	keys_manager: &Arc<KeysManager>, logger: &Arc<Logger>, channel_id: &ChannelId,
) -> Result<ChannelAttestation, Error> {
	// Pull channel metadata (capacity, counterparty, is_outbound, HTLCs, balances).
	let details = channel_manager
		.list_channels()
		.into_iter()
		.find(|c| c.channel_id == *channel_id)
		.ok_or(Error::ChannelConfigUpdateFailed)?;

	// Locate the ChannelMonitor for this channel.
	let monitor = chain_monitor.get_monitor(*channel_id).map_err(|_| Error::ChannelConfigUpdateFailed)?;
	let ldk_funding = monitor.get_funding_txo();
	// lightning::chain::transaction::OutPoint → bitcoin::OutPoint
	let funding_outpoint = bitcoin::OutPoint {
		txid: ldk_funding.txid,
		vout: ldk_funding.index as u32,
	};
	let channel_keys_id = monitor.channel_keys_id();

	// Re-derive the holder signer — read-only — to obtain the holder's funding pubkey.
	let signer = keys_manager.derive_channel_signer(channel_keys_id);
	let secp = Secp256k1::new();
	let holder_funding_pubkey = {
		use lightning::sign::ChannelSigner;
		signer.pubkeys(&secp).funding_pubkey
	};

	// Extract the latest signed holder commitment tx. Despite the scary name, this
	// method is pure read (no broadcast side-effect) — see rust-lightning's docs.
	let signed_txs = monitor.unsafe_get_latest_holder_commitment_txn(logger.as_ref());
	let signed_tx = signed_txs.into_iter().next().ok_or(Error::ChannelConfigUpdateFailed)?;

	// Parse the funding-input witness: [empty, sig_A, sig_B, redeem_script].
	let (funding_pk_a, funding_pk_b, sig_a, sig_b) = parse_funding_witness(&signed_tx)?;

	// Identify counterparty: the pubkey that is NOT ours.
	let (counterparty_funding_pubkey, counterparty_signature) =
		if holder_funding_pubkey == funding_pk_a {
			// We're pk_a; counterparty is pk_b. The sig at the counterparty's slot in the
			// witness is sig_b (slots are lex-ordered by pubkey; the script is
			// OP_2 <pk_a> <pk_b> OP_2 OP_CHECKMULTISIG).
			(funding_pk_b, sig_b)
		} else if holder_funding_pubkey == funding_pk_b {
			(funding_pk_a, sig_a)
		} else {
			return Err(Error::ChannelConfigUpdateFailed);
		};

	// Strip the witness from the commitment tx to produce the unsigned tx a verifier
	// will sighash.
	let mut unsigned_tx = signed_tx.clone();
	for input in unsigned_tx.input.iter_mut() {
		input.witness = Witness::new();
	}

	// Aggregate HTLCs (inbound = received = offered=false, outbound = sent = offered=true).
	let mut pending_htlcs = Vec::new();
	for htlc in &details.pending_inbound_htlcs {
		pending_htlcs.push(AttestationHtlc {
			offered: false,
			amount_msat: htlc.amount_msat,
			payment_hash: htlc.payment_hash.0,
			cltv_expiry: htlc.cltv_expiry,
		});
	}
	for htlc in &details.pending_outbound_htlcs {
		pending_htlcs.push(AttestationHtlc {
			offered: true,
			amount_msat: htlc.amount_msat,
			payment_hash: htlc.payment_hash.0,
			cltv_expiry: htlc.cltv_expiry,
		});
	}

	Ok(ChannelAttestation {
		channel_id: *channel_id,
		funding_outpoint,
		capacity_sats: details.channel_value_satoshis,
		counterparty_node_id: details.counterparty.node_id,
		holder_funding_pubkey,
		counterparty_funding_pubkey,
		holder_commitment_tx: unsigned_tx,
		counterparty_signature,
		holder_balance_msat: details.outbound_capacity_msat,
		counterparty_balance_msat: details.inbound_capacity_msat,
		pending_htlcs,
		// Commitment number could be recovered by de-obscuring the locktime+sequence
		// using both parties' payment_basepoints, but those aren't on ChannelMonitor's
		// public API today. Left as 0; the cryptographic BIP-143 verification does not
		// depend on this field.
		commitment_number: 0,
		is_outbound: details.is_outbound,
	})
}

/// Parse the funding-input witness of a Lightning commitment tx.
///
/// The witness layout (per BOLT #3) is:
///   `[empty, sig_pk_lex0, sig_pk_lex1, redeem_script]`
/// where `redeem_script = OP_2 <33-byte pk_lex0> <33-byte pk_lex1> OP_2 OP_CHECKMULTISIG`
/// and the pubkeys are sorted lexicographically by serialized bytes.
///
/// Returns `(pk_lex0, pk_lex1, sig_pk_lex0, sig_pk_lex1)`. Both signatures include
/// a trailing SIGHASH byte which is stripped before parsing into `Signature`.
fn parse_funding_witness(tx: &Transaction) -> Result<(PublicKey, PublicKey, Signature, Signature), Error> {
	let input = tx.input.first().ok_or(Error::ChannelConfigUpdateFailed)?;
	let witness = &input.witness;
	if witness.len() != 4 {
		return Err(Error::ChannelConfigUpdateFailed);
	}
	let sig_a_bytes = witness.nth(1).ok_or(Error::ChannelConfigUpdateFailed)?;
	let sig_b_bytes = witness.nth(2).ok_or(Error::ChannelConfigUpdateFailed)?;
	let redeem_script_bytes = witness.nth(3).ok_or(Error::ChannelConfigUpdateFailed)?;

	// DER + SIGHASH byte → strip last byte.
	let sig_a = parse_der_sig(sig_a_bytes).ok_or(Error::ChannelConfigUpdateFailed)?;
	let sig_b = parse_der_sig(sig_b_bytes).ok_or(Error::ChannelConfigUpdateFailed)?;

	// Parse multisig redeem script: OP_2 <33> pk_a <33> pk_b OP_2 OP_CHECKMULTISIG.
	let (pk_a, pk_b) = parse_2of2_redeem_script(redeem_script_bytes)
		.ok_or(Error::ChannelConfigUpdateFailed)?;

	Ok((pk_a, pk_b, sig_a, sig_b))
}

fn parse_der_sig(bytes: &[u8]) -> Option<Signature> {
	if bytes.is_empty() {
		return None;
	}
	// Strip the trailing SIGHASH byte.
	let der = &bytes[..bytes.len() - 1];
	Signature::from_der(der).ok()
}

/// Parse a 2-of-2 multisig redeem script and return the two pubkeys.
///
/// Expected layout:
/// ```text
///   OP_2 (0x52)
///   0x21 (push 33) <33 bytes pk_a>
///   0x21 (push 33) <33 bytes pk_b>
///   OP_2 (0x52)
///   OP_CHECKMULTISIG (0xae)
/// ```
fn parse_2of2_redeem_script(bytes: &[u8]) -> Option<(PublicKey, PublicKey)> {
	if bytes.len() != 71 {
		return None;
	}
	if bytes[0] != 0x52 || bytes[68] != 0x52 || bytes[69..] != [0xae] && bytes[69] != 0xae {
		return None;
	}
	if bytes[1] != 33 || bytes[35] != 33 {
		return None;
	}
	let pk_a = PublicKey::from_slice(&bytes[2..35]).ok()?;
	let pk_b = PublicKey::from_slice(&bytes[36..69]).ok()?;
	Some((pk_a, pk_b))
}

// Suppress warnings for the (unused-by-default) Hash import when the ChannelSigner
// import pattern pulls it in indirectly.
#[allow(dead_code)]
fn _use_hash() -> Option<[u8; 32]> {
	let b = bitcoin::hashes::sha256::Hash::hash(b"x");
	Some(b.to_byte_array())
}
