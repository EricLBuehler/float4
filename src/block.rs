//! MXFP4 block format implementation.
//!
//! This module provides the MXFP4Block type for efficient storage of multiple
//! 4-bit floating-point values with a shared scale factor.

use crate::{E8M0, F4E2M1};

/// A compressed block of 32 F4E2M1 values with a shared E8M0 scale factor.
///
/// # Overview
///
/// `MXFP4Block` implements the MXFP4 block format from the OCP MX specification,
/// designed for efficient storage and computation in machine learning applications.
/// This format achieves 4x compression by:
///
/// - Storing 32 4-bit F4E2M1 values in just 16 bytes
/// - Using a single shared E8M0 scale factor for the entire block
/// - Enabling vectorized operations on compressed data
///
/// # Format Details
///
/// The block consists of:
/// - **Data**: 16 bytes containing 32 packed F4E2M1 values (2 per byte)
/// - **Scale**: 1 byte E8M0 scale factor (power of two from 2^-127 to 2^127)
///
/// Total size: 17 bytes for 32 values (compared to 128 bytes for f32)
///
/// # Use Cases
///
/// MXFP4Block is particularly useful for:
/// - Neural network weight compression
/// - Activation storage in quantized models
/// - Gradient accumulation in low-precision training
/// - Memory-bandwidth limited applications
///
/// # Examples
///
/// ## Creating from F4E2M1 values
///
/// ```rust
/// use float4::{F4E2M1, E8M0, MXFP4Block};
///
/// // Create 32 F4E2M1 values
/// let mut values = [F4E2M1::from_f64(0.0); 32];
/// for i in 0..32 {
///     values[i] = F4E2M1::from_f64((i as f64) * 0.1);
/// }
///
/// // Create scale factor
/// let scale = E8M0::from(1.0);
///
/// // Pack into block
/// let block = MXFP4Block::from_f32_slice(values, scale);
///
/// // Convert back to f32
/// let f32_values = block.to_f32_array();
/// ```
///
/// ## Quantizing f32 data
///
/// ```rust
/// use float4::{F4E2M1, E8M0, MXFP4Block};
///
/// // Original f32 data
/// let f32_data = [1.5, 2.0, -0.5, 3.0, /* ... 28 more values ... */];
///
/// // Compute appropriate scale
/// let scale = E8M0::from_f32_slice(&f32_data[..]);
/// let scale_val = scale.to_f64();
///
/// // Quantize to F4E2M1
/// let mut quantized = [F4E2M1::from_f64(0.0); 32];
/// for i in 0..32 {
///     if i < f32_data.len() {
///         quantized[i] = F4E2M1::from_f64(f32_data[i] as f64 / scale_val);
///     }
/// }
///
/// // Create compressed block
/// let block = MXFP4Block::from_f32_slice(quantized, scale);
/// ```
///
/// # Memory Layout
///
/// The 16-byte data array packs values as follows:
/// ```text
/// Byte 0: [Value1 bits 0-3][Value0 bits 0-3]
/// Byte 1: [Value3 bits 0-3][Value2 bits 0-3]
/// ...
/// Byte 15: [Value31 bits 0-3][Value30 bits 0-3]
/// ```
///
/// Each byte contains two 4-bit values, with even-indexed values in the
/// lower nibble and odd-indexed values in the upper nibble.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MXFP4Block {
    /// 16 bytes containing 32 packed F4E2M1 values (2 per byte).
    block: [u8; 16],
    /// Shared E8M0 scale factor for all values in the block.
    scale: E8M0,
}

const _: () = assert!(std::mem::size_of::<MXFP4Block>() == 17);

impl MXFP4Block {
    /// Creates a new MXFP4Block from pre-quantized F4E2M1 values and a scale factor.
    ///
    /// This function packs 32 F4E2M1 values into a compressed block format. The values
    /// should already be quantized to F4E2M1 format and the scale should be chosen
    /// appropriately for the data range.
    ///
    /// # Arguments
    ///
    /// * `xs` - Array of exactly 32 F4E2M1 values to pack
    /// * `scale` - E8M0 scale factor that will be applied when unpacking
    ///
    /// # Packing Details
    ///
    /// Values are packed two per byte in little-endian nibble order:
    /// - Even indices (0, 2, 4, ...) go in the lower 4 bits
    /// - Odd indices (1, 3, 5, ...) go in the upper 4 bits
    ///
    /// # Examples
    ///
    /// ```rust
    /// use float4::{F4E2M1, E8M0, MXFP4Block};
    ///
    /// // Create normalized values in [-6, 6] range
    /// let mut values = [F4E2M1::from_f64(0.0); 32];
    /// for i in 0..32 {
    ///     values[i] = F4E2M1::from_f64(((i as f64) - 16.0) / 4.0);
    /// }
    ///
    /// // Pack with scale factor of 8
    /// let scale = E8M0::from(8.0);
    /// let block = MXFP4Block::from_f32_slice(values, scale);
    /// ```
    #[inline(always)]
    pub fn from_f32_slice(xs: [F4E2M1; 32], scale: E8M0) -> Self {
        let mut block = [0u8; 16];

        // Pack two F4E2M1 values into each byte
        for (i, byte) in block.iter_mut().enumerate() {
            let idx = i * 2;
            // First value goes in lower 4 bits
            let low = xs[idx].to_bits() & 0x0F;
            // Second value goes in upper 4 bits
            let high = (xs[idx + 1].to_bits() & 0x0F) << 4;
            *byte = low | high;
        }

        Self { block, scale }
    }

    /// Unpacks the compressed block into individual F4E2M1 values.
    ///
    /// This extracts the 32 packed F4E2M1 values from the compressed format,
    /// returning them as an array. The scale factor is not applied - the
    /// returned values are exactly as stored in the block.
    ///
    /// # Returns
    ///
    /// An array of 32 F4E2M1 values in the same order they were packed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use float4::{F4E2M1, E8M0, MXFP4Block};
    ///
    /// let values = [F4E2M1::from_f64(1.5); 32];
    /// let block = MXFP4Block::from_f32_slice(values, E8M0::from(1.0));
    ///
    /// let unpacked = block.to_f4_array();
    /// assert_eq!(unpacked[0].to_f64(), 1.5);
    /// ```
    #[inline(always)]
    pub fn to_f4_array(&self) -> [F4E2M1; 32] {
        let mut result = [F4E2M1::from_bits(0); 32];

        for i in 0..16 {
            let byte = self.block[i];
            let idx = i * 2;
            // Extract lower 4 bits
            result[idx] = F4E2M1::from_bits(byte & 0x0F);
            // Extract upper 4 bits
            result[idx + 1] = F4E2M1::from_bits((byte >> 4) & 0x0F);
        }

        result
    }

    /// Returns the E8M0 scale factor associated with this block.
    ///
    /// The scale factor is a power of two that should be multiplied with
    /// the F4E2M1 values to recover the original data range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use float4::{F4E2M1, E8M0, MXFP4Block};
    ///
    /// let block = MXFP4Block::from_f32_slice(
    ///     [F4E2M1::from_f64(1.0); 32],
    ///     E8M0::from(16.0)
    /// );
    /// assert_eq!(block.scale().to_f64(), 16.0);
    /// ```
    #[inline(always)]
    pub fn scale(&self) -> E8M0 {
        self.scale
    }

    /// Converts the block to an array of f32 values by applying the scale factor.
    ///
    /// This method unpacks all F4E2M1 values and multiplies each by the block's
    /// scale factor, producing the final decompressed values. This is the typical
    /// way to retrieve usable floating-point data from an MXFP4Block.
    ///
    /// # Returns
    ///
    /// An array of 32 f32 values computed as: `F4E2M1_value * scale_factor`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use float4::{F4E2M1, E8M0, MXFP4Block};
    ///
    /// // Create block with values [0.5, 1.0, 1.5, ...] and scale 4.0
    /// let mut values = [F4E2M1::from_f64(0.0); 32];
    /// values[0] = F4E2M1::from_f64(0.5);
    /// values[1] = F4E2M1::from_f64(1.0);
    /// values[2] = F4E2M1::from_f64(1.5);
    ///
    /// let block = MXFP4Block::from_f32_slice(values, E8M0::from(4.0));
    /// let f32_array = block.to_f32_array();
    ///
    /// assert_eq!(f32_array[0], 2.0);  // 0.5 * 4.0
    /// assert_eq!(f32_array[1], 4.0);  // 1.0 * 4.0
    /// assert_eq!(f32_array[2], 6.0);  // 1.5 * 4.0
    /// ```
    ///
    /// # Precision Considerations
    ///
    /// The conversion involves:
    /// 1. F4E2M1 → f64 (exact)
    /// 2. Multiplication by scale (exact for power-of-two scales)
    /// 3. f64 → f32 (may round)
    ///
    /// For maximum precision, consider working with f64 if your scale
    /// factors and values require it.
    #[inline(always)]
    pub fn to_f32_array(&self) -> [f32; 32] {
        let scale = self.scale.to_f64();
        let values = self.to_f4_array();
        let mut result = [0.0f32; 32];

        for i in 0..32 {
            result[i] = (values[i].to_f64() * scale) as f32;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        // Create test data
        let mut values = [F4E2M1::from_bits(0); 32];
        for (i, value) in values.iter_mut().enumerate() {
            // Use different values that fit in 4 bits
            *value = F4E2M1::from_bits((i % 16) as u8);
        }
        let scale = E8M0::from_f64(4.0f64);

        // Pack into block
        let block = MXFP4Block::from_f32_slice(values, scale);

        // Verify scale
        assert_eq!(block.scale().to_f64(), 4.0);

        // Unpack and verify
        let unpacked = block.to_f4_array();
        for i in 0..32 {
            assert_eq!(unpacked[i].to_bits(), values[i].to_bits());
        }
    }

    #[test]
    fn test_to_f32_array() {
        // Create simple test values
        let mut values = [F4E2M1::from_bits(0); 32];
        values[0] = F4E2M1::from_f64(1.0); // 1.0
        values[1] = F4E2M1::from_f64(2.0); // 2.0
        values[2] = F4E2M1::from_f64(0.5); // 0.5

        let scale = E8M0::from_f64(2.0f64); // scale = 2.0

        let block = MXFP4Block::from_f32_slice(values, scale);
        let f32_array = block.to_f32_array();

        // Check scaled values
        assert_eq!(f32_array[0], 2.0); // 1.0 * 2.0
        assert_eq!(f32_array[1], 4.0); // 2.0 * 2.0
        assert_eq!(f32_array[2], 1.0); // 0.5 * 2.0

        // Rest should be 0.0
        for value in f32_array.iter().skip(3) {
            assert_eq!(*value, 0.0);
        }
    }

    #[test]
    fn test_packing_layout() {
        // Test specific packing layout
        let mut values = [F4E2M1::from_bits(0); 32];

        // Set specific patterns
        values[0] = F4E2M1::from_bits(0x5); // 0101
        values[1] = F4E2M1::from_bits(0xA); // 1010
        values[2] = F4E2M1::from_bits(0x3); // 0011
        values[3] = F4E2M1::from_bits(0xC); // 1100

        let scale = E8M0::from_f64(1.0f64);
        let block = MXFP4Block::from_f32_slice(values, scale);

        // Check packed bytes
        assert_eq!(block.block[0], 0xA5); // 1010_0101
        assert_eq!(block.block[1], 0xC3); // 1100_0011

        // Remaining bytes should be 0
        for i in 2..16 {
            assert_eq!(block.block[i], 0x00);
        }
    }
}
