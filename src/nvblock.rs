//! NVPF4 block format implementation with fp8 scale factor
//!
//!
//! ```rust
//! use float4::{F4E2M1, NVFP4Block};
//!
//! // Original data
//! let data = vec![1.5, -2.0, 0.5, 3.0, 1.0, -0.5];
//!
//! let global_scale=NVFP4Block::global_scale_from_f32_slice(&data);
//! let block_scale = NVFP4Block::block_scale_from_f32_slice(&data, global_scale);
//! assert_eq!(global_scale, 0.0012019231);
//! assert_eq!(block_scale.to_f64(), 416.0);
//!
//! // Quantize to F4E2M1
//! let mut quantized = [F4E2M1::from_f64(0.0); 16];
//! for (i, &value) in data.iter().enumerate() {
//!     quantized[i] = F4E2M1::from_f64(value as f64  / (block_scale.to_f64()  *global_scale as f64));
//! }
//!
//! let block = NVFP4Block::new_from_f32_slice(quantized, block_scale);
//!
//! // Retrieve values
//! let restored = block.to_f32_array(global_scale);
//! assert_eq!(restored[0], 1.5);
//! assert_eq!(restored[1], -2.0);
//! ```
//!

use crate::F4E2M1;
use float8::F8E4M3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NVFP4Block {
    /// 8 bytes containing 16 packed F4E2M1 values (2 per byte).
    block: [u8; 8],
    /// shared F8E4M3 scale
    scale: float8::F8E4M3,
}

const _: () = assert!(std::mem::size_of::<NVFP4Block>() == 9);

impl NVFP4Block {
    /// Creates the global f32 scale of a block
    ///
    /// This function computes the global scale factor as needed for mvfp4.
    /// It finds the maximum absolute value in the slice and converts it to a power of two
    /// scale factor following F8E4M3 conversion rules.
    ///
    /// # Arguments
    ///
    /// * `values` - A slice of f32 values to compute the scale from
    ///
    /// # Returns
    ///
    /// An F8E4M3 scale factor that can represent the largest value in the slice when
    /// multiplied by the quantized values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use float8::F8E4M3;
    /// use float4::NVFP4Block;
    ///
    /// // Scale for values within a small range
    /// let values = [10.0,20.0,30.0,40.0];
    /// let scale = NVFP4Block::global_scale_from_f32_slice(&values);
    /// assert_eq!(scale, 0.016025642);  
    ///
    /// // Scale for larger values
    /// let values = [1.0, 5.0, -3.5];
    /// let scale = NVFP4Block::global_scale_from_f32_slice(&values);
    /// assert_eq!(scale, 0.0020032052);
    ///
    /// // Empty slice returns 0 scale
    /// let scale = NVFP4Block::global_scale_from_f32_slice(&[]);
    /// assert_eq!(scale, 0.0);
    /// ```
    pub fn global_scale_from_f32_slice(values: &[f32]) -> f32 {
        let max_abs = values.iter().map(|&x| x.abs()).fold(0.0f32, |a, b| {
            if b.is_nan() || a.is_nan() {
                f32::NAN
            } else if b > a {
                b
            } else {
                a
            }
        });
        max_abs / (F4E2M1::MAX.to_f64() as f32 * F8E4M3::MAX.to_f32())
    }

    /// Creates an F8E4M3 scale factor from a slice of f32 values.
    ///
    /// This function computes an appropriate scale factor for quantizing the given values.
    /// It finds the maximum absolute value in the slice and converts it to a power of two
    /// scale factor following F8E4M3 conversion rules.
    ///
    /// # Arguments
    ///
    /// * `values` - A slice of f32 values to compute the scale from
    ///
    /// # Returns
    ///
    /// An F8E4M3 scale factor that can represent the largest value in the slice when
    /// multiplied by the quantized values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use float8::F8E4M3;
    /// use float4::NVFP4Block;
    ///
    /// // assume a global scale of 1f32
    /// let global_scale = 1.0;
    /// // Scale for values within a small range
    /// let values = [0.5, -0.75, 0.25];
    /// let scale = NVFP4Block::block_scale_from_f32_slice(&values, global_scale);
    /// assert_eq!(scale.to_f32(), 0.125);  // 0.75 / F4E2M1::MAX == 0.75 / 6
    ///
    /// let scale = NVFP4Block::block_scale_from_f32_slice(&values, 0.5);
    /// assert_eq!(scale.to_f32(),0.25);  
    ///
    /// let scale = NVFP4Block::block_scale_from_f32_slice(&values, 2.0);
    /// assert_eq!(scale.to_f32(),0.0625);  
    ///
    /// // Scale for larger values
    /// let values = [1.0, 5.0, -3.5];
    /// let scale = NVFP4Block::block_scale_from_f32_slice(&values, global_scale);
    /// assert_eq!(scale.to_f32(), 0.8125);
    ///
    /// // Empty slice returns 0 scale
    /// let scale = NVFP4Block::block_scale_from_f32_slice(&[], global_scale);
    /// assert_eq!(scale.to_f32(), 0.0);
    /// ```
    #[inline(always)]
    pub fn block_scale_from_f32_slice(values: &[f32], global_scale: f32) -> F8E4M3 {
        let max_abs = values.iter().map(|x| x.abs()).fold(0.0f32, |a, b| {
            if b.is_nan() || a.is_nan() {
                f32::NAN
            } else if b > a {
                b
            } else {
                a
            }
        });
        let with_global_scale = max_abs / (global_scale * F4E2M1::MAX.to_f64() as f32);
        F8E4M3::from_f32(with_global_scale)
    }

    /// Creates a new NVFP4Block from pre-quantized F4E2M1 values and a scale factor.
    ///
    /// This function packs 16 F4E2M1 values into a compressed block format. The values
    /// should already be quantized to F4E2M1 format and the scale should be chosen
    /// appropriately for the data range.
    ///
    /// # Arguments
    ///
    /// * `xs` - Array of exactly 16 F4E2M1 values to pack
    /// * `scale` - F8E4M3 scale factor that will be applied when unpacking
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
    /// use float4::{F4E2M1, E8M0, NVFP4Block};
    /// use float8::F8E4M3;
    ///
    /// // Create normalized values in [-6, 6] range
    /// let mut values = [F4E2M1::from_f64(0.0); 16];
    /// for i in 0..16 {
    ///     values[i] = F4E2M1::from_f64(((i as f64) - 16.0) / 4.0);
    /// }
    ///
    /// // Pack with scale factor of 8
    /// let scale = F8E4M3::from(8.0);
    /// let block = NVFP4Block::new_from_f32_slice(values, scale);
    /// ```
    #[inline(always)]
    pub fn new_from_f32_slice(xs: [F4E2M1; 16], scale: float8::F8E4M3) -> Self {
        let mut block = [0u8; 8];

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
    /// This extracts the 16 packed F4E2M1 values from the compressed format,
    /// returning them as an array. The scale factor is not applied - the
    /// returned values are exactly as stored in the block.
    ///
    /// # Returns
    ///
    /// An array of 16 F4E2M1 values in the same order they were packed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use float4::{F4E2M1, E8M0, NVFP4Block};
    /// use float8::F8E4M3;
    ///
    /// let values = [F4E2M1::from_f64(1.5); 16];
    /// let block = NVFP4Block::new_from_f32_slice(values, F8E4M3::from(1.0));
    ///
    /// let unpacked = block.to_f4_array();
    /// assert_eq!(unpacked[0].to_f64(), 1.5);
    /// ```
    #[inline(always)]
    pub fn to_f4_array(&self) -> [F4E2M1; 16] {
        let mut result = [F4E2M1::from_bits(0); 16];

        for i in 0..8 {
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
    /// use float4::{F4E2M1, E8M0, NVFP4Block};
    /// use float8::F8E4M3;
    ///
    /// let block = NVFP4Block::new_from_f32_slice(
    ///     [F4E2M1::from_f64(1.0); 16],
    ///     F8E4M3::from(16.0)
    /// );
    /// assert_eq!(block.scale().to_f64(), 16.0);
    /// ```
    #[inline(always)]
    pub fn scale(&self) -> F8E4M3 {
        self.scale
    }

    pub fn from_f32_slice(values: [f32; 16], global_scale: f32) -> Self {
        let block_scale = Self::block_scale_from_f32_slice(&values, global_scale);

        let mut block = [0u8; 8];

        let block_scale_f32 = block_scale.to_f32();
        // Pack two F4E2M1 values into each byte
        for (i, byte) in block.iter_mut().enumerate() {
            let idx = i * 2;
            let val1a = values[idx] / (global_scale * block_scale_f32);
            let val2b = values[idx + 1] / (global_scale * block_scale_f32);
            let val1 = F4E2M1::from_f64(val1a as f64);
            let val2 = F4E2M1::from_f64(val2b as f64);
            // First value goes in lower 4 bits
            let low = val1.to_bits() & 0x0F;
            // Second value goes in upper 4 bits
            let high = (val2.to_bits() & 0x0F) << 4;
            *byte = low | high;
        }

        Self {
            block,
            scale: block_scale,
        }
    }

    /// Converts the block to an array of f32 values by applying the scale factor.
    ///
    /// This method unpacks all F4E2M1 values and multiplies each by the block's
    /// scale factor, producing the final decompressed values. This is the typical
    /// way to retrieve usable floating-point data from an NVFP4Block.
    ///
    /// # Returns
    ///
    /// An array of 16 f32 values computed as: `F4E2M1_value * scale_factor`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use float4::{F4E2M1, E8M0, NVFP4Block};
    /// use float8::F8E4M3;
    ///
    /// // Create block with values [0.5, 1.0, 1.5, ...] and scale 4.5
    /// let mut values = [F4E2M1::from_f64(0.0); 16];
    /// values[0] = F4E2M1::from_f64(0.5);
    /// values[1] = F4E2M1::from_f64(1.0);
    /// values[2] = F4E2M1::from_f64(1.5);
    ///
    ///
    ///
    /// let block = NVFP4Block::new_from_f32_slice(values, F8E4M3::from(4.5));
    /// let f32_array = block.to_f32_array(1.0);
    ///
    /// assert_eq!(f32_array[0], 2.25);  // 0.5 * 4.5
    /// assert_eq!(f32_array[1], 4.5);  // 1.0 * 4.5
    /// assert_eq!(f32_array[2], 6.75);  // 1.5 * 4.5
    /// ```
    ///
    /// # Precision Considerations
    ///
    /// The conversion involves:
    /// 1. F4E2M1 → f64 (exact)
    /// 2. Multiplication by scale (lossy, F8E4M3 to f16 to f64)
    /// 3. f64 → f32 (may round)
    #[inline(always)]
    pub fn to_f32_array(&self, global_scale: f32) -> [f32; 16] {
        let scale = self.scale.to_f32();
        let values = self.to_f4_array();
        let mut result = [0.0f32; 16];

        for i in 0..16 {
            result[i] = (values[i].to_f64() as f32 * (global_scale * scale)) as f32;
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
        let mut values = [F4E2M1::from_bits(0); 16];
        for (i, value) in values.iter_mut().enumerate() {
            // Use different values that fit in 4 bits
            *value = F4E2M1::from_bits((i % 16) as u8);
        }
        let scale = F8E4M3::from_f64(4.5f64);

        // Pack into block
        let block = NVFP4Block::new_from_f32_slice(values, scale);

        // Verify scale
        assert_eq!(block.scale().to_f64(), 4.5);

        // Unpack and verify
        let unpacked = block.to_f4_array();
        for i in 0..16 {
            assert_eq!(unpacked[i].to_bits(), values[i].to_bits());
        }
    }

    #[test]
    fn test_to_f32_array() {
        // Create simple test values
        let mut values = [F4E2M1::from_bits(0); 16];
        values[0] = F4E2M1::from_f64(1.0); // 1.0
        values[1] = F4E2M1::from_f64(2.0); // 2.0
        values[2] = F4E2M1::from_f64(0.5); // 0.5

        let scale = F8E4M3::from_f64(2.5f64); // scale = 2.5

        let block = NVFP4Block::new_from_f32_slice(values, scale);
        let f32_array = block.to_f32_array(1.0);

        // Check scaled values
        assert_eq!(f32_array[0], 2.5); // 1.0 * 2.5
        assert_eq!(f32_array[1], 5.0); // 2.0 * 2.5
        assert_eq!(f32_array[2], 1.25); // 0.5 * 2.5

        // Rest should be 0.0
        for value in f32_array.iter().skip(3) {
            assert_eq!(*value, 0.0);
        }
    }

    #[test]
    fn test_packing_layout() {
        // Test specific packing layout
        let mut values = [F4E2M1::from_bits(0); 16];

        // Set specific patterns
        values[0] = F4E2M1::from_bits(0x5); // 0101
        values[1] = F4E2M1::from_bits(0xA); // 1010
        values[2] = F4E2M1::from_bits(0x3); // 0011
        values[3] = F4E2M1::from_bits(0xC); // 1100

        let scale = F8E4M3::from_f64(1.0f64);
        let block = NVFP4Block::new_from_f32_slice(values, scale);

        // Check packed bytes
        assert_eq!(block.block[0], 0xA5); // 1010_0101
        assert_eq!(block.block[1], 0xC3); // 1100_0011

        // Remaining bytes should be 0
        for i in 2..8 {
            assert_eq!(block.block[i], 0x00);
        }
    }

    #[test]
    fn test_f32_conversion_global_factor_1() {
        // Test specific packing layout
        let mut values = [0f32; 16];
        values[0] = 10.0;
        values[1] = 20.0;
        values[2] = 30.0;
        values[3] = 40.0;

        let global_scale = 1.0;

        let block = NVFP4Block::from_f32_slice(values, global_scale);
        let result = block.to_f32_array(global_scale);

        assert_eq!(result[0], 9.75);
        assert_eq!(result[1], 19.5);
        assert_eq!(result[2], 26.0);
        assert_eq!(result[3], 39.0);
    }

    #[test]
    fn test_f32_conversion() {
        let mut values = [0f32; 16];
        values[0] = 10.0;
        values[1] = 20.0;
        values[2] = 30.0;
        values[3] = 40.0;

        let global_scale = NVFP4Block::global_scale_from_f32_slice(&values);

        let block = NVFP4Block::from_f32_slice(values, global_scale);
        let result = block.to_f32_array(global_scale);

        assert_eq!(result[0], 10.0);
        assert_eq!(result[1], 20.0);
        assert_eq!(f32::trunc(result[2] * 100.0) / 100.0, 26.66);
        assert_eq!(result[3], 40.0);
    }
    #[test]
    fn test_f32_conversion_big() {
        let mut values = [0f32; 16];
        values[0] = 15.0;
        values[1] = 30.0;
        values[2] = 120.0;
        values[3] = 180.0;

        let global_scale = NVFP4Block::global_scale_from_f32_slice(&values);

        let block = NVFP4Block::from_f32_slice(values, global_scale);
        let result = block.to_f32_array(global_scale);

        assert_eq!(result[0], 15.0);
        assert_eq!(result[1], 30.0);
        assert_eq!(result[2], 120.0);
        assert_eq!(result[3], 180.0);
    }
}
