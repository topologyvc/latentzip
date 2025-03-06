/// LatentZip compression library
///
/// Provides LLM-based compression using arithmetic coding and language model predictions
const std = @import("std");

/// Arithmetic coding implementation for compression and decompression
pub const coder = @import("coder.zig");
/// Hugging Face model integration for loading models from repositories
pub const hf = @import("hf.zig");
/// Language model interface with token probability distributions
pub const llm = @import("llm.zig");
/// Main compression/decompression functionality
pub const zip = @import("zip.zig");

test {
    // Run all tests from our modules
    std.testing.refAllDecls(@This());
}
