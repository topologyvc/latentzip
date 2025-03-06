/// Arithmetic coding implementation for efficient data compression
/// This module provides the core arithmetic coding algorithms used by LatentZip
const std = @import("std");
const llm = @import("llm.zig");
const Allocator = std.mem.Allocator;

/// Number of bits used for internal state representation
/// Larger values provide more precision but require more memory
const NUM_STATE_BITS: u8 = 52;

/// Base implementation of arithmetic coding functionality
/// Shared between encoder and decoder implementations
const ArithmeticCoderBase = struct {
    half_range: u64,
    quarter_range: u64,
    state_mask: u64,
    low: u64,
    high: u64,

    /// Initialize a new arithmetic coder base with default state values
    ///
    /// Returns: Configured ArithmeticCoderBase instance
    pub fn init() ArithmeticCoderBase {
        const half_range: u64 = 1 << (NUM_STATE_BITS - 1);
        return .{
            .half_range = half_range,
            .quarter_range = half_range >> 1,
            .state_mask = (1 << NUM_STATE_BITS) - 1,
            .low = 0,
            .high = (1 << NUM_STATE_BITS) - 1,
        };
    }

    /// Update the arithmetic coder's state based on a symbol and its probability
    ///
    /// Params:
    ///   Handler: Type of the handler (either Encoder or Decoder)
    ///   handler: Instance of the handler for bit operations
    ///   base: Arithmetic coder base instance to update
    ///   cdf: Cumulative distribution function representing token probabilities
    ///   symbol: Index of the symbol being encoded/decoded
    pub fn update(comptime Handler: type, handler: *Handler, base: *ArithmeticCoderBase, cdf: []const llm.CdfToken, symbol: usize) void {
        const total: u128 = cdf[cdf.len - 1].probability;
        const range: u128 = base.high - base.low + 1;
        const symhigh: u128 = cdf[symbol].probability;
        const symlow: u128 = if (symbol > 0) cdf[symbol - 1].probability else 0;
        base.high = base.low + @as(u64, @intCast(symhigh * range / total - 1));
        base.low = base.low + @as(u64, @intCast(symlow * range / total));
        while (((base.low ^ base.high) & base.half_range) == 0) {
            handler.shift();
            base.low = (base.low << 1) & base.state_mask;
            base.high = ((base.high << 1) & base.state_mask) | 1;
        }
        while ((base.low & ~base.high & base.quarter_range) != 0) {
            handler.underflow();
            base.low = (base.low << 1) ^ base.half_range;
            base.high = ((base.high ^ base.half_range) << 1) | base.half_range | 1;
        }
    }
};

/// Encoder implementation for arithmetic coding
/// Handles the compression of symbols based on their probability distributions
pub const Encoder = struct {
    base: ArithmeticCoderBase,
    encoded_data: std.ArrayList(u8),
    bit_index: u3,
    num_underflow: usize,
    allocator: Allocator,

    /// Initialize a new encoder instance with the given allocator
    ///
    /// Returns: Configured Encoder instance
    pub fn init(allocator: Allocator) !Encoder {
        return .{
            .base = ArithmeticCoderBase.init(),
            .encoded_data = std.ArrayList(u8).init(allocator),
            .bit_index = 0,
            .num_underflow = 0,
            .allocator = allocator,
        };
    }

    /// Deinitialize the encoder instance
    pub fn deinit(self: *Encoder) void {
        self.encoded_data.deinit();
    }

    /// Encode a symbol based on its probability distribution
    ///
    /// Params:
    ///   cdf: Cumulative distribution function representing token probabilities
    ///   symbol: Index of the symbol being encoded
    pub fn encodeSymbol(self: *Encoder, cdf: []const llm.CdfToken, symbol: usize) void {
        ArithmeticCoderBase.update(Encoder, self, &self.base, cdf, symbol);
    }

    /// Finish encoding and finalize the compressed data
    pub fn finish(self: *Encoder) !void {
        try self.appendBit(1);
    }

    /// Get the encoded data as a byte slice
    pub fn getEncoded(self: *const Encoder) []const u8 {
        return self.encoded_data.items;
    }

    /// Shift the encoder's state by one bit
    fn shift(self: *Encoder) void {
        const bit: u1 = @intCast(self.base.low >> (NUM_STATE_BITS - 1));
        self.appendBit(bit) catch unreachable;
        var i: usize = 0;
        while (i < self.num_underflow) : (i += 1) {
            self.appendBit(bit ^ 1) catch unreachable;
        }
        self.num_underflow = 0;
    }

    /// Handle an underflow event in the encoder's state
    fn underflow(self: *Encoder) void {
        self.num_underflow += 1;
    }

    /// Append a bit to the encoded data
    ///
    /// Params:
    ///   bit: Bit value to append (0 or 1)
    fn appendBit(self: *Encoder, bit: u1) !void {
        if (self.bit_index == 0) {
            try self.encoded_data.append(0);
        }
        self.encoded_data.items[self.encoded_data.items.len - 1] |= @as(u8, bit) << (7 - self.bit_index);
        self.bit_index = @intCast((@as(u4, self.bit_index) + 1) % 8);
    }
};

/// Decoder implementation for arithmetic coding
/// Handles the decompression of encoded data using probability distributions
pub const Decoder = struct {
    base: ArithmeticCoderBase,
    input: []const u8,
    byte_index: usize,
    bit_index: u3,
    code: u64,

    /// Initialize a new decoder instance with the given input data
    ///
    /// Returns: Configured Decoder instance
    pub fn init(input: []const u8) !Decoder {
        var decoder = Decoder{
            .base = ArithmeticCoderBase.init(),
            .input = input,
            .byte_index = 0,
            .bit_index = 0,
            .code = 0,
        };
        var i: u6 = NUM_STATE_BITS - 1;
        while (true) : (i -= 1) {
            decoder.code = (decoder.code << 1) | decoder.readCodeBit();
            if (i == 0) break;
        }
        return decoder;
    }

    /// Decode a symbol based on its probability distribution
    ///
    /// Params:
    ///   cdf: Cumulative distribution function representing token probabilities
    ///
    /// Returns: Index of the decoded symbol
    pub fn decodeSymbol(self: *Decoder, cdf: []const llm.CdfToken) usize {
        const total: u128 = cdf[cdf.len - 1].probability;
        const range: u128 = self.base.high - self.base.low + 1;
        const offset: u128 = self.code - self.base.low;
        const value: u128 = ((offset + 1) * total - 1) / range;
        var symbol: usize = 0;
        while (symbol < cdf.len - 1 and cdf[symbol].probability <= value) {
            symbol += 1;
        }
        ArithmeticCoderBase.update(Decoder, self, &self.base, cdf, symbol);
        return symbol;
    }

    /// Shift the decoder's state by one bit
    fn shift(self: *Decoder) void {
        self.code = ((self.code << 1) & self.base.state_mask) | self.readCodeBit();
    }

    /// Handle an underflow event in the decoder's state
    fn underflow(self: *Decoder) void {
        self.code = (self.code & self.base.half_range) |
            ((self.code << 1) & (self.base.state_mask >> 1)) |
            self.readCodeBit();
    }

    /// Read a bit from the input data
    fn readCodeBit(self: *Decoder) u1 {
        if (self.byte_index >= self.input.len) {
            return 0;
        }
        const bit: u1 = @intCast((self.input[self.byte_index] >> (7 - self.bit_index)) & 1);
        self.bit_index = @intCast((@as(usize, self.bit_index) + 1) % 8);
        if (self.bit_index == 0) {
            self.byte_index += 1;
        }
        return bit;
    }
};

test "coder" {
    const allocator = std.testing.allocator;
    var encoder = try Encoder.init(allocator);
    defer encoder.deinit();
    const symbols = [_]usize{ 0, 1, 2, 2, 3 };
    const cdf = [_]llm.CdfToken{
        .{ .token = "H", .probability = 10 },
        .{ .token = "e", .probability = 50 },
        .{ .token = "l", .probability = 60 },
        .{ .token = "o", .probability = 100 },
    };
    for (symbols) |symbol| {
        encoder.encodeSymbol(&cdf, symbol);
    }
    try encoder.finish();
    const encoded = encoder.getEncoded();
    var decoder = try Decoder.init(encoded);
    for (symbols) |symbol| {
        const decoded = decoder.decodeSymbol(&cdf);
        try std.testing.expectEqual(decoded, symbol);
    }
}

// Test the initialization and basic properties of the arithmetic coder base
test "ArithmeticCoderBase initialization" {
    const base = ArithmeticCoderBase.init();

    // Verify base properties are correctly set
    const expected_half_range: u64 = 1 << (NUM_STATE_BITS - 1);
    try std.testing.expectEqual(expected_half_range, base.half_range);
    try std.testing.expectEqual(expected_half_range >> 1, base.quarter_range);
    try std.testing.expectEqual((1 << NUM_STATE_BITS) - 1, base.state_mask);
    try std.testing.expectEqual(@as(u64, 0), base.low);
    try std.testing.expectEqual((1 << NUM_STATE_BITS) - 1, base.high);
}

// Test encoder initialization and state
test "Encoder initialization" {
    const allocator = std.testing.allocator;
    var encoder = try Encoder.init(allocator);
    defer encoder.deinit();

    // Verify encoder properties
    try std.testing.expectEqual(@as(u3, 0), encoder.bit_index);
    try std.testing.expectEqual(@as(usize, 0), encoder.num_underflow);
    try std.testing.expectEqual(@as(usize, 0), encoder.encoded_data.items.len);
}

// Test encoding with uniform distribution
test "Encode with uniform distribution" {
    const allocator = std.testing.allocator;
    var encoder = try Encoder.init(allocator);
    defer encoder.deinit();

    // Create a uniform distribution
    const uniform_cdf = [_]llm.CdfToken{
        .{ .token = "A", .probability = 25 },
        .{ .token = "B", .probability = 50 },
        .{ .token = "C", .probability = 75 },
        .{ .token = "D", .probability = 100 },
    };

    // Encode alternating symbols
    const symbols = [_]usize{ 0, 1, 2, 3, 0, 1, 2, 3 };
    for (symbols) |symbol| {
        encoder.encodeSymbol(&uniform_cdf, symbol);
    }

    try encoder.finish();
    const encoded = encoder.getEncoded();
    try std.testing.expect(encoded.len > 0);

    // Decode and verify
    var decoder = try Decoder.init(encoded);
    for (symbols) |symbol| {
        const decoded = decoder.decodeSymbol(&uniform_cdf);
        try std.testing.expectEqual(symbol, decoded);
    }
}

// Test encoding with skewed distribution
test "Encode with skewed distribution" {
    const allocator = std.testing.allocator;
    var encoder = try Encoder.init(allocator);
    defer encoder.deinit();

    // Create a skewed distribution (symbol 0 is most probable)
    const skewed_cdf = [_]llm.CdfToken{
        .{ .token = "A", .probability = 70 }, // 70% probability
        .{ .token = "B", .probability = 85 }, // 15% probability
        .{ .token = "C", .probability = 95 }, // 10% probability
        .{ .token = "D", .probability = 100 }, // 5% probability
    };

    // Encode with mostly the most probable symbol
    const symbols = [_]usize{ 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0 };
    for (symbols) |symbol| {
        encoder.encodeSymbol(&skewed_cdf, symbol);
    }

    try encoder.finish();
    const encoded = encoder.getEncoded();

    // Decode and verify
    var decoder = try Decoder.init(encoded);
    for (symbols) |symbol| {
        const decoded = decoder.decodeSymbol(&skewed_cdf);
        try std.testing.expectEqual(symbol, decoded);
    }
}

// Test encoding and decoding with a single symbol
test "Encode single symbol" {
    const allocator = std.testing.allocator;
    var encoder = try Encoder.init(allocator);
    defer encoder.deinit();

    const cdf = [_]llm.CdfToken{
        .{ .token = "A", .probability = 100 },
    };

    // Encode a single symbol multiple times
    const count = 5;
    var i: usize = 0;
    while (i < count) : (i += 1) {
        encoder.encodeSymbol(&cdf, 0);
    }

    try encoder.finish();
    const encoded = encoder.getEncoded();

    // Decode and verify
    var decoder = try Decoder.init(encoded);
    i = 0;
    while (i < count) : (i += 1) {
        const decoded = decoder.decodeSymbol(&cdf);
        try std.testing.expectEqual(@as(usize, 0), decoded);
    }
}

// Test encoding and decoding with large probabilities
test "Encode with large probabilities" {
    const allocator = std.testing.allocator;
    var encoder = try Encoder.init(allocator);
    defer encoder.deinit();

    // Use large numbers for probabilities
    const large_cdf = [_]llm.CdfToken{
        .{ .token = "A", .probability = 1_000_000 },
        .{ .token = "B", .probability = 5_000_000 },
        .{ .token = "C", .probability = 8_000_000 },
        .{ .token = "D", .probability = 10_000_000 },
    };

    const symbols = [_]usize{ 0, 1, 2, 3, 1, 2 };
    for (symbols) |symbol| {
        encoder.encodeSymbol(&large_cdf, symbol);
    }

    try encoder.finish();
    const encoded = encoder.getEncoded();

    // Decode and verify
    var decoder = try Decoder.init(encoded);
    for (symbols) |symbol| {
        const decoded = decoder.decodeSymbol(&large_cdf);
        try std.testing.expectEqual(symbol, decoded);
    }
}

// Test bit operations in the encoder
test "Encoder bit operations" {
    const allocator = std.testing.allocator;
    var encoder = try Encoder.init(allocator);
    defer encoder.deinit();

    // Manually append bits
    try encoder.appendBit(1);
    try encoder.appendBit(0);
    try encoder.appendBit(1);
    try encoder.appendBit(0);
    try encoder.appendBit(1);
    try encoder.appendBit(1);
    try encoder.appendBit(1);
    try encoder.appendBit(0);

    // Should have created exactly one byte
    try std.testing.expectEqual(@as(usize, 1), encoder.encoded_data.items.len);
    try std.testing.expectEqual(@as(u8, 0b10101110), encoder.encoded_data.items[0]);

    // Append more bits to test wrapping to next byte
    try encoder.appendBit(1);
    try encoder.appendBit(1);

    // Should now have two bytes
    try std.testing.expectEqual(@as(usize, 2), encoder.encoded_data.items.len);
    try std.testing.expectEqual(@as(u8, 0b11000000), encoder.encoded_data.items[1] & 0b11000000);
}

// Test encoder and decoder with repeated symbols
test "Encode and decode repeated symbols" {
    const allocator = std.testing.allocator;
    var encoder = try Encoder.init(allocator);
    defer encoder.deinit();

    const cdf = [_]llm.CdfToken{
        .{ .token = "A", .probability = 50 },
        .{ .token = "B", .probability = 100 },
    };

    // Create a pattern of repeated symbols
    var symbols = std.ArrayList(usize).init(allocator);
    defer symbols.deinit();

    const pattern = [_]usize{ 0, 0, 0, 1, 1, 0, 0, 1, 1, 1 };
    for (pattern) |sym| {
        try symbols.append(sym);
    }

    // Encode all symbols
    for (symbols.items) |symbol| {
        encoder.encodeSymbol(&cdf, symbol);
    }

    try encoder.finish();
    const encoded = encoder.getEncoded();

    // Decode and verify
    var decoder = try Decoder.init(encoded);
    for (symbols.items) |symbol| {
        const decoded = decoder.decodeSymbol(&cdf);
        try std.testing.expectEqual(symbol, decoded);
    }
}

// Test encoding and decoding with underflow handling
test "Underflow handling" {
    const allocator = std.testing.allocator;
    var encoder = try Encoder.init(allocator);
    defer encoder.deinit();

    // This distribution is designed to trigger underflow handling
    const cdf = [_]llm.CdfToken{
        .{ .token = "A", .probability = 25 },
        .{ .token = "B", .probability = 50 },
        .{ .token = "C", .probability = 75 },
        .{ .token = "D", .probability = 100 },
    };

    // Create a sequence that should trigger underflow
    // Alternating between symbols near the middle range
    const symbols = [_]usize{ 1, 2, 1, 2, 1, 2, 1, 2, 1, 2 };
    for (symbols) |symbol| {
        encoder.encodeSymbol(&cdf, symbol);
    }

    try encoder.finish();
    const encoded = encoder.getEncoded();

    // Decode and verify
    var decoder = try Decoder.init(encoded);
    for (symbols) |symbol| {
        const decoded = decoder.decodeSymbol(&cdf);
        try std.testing.expectEqual(symbol, decoded);
    }
}

// Test encoding/decoding with many symbols to verify stability
test "Long sequence stability" {
    const allocator = std.testing.allocator;
    var encoder = try Encoder.init(allocator);
    defer encoder.deinit();

    const cdf = [_]llm.CdfToken{
        .{ .token = "A", .probability = 25 },
        .{ .token = "B", .probability = 50 },
        .{ .token = "C", .probability = 75 },
        .{ .token = "D", .probability = 100 },
    };

    // Create a larger sequence of random-like symbols
    var symbols = std.ArrayList(usize).init(allocator);
    defer symbols.deinit();

    // Use a simple pattern for reproducibility
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try symbols.append(i % 4);
    }

    // Encode all symbols
    for (symbols.items) |symbol| {
        encoder.encodeSymbol(&cdf, symbol);
    }

    try encoder.finish();
    const encoded = encoder.getEncoded();

    // Decode and verify
    var decoder = try Decoder.init(encoded);
    for (symbols.items) |symbol| {
        const decoded = decoder.decodeSymbol(&cdf);
        try std.testing.expectEqual(symbol, decoded);
    }
}
