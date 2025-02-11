const std = @import("std");
const llm = @import("llm.zig");
const Allocator = std.mem.Allocator;

// ---------------------------------------------------------------------
// Constants
const NUM_STATE_BITS: u8 = 52;

// ---------------------------------------------------------------------
// Base arithmetic coder functionality
const ArithmeticCoderBase = struct {
    half_range: u64,
    quarter_range: u64,
    state_mask: u64,
    low: u64,
    high: u64,

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

// ---------------------------------------------------------------------
// Encoder implementation
pub const Encoder = struct {
    base: ArithmeticCoderBase,
    encoded_data: std.ArrayList(u8),
    bit_index: u3,
    num_underflow: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator) !Encoder {
        return .{
            .base = ArithmeticCoderBase.init(),
            .encoded_data = std.ArrayList(u8).init(allocator),
            .bit_index = 0,
            .num_underflow = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Encoder) void {
        self.encoded_data.deinit();
    }

    pub fn encodeSymbol(self: *Encoder, cdf: []const llm.CdfToken, symbol: usize) void {
        ArithmeticCoderBase.update(Encoder, self, &self.base, cdf, symbol);
    }

    pub fn finish(self: *Encoder) !void {
        try self.appendBit(1);
    }

    pub fn getEncoded(self: *const Encoder) []const u8 {
        return self.encoded_data.items;
    }

    fn shift(self: *Encoder) void {
        const bit: u1 = @intCast(self.base.low >> (NUM_STATE_BITS - 1));
        self.appendBit(bit) catch unreachable;
        var i: usize = 0;
        while (i < self.num_underflow) : (i += 1) {
            self.appendBit(bit ^ 1) catch unreachable;
        }
        self.num_underflow = 0;
    }

    fn underflow(self: *Encoder) void {
        self.num_underflow += 1;
    }

    fn appendBit(self: *Encoder, bit: u1) !void {
        if (self.bit_index == 0) {
            try self.encoded_data.append(0);
        }
        self.encoded_data.items[self.encoded_data.items.len - 1] |= @as(u8, bit) << (7 - self.bit_index);
        self.bit_index = @intCast((@as(u4, self.bit_index) + 1) % 8);
    }
};

// ---------------------------------------------------------------------
// Decoder implementation
pub const Decoder = struct {
    base: ArithmeticCoderBase,
    input: []const u8,
    byte_index: usize,
    bit_index: u3,
    code: u64,

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

    fn shift(self: *Decoder) void {
        self.code = ((self.code << 1) & self.base.state_mask) | self.readCodeBit();
    }

    fn underflow(self: *Decoder) void {
        self.code = (self.code & self.base.half_range) |
            ((self.code << 1) & (self.base.state_mask >> 1)) |
            self.readCodeBit();
    }

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
