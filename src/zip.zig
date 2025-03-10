const std = @import("std");
const llm = @import("llm.zig");
const coder = @import("coder.zig");
const Allocator = std.mem.Allocator;

/// LLM-based compression implementation using arithmetic coding guided by language model predictions
pub const LatentZip = struct {
    llm_interface: *llm.Interface,
    verbose: bool,

    /// Initialize a new LatentZip compressor with the specified language model
    ///
    /// Params:
    ///   allocator: Memory allocator for internal buffers
    ///   model_path: Path to the language model repository
    ///   verbose: Enable progress logging during operations
    ///
    /// Returns: Initialized LatentZip compressor or error if model cannot be loaded
    pub fn init(allocator: Allocator, model_path: []const u8, comptime verbose: bool) !LatentZip {
        return LatentZip{ .llm_interface = try llm.Interface.init(allocator, .{
            .hf_repo = model_path,
            .inference_mode = .llama_cpp,
        }), .verbose = verbose };
    }

    /// Compress data using LLM-guided arithmetic coding
    ///
    /// Params:
    ///   allocator: Memory allocator for compression buffers
    ///   uncompressed: Input data to compress
    ///
    /// Returns: Compressed data as byte array
    ///
    /// Error: Fails if token prediction or arithmetic coding fails
    pub fn compress(self: *const LatentZip, allocator: Allocator, uncompressed: []const u8) ![]const u8 {
        var encoder = try coder.Encoder.init(allocator);
        var i: usize = 0;
        while (i < uncompressed.len) {
            const llm_cdf = try self.llm_interface.getLlmCdf(allocator, uncompressed[0..i]);
            defer llm_cdf.deinit();
            const symbol = try findTokenIndex(llm_cdf.tokens, uncompressed, i);
            encoder.encodeSymbol(llm_cdf.tokens, symbol);
            i += llm_cdf.tokens[symbol].token.len;
            const progress = if (uncompressed.len == 0) 100 else (i * 100) / uncompressed.len;
            self.log("\rCompression Progress: {}%", .{progress});
        }
        const llm_cdf = try self.llm_interface.getLlmCdf(allocator, uncompressed[0..i]);
        defer llm_cdf.deinit();
        const symbol = try findTerminatingTokenIndex(llm_cdf.tokens);
        encoder.encodeSymbol(llm_cdf.tokens, symbol);
        try encoder.finish();
        self.log("\rCompression Progress: 100%", .{});
        return encoder.getEncoded();
    }

    /// Decompress previously compressed data
    ///
    /// Params:
    ///   allocator: Memory allocator for decompression buffers
    ///   input: Compressed data to decompress
    ///
    /// Returns: Original uncompressed data
    ///
    /// Error: Fails if data is corrupted or token prediction fails
    pub fn decompress(self: *const LatentZip, allocator: Allocator, input: []const u8) ![]u8 {
        var decoder = try coder.Decoder.init(input);
        var decompressed = std.ArrayList(u8).init(allocator);
        while (true) {
            const llm_cdf = try self.llm_interface.getLlmCdf(allocator, decompressed.items);
            defer llm_cdf.deinit();
            const symbol = decoder.decodeSymbol(llm_cdf.tokens);
            const token = llm_cdf.tokens[symbol].token;
            if (token.len == 0) break;
            try decompressed.appendSlice(token);
            self.log("\rDecompression Progress: {} bytes decompressed", .{decompressed.items.len});
        }
        return decompressed.items;
    }

    /// Find the index of the token in the LLM CDF that matches the input data
    ///
    /// Params:
    ///   llm_cdf: Token probability distribution from the LLM
    ///   uncompressed: Input data being compressed
    ///   i: Current position in the input data
    ///
    /// Returns: Index of the matching token
    ///
    /// Error: Returns error.NoTokenFound if no matching token found
    fn findTokenIndex(llm_cdf: []const llm.CdfToken, uncompressed: []const u8, i: usize) !usize {
        var symbol: usize = 0;
        for (llm_cdf) |token| {
            if (i + token.token.len <= uncompressed.len and std.mem.eql(u8, token.token, uncompressed[i .. i + token.token.len]) and token.token.len > 0) {
                return symbol;
            }
            symbol += 1;
        }
        return error.NoTokenFound;
    }

    /// Find the index of the terminating (EOS) token in the LLM CDF
    ///
    /// Params:
    ///   llm_cdf: Token probability distribution from the LLM
    ///
    /// Returns: Index of the terminating token
    ///
    /// Error: Returns error.NoTerminatingTokenFound if not found
    fn findTerminatingTokenIndex(llm_cdf: []const llm.CdfToken) !usize {
        var symbol: usize = 0;
        for (llm_cdf) |token| {
            if (token.token.len == 0) {
                return symbol;
            }
            symbol += 1;
        }
        return error.NoTerminatingTokenFound;
    }

    /// Display compression/decompression progress with spinner animation
    ///
    /// Params:
    ///   message: Format string for the log message
    ///   args: Arguments for the format string
    fn log(self: *const LatentZip, comptime message: []const u8, args: anytype) void {
        if (self.verbose) {
            const spinner = [_]u8{ '|', '/', '-', '\\' };
            const tick: usize = @intCast(@divTrunc(@mod(std.time.milliTimestamp(), 1000), 250));
            std.debug.print(message ++ " {c}", args ++ .{spinner[tick]});
        }
    }
};
