/// Language model integration for token probability prediction
/// Provides interfaces to work with language models for compression
const std = @import("std");
const llama = @import("llama");
const hf = @import("hf.zig");

const Allocator = std.mem.Allocator;

/// Inner structure for token probability representation in LLama.cpp
/// Contains detailed information about a single token prediction
const LlamaCompletionProbabilityInner = struct {
    id: u32,
    token: []const u8,
    bytes: []const u8,
    logprob: f32,
};

/// Completion probability structure for LLama.cpp model responses
/// Includes the token, its log probability, and related top predictions
const LlamaCompletionProbability = struct {
    id: u32,
    token: []const u8,
    bytes: []const u8,
    logprob: f32,
    top_logprobs: []LlamaCompletionProbabilityInner,
};

/// Container for LLama.cpp completion probabilities
const LlamaData = struct {
    completion_probabilities: []LlamaCompletionProbability,
};

/// Intermediate representation of logit values during token probability calculation
const IntermediateLogit = struct {
    logit: f32,
    token_idx: llama.Token,
};

/// Token representation with cumulative distribution function probability
/// Used for arithmetic coding in compression/decompression
pub const CdfToken = struct {
    token: []const u8,
    probability: u64,
};

/// Collection of CDF tokens with memory management capabilities
/// Provides ownership and cleanup of token arrays
pub const CdfTokens = struct {
    tokens: []CdfToken,
    allocator: Allocator,

    /// Initialize a new CdfTokens collection
    ///
    /// Params:
    ///   allocator: Memory allocator for tokens management
    ///   tokens: Array of CdfToken structures
    ///
    /// Returns: Pointer to initialized CdfTokens structure
    pub fn init(allocator: Allocator, tokens: []CdfToken) !*CdfTokens {
        const self = try allocator.create(CdfTokens);
        self.* = .{ .tokens = tokens, .allocator = allocator };
        return self;
    }

    /// Clean up all allocated memory for tokens
    pub fn deinit(self: *@This()) void {
        for (self.tokens) |token| {
            self.allocator.free(token.token);
        }
        self.allocator.free(self.tokens);
        self.allocator.destroy(self);
    }
};

/// Server interface for LLama.cpp model interactions
const LlamaCppServer = struct {
    /// Call the LLama.cpp server for completion probability predictions
    ///
    /// Params:
    ///   allocator: Memory allocator for response handling
    ///   prompt: Input prompt for completion probability prediction
    ///
    /// Returns: Pointer to LlamaData structure containing completion probabilities
    fn callLlamaCpp(allocator: Allocator, prompt: []const u8) !LlamaData {
        const data = .{
            .prompt = prompt,
            .n_predict = 1,
            .n_probs = 128000,
        };
        var json_string = std.ArrayList(u8).init(allocator);
        defer json_string.deinit();
        try std.json.stringify(data, .{}, json_string.writer());
        var client = std.http.Client{ .allocator = allocator };
        defer client.deinit();
        var buf: [2048]u8 = undefined;
        const uri = try std.Uri.parse("http://localhost:8080/completion");
        var request = try client.open(.POST, uri, .{
            .server_header_buffer = &buf,
        });
        defer request.deinit();
        request.transfer_encoding = .chunked;
        try request.send();
        try request.writeAll(json_string.items);
        try request.finish();
        try request.wait();
        const json_resp = try request.reader().readAllAlloc(allocator, std.math.maxInt(usize));
        const llama_data = try std.json.parseFromSlice(LlamaData, allocator, json_resp, .{ .ignore_unknown_fields = true });
        return llama_data.value;
    }

    /// Convert log probabilities to cumulative distribution function (CDF) values
    ///
    /// Params:
    ///   allocator: Memory allocator for CDF calculation
    ///   llama_data: Pointer to LlamaData structure containing completion probabilities
    ///
    /// Returns: Array of CDF values
    fn logprobToCdf(allocator: Allocator, llama_data: *const LlamaData) ![]u64 {
        const scale: u64 = 1000;
        var cdf = std.ArrayList(u64).init(allocator);
        const max_logprob = llama_data.completion_probabilities[0].top_logprobs[0].logprob;
        var cumlative_prob: u64 = 0;
        for (llama_data.completion_probabilities) |completion_probability| {
            for (completion_probability.top_logprobs) |top_logprob| {
                const prob: u64 = @intFromFloat(std.math.exp(top_logprob.logprob - max_logprob) * scale);
                cumlative_prob += prob + 1;
                try cdf.append(cumlative_prob);
            }
        }
        return cdf.items;
    }

    /// Get the CDF tokens for a given prompt using the LLama.cpp server
    ///
    /// Params:
    ///   allocator: Memory allocator for response handling
    ///   prompt: Input prompt for completion probability prediction
    ///
    /// Returns: Pointer to CdfTokens structure containing CDF tokens
    pub fn getLlmCdf(allocator: Allocator, prompt: []const u8) !*CdfTokens {
        const llama_data = try LlamaCppServer.callLlamaCpp(allocator, prompt);
        const cdf = try LlamaCppServer.logprobToCdf(allocator, &llama_data);
        const entries = try allocator.alloc(CdfToken, cdf.len);
        for (0..cdf.len) |i| {
            entries[i] = .{ .token = llama_data.completion_probabilities[0].top_logprobs[i].token, .probability = cdf[i] };
        }
        return try CdfTokens.init(allocator, entries);
    }
};

/// Comparison function for sorting IntermediateLogit structures
fn compare_logits(_: void, a: IntermediateLogit, b: IntermediateLogit) bool {
    return a.logit > b.logit;
}

/// LLama.cpp model interface for token probability prediction
const LlamaCpp = struct {
    model: *llama.Model,
    sampler: *llama.Sampler,
    vocab: *llama.Vocab,
    ctx: *llama.Context,
    batch: *llama.Batch,

    /// Initialize the LLama.cpp model
    ///
    /// Params:
    ///   allocator: Memory allocator for model initialization
    ///   hf_repo: Path to the Hugging Face repository for model loading
    ///
    /// Returns: Pointer to initialized LlamaCpp structure
    pub fn init(allocator: Allocator, hf_repo: []const u8) !*LlamaCpp {
        const self = try allocator.create(LlamaCpp);
        const path = try hf.loadHfRepo(allocator, hf_repo);
        const c_path: [:0]const u8 = try std.mem.Allocator.dupeZ(allocator, u8, path);
        defer allocator.free(path);
        defer allocator.free(c_path);
        llama.Backend.init();
        self.model = try llama.Model.initFromFile(c_path.ptr, llama.Model.defaultParams());
        var sampler = llama.Sampler.initChain(.{ .no_perf = false });
        sampler.add(llama.Sampler.initGreedy());
        self.sampler = sampler;
        var cparams = llama.Context.defaultParams();
        const cpu_threads = try std.Thread.getCpuCount(); // logical cpu cores
        cparams.n_threads = @intCast(@min(cpu_threads, 4));
        cparams.n_threads_batch = @intCast(cpu_threads / 2);
        cparams.no_perf = false;
        self.ctx = try llama.Context.initWithModel(self.model, cparams);
        self.batch = try allocator.create(llama.Batch);
        self.batch.* = llama.Batch.init(@intCast(self.ctx.nBatch()), 0, 1);
        return self;
    }

    /// Clean up the LLama.cpp model
    pub fn deinit(self: *LlamaCpp) void {
        llama.Backend.deinit();
        self.model.?.deinit();
        self.sampler.?.deinit();
        self.ctx.?.deinit();
        self.batch.?.deinit();
    }

    /// Get the CDF tokens for a given prompt using the LLama.cpp model
    ///
    /// Params:
    ///   allocator: Memory allocator for response handling
    ///   prompt: Input prompt for completion probability prediction
    ///
    /// Returns: Pointer to CdfTokens structure containing CDF tokens
    pub fn getLlmCdf(self: *LlamaCpp, allocator: Allocator, prompt: []const u8) !*CdfTokens {
        if (prompt.len == 0) {
            return self.zeroPrompt(allocator);
        }
        self.batch.clear();
        const vocab = self.model.vocab() orelse unreachable;
        var tokenizer = llama.Tokenizer.init(allocator);
        defer tokenizer.deinit();
        try tokenizer.tokenize(vocab, prompt, false, true);
        var detokenizer = llama.Detokenizer.init(allocator);
        defer detokenizer.deinit();
        const start_index = tokenizer.getTokens().len - 1;
        const batch_tokens = tokenizer.getTokens()[start_index..];
        self.batch.add(batch_tokens[0], @intCast(start_index), &[_]llama.SeqId{0}, true);
        self.batch.decode(self.ctx) catch |err| {
            if (err == error.NoKvSlotWarning) {
                self.ctx.kvCacheClear();
            } else {
                return err;
            }
        };
        const logits = self.ctx.getLogitsIth(@intCast(batch_tokens.len - 1));
        const n_vocab: usize = @intCast(vocab.nVocab());
        const intermediate_entries = try allocator.alloc(IntermediateLogit, n_vocab);
        defer allocator.free(intermediate_entries);
        for (0..n_vocab) |i| {
            intermediate_entries[i] = .{ .token_idx = @intCast(i), .logit = std.math.exp(logits[i]) };
        }
        std.mem.sort(IntermediateLogit, intermediate_entries, {}, compare_logits);
        const max_logit = intermediate_entries[0].logit;
        var running_prob: u64 = 0;
        const entries = try allocator.alloc(CdfToken, n_vocab);
        for (0..n_vocab) |i| {
            var token_str = try detokenizer.detokenize(vocab, intermediate_entries[i].token_idx);
            if (vocab.isEog(intermediate_entries[i].token_idx)) {
                token_str = "";
            }
            const token_str_copy = try allocator.dupe(u8, token_str);
            running_prob += @intFromFloat((intermediate_entries[i].logit / max_logit * 100000) + 1);
            entries[i] = .{ .token = token_str_copy, .probability = running_prob };
            detokenizer.clearRetainingCapacity();
        }
        return try CdfTokens.init(allocator, entries);
    }

    /// Handle zero-length prompts by returning a default CDF token set
    fn zeroPrompt(_: *LlamaCpp, allocator: Allocator) !*CdfTokens {
        const entries = try allocator.alloc(CdfToken, 256);
        var i: u8 = 1;
        while (i < 255) : (i += 1) {
            const char_str = try allocator.dupe(u8, &[_]u8{i});
            entries[i] = .{ .token = char_str, .probability = i + 1 };
        }
        const empty_str = try allocator.dupe(u8, "");
        entries[255] = .{ .token = empty_str, .probability = 256 };
        entries[0] = .{ .token = "", .probability = 1 };
        return try CdfTokens.init(allocator, entries);
    }
};

/// Configuration structure for interface initialization
pub const InterfaceConfig = struct {
    hf_repo: []const u8,
    inference_mode: enum {
        llama_cpp_server,
        llama_cpp,
    },
};

/// Interface for language model interactions
pub const Interface = struct {
    config: InterfaceConfig,
    llama_cpp: ?*LlamaCpp,

    /// Initialize the interface
    ///
    /// Params:
    ///   allocator: Memory allocator for interface initialization
    ///   config: Interface configuration structure
    ///
    /// Returns: Pointer to initialized Interface structure
    pub fn init(allocator: Allocator, config: InterfaceConfig) !*Interface {
        const self = try allocator.create(Interface);
        self.* = .{ .config = config, .llama_cpp = null };
        if (config.inference_mode == .llama_cpp) {
            self.llama_cpp = try LlamaCpp.init(allocator, config.hf_repo);
        }
        return self;
    }

    /// Clean up the interface
    pub fn deinit(self: *Interface) void {
        if (self.config.inference_mode == .llama_cpp) {
            self.llama_cpp.?.deinit();
        }
    }

    /// Get the CDF tokens for a given prompt using the interface
    ///
    /// Params:
    ///   allocator: Memory allocator for response handling
    ///   prompt: Input prompt for completion probability prediction
    ///
    /// Returns: Pointer to CdfTokens structure containing CDF tokens
    pub fn getLlmCdf(self: *Interface, allocator: Allocator, prompt: []const u8) !*CdfTokens {
        switch (self.config.inference_mode) {
            .llama_cpp_server => return LlamaCppServer.getLlmCdf(allocator, prompt),
            .llama_cpp => return self.llama_cpp.?.getLlmCdf(allocator, prompt),
        }
    }
};
