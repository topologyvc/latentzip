/// Hugging Face model integration for LatentZip
/// Handles model downloading, caching, and file path management
const std = @import("std");
const llama = @import("llama");
const builtin = @import("builtin");

/// Structure representing a GGUF model file
/// Contains the relative filename for the model
const GgufFile = struct {
    rfilename: []const u8,
};

/// Model manifest structure for Hugging Face repositories
/// Contains metadata about the model files
const Manifest = struct {
    ggufFile: GgufFile,
};

/// Get the platform-specific cache directory for models
///
/// Params:
///   allocator: Memory allocator for path manipulation
///
/// Returns: Path to the cache directory
///
/// Error: UnsupportedPlatform if the OS is not supported
fn getCacheDir(allocator: std.mem.Allocator) ![]const u8 {
    const cache_dir = if (builtin.os.tag == .macos) blk: {
        const home = try std.process.getEnvVarOwned(allocator, "HOME");
        defer allocator.free(home);
        break :blk try std.fs.path.join(allocator, &.{ home, "Library", "Caches", "llama.cpp" });
    } else if (builtin.os.tag == .linux) blk: {
        const home = try std.process.getEnvVarOwned(allocator, "HOME");
        defer allocator.free(home);
        break :blk try std.fs.path.join(allocator, &.{ home, ".cache", "llama.cpp" });
    } else {
        return error.UnsupportedPlatform;
    };
    return cache_dir;
}

/// Get the file path for a Hugging Face model
///
/// Params:
///   allocator: Memory allocator for path manipulation
///   hf_repo: Hugging Face repository identifier (e.g., "unsloth/Llama-3.2-3B-Instruct-GGUF")
///
/// Returns: Absolute path to the model file
///
/// Error: Returns error if the model cannot be found or downloaded
pub fn getHfFilePath(allocator: std.mem.Allocator, hf_repo: []const u8) ![]const u8 {
    const hf_repo_prefix = try allocator.dupe(u8, hf_repo);
    defer allocator.free(hf_repo_prefix);
    std.mem.replaceScalar(u8, hf_repo_prefix, '/', '_');
    const manifest = try getHfManifest(allocator, hf_repo);
    const cache_dir = try getCacheDir(allocator);
    defer allocator.free(cache_dir);
    return try std.fmt.allocPrint(allocator, "{s}/{s}_{s}", .{ cache_dir, hf_repo_prefix, manifest.ggufFile.rfilename });
}

/// Fetch the manifest for a Hugging Face repository
///
/// Params:
///   allocator: Memory allocator for HTTP request
///   hf_repo: Hugging Face repository identifier
///
/// Returns: Manifest structure with model metadata
///
/// Error: Returns error if the manifest cannot be fetched or parsed
fn getHfManifest(allocator: std.mem.Allocator, hf_repo: []const u8) !Manifest {
    const host = "huggingface.co";
    const path = try std.fmt.allocPrint(allocator, "/v2/{s}/manifests/latest", .{hf_repo});
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();
    var buf: [2048]u8 = undefined;
    const url = try std.fmt.allocPrint(allocator, "https://{s}{s}", .{ host, path });
    const uri = try std.Uri.parse(url);
    var request = try client.open(.GET, uri, .{
        .server_header_buffer = &buf,
        .headers = .{
            .user_agent = .{ .override = "llama-cpp" },
        },
    });
    defer request.deinit();
    try request.send();
    try request.finish();
    try request.wait();
    const json_resp = try request.reader().readAllAlloc(allocator, std.math.maxInt(usize));
    const parsed = try std.json.parseFromSlice(Manifest, allocator, json_resp, .{ .ignore_unknown_fields = true });
    return parsed.value;
}

/// Downloads a Hugging Face repository model, downloading if not in cache
///
/// Params:
///   allocator: Memory allocator for file operations
///   hf_repo: Hugging Face repository identifier
///
/// Error: Returns error if the model cannot be downloaded
fn downloadHfRepo(allocator: std.mem.Allocator, hf_repo: []const u8) !void {
    const stdout = std.io.getStdOut().writer();
    const host = "huggingface.co";
    const manifest = try getHfManifest(allocator, hf_repo);
    const path = try std.fmt.allocPrint(allocator, "/{s}/resolve/main/{s}", .{ hf_repo, manifest.ggufFile.rfilename });
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();
    var buf: [4096]u8 = undefined;
    const url = try std.fmt.allocPrint(allocator, "https://{s}{s}", .{ host, path });
    const uri = try std.Uri.parse(url);
    var request = try client.open(.GET, uri, .{
        .server_header_buffer = &buf,
    });
    defer request.deinit();
    try request.send();
    try request.finish();
    try request.wait();
    const response_data = try request.reader().readAllAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(response_data);
    const output_path = try getHfFilePath(allocator, hf_repo);
    var output_file = try std.fs.cwd().createFile(output_path, .{
        .truncate = true,
    });
    defer output_file.close();
    try output_file.writeAll(response_data);
    try stdout.print("Downloaded file saved as {s}\n", .{output_path});
}

/// Load a Hugging Face repository model, downloading if not in cache
///
/// Params:
///   allocator: Memory allocator for file operations
///   hf_repo: Hugging Face repository identifier
///
/// Returns: Path to the loaded model file
///
/// Error: Returns error if the model cannot be loaded
pub fn loadHfRepo(allocator: std.mem.Allocator, hf_repo: []const u8) ![]const u8 {
    const output_path = try getHfFilePath(allocator, hf_repo);
    std.debug.print("Checking for model {s} at path {s}\n", .{ hf_repo, output_path });
    const fs = std.fs.cwd();
    const file = fs.openFile(output_path, .{ .mode = .read_only }) catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print("Model not found, downloading model {s}...\n", .{hf_repo});
            try downloadHfRepo(allocator, hf_repo);
            return output_path;
        }
        return err;
    };
    defer file.close();
    return output_path;
}
