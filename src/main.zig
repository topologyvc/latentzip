const std = @import("std");
const clap = @import("clap");
const llm = @import("llm.zig");
const coder = @import("coder.zig");
const zip = @import("zip.zig");

const Allocator = std.mem.Allocator;
const stdout = std.io.getStdOut().writer();

// ---------------------------------------------------------------------
// Main entry: argument parsing and compression/decompression modes.
pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const params = comptime clap.parseParamsComptime(
        \\-h, --help                   Display this help and exit.
        \\-c, --compress <str>         Compress input file.
        \\-d, --decompress <str>       Decompress input file.
        \\-o, --output <str>           Output file.
        \\-v, --verbose                Verbose output.
        \\-m, --hf_repo <str>          Hugging Face repository name.
        \\
    );

    var diag = clap.Diagnostic{};
    var res = clap.parse(clap.Help, &params, clap.parsers.default, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        diag.report(std.io.getStdErr().writer(), err) catch {};
        return err;
    };
    defer res.deinit();

    var model: []const u8 = "unsloth/Llama-3.2-3B-Instruct-GGUF";
    if (res.args.hf_repo) |repo| {
        model = repo;
    }

    if (res.args.help != 0)
        return clap.help(std.io.getStdErr().writer(), clap.Help, &params, .{});
    if (res.args.compress) |filename| {
        const file = try std.fs.cwd().openFile(filename, .{});
        defer file.close();
        const input = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
        defer allocator.free(input);
        const latent_zip = try zip.LatentZip.init(allocator, model, true);
        const compressed = try latent_zip.compress(allocator, input);
        if (res.args.verbose != 0) {
            stdout.print("\nUncompressed byte size: {}\n", .{input.len}) catch unreachable;
            stdout.print("Compressed byte size: {}\n", .{compressed.len}) catch unreachable;
            stdout.print("Compression ratio: {}\n", .{input.len / compressed.len}) catch unreachable;
        }
        if (res.args.output) |output_filename| {
            const output_file = try std.fs.cwd().createFile(output_filename, .{});
            defer output_file.close();
            try output_file.writeAll(compressed);
        }
    }
    if (res.args.decompress) |filename| {
        const file = try std.fs.cwd().openFile(filename, .{});
        defer file.close();
        const input = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
        defer allocator.free(input);
        const latent_zip = try zip.LatentZip.init(allocator, model, true);
        const decompressed = try latent_zip.decompress(allocator, input);
        if (res.args.output) |output_filename| {
            const output_file = try std.fs.cwd().createFile(output_filename, .{});
            defer output_file.close();
            try output_file.writeAll(decompressed);
        } else {
            stdout.print("\n{s}\n", .{decompressed}) catch unreachable;
        }
    }
}
