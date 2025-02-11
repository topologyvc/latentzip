# LatentZip

                    ╔═══════════════════════════════════╗
                    ║   01110100 01101111 01110000      ║ 
                    ║   01101111 01101100 01101111      ║
                    ║   01100111 01111001               ║
                    ╚═══════════════════════════════════╝

                    [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░] 78% compressed

LatentZip is a utf8 lossless compression and decompression tool that leverages a large language models and arithmetic coding to achieve state-of-the-art compression ratios. 

The project is implemented in Zig with hooks to llama.cpp

*LatentZip is inspired by AlexBuz's project [LlamaZip](https://github.com/AlexBuz/LlamaZip)*

## Features

- Compress a file using a large language model and arithmetic coding
- Decompress previously compressed files
- Easy-to-use command-line interface

## Installation

1. Install [Zig](https://ziglang.org/download/).
2. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/latentzip.git
   cd latentzip
   ```

3. Build the project (for example, in release-safe mode):

   ```bash
   zig build
   ```

## CLI Usage

LatentZip provides a command-line interface with the following options:

| Option | Description | Optional |
|--------|-------------|-----------|
| `-h, --help` | Display help information and exit. | Yes |
| `-c, --compress <input_file>` | Compress the specified input file. | No |
| `-d, --decompress <input_file>` | Decompress the specified input file. | No |
| `-o, --output <output_file>` | Specify the output file name. | Yes |
| `-v, --verbose` | Enable verbose mode to display additional logs during compression or decompression. | Yes |
| `-m, --hf_repo <model_repo>` | Specify the Hugging Face repository name for the model to use. The default value is `unsloth/Llama-3.2-3B-Instruct-GGUF`. | Yes |

### Examples

**Compress a file:**

```bash
latentzip --compress input.txt --output compressed.lz
```

**Decompress a file:**

```bash
latentzip --decompress compressed.lz --output output.txt 
```

## Development

Contributions to this project are welcome - please feel free to submit a pull request.

## License

This project is licensed under the MIT License.
