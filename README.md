# LatentZip

                    ╔═══════════════════════════════════╗
                    ║   01110100 01101111 01110000      ║
                    ║   01101111 01101100 01101111      ║
                    ║   01100111 01111001               ║
                    ╚═══════════════════════════════════╝

                    [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░] 78% compressed

LatentZip is a **UTF-8 lossless compression and decompression tool** that fuses the power of large language models with  arithmetic coding to deliver state-of-the-art compression ratios. It's built in Zig with cutting-edge hooks to llama.cpp, pushing the limits of what compression can be.

*Inspired by AlexBuz's project [LlamaZip](https://github.com/AlexBuz/LlamaZip)* 

---

## Features

- ⚡ **Transformer-Powered Compression:** Compress files using a large language model and arithmetic coding.
- 🚀 **Native Llama Bindings:** Llama.cpp is linked natively into the LatentZip binary.
- 🎛 **CLI:** Command-line interface with an expandable feature set.

---

## Installation

1. **Install [Zig](https://ziglang.org/download/).**
2. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/latentzip.git
   cd latentzip
   ```

3. **Build the project:**

   For example, in release mode:

   ```bash
   zig build
   ```

---

## CLI Usage

LatentZip provides a command-line interface with the following options:

| Option                             | Description                                                                                     | Optional |
|------------------------------------|-------------------------------------------------------------------------------------------------|----------|
| `-h, --help`                      | Display help information and exit.                                                            | Yes      |
| `-c, --compress <input_file>`       | Compress the specified input file.                                                           | No       |
| `-d, --decompress <input_file>`     | Decompress the specified input file.                                                         | No       |
| `-o, --output <output_file>`        | Specify the output file name (default: stdout).                                       | Yes      |
| `-v, --verbose`                   | Enable verbose mode for additional logs.                               | Yes      |
| `-m, --hf_repo <model_repo>`        | Set the Hugging Face model repository. Default: `unsloth/Llama-3.2-3B-Instruct-GGUF`.           | Yes      |

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
