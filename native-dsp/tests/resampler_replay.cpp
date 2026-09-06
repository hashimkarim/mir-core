#include "mir_dsp.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 6) throw std::runtime_error("usage: replay input-rate output-rate chunk-samples input.f32 output.f32");
        std::unique_ptr<mir_resampler, decltype(&mir_resampler_destroy)> stream(
            mir_resampler_create(std::stoul(argv[1]), std::stoul(argv[2])), mir_resampler_destroy);
        if (!stream) throw std::runtime_error(mir_dsp_last_error());
        const size_t chunk = std::stoul(argv[3]);
        if (!chunk || chunk > 16 * 1024 * 1024) throw std::runtime_error("invalid chunk size");
        std::ifstream input(argv[4], std::ios::binary);
        std::ofstream output(argv[5], std::ios::binary);
        if (!input || !output) throw std::runtime_error("cannot open replay files");
        std::vector<float> buffer(chunk);
        while (input) {
            input.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(float));
            if (input.gcount() % sizeof(float)) throw std::runtime_error("truncated float input");
            const float* values = nullptr;
            size_t count = 0;
            if (mir_resampler_process(stream.get(), buffer.data(), input.gcount() / sizeof(float),
                                       input.eof(), &values, &count))
                throw std::runtime_error(mir_dsp_last_error());
            output.write(reinterpret_cast<const char*>(values), count * sizeof(float));
        }
        if (!output) throw std::runtime_error("cannot write replay output");
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
