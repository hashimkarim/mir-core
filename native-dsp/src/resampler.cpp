#include "mir_dsp.h"
#include "soxr.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
void check(soxr_error_t error) {
    if (error) throw std::runtime_error(error);
}
}

struct mir_resampler {
    soxr_t state = nullptr;
    double ratio = 1;
    size_t division = 48000;
    bool ended = false;
    std::vector<float> output;
    ~mir_resampler() { soxr_delete(state); }
};

uint32_t mir_dsp_abi_version() { return 1; }
const char* mir_dsp_resampling_contract() { return "mir.soxr-hq/mono-f32-v1"; }
const char* mir_dsp_last_error() { return last_error.c_str(); }

mir_resampler* mir_resampler_create(uint32_t input_rate, uint32_t output_rate) {
    try {
        require(input_rate >= 1000 && input_rate <= 768000 &&
                output_rate >= 1000 && output_rate <= 768000, "sample rate outside 1000..768000 Hz");
        auto result = std::make_unique<mir_resampler>();
        result->ratio = double(output_rate) / input_rate;
        result->division = size_t(std::max(1000.0, 48000.0 / result->ratio));
        auto io = soxr_io_spec(SOXR_FLOAT32_I, SOXR_FLOAT32_I);
        auto quality = soxr_quality_spec(SOXR_HQ, 0);
        soxr_error_t error = nullptr;
        result->state = soxr_create(input_rate, output_rate, 1, &error, &io, &quality, nullptr);
        check(error);
        require(result->state != nullptr, "unable to allocate resampler");
        return result.release();
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}

int mir_resampler_process(mir_resampler* self, const float* input, size_t count, int final,
                          const float** output, size_t* output_count) {
    if (output) *output = nullptr;
    if (output_count) *output_count = 0;
    try {
        require(self && output && output_count, "null resampler/output argument");
        require(!self->ended, "input after final block; reset required");
        require(input || count == 0, "null audio input");
        require(final == 0 || final == 1, "final must be 0 or 1");
        require(count <= 16 * 1024 * 1024, "input block too large");
        for (size_t i = 0; i < count; ++i)
            require(std::isfinite(input[i]), "non-finite audio input");
        // Match Python-SoXR's allocation and long-input segmentation. The +1
        // leaves enough room for final rounding and drains without truncation.
        const double capacity = soxr_delay(self->state) + count * self->ratio + 1;
        require(capacity <= 32 * 1024 * 1024, "output block too large");
        self->output.resize(size_t(capacity));
        size_t produced = 0;
        for (size_t i = 0; i < count; i += self->division) {
            size_t written = 0;
            check(soxr_process(self->state, input + i, std::min(self->division, count - i),
                               nullptr, self->output.data() + produced,
                               self->output.size() - produced, &written));
            produced += written;
        }
        if (final) {
            size_t written = 0;
            check(soxr_process(self->state, nullptr, 0, nullptr,
                               self->output.data() + produced,
                               self->output.size() - produced, &written));
            produced += written;
            self->ended = true;
        }
        self->output.resize(produced);
        *output = self->output.data();
        *output_count = produced;
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

int mir_resampler_reset(mir_resampler* self) {
    try {
        require(self != nullptr, "null resampler");
        check(soxr_clear(self->state));
        self->ended = false;
        self->output.clear();
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}

double mir_resampler_delay(const mir_resampler* self) {
    return self ? soxr_delay(self->state) : std::numeric_limits<double>::quiet_NaN();
}
void mir_resampler_destroy(mir_resampler* self) { delete self; }
