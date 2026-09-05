#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define MIR_DSP_API __declspec(dllexport)
#else
#define MIR_DSP_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Mono float32, libsoxr HQ, streaming delay in output samples. Each handle is
 * exclusively owned by its caller. Reset discards pending input/output; final
 * processing drains the filter and disallows more input until reset. Returned
 * output is owned by the handle until the next operation. No C++ exceptions
 * cross this boundary. Non-finite audio is rejected before state changes. */
typedef struct mir_resampler mir_resampler;
MIR_DSP_API uint32_t mir_dsp_abi_version(void);
MIR_DSP_API const char* mir_dsp_resampling_contract(void);
MIR_DSP_API mir_resampler* mir_resampler_create(uint32_t input_rate, uint32_t output_rate);
MIR_DSP_API const char* mir_dsp_last_error(void);
MIR_DSP_API int mir_resampler_process(mir_resampler*, const float*, size_t, int final,
                                     const float** output, size_t* output_count);
MIR_DSP_API int mir_resampler_reset(mir_resampler*);
MIR_DSP_API double mir_resampler_delay(const mir_resampler*);
MIR_DSP_API void mir_resampler_destroy(mir_resampler*);
#ifdef __cplusplus
}
#endif
