#include "mir_dsp.h"
#include <jni.h>
#include <cstdint>
#include <exception>
#include <vector>

namespace {
mir_resampler* pointer(jlong handle) { return reinterpret_cast<mir_resampler*>(intptr_t(handle)); }
void fail(JNIEnv* env, const char* message) {
    env->ThrowNew(env->FindClass("java/lang/IllegalStateException"), message);
}
}

extern "C" JNIEXPORT jlong JNICALL
Java_nl_tudelft_ritmomaestro_dsp_StreamingResampler_nativeCreate(
    JNIEnv* env, jobject, jint source, jint target) {
    auto* result = mir_resampler_create(source, target);
    if (!result) fail(env, mir_dsp_last_error());
    return jlong(reinterpret_cast<intptr_t>(result));
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_nl_tudelft_ritmomaestro_dsp_StreamingResampler_nativeProcess(
    JNIEnv* env, jobject, jlong handle, jfloatArray input, jboolean final) {
    try {
        std::vector<float> values(env->GetArrayLength(input));
        if (!values.empty()) env->GetFloatArrayRegion(input, 0, values.size(), values.data());
        if (env->ExceptionCheck()) return nullptr;
        const float* output = nullptr;
        size_t count = 0;
        if (mir_resampler_process(pointer(handle), values.data(), values.size(), final, &output, &count)) {
            fail(env, mir_dsp_last_error());
            return nullptr;
        }
        auto result = env->NewFloatArray(count);
        if (result && count) env->SetFloatArrayRegion(result, 0, count, output);
        return result;
    } catch (const std::exception& e) { fail(env, e.what()); return nullptr; }
}

extern "C" JNIEXPORT void JNICALL
Java_nl_tudelft_ritmomaestro_dsp_StreamingResampler_nativeReset(JNIEnv* env, jobject, jlong handle) {
    if (mir_resampler_reset(pointer(handle))) fail(env, mir_dsp_last_error());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_nl_tudelft_ritmomaestro_dsp_StreamingResampler_nativeDelay(JNIEnv*, jobject, jlong handle) {
    return mir_resampler_delay(pointer(handle));
}

extern "C" JNIEXPORT void JNICALL
Java_nl_tudelft_ritmomaestro_dsp_StreamingResampler_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    mir_resampler_destroy(pointer(handle));
}
