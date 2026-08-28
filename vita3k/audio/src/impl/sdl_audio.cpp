// Vita3K emulator project
// Copyright (C) 2026 Vita3K team
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#include "audio/impl/sdl_audio.h"
#include "util/log.h"
#include <SDL3/SDL_audio.h>
#include <SDL3/SDL_hints.h>
#include <cstdlib>
#include <string>

#ifdef __ANDROID__
#include <SDL3/SDL_system.h>
#include <jni.h>
#endif

#define SDL_CHECK_EXT(condition, ret)                         \
    do {                                                      \
        if (!(condition)) {                                   \
            LOG_ERROR("SDL audio error: {}", SDL_GetError()); \
            return ret;                                       \
        }                                                     \
    } while (0)

#define SDL_CHECK(f_call) SDL_CHECK_EXT(f_call, {})
#define SDL_CHECK_VOID(f_call) SDL_CHECK_EXT(f_call, )
#define SDL_CHECK_NEG(f_call) SDL_CHECK_EXT((f_call) >= 0, {})

// How many device-buffer-periods of audio we let queue up in the stream before the
// feeder thread refills. This is pure added latency on top of the mixer's own period,
// so it's kept as tight as possible. 2x is a starting point for the normal (non-low-latency)
// path -- if you see underrun crackle on lower-end devices, raise it in small steps (2.5, 3)
// rather than jumping back to 4.
static constexpr int kQueueThresholdMultiplier = 2;

static int get_threshold_samples(const int device_buffer_samples) {
    return kQueueThresholdMultiplier * device_buffer_samples;
}

#ifdef __ANDROID__
// Ask Android for its reported optimal output buffer size (AudioManager.PROPERTY_OUTPUT_FRAMES_PER_BUFFER)
// so we can hint SDL/AAudio to use that instead of whatever conservative default the OEM's
// normal-mixer path would otherwise pick. Falls back to 0 (== "let SDL decide") on any failure.
static int get_android_optimal_frames_per_buffer() {
    JNIEnv *env = static_cast<JNIEnv *>(SDL_GetAndroidJNIEnv());
    jobject activity = static_cast<jobject>(SDL_GetAndroidActivity());
    if (!env || !activity)
        return 0;

    jclass activity_class = env->GetObjectClass(activity);
    jmethodID get_system_service = env->GetMethodID(activity_class, "getSystemService", "(Ljava/lang/String;)Ljava/lang/Object;");
    jclass context_class = env->FindClass("android/content/Context");
    jfieldID audio_service_field = env->GetStaticFieldID(context_class, "AUDIO_SERVICE", "Ljava/lang/String;");
    jstring audio_service = static_cast<jstring>(env->GetStaticObjectField(context_class, audio_service_field));
    jobject audio_manager = env->CallObjectMethod(activity, get_system_service, audio_service);
    if (!audio_manager || env->ExceptionCheck()) {
        env->ExceptionClear();
        return 0;
    }

    jclass audio_manager_class = env->GetObjectClass(audio_manager);
    jmethodID get_property = env->GetMethodID(audio_manager_class, "getProperty", "(Ljava/lang/String;)Ljava/lang/String;");
    jfieldID prop_field = env->GetStaticFieldID(audio_manager_class, "PROPERTY_OUTPUT_FRAMES_PER_BUFFER", "Ljava/lang/String;");
    jstring prop_name = static_cast<jstring>(env->GetStaticObjectField(audio_manager_class, prop_field));
    jstring result = static_cast<jstring>(env->CallObjectMethod(audio_manager, get_property, prop_name));
    if (!result || env->ExceptionCheck()) {
        env->ExceptionClear();
        return 0;
    }

    const char *chars = env->GetStringUTFChars(result, nullptr);
    const int frames = chars ? std::atoi(chars) : 0;
    if (chars)
        env->ReleaseStringUTFChars(result, chars);
    return frames;
}
#endif

void SDLCALL SDLAudioAdapter::thread_wakeup_callback(void *userdata, SDL_AudioStream *stream, int additional_amount, int total_amount) {
    assert(userdata != nullptr);
    assert(stream != nullptr);
    SDLAudioOutPort *port = static_cast<SDLAudioOutPort *>(userdata);
    const int samples_available = port->adapter.get_rest_sample(*port);
    if (samples_available < get_threshold_samples(port->adapter.device_buffer_samples) || additional_amount > 0) {
        port->cond_var.notify_one();
    }
}

SDLAudioAdapter::SDLAudioAdapter(AudioState &audio_state)
    : AudioAdapter(audio_state) {}

SDLAudioAdapter::~SDLAudioAdapter() {
    if (device_id > 0)
        SDL_CloseAudioDevice(device_id);
}

bool SDLAudioAdapter::init() {
    // Disabling SDL_HINT_ANDROID_LOW_LATENCY_AUDIO (defaults to true) keeps SDL's AAudio
    // backend from requesting AAUDIO_PERFORMANCE_MODE_LOW_LATENCY, so every port -- gameplay
    // audio included -- stays on Android's normal mixer thread instead of the fast/MMAP path.
    // That's the path OS-level audio effects (equalizer, DTS/Dolby-style processing) are
    // applied on. Trade-off: gameplay audio latency goes from AAudio's fast/MMAP path
    // (~2-3ms mixer period) to the normal mixer thread's (~20ms), which can be noticeable in
    // latency-sensitive games.
    SDL_SetHint(SDL_HINT_ANDROID_LOW_LATENCY_AUDIO, "0");

#ifdef __ANDROID__
    // Even on the normal mixer path, some OEMs default to a larger buffer than the device
    // actually needs. Hint SDL to the device's own reported optimal frame count instead of
    // leaving it to whatever conservative default AAudio picks for non-low-latency streams.
    const int optimal_frames = get_android_optimal_frames_per_buffer();
    if (optimal_frames > 0) {
        SDL_SetHint(SDL_HINT_AUDIO_DEVICE_SAMPLE_FRAMES, std::to_string(optimal_frames).c_str());
        LOG_INFO("SDL audio: requesting Android-reported optimal buffer size of {} frames", optimal_frames);
    }
#endif

    device_id = SDL_OpenAudioDevice(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, nullptr);
    SDL_CHECK_EXT(device_id > 0, false);
    return true;
}

void SDLAudioAdapter::switch_state(const bool pause) {
    if (pause)
        SDL_CHECK_VOID(SDL_PauseAudioDevice(device_id));
    else
        SDL_CHECK_VOID(SDL_ResumeAudioDevice(device_id));
}

AudioOutPortPtr SDLAudioAdapter::open_port(int nb_channels, int freq, int nb_sample, int port_type) {
    // every port shares the same (non-low-latency) device now -- see the comment in init().
    (void)port_type;

    SDL_AudioSpec src_spec = {
        .format = SDL_AUDIO_S16LE,
        .channels = nb_channels,
        .freq = freq
    };
    SDL_CHECK(SDL_GetAudioDeviceFormat(device_id, &dst_spec, &device_buffer_samples));
    const AudioStreamPtr stream(SDL_CreateAudioStream(&src_spec, &dst_spec), SDL_DestroyAudioStream);
    SDL_CHECK(stream);
    SDL_CHECK(SDL_BindAudioStream(device_id, stream.get()));
    auto port = std::make_shared<SDLAudioOutPort>(stream, *this);
    SDL_CHECK(SDL_SetAudioStreamGetCallback(stream.get(), SDLAudioAdapter::thread_wakeup_callback, port.get()));
    port->channels = nb_channels;
    port->len_microseconds = (nb_sample * 1'000'000ULL) / freq;
    port->len_bytes = nb_sample * nb_channels * sizeof(int16_t);
    switch_state(false);
    return port;
}

void SDLAudioAdapter::audio_output(AudioOutPort &out_port, const void *buffer) {
    if (out_port.stopping)
        return;

    //  Put audio to the port's stream and see how much is left to play.
    SDLAudioOutPort &port = static_cast<SDLAudioOutPort &>(out_port);
    // If there's lots of audio left to play, stop this thread.
    // The audio callback will wake it up later when it's running out of data.
    const int samples_available = get_rest_sample(port);
    if (samples_available > get_threshold_samples(device_buffer_samples)) {
        std::unique_lock<std::mutex> lock(port.mutex);
        // Was len_microseconds * 2. A tighter sleep cap means the feeder catches the
        // threshold crossing sooner instead of oversleeping past it, which is what lets
        // the queue threshold above stay low without risking underrun on wakeup jitter.
        port.cond_var.wait_for(lock, std::chrono::microseconds(port.len_microseconds));
        if (out_port.stopping)
            return;
    }
    SDL_CHECK_VOID(SDL_PutAudioStreamData(port.stream.get(), buffer, out_port.len_bytes));
}

void SDLAudioAdapter::set_volume(AudioOutPort &out_port, float volume) {
    SDL_CHECK_VOID(SDL_SetAudioStreamGain(static_cast<SDLAudioOutPort &>(out_port).stream.get(), volume));
}

int SDLAudioAdapter::get_rest_sample(AudioOutPort &out_port) {
    auto &port = static_cast<SDLAudioOutPort &>(out_port);
    const int bytes_available = SDL_GetAudioStreamAvailable(port.stream.get());
    SDL_CHECK_NEG(bytes_available);
    // we have the number of bytes left, we can convert it back to the number of samples left
    return bytes_available / SDL_AUDIO_FRAMESIZE(dst_spec);
}

void SDLAudioAdapter::wake_all_ports() {
    for (auto &[_, port_ptr] : state.out_ports) {
        auto &port = static_cast<SDLAudioOutPort &>(*port_ptr);
        {
            std::lock_guard<std::mutex> lock(port.mutex);
        }
        port.cond_var.notify_all();
    }
}
