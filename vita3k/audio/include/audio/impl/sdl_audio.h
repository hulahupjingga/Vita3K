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

#pragma once

#include "../state.h"
#include <SDL3/SDL_audio.h>
#include <condition_variable>

class SDLAudioAdapter : public AudioAdapter {
private:
    // low-latency device: used for MAIN/VOICE ports, where the game needs tight audio timing
    // for interactive sound effects/gameplay audio.
    SDL_AudioDeviceID device_id_fast = 0;
    // normal-latency device: used for BGM ports (e.g. movie/cutscene audio via SceAvPlayer),
    // which don't need tight timing. Opened without the low-latency hint so it stays off the
    // MMAP fast path on platforms like Android, keeping it on the regular mixer output that
    // OS-level audio effects (equalizer, DTS/Dolby-style processing) are attached to.
    SDL_AudioDeviceID device_id_normal = 0;
    int device_buffer_samples = 0;
    SDL_AudioSpec dst_spec;

    static void SDLCALL thread_wakeup_callback(void *userdata, SDL_AudioStream *stream, int additional_amount, int total_amount);

    SDL_AudioDeviceID device_for_port_type(int port_type) const;

public:
    explicit SDLAudioAdapter(AudioState &audio_state);
    ~SDLAudioAdapter() override;

    bool init() override;
    void switch_state(const bool pause) override;
    AudioOutPortPtr open_port(int nb_channels, int freq, int nb_sample, int port_type) override;
    void audio_output(AudioOutPort &out_port, const void *buffer) override;
    void set_volume(AudioOutPort &out_port, float volume) override;
    int get_rest_sample(AudioOutPort &out_port) override;
    void wake_all_ports() override;
};

using AudioStreamPtr = std::shared_ptr<SDL_AudioStream>;

struct SDLAudioOutPort : public AudioOutPort {
    int channels = 2;
    AudioStreamPtr stream;
    SDLAudioAdapter &adapter;
    std::mutex mutex;
    std::condition_variable cond_var;
    SDLAudioOutPort(AudioStreamPtr stream, AudioAdapter &adapter)
        : stream(std::move(stream))
        , adapter(dynamic_cast<SDLAudioAdapter &>(adapter)) {}
};
