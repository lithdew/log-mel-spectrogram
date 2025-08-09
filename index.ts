// log-mel spectrogram (128 x N frames)
// Written by Kenta Iwasaki. All rights reserved.
// 2025-08-10

import { dlopen, FFIType, suffix } from "bun:ffi";

export const lib = dlopen(`./libmel.${suffix}`, {
  log_mel_spectrogram: {
    args: [FFIType.ptr, FFIType.i32, FFIType.i32, FFIType.ptr],
    returns: FFIType.void,
  },
  init_globals: { args: [], returns: FFIType.void },
});

lib.symbols.init_globals();

/**
 * Computes the log-Mel spectrogram of a signal.
 * @param audio The input signal as a Float32Array.
 * @param padTail The number of zero samples to pad at the end of the signal.
 * @returns A Float32Array containing the flattened spectrogram data.
 */
export function logMelSpectrogram(
  audio: Float32Array,
  padTail = 0
): Float32Array {
  const numFrames = ((audio.length + padTail) / 160) | 0;
  const result = new Float32Array(128 * numFrames);
  lib.symbols.log_mel_spectrogram(audio, audio.length, padTail, result);
  return result;
}
