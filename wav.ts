// log-mel spectrogram (128 x N frames)
// Written by Kenta Iwasaki. All rights reserved.
// 2025-08-10

import fs from "node:fs";

export async function loadWavFile(filename: string): Promise<{
  data: Float32Array;
  sampleRate: number;
  numChannels: number;
  bitsPerSample: number;
  numSamples: number;
}> {
  const buffer = await fs.promises.readFile(filename);

  const riffHeader = buffer.toString("ascii", 0, 4);
  if (riffHeader !== "RIFF") throw new Error("Not a valid WAV file");

  const waveHeader = buffer.toString("ascii", 8, 12);
  if (waveHeader !== "WAVE") throw new Error("Not a valid WAV file");

  let offset = 12;
  let sampleRate = 0;
  let numChannels = 0;
  let bitsPerSample = 0;
  let dataOffset = 0;
  let dataSize = 0;

  while (offset < buffer.length) {
    const chunkId = buffer.toString("ascii", offset, offset + 4);
    const chunkSize = buffer.readUInt32LE(offset + 4);

    if (chunkId === "fmt ") {
      const audioFormat = buffer.readUInt16LE(offset + 8);
      if (audioFormat !== 1) throw new Error("Only PCM format supported");

      numChannels = buffer.readUInt16LE(offset + 10);
      sampleRate = buffer.readUInt32LE(offset + 12);
      bitsPerSample = buffer.readUInt16LE(offset + 22);
    } else if (chunkId === "data") {
      dataOffset = offset + 8;
      dataSize = chunkSize;
      break;
    }

    offset += 8 + chunkSize;
  }

  if (dataOffset === 0) throw new Error("No data chunk found");

  const bytesPerSample = bitsPerSample / 8;
  const numSamples = dataSize / bytesPerSample;
  const audio = new Float32Array(numSamples);

  for (let i = 0; i < numSamples; i++) {
    const sampleOffset = dataOffset + i * bytesPerSample;

    if (bitsPerSample === 16) {
      audio[i] = buffer.readInt16LE(sampleOffset) / 32768.0;
    } else if (bitsPerSample === 32) {
      audio[i] = buffer.readFloatLE(sampleOffset);
    } else {
      throw new Error(`Unsupported bits per sample: ${bitsPerSample}`);
    }
  }

  // Convert to mono if stereo

  if (numChannels >= 2) {
    const monoAudio = new Float32Array(numSamples / numChannels);
    for (let i = 0; i < numSamples / numChannels; i++) {
      let sum = 0;
      for (let j = 0; j < numChannels; j++) {
        sum += audio[i * numChannels + j]!;
      }
      monoAudio[i] = sum / numChannels;
    }

    return {
      data: monoAudio,
      sampleRate,
      numChannels: 1,
      bitsPerSample,
      numSamples: numSamples / numChannels,
    };
  }

  return {
    data: audio,
    sampleRate,
    numChannels,
    bitsPerSample,
    numSamples,
  };
}
