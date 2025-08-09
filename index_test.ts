// log-mel spectrogram (128 x N frames)
// Written by Kenta Iwasaki. All rights reserved.
// 2025-08-10

import { test, expect } from "bun:test";
import { loadWavFile } from "./wav";
import { logMelSpectrogram } from ".";

test("correctly loads a wav file", async () => {
  const { data, sampleRate, numChannels, bitsPerSample, numSamples } =
    await loadWavFile("./samples_jfk.wav");
  expect(data).toBeDefined();
  expect(sampleRate).toBe(16000);
  expect(numChannels).toBe(1);
  expect(bitsPerSample).toBe(16);
  expect(numSamples).toBe(176000);
});

test("produces correct log mel spectrogram", async () => {
  const json: number[][] = await Bun.file("mel_example.json").json();
  const expected = new Float32Array(json.flat());

  const { data } = await loadWavFile("./samples_jfk.wav");
  const actual = logMelSpectrogram(data, 30 * 16_000);

  expect(actual.length).toBe(expected.length);

  let maxDiff = 0;
  let maxDiffIdx = -1;
  const diffs: number[] = [];

  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i]! - expected[i]!);
    diffs.push(diff);
    if (diff > maxDiff) {
      maxDiff = diff;
      maxDiffIdx = i;
    }
  }

  const mel_bin = maxDiffIdx % 128;
  const frame = Math.floor(maxDiffIdx / 128);
  console.log(
    `\nMax difference: ${maxDiff} at index ${maxDiffIdx} (mel_bin=${mel_bin}, frame=${frame})`
  );
  console.log(
    `Actual value: ${actual[maxDiffIdx]}, Expected: ${expected[maxDiffIdx]}`
  );
  console.log(
    `Mean difference: ${diffs.reduce((a, b) => a + b) / diffs.length}`
  );
  console.log(
    `Values with diff > 0.0001: ${diffs.filter((d) => d > 0.0001).length} (${(
      (diffs.filter((d) => d > 0.0001).length / diffs.length) *
      100
    ).toFixed(1)}%)`
  );

  // Check if there's a pattern in the differences
  const firstFrameDiffs = diffs.slice(0, 128).reduce((a, b) => a + b) / 128;
  const lastFrameDiffs = diffs.slice(-128).reduce((a, b) => a + b) / 128;
  console.log(`\nFirst frame mean diff: ${firstFrameDiffs}`);
  console.log(`Last frame mean diff: ${lastFrameDiffs}`);

  // Verify close match to Python implementation (tightened for new FFT)
  expect(maxDiff).toBeLessThan(0.00005); // Max difference < 5e-5
  expect(diffs.reduce((a, b) => a + b) / diffs.length).toBeLessThan(0.0000001); // Mean diff < 1e-7

  const sorted = [...diffs].sort((a, b) => a - b);
  const percentile = (p: number) => {
    const idx = (sorted.length - 1) * p;
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    const h = idx - lo;
    return sorted[lo]! * (1 - h) + sorted[hi]! * h;
  };
  const p99 = percentile(0.99);
  const p999 = percentile(0.999);
  console.log(`p99=${p99}, p999=${p999}`);
  expect(p99).toBeLessThan(1e-6);
  expect(p999).toBeLessThan(5e-6);
});
