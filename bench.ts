// log-mel spectrogram (128 x N frames)
// Written by Kenta Iwasaki. All rights reserved.
// 2025-08-10

import os from "node:os";
import { basename } from "node:path";
import { lib, logMelSpectrogram } from "./index.ts";
import { loadWavFile } from "./wav";

type BenchOptions = {
  file: string;
  padTailSeconds: number; // seconds of zero-padding
  warmup: number; // number of warmup iterations
  iterations: number; // number of measured iterations
  runs: number; // number of distinct runs (report best & median)
  useApiWrapper: boolean; // true to call logMelSpectrogram, false to call FFI symbol
  reuseOutput: boolean; // true to reuse a preallocated output buffer
};

function parseArgs(argv: string[]): BenchOptions {
  const defaults: BenchOptions = {
    file: "samples_jfk.wav",
    padTailSeconds: 30,
    warmup: 5,
    iterations: 30,
    runs: 1,
    useApiWrapper: false,
    reuseOutput: true,
  };

  const args = new Map<string, string | boolean>();
  for (let i = 2; i < argv.length; i++) {
    const token = argv[i]!;
    if (token.startsWith("--")) {
      const [key, rawVal]: [string, string | undefined] = token
        .slice(2)
        .split("=", 2) as [string, string | undefined];
      if (rawVal === undefined) {
        args.set(key, true);
      } else {
        args.set(key, rawVal);
      }
    }
  }

  const file = (args.get("file") as string) ?? defaults.file;
  const padTailSeconds = args.has("pad")
    ? Number(args.get("pad"))
    : args.has("padTailSeconds")
    ? Number(args.get("padTailSeconds"))
    : defaults.padTailSeconds;
  const warmup = args.has("warmup")
    ? Number(args.get("warmup"))
    : defaults.warmup;
  const iterations = args.has("iterations")
    ? Number(args.get("iterations"))
    : defaults.iterations;
  const runs = args.has("runs") ? Number(args.get("runs")) : defaults.runs;
  const useApiWrapper = args.has("api") ? true : defaults.useApiWrapper;
  const reuseOutput = args.has("no-prealloc") ? false : defaults.reuseOutput;

  return {
    file,
    padTailSeconds: Number.isFinite(padTailSeconds)
      ? padTailSeconds
      : defaults.padTailSeconds,
    warmup: Number.isFinite(warmup) ? warmup : defaults.warmup,
    iterations: Number.isFinite(iterations) ? iterations : defaults.iterations,
    runs: Number.isFinite(runs) ? runs : defaults.runs,
    useApiWrapper,
    reuseOutput,
  };
}

function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"] as const;
  let n = bytes;
  let u = 0;
  while (n >= 1024 && u < units.length - 1) {
    n /= 1024;
    u++;
  }
  return `${n.toFixed(2)} ${units[u]}`;
}

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return NaN;
  if (sorted.length === 1) return sorted[0]!;
  const idx = (sorted.length - 1) * p;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  const h = idx - lo;
  return sorted[lo]! * (1 - h) + sorted[hi]! * h;
}

function summarize(timesMs: number[]) {
  const sorted = [...timesMs].sort((a, b) => a - b);
  const n = sorted.length;
  const sum = sorted.reduce((a, b) => a + b, 0);
  const avg = sum / n;
  const min = sorted[0]!;
  const max = sorted[n - 1]!;
  const p50 = percentile(sorted, 0.5);
  const p90 = percentile(sorted, 0.9);
  const p99 = percentile(sorted, 0.99);
  const variance =
    sorted.reduce((acc, v) => acc + (v - avg) * (v - avg), 0) / n;
  const stdev = Math.sqrt(variance);
  return { n, avg, min, max, p50, p90, p99, stdev };
}

function nowMs(): number {
  return performance.now();
}

function getMemorySnapshot() {
  try {
    const mu = process.memoryUsage?.();
    if (mu === undefined) return null;

    return {
      rss: mu.rss,
      heapTotal: mu.heapTotal,
      heapUsed: mu.heapUsed,
      external: (mu as any).external ?? 0,
    };
  } catch {
    return null;
  }
}

async function main() {
  const opts = parseArgs(process.argv);

  const { data, sampleRate, numSamples } = await loadWavFile(opts.file);
  if (sampleRate !== 16000) {
    console.warn(
      `Warning: WAV sampleRate=${sampleRate}. The spectrogram implementation assumes 16kHz (hop=160).`
    );
  }

  const padTailSamples = Math.max(
    0,
    Math.round(opts.padTailSeconds * sampleRate)
  );
  const numFrames = ((data.length + padTailSamples) / 160) | 0;
  const outputLength = 128 * numFrames;
  const method = opts.useApiWrapper ? "API" : "FFI";

  console.log(`Benchmarking log-mel spectrogram`);
  console.log(
    `- file: ${basename(opts.file)} (${numSamples} samples @ ${sampleRate} Hz)`
  );
  console.log(
    `- pad tail: ${opts.padTailSeconds}s (${padTailSamples} samples)`
  );
  console.log(`- frames: ${numFrames} (128 mel bins → ${outputLength} floats)`);
  console.log(
    `- engine: ${method} call; output ${
      opts.reuseOutput ? "reused" : "reallocated"
    }`
  );
  console.log(
    `- iterations: ${opts.iterations}, warmup: ${opts.warmup}, runs: ${opts.runs}`
  );
  console.log(
    `- env: Bun ${Bun.version} | ${os.type()} ${os.release()} | ${
      os.cpus()[0]?.model ?? "CPU"
    } x${os.cpus().length}`
  );

  let preallocated: Float32Array | null = null;
  if (opts.reuseOutput) {
    preallocated = new Float32Array(outputLength);
  }

  // Warmup
  for (let i = 0; i < opts.warmup; i++) {
    if (opts.useApiWrapper) {
      void logMelSpectrogram(data, padTailSamples);
    } else {
      const out = preallocated ?? new Float32Array(outputLength);
      lib.symbols.log_mel_spectrogram(data, data.length, padTailSamples, out);
    }
  }

  const runSummaries: ReturnType<typeof summarize>[] = [];
  for (let r = 0; r < opts.runs; r++) {
    const memBefore = getMemorySnapshot();
    const times: number[] = [];
    for (let i = 0; i < opts.iterations; i++) {
      const out = preallocated ?? new Float32Array(outputLength);
      const t0 = nowMs();
      if (opts.useApiWrapper) {
        const result = logMelSpectrogram(data, padTailSamples);
        if (!opts.reuseOutput) {
          // Use result to avoid DCE
          (result[0] ?? 0) + (result[result.length - 1] ?? 0);
        } else {
          // copy into preallocated to keep amount of work similar
          preallocated!.set(result);
        }
      } else {
        lib.symbols.log_mel_spectrogram(data, data.length, padTailSamples, out);
      }
      const t1 = nowMs();
      times.push(t1 - t0);
    }
    const memAfter = getMemorySnapshot();
    const s = summarize(times);
    runSummaries.push(s);

    const avgMs = s.avg;
    const framesPerIter = numFrames;
    const framesPerSec = framesPerIter / (avgMs / 1000);
    const samplesPerSec = framesPerSec * 160; // hop size in samples

    console.log(
      `Run ${r + 1}/${opts.runs}: avg=${avgMs.toFixed(3)}ms p50=${s.p50.toFixed(
        3
      )}ms p90=${s.p90.toFixed(3)}ms p99=${s.p99.toFixed(
        3
      )}ms (min=${s.min.toFixed(3)}ms, max=${s.max.toFixed(
        3
      )}ms, sd=${s.stdev.toFixed(3)})`
    );
    console.log(
      `  Throughput: ${framesPerSec.toFixed(1)} frames/s | ${(
        samplesPerSec / 1e6
      ).toFixed(3)} MSamples/s | ${(
        (outputLength * 4) /
        (avgMs / 1000) /
        1e6
      ).toFixed(2)} MB/s out`
    );
    if (memBefore && memAfter) {
      const rssDelta = memAfter.rss - memBefore.rss;
      const heapDelta = memAfter.heapUsed - memBefore.heapUsed;
      console.log(
        `  Memory: rss=${formatBytes(memAfter.rss)} (${
          rssDelta >= 0 ? "+" : ""
        }${formatBytes(rssDelta)}) | heapUsed=${formatBytes(
          memAfter.heapUsed
        )} (${heapDelta >= 0 ? "+" : ""}${formatBytes(heapDelta)})`
      );
    }
  }

  if (runSummaries.length > 1) {
    const best = runSummaries.reduce((a, b) => (a.avg < b.avg ? a : b));
    const med = runSummaries.toSorted((a, b) => a.avg - b.avg)[
      Math.floor(runSummaries.length / 2)
    ]!;
    console.log(
      `Best avg=${best.avg.toFixed(3)}ms | Median avg=${med.avg.toFixed(
        3
      )}ms over ${runSummaries.length} runs`
    );
  }

  console.log("Done.");
}

await main();
