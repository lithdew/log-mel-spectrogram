// log-mel spectrogram (128 x N frames)
// Written by Kenta Iwasaki. All rights reserved.
// 2025-08-10

const inputPath = process.argv[2] ?? "mel_filterbank.json";
const outputPath = process.argv[3] ?? "mel_mapping_tables.h";

// Step 1: Load JSON
const mel: number[][] = await Bun.file(inputPath).json();

const nRows = mel.length;
const nCols = mel[0]?.length ?? 0;

// Step 2: Convert to CSR sparse format
const rowPtr: number[] = [0];
const colIdx: number[] = [];
const weight: number[] = [];

for (let r = 0; r < nRows; r++) {
  for (let c = 0; c < nCols; c++) {
    const val = mel[r]![c]!;
    if (val !== 0) {
      colIdx.push(c);
      weight.push(val);
    }
  }
  rowPtr.push(colIdx.length);
}

// Step 3: Make header text
const arrayToC = (
  type: string,
  name: string,
  arr: number[],
  fmt = (v: number) => v.toString()
) => {
  const vals = arr.map(fmt).join(", ");
  return `static const ${type} ${name}[${arr.length}] = { ${vals} };`;
};

const header = `#pragma once

#include <stdint.h>

${arrayToC("uint16_t", "mel_row_ptr", rowPtr)}
${arrayToC("uint8_t", "mel_col_idx", colIdx)}
${arrayToC("float", "mel_weight", weight, (v) => v.toExponential() + "f")}
`;

// Step 4: Save
await Bun.write(outputPath, header);

console.log(`Wrote sparse mel filterbank header: ${outputPath}`);
console.log(`nRows=${nRows}, nCols=${nCols}, nnz=${weight.length}`);
