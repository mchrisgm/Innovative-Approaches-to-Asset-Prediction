# Model checkpoints

## Filenames

- Each model checkpoint contains 2 files:

  - **JSON**: Model config information
  - **PTH** PyTorch weights file
- Filename meaning:

  `{origin}`.`{number of stocks}`.`{lookback period}`.`{image dimension}`.`{RGB channels}`.`{MONO (integer) or RANGE (float) target output}`.`{output classes}`.`{validation accuracy}`
