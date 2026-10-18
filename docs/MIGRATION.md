# Versions and checkpoint compatibility

The public downloads are **Windows 1.1.0rc10** and **Linux 1.1.0rc9-tutorial1**. The GitHub tag `v1.1.0rc10` identifies the combined distribution; it does not rename the Linux executable. [Release status](RELEASE_STATUS.md)

KaKsTurbo is the public name. `kaksforge` remains an import compatibility alias. Historical engine paths and hash-domain identifiers retain their names so source provenance and checks remain reproducible.

## Interface version and engine identity

Program versions describe the packaged CLI/runtime interface. Native scientific assets are pinned separately for each platform in [engine_pins.json](../engine_pins.json). These hashes, rather than an old development label alone, identify the loaded scientific implementation. Wrapper releases can add input, output or path support without changing the numerical code.

## Resume a run with its original release

Resume checks the interface, settings, input bytes, pinned engine and completed chunks. Finish an existing checkpoint with its original matching environment. A new release must not bypass a mismatch by editing a manifest or substituting engine files.

Relocating identical input files is supported through the recorded resume options; configuration or scientific-policy changes are not silent migrations. Keep checkpoint-referenced chunks for unfinished runs. Completed output can be inspected or exported without refitting through a compatible installed interface. [CLI details](USAGE.md)

## Documentation history

The original public source and portable assets remain at tag `v1.1.0rc10`. The repository's `main` branch carries subsequent documentation and presentation improvements. The homepage's figures and timing summaries document existing measurements; they are not a new engine build or benchmark.
