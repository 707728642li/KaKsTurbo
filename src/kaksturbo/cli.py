"""Human-readable commands and explicit JSON for workflow managers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from ._paths import absolute_path
import sys
from . import api
from .errors import ConfigurationError, KaKsTurboError
from .models import RunConfig, VERSION, default_threads
from .preparation import prepare_aligned, write_input_examples
from .progress import TerminalProgress
from .report import create_report

class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ConfigurationError(message)

def _display(parser):
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="Machine-readable JSON, including errors")
    parser.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS, help="Hide progress")

def _input(parser):
    parser.add_argument("--input", required=True, help="Codon-aligned AXT or FASTA; .gz supported")
    parser.add_argument("--input-format", choices=("auto", "axt", "fasta"), default="auto")
    parser.add_argument("--pairs", dest="pairs_manifest", help="FASTA pair TSV: pair_id, seq1, seq2")
    parser.add_argument("--metadata", dest="input_metadata", help="Prepared AXT metadata JSONL; verified against every directed pair")

def _report_options(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--report", action="store_true", help="Also create an optional offline HTML report")
    group.add_argument("--no-report", action="store_true", help="Compatibility option; reports are disabled by default")

def build_parser():
    parser = Parser(prog="kaksturbo", description="Calculate Ka, Ks and Ka/Ks from codon-aligned gene pairs on CPU or GPU.",
                    epilog="Start with: kaksturbo examples --output example_inputs")
    parser.add_argument("--version", action="version", version="KaKsTurbo " + VERSION)
    _display(parser)
    commands = parser.add_subparsers(dest="command", required=True, parser_class=Parser)
    def command(name, help):
        sub = commands.add_parser(name, help=help)
        _display(sub)
        return sub
    run = command("run", "Fit aligned pairs and write standardized result files")
    _input(run)
    run.add_argument("--output", required=True, help="A new results directory")
    _report_options(run)
    run.add_argument("--mode", choices=("robust", "reproduce"), default="robust")
    run.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto", help="auto selects CPU; choose gpu explicitly")
    run.add_argument("--threads", type=int, default=default_threads(), help="CPU workers: Windows 1–32, Linux 1–128 (default: up to 16 available CPUs)")
    run.add_argument("--device", default="auto", help="auto, cuda:0, cuda:1, or physical GPU UUID")
    run.add_argument("--chunk-pairs", type=int, default=1000, help="Pairs per resumable chunk (1–1000)")
    run.add_argument("--method", choices=("MA",), default="MA")
    run.add_argument("--genetic-code", type=int, choices=(1,), default=1)
    run.add_argument("--max-seconds", type=float, default=600.0, help="Time limit per chunk, in seconds")
    for name, default in (("batch-tasks", 512), ("gpu-cache-buckets", 256), ("request-capacity", 16384)):
        run.add_argument("--" + name, type=int, default=default, help=argparse.SUPPRESS)
    validation = command("validate", "List input problems before any fitting")
    _input(validation)
    validation.add_argument("--output", help="New directory for validation.json, errors.tsv and QC")
    validation.add_argument("--max-errors", type=int, default=1000)
    prepare = command("prepare", "Convert already aligned input to AXT; does not align raw CDS")
    _input(prepare)
    prepare.add_argument("--output", required=True)
    examples = command("examples", "Write runnable examples and an empty pair-table template")
    examples.add_argument("--output", required=True)
    doctor = command("doctor", "Check installation and CPU/GPU availability")
    doctor.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    doctor.add_argument("--device", default="auto")
    doctor.add_argument("--probe-gpu", action="store_true", help="Initialize driver after admission; no GPU allocation or numerical fit")
    resume = command("resume", "Continue an interrupted run with its original configuration")
    resume.add_argument("directory")
    resume.add_argument("--max-seconds", type=float, help="Increase the per-chunk time limit; scientific parameters stay fixed")
    resume.add_argument("--input", help="Relocate the original input file; bytes must match")
    resume.add_argument("--pairs", dest="pairs_manifest", help="Relocate the original FASTA pairs TSV; bytes must match")
    resume.add_argument("--metadata", dest="input_metadata", help="Relocate the original prepared metadata; bytes must match")
    _report_options(resume)
    status = command("status", "Show completed pairs, current chunk and process status")
    status.add_argument("directory")
    export = command("export", "Export saved results without fitting; originals stay unchanged")
    export.add_argument("directory", help="Completed public results directory")
    export.add_argument("--format", choices=("kaks-calculator",), required=True)
    export.add_argument("--output", required=True, help="New export directory outside the original run")
    report = command("report", "Create a searchable offline report with QC and downloads")
    report.add_argument("directory")
    report.add_argument("--output")
    return parser

def _config(args):
    values = {key: value for key, value in vars(args).items() if key in RunConfig.__dataclass_fields__}
    if values.get("input_format") == "auto":
        from .io import infer_input_format
        values["input_format"] = "axt" if absolute_path(args.input).is_dir() else infer_input_format(args.input)
    return RunConfig(**values)

def _progress(event):
    # An explicit status query is a single plain snapshot, not a live bar.
    with TerminalProgress(tty=False) as display:
        display(event)

def _error(payload):
    print("Error: " + payload["message"], file=sys.stderr)
    for key, value in payload.get("details", {}).items():
        if key == "errors" and isinstance(value, list):
            for issue in value[:5]:
                print("  " + issue.get("message", str(issue)), file=sys.stderr)
            if len(value) > 5:
                print(f"  {len(value) - 5} more errors are listed in the validation report.", file=sys.stderr)
            continue
        print(f"  {key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}", file=sys.stderr)
    if payload.get("error") == "invalid_input":
        print("Use 'kaksturbo validate --input FILE --output qc' to locate problems. Input must already be codon-aligned.", file=sys.stderr)

def _human(command, payload):
    if command in {"run", "resume"}:
        print(f"Completed {payload['count']} gene pairs ({payload['backend']}).")
        print("Results: " + str(Path(payload["directory"]) / "result.tsv"))
        print("Original engine output: " + str(Path(payload["directory"]) / "native_result.tsv"))
        if payload.get("quality_summary"):
            summary = payload["quality_summary"]
            p, m = summary["pairs"], summary["models"]
            print(f"Quality: {p['finite_Ka_Ks']} finite Ka/Ks, {p['NA_Ka_Ks']} NA; "
                  f"{m['iteration_capped']} capped model fits; "
                  f"{m['gradient_above_1e_minus_8']} model gradients above 1e-8, {m['gradient_unknown']} unknown.")
            print("Detailed counts and interpretation: " + str(Path(payload["directory"]) / "quality_summary.txt"))
        if payload.get("report"):
            print("Report: " + payload["report"])
        if payload.get("report_error"):
            print("Fitting completed; report generation failed: " + payload["report_error"], file=sys.stderr)
    elif command == "validate":
        print("Input is valid." if payload["valid"] else f"Input has {payload.get('error_count', 0)} problem(s).")
        print(f"Checked {payload.get('pairs_seen', payload.get('pairs', 0))} pairs. No fitting performed.")
        for item in payload.get("errors", [])[:5]:
            _error(item)
        if payload.get("errors_path"):
            print("All reported problems: " + payload["errors_path"])
    elif command == "status":
        _progress(payload)
        print("Process active: " + ("yes" if payload["process_active"] else "no"))
        if payload["phase"] == "interrupted":
            print("Resume with: kaksturbo resume " + json.dumps(payload["directory"]))
    elif command == "report":
        print("Report: " + payload["report"])
    elif command == "export":
        print(f"Exported {payload['count']} pairs: {payload['result']}")
        print("22-column table-layout compatibility; source estimator is unchanged.")
        print("Missing statistics are NA. Read missing_fields.tsv and compatibility.json.")
        print("No fitting performed; original results remain unchanged.")
    elif command in {"prepare", "examples"}:
        print("Prepared aligned input; no alignment or fitting performed." if command == "prepare" else "Examples and pair-table template written.")
        print("Output: " + str(payload.get("directory", payload.get("output_dir", ""))))
    elif command == "doctor":
        print("CPU: " + payload["CPU_status"])
        print("GPU: " + payload["GPU_status"])
        if payload.get("GPU_package_diagnostic"):
            print("  " + payload["GPU_package_diagnostic"]["message"])
        if payload.get("GPU_diagnostic"):
            info=payload["GPU_diagnostic"]
            print("  " + info["message"])
            print("  " + info["details"]["action"])
        print("Backend: " + payload["selected_backend"] + "; " + payload["auto_policy"])
        print("Modes in this package: " + ", ".join(payload["available_modes"]))
        print("No numerical fit performed. Use --json for the complete diagnostic record.")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))

def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    machine = "--json" in raw
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="strict")
    try:
        args = build_parser().parse_args(raw)
        machine = getattr(args, "json", False)
        if args.command in {"run", "resume"}:
            callback = None if machine or getattr(args, "quiet", False) else TerminalProgress()
            try:
                from ._cli_affinity import temporary_budget
                from .checkpoint import load_json
                configuration = (_config(args) if args.command == "run" else
                                 RunConfig(**load_json(absolute_path(args.directory) / "checkpoint.json")["config"]))
                with temporary_budget(configuration.threads):
                    result = (api.fit_file(args.input, args.output, configuration, progress=callback) if args.command == "run"
                              else api.resume(args.directory, configuration, progress=callback, max_seconds=args.max_seconds,
                                              input_path=args.input, pairs_manifest=args.pairs_manifest,
                                              input_metadata=args.input_metadata))
            finally:
                if callback is not None:
                    callback.close()
            payload = {"schema": "kaksturbo.command_result/2", "complete": True, "directory": str(result.directory),
                       "count": result.manifest["count"], "version": VERSION, "backend": ",".join(result.manifest.get("backends_used", []))}
            payload["quality_summary"] = result.manifest.get("quality_summary")
            if args.report:
                try:
                    payload["report"] = str(create_report(result.directory))
                except (KaKsTurboError, OSError, ValueError) as exc:
                    payload["report_error"] = str(exc)
        elif args.command == "validate":
            payload = api.validate(args.input, _config(args), args.output, max_errors=args.max_errors)
        elif args.command == "prepare":
            payload = prepare_aligned(args.input, args.output, _config(args))
        elif args.command == "examples":
            payload = write_input_examples(args.output)
        elif args.command == "doctor":
            payload = api.doctor(vars(args))
        elif args.command == "status":
            payload = api.status(args.directory)
        elif args.command == "export":
            from .compat_export import export_kaks_calculator
            payload = export_kaks_calculator(args.directory, args.output)
        else:
            payload = {"schema": "kaksturbo.command_result/2", "report": str(create_report(args.directory, args.output)), "version": VERSION}
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False)) if machine else _human(args.command, payload)
        if args.command == "doctor" and not payload.get("PASS", False):
            return 4
        return 2 if args.command == "validate" and not payload["valid"] else 0
    except KaKsTurboError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False), file=sys.stderr) if machine else _error(exc.as_dict())
        return exc.exit_code
    except KeyboardInterrupt:
        payload = {"schema": "kaksturbo.error/1", "error": "interrupted", "message": "Interrupted; resume this results directory to continue.", "details": {}}
        print(json.dumps(payload) if machine else payload["message"], file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        payload = {"schema": "kaksturbo.error/1", "error": "runtime_error", "message": str(exc), "details": {}}
        print(json.dumps(payload, ensure_ascii=False) if machine else "Error: " + str(exc), file=sys.stderr)
        return 4

if __name__ == "__main__":
    raise SystemExit(main())
