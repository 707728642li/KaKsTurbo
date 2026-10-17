"""Offline, paginated result browser with typed data and no remote dependencies."""
from __future__ import annotations

from collections import Counter
import csv
from html import escape
import json
import os
from pathlib import Path
from ._paths import absolute_path
from urllib.parse import quote
import uuid

from .checkpoint import contained, load_json, verify_hashes
from .errors import ConfigurationError, IntegrityError
from .results import CORE_FIELDS, DESCRIPTIONS, core_row, parse_row

MAX_REPORT_ROWS = 100_000
DOWNLOADS = ("result.tsv", "results.jsonl", "native_result.tsv", "native_results.jsonl",
             "models.tsv", "models.jsonl", "pair_statistics.jsonl", "qc.tsv", "data_dictionary.json")


def _safe_json(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _download_url(path, target):
    try:
        return quote(Path(os.path.relpath(path, target.parent)).as_posix(), safe="/.")
    except ValueError:  # Different Windows drive letters.
        return path.as_uri()


def _result_rows(directory, manifest):
    qc_path = directory / "qc.tsv"
    qc_handle = qc_path.open(encoding="utf-8-sig", newline="") if qc_path.is_file() else None
    try:
        qc_reader = csv.DictReader(qc_handle, delimiter="\t") if qc_handle else None
        with (directory / "result.tsv").open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            modern = reader.fieldnames == list(CORE_FIELDS)
            if not modern and not {"Sequence", "Method", "Ka", "Ks", "Ka/Ks"}.issubset(reader.fieldnames or []):
                raise IntegrityError("Unsupported result TSV header")
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise IntegrityError("Malformed result TSV row")
                pair = row["pair_id"] if modern else row["Sequence"]
                qc = next(qc_reader, None) if qc_reader else {}
                if qc is None or (qc_reader and qc.get("pair_id") != pair):
                    raise IntegrityError("Result and QC ordered identifiers differ")
                if modern:
                    result = parse_row(row)
                else:
                    metadata = dict(qc)
                    metadata["pair_id"] = pair
                    backends = manifest.get("backends_used", [])
                    if len(backends) == 1:
                        metadata["backend"] = backends[0]
                    result = core_row(row, {}, metadata, mode=manifest["mode"])
                yield result, qc
            if qc_reader and next(qc_reader, None) is not None:
                raise IntegrityError("QC contains extra ordered requests")
    finally:
        if qc_handle:
            qc_handle.close()


STYLE = """
:root{color-scheme:light;--ink:#172b3a;--blue:#0f4d92;--line:#d8e1e7}
*{box-sizing:border-box}body{margin:0;background:#f5f8fa;color:var(--ink);font:15px/1.5 Arial,system-ui,sans-serif}
main{max-width:1600px;margin:auto;padding:28px}h1{margin:0;font-size:28px}h2{font-size:20px;margin:0 0 12px}p{margin:8px 0}
section{background:white;border:1px solid var(--line);border-radius:10px;padding:20px;margin:18px 0}
a{color:var(--blue)}.muted{color:#536674}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:12px;margin:20px 0}
.card{background:white;border:1px solid var(--line);border-radius:8px;padding:14px}.card strong{display:block;font-size:24px;font-variant-numeric:tabular-nums}
.counts{display:flex;flex-wrap:wrap;gap:8px 22px}.filters{display:flex;flex-wrap:wrap;gap:10px;align-items:end}.filters label{display:flex;flex-direction:column;font-size:13px;gap:4px}
input,select,button{font:inherit;border:1px solid #aabcc8;border-radius:5px;padding:7px;background:white;color:var(--ink)}input[type=search]{min-width:260px}
button{cursor:pointer}button:hover{background:#eef4f8}button:disabled{opacity:.45;cursor:default}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #79b9dd;outline-offset:2px}
.table-wrap{overflow:auto;max-height:65vh;margin:14px 0;border:1px solid var(--line)}table{width:100%;border-collapse:collapse;font-size:13px}
td,th{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}td{white-space:nowrap;font-variant-numeric:tabular-nums}
th{position:sticky;top:0;background:#eef4f8;z-index:1}th button{border:0;background:none;padding:0;font-weight:bold;white-space:nowrap}
tbody tr:nth-child(even){background:#f8fafc}.pager{display:flex;flex-wrap:wrap;align-items:center;gap:12px}.downloads{display:flex;gap:12px;flex-wrap:wrap}
.dictionary td{white-space:normal}.dictionary th{position:static}.dictionary td:first-child{font-family:monospace}.small{font-size:13px}
@media(max-width:600px){main{padding:12px}section{padding:14px}.filters label,input[type=search]{width:100%;min-width:0}}
"""

SCRIPT = r"""
'use strict';
const payload=JSON.parse(document.getElementById('result-data').textContent);
const fields=payload.fields, rows=payload.rows, col=Object.fromEntries(fields.map((f,i)=>[f,i]));
const el=id=>document.getElementById(id);
const display=v=>v===null?'NA':Array.isArray(v)?(v.length?v.join(', '):'none'):String(v);
const text=rows.map(r=>r.map(v=>v===null?'':Array.isArray(v)?v.join(','):String(v)).join('\t').toLowerCase());
let filtered=[], page=0, sortField=null, direction=1;
for(const [id,field] of [['status-filter','status'],['mode-filter','mode']]){
  for(const value of [...new Set(rows.map(r=>r[col[field]]))].sort()){
    const option=document.createElement('option');option.value=value;option.textContent=value;el(id).appendChild(option);
  }
}
for(const field of fields){
  const th=document.createElement('th'), button=document.createElement('button');
  th.scope='col';th.dataset.field=field;th.setAttribute('aria-sort','none');button.type='button';button.textContent=field;
  button.addEventListener('click',()=>{direction=sortField===field?-direction:1;sortField=field;sortRows();page=0;render();});
  th.appendChild(button);el('head-row').appendChild(th);
}
function listMatches(value,choice){return choice==='all'||(choice==='unknown'&&value===null)||(choice==='yes'&&Array.isArray(value)&&value.length>0)||(choice==='none'&&Array.isArray(value)&&value.length===0);}
function sortRows(){
  if(sortField===null)return;
  const index=col[sortField];
  filtered.sort((a,b)=>{
    const x=rows[a][index],y=rows[b][index];
    if(x===null||y===null)return x===y?a-b:x===null?1:-1;
    const cmp=typeof x==='number'&&typeof y==='number'?x-y:display(x).localeCompare(display(y));
    return cmp===0?a-b:cmp*direction;
  });
}
function applyFilters(){
  const query=el('search').value.toLowerCase(),status=el('status-filter').value,mode=el('mode-filter').value,missing=el('missing-filter').value;
  filtered=[];
  for(let i=0;i<rows.length;i++){
    const r=rows[i];
    if(query&&!text[i].includes(query))continue;
    if(status!=='all'&&r[col.status]!==status)continue;
    if(mode!=='all'&&r[col.mode]!==mode)continue;
    if(missing==='missing'&&r[col.Ka_Ks]!==null||missing==='finite'&&r[col.Ka_Ks]===null)continue;
    if(!listMatches(r[col.iteration_capped_models],el('cap-filter').value)||!listMatches(r[col.boundary_models],el('bound-filter').value))continue;
    filtered.push(i);
  }
  page=0;sortRows();render();
}
function render(){
  const size=Number(el('page-size').value),pages=Math.max(1,Math.ceil(filtered.length/size));page=Math.max(0,Math.min(page,pages-1));
  const fragment=document.createDocumentFragment();
  for(const index of filtered.slice(page*size,(page+1)*size)){
    const tr=document.createElement('tr');
    for(const value of rows[index]){const td=document.createElement('td');td.textContent=display(value);tr.appendChild(td);}
    fragment.appendChild(tr);
  }
  el('result-body').replaceChildren(fragment);
  el('row-count').textContent=filtered.length.toLocaleString()+' matching / '+rows.length.toLocaleString()+' total ordered pairs';
  el('page-info').textContent='Page '+(page+1)+' / '+pages;
  el('previous').disabled=page===0;el('next').disabled=page===pages-1;el('empty').hidden=filtered.length!==0;
  for(const th of el('head-row').children)th.setAttribute('aria-sort',th.dataset.field===sortField?(direction===1?'ascending':'descending'):'none');
}
let timer;
el('search').addEventListener('input',()=>{clearTimeout(timer);timer=setTimeout(applyFilters,120);});
for(const id of ['status-filter','mode-filter','missing-filter','cap-filter','bound-filter'])el(id).addEventListener('change',applyFilters);
el('page-size').addEventListener('change',()=>{page=0;render();});
el('previous').addEventListener('click',()=>{page--;render();});el('next').addEventListener('click',()=>{page++;render();});
el('reset').addEventListener('click',()=>{el('search').value='';for(const id of ['status-filter','mode-filter','missing-filter','cap-filter','bound-filter'])el(id).value='all';sortField=null;direction=1;applyFilters();});
applyFilters();
"""


def _list_filter(identifier, label):
    return ('<label>' + label + '<select id="' + identifier + '"><option value="all">All</option>'
            '<option value="yes">Recorded hits</option><option value="none">Known none</option>'
            '<option value="unknown">Unknown</option></select></label>')


def create_report(output_dir, destination=None, *, max_rows=MAX_REPORT_ROWS) -> Path:
    """Write a self-contained browser for every result, up to a documented row limit.

    Data are streamed into HTML; only the current page is rendered in the browser.
    Native 1.0 runs are adapted for display without changing their source artifacts.
    """
    if type(max_rows) is not int or not 1 <= max_rows <= MAX_REPORT_ROWS:
        raise ConfigurationError(f"max_rows must be 1..{MAX_REPORT_ROWS}")
    directory = absolute_path(output_dir)
    manifest = load_json(directory / "run_manifest.json")
    if not manifest.get("complete") or manifest.get("schema") not in ("kaksturbo.run_manifest/1", "kaksturbo.run_manifest/2", "kaksforge.run_manifest/1", "kaksforge.run_manifest/2"):
        raise IntegrityError("HTML report requires a completed KaKsTurbo run")
    count = manifest.get("count")
    if type(count) is not int or count < 0:
        raise IntegrityError("Invalid completed-run pair count")
    if count > max_rows:
        raise ConfigurationError(f"Offline report limit is {max_rows} pairs; use the complete TSV/JSONL files for larger runs")
    hashes = manifest.get("artifacts_sha256", {})
    if "result.tsv" not in hashes:
        raise IntegrityError("Completed run does not register result.tsv")
    read_hashes = {"result.tsv": hashes["result.tsv"]}
    if (directory / "qc.tsv").is_file():
        if "qc.tsv" not in hashes:
            raise IntegrityError("QC file is not registered in the completed run")
        read_hashes["qc.tsv"] = hashes["qc.tsv"]
    verify_hashes(directory, read_hashes)
    target = absolute_path(destination) if destination else directory / "report.html"
    protected = {contained(directory, name) for name in hashes}
    protected.update({directory / "run_manifest.json", directory / "checkpoint.json", directory / ".kaksturbo.lock"})
    source = manifest.get("input") or {}
    if source.get("path"):
        protected.add(Path(source["path"]).resolve())
    if source.get("pairs_manifest"):
        protected.add(Path(source["pairs_manifest"]["path"]).resolve())
    if target in protected or target.is_relative_to(directory / ".work"):
        raise ConfigurationError("Report destination would overwrite a scientific input, output or restart artifact")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
    title = "KaKsTurbo " + str(manifest.get("version", ""))
    statuses, missing, diagnostics, qc_counts = Counter(), Counter(), Counter(), Counter()
    observed = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                         '<title>' + escape(title) + ' results</title><style>' + STYLE + '</style></head><body><main><header><h1>' + escape(title) + '</h1>'
                         '<p class="muted">Offline results · ordered, directed sequence pairs</p></header>')
            handle.write('<script id="result-data" type="application/json">{"fields":' + _safe_json(CORE_FIELDS) + ',"rows":[')
            for row, qc in _result_rows(directory, manifest):
                if observed >= count or observed >= max_rows:
                    raise IntegrityError("Report result count exceeds the completed-run count or report limit")
                handle.write((',' if observed else '') + _safe_json([row[field] for field in CORE_FIELDS]))
                observed += 1
                statuses[row["status"]] += 1
                for field in ("Ka", "Ks", "Ka_Ks", "lnL"):
                    missing[field] += row[field] is None
                for field in ("iteration_capped_models", "boundary_models"):
                    diagnostics[field + "_unknown"] += row[field] is None
                    diagnostics[field + "_hits"] += bool(row[field])
                for field in ("gap_codons", "ambiguous_codons", "stop_codons"):
                    if qc.get(field) not in (None, "", "NA"):
                        try:
                            value = int(qc[field])
                        except (TypeError, ValueError) as exc:
                            raise IntegrityError("Invalid QC codon count") from exc
                        if value < 0:
                            raise IntegrityError("Negative QC codon count")
                        qc_counts[field] += value
                        qc_counts[field + "_recorded"] += 1
            if observed != count:
                raise IntegrityError("Report result count differs from manifest")
            handle.write(']}</script>')
            cards = (("Ordered pairs", count), ("Available MA", statuses["available"]), ("Missing Ka/Ks", missing["Ka_Ks"]),
                     ("Pairs with iteration limits", diagnostics["iteration_capped_models_hits"]),
                     ("Pairs with bound hits", diagnostics["boundary_models_hits"]))
            handle.write('<div class="cards">' + ''.join('<div class="card"><strong>' + f'{value:,}' + '</strong>' + label + '</div>' for label, value in cards) + '</div>')
            handle.write('<section><h2>Result and input summary</h2><p>Mode: <strong>' + escape(str(manifest.get("mode", "unknown")))
                         + '</strong>. Backends: ' + escape(', '.join(manifest.get("backends_used", []))) + '. Method: MA; all 14 models; genetic code 1.</p>')
            handle.write('<p class="counts">' + ''.join('<span>' + escape(status) + ': <strong>' + f'{n:,} / {count:,}' + '</strong></span>' for status, n in sorted(statuses.items())) + '</p>')
            handle.write('<p>Missing values: ' + '; '.join(escape(field) + ' ' + f'{missing[field]:,} / {count:,}' for field in ("Ka", "Ks", "Ka_Ks", "lnL")) + '.</p>')
            handle.write('<p>Unknown diagnostics: iteration limits ' + f'{diagnostics["iteration_capped_models_unknown"]:,} / {count:,}'
                         + '; parameter bounds ' + f'{diagnostics["boundary_models_unknown"]:,} / {count:,}' + '. Counts refer to pairs, not models.</p>')
            if qc_counts:
                handle.write('<p>Recorded QC codon counts: ' + '; '.join(escape(field) + ' ' + f'{qc_counts[field]:,}' + ' across '
                             + f'{qc_counts[field + "_recorded"]:,}' + ' requests' for field in ("gap_codons", "ambiguous_codons", "stop_codons") if qc_counts[field + "_recorded"]) + '. Categories may overlap.</p>')
            else:
                handle.write('<p>Detailed input QC counts were not recorded for this run.</p>')
            handle.write('<p class="small muted">Availability does not establish optimizer convergence. Ka/Ks is an effect estimate; Ka/Ks &gt; 1 is not a positive-selection significance test. Unknown historical diagnostics stay unknown. Native files retain estimator-specific statistics and original printed precision.</p></section>')
            handle.write('<section><h2>Browse all results</h2><noscript><p>Enable JavaScript to search and page through the full result table, or use the TSV/JSONL downloads below.</p></noscript>'
                         '<div class="filters"><label>Search any field<input id="search" type="search" placeholder="Pair ID, sequence ID, model, reason…"></label>'
                         '<label>Native status<select id="status-filter"><option value="all">All</option></select></label>'
                         '<label>Mode<select id="mode-filter"><option value="all">All</option></select></label>'
                         '<label>Ka/Ks<select id="missing-filter"><option value="all">All</option><option value="finite">Finite</option><option value="missing">Missing</option></select></label>'
                         + _list_filter('cap-filter', 'Iteration limits') + _list_filter('bound-filter', 'Parameter bounds')
                         + '<button type="button" id="reset">Reset</button></div><p id="row-count" aria-live="polite"></p>'
                         '<div class="table-wrap"><table aria-label="All pair estimates"><thead><tr id="head-row"></tr></thead><tbody id="result-body"></tbody></table></div>'
                         '<p id="empty" hidden>No matching results.</p><div class="pager"><button type="button" id="previous">Previous</button>'
                         '<span id="page-info" aria-live="polite"></span><button type="button" id="next">Next</button><label>Rows per page <select id="page-size">'
                         '<option>25</option><option selected>50</option><option>100</option><option>200</option></select></label></div>'
                         '<p class="small muted">Click a column name to sort. Every result participates in search and sorting; only the current page is rendered. NA means missing or unknown. A model list shown as “none” means recorded absence.</p></section>')
            handle.write('<section><h2>Download results</h2><div class="downloads">')
            for name in DOWNLOADS:
                path = directory / name
                if name in hashes and path.is_file():
                    handle.write('<a download href="' + escape(_download_url(path, target), quote=True) + '">' + escape(name) + '</a>')
            handle.write('</div><p class="small muted">Links open companion files; keep the run directory when sharing downloads. This HTML contains all displayed rows and works offline on its own.</p></section>'
                         '<section><h2>Data dictionary</h2><p>JSON uses null for unavailable values. TSV uses NA for unavailable numbers or lists and an empty field for unavailable text. [] is a known empty model list. Floating-point values are serialized with round-trip precision; historical rounded values do not gain precision.</p>'
                         '<table class="dictionary"><thead><tr><th>Field</th><th>Meaning</th></tr></thead><tbody>')
            for field in CORE_FIELDS:
                handle.write('<tr><td>' + escape(field) + '</td><td>' + escape(DESCRIPTIONS[field]) + '</td></tr>')
            handle.write('</tbody></table></section></main><script>' + SCRIPT + '</script></body></html>\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
