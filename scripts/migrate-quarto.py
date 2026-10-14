"""One-time import. Never overwrites an existing Quartz note. Requires PyYAML + pandoc."""
from pathlib import Path
import base64
import datetime
import json
import re
import shutil
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'Karem Site'
CONTENT = ROOT / 'content'
report = {'source': str(SOURCE), 'imported': [], 'drafts_left_in_quarto': [], 'missing_assets': [], 'notebook_outputs': []}
assets = [p for p in SOURCE.rglob('*') if p.is_file() and not any(x.startswith('.') or x in ('_site', '_extensions', '__marimo__') for x in p.relative_to(SOURCE).parts)]
asset_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.svg', '.gif', '.ico', '.pdf', '.mp3', '.mp4', '.wav', '.webm', '.ogg'}

def split(text):
    match = re.match(r'^---\s*\n(.*?)\n---\s*\n?', text, re.S)
    return (yaml.safe_load(match[1]) or {}, text[match.end():]) if match else ({}, text)

def normal_date(value):
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()[:10]
    match = re.fullmatch(r'(\d{4})-(\d{1,2})-(\d{1,2})', str(value))
    if match:
        return f'{match[1]}-{int(match[2]):02}-{int(match[3]):02}'
    return str(value)

def copy_asset(ref, page):
    ref = ref.strip().split('|')[0]
    if re.match(r'^(?:[a-z]+:|#|//)', ref, re.I):
        return
    from urllib.parse import unquote
    ref = unquote(ref.split('#')[0].split('?')[0])
    candidate = (SOURCE / ref.lstrip('/')) if ref.startswith('/') else page.parent / ref
    candidate = candidate.resolve()
    if not candidate.is_relative_to(SOURCE.resolve()):
        report['missing_assets'].append({'page': str(page.relative_to(SOURCE)), 'ref': ref})
        return
    if candidate.suffix.lower() not in asset_extensions:
        return
    if not candidate.is_file():
        report['missing_assets'].append({'page': str(page.relative_to(SOURCE)), 'ref': ref})
        return
    dest = CONTENT / candidate.relative_to(SOURCE.resolve())
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, dest)

def notebook_text(path):
    notebook = json.loads(path.read_text())
    blocks = []
    html_outputs = []
    language = notebook.get('metadata', {}).get('kernelspec', {}).get('language', 'python')
    for i, cell in enumerate(notebook['cells']):
        source = ''.join(cell.get('source', []))
        if cell['cell_type'] == 'markdown':
            blocks.append(source)
        elif cell['cell_type'] == 'code':
            source = re.sub(r'^#\|.*\n?', '', source, flags=re.M)
            fence = '`' * max(3, 1 + max([len(m[0]) for m in re.finditer(r'`+', source)] or [0]))
            blocks.append(f'{fence}{language}\n{source.rstrip()}\n{fence}')
            for j, output in enumerate(cell.get('outputs', [])):
                data = output.get('data', {})
                image = next((t for t in ('image/png', 'image/jpeg') if t in data), None)
                if image:
                    name = f'{path.stem}-{i}-{j}.{image.split("/")[1]}'
                    dest = CONTENT / path.relative_to(SOURCE).parent / 'notebook-images' / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(base64.b64decode(''.join(data[image])))
                    blocks.append(f'![](notebook-images/{name})')
                    report['notebook_outputs'].append(str(dest.relative_to(CONTENT)))
                elif 'text/html' in data:
                    html_outputs.append(''.join(data['text/html']))
                elif 'text/plain' in data or output.get('output_type') == 'stream':
                    text = ''.join(data.get('text/plain', output.get('text', [])))
                    blocks.append('```text\n' + re.sub(r'\x1b\[[0-9;]*m', '', text).rstrip() + '\n```')
    if html_outputs:
        filename = f'notebook-outputs/{path.stem}.htm'
        dest = CONTENT / path.relative_to(SOURCE).parent / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>' + '\n'.join(html_outputs) + '</body></html>')
        blocks.append(f'<iframe src="{filename}" title="Saved notebook visualizations" width="100%" height="640" loading="lazy"></iframe>')
        report['notebook_outputs'].append(str(dest.relative_to(CONTENT)))
    return '\n\n'.join(blocks) + '\n'

for page in sorted(assets):
    rel = page.relative_to(SOURCE)
    if page.suffix not in ('.md', '.qmd', '.ipynb') or page.name == 'README.md' or rel == Path('404.qmd'):
        continue
    if '--notebooks-only' in sys.argv and page.suffix != '.ipynb':
        continue
    if len(rel.parts) > 1 and rel.parts[0] not in ('blog', 'til'):
        continue
    destination = CONTENT / rel.with_suffix('.md')
    if destination.exists():
        raise SystemExit(f'Refusing to overwrite {destination}')
    meta, body = split(notebook_text(page) if page.suffix == '.ipynb' else page.read_text())
    if meta.get('draft') is True:
        report['drafts_left_in_quarto'].append(str(rel))
        continue
    inherited = {}
    for directory in reversed([page.parent, *page.parent.parents]):
        if directory == SOURCE or SOURCE in directory.parents:
            metadata = directory / '_metadata.yml'
            if metadata.exists():
                inherited.update(yaml.safe_load(metadata.read_text()) or {})
    merged = {**inherited, **meta}
    output = {k: merged[k] for k in ('title', 'description', 'lang', 'dir', 'author', 'comments', 'toc') if k in merged}
    output['title'] = output.get('title') or ('عبدالكريم الخطيب' if rel == Path('index.qmd') else page.stem)
    output['publish'] = True
    output['lang'] = output.get('lang', 'ar')
    output['dir'] = output.get('dir', 'rtl' if output['lang'].startswith('ar') else 'ltr')
    if meta.get('pagetitle'):
        output['seoTitle'] = meta['pagetitle']
    for name in ('date', 'updated'):
        if merged.get(name):
            output['date' if name == 'date' else 'modified'] = normal_date(merged[name])
    tags = []
    for field in ('tags', 'categories'):
        values = merged.get(field, [])
        if isinstance(values, str): values = [values]
        tags.extend(str(x).replace(' ', '-') for x in values)
    if tags: output['tags'] = list(dict.fromkeys(tags))
    image = merged.get('image')
    if isinstance(image, str) and image:
        output['socialImage'] = image
        copy_asset(image, page)
    if 'listing' in meta:
        output['listing'] = 'til' if rel.parts[0] == 'til' else ('embedding' if 'embedding_world' in rel.parts else 'blog')
    # These three portfolio pages contain presentational HTML, not article code.
    # Flatten their wrappers into readable Obsidian-native Markdown once.
    if rel.as_posix() in ('index.qmd', 'about.qmd', 'papers.qmd'):
        body = re.sub(r'^:::+.*$', '', body, flags=re.M)
        body = subprocess.run(['pandoc', '-f', 'html', '-t', 'gfm', '--wrap=none'], input=subprocess.run(['pandoc', '-f', 'markdown', '-t', 'html'], input=body, text=True, capture_output=True, check=True).stdout, text=True, capture_output=True, check=True).stdout
    body = re.sub(r'(?<=\])\(([^)\n]+)\.(?:qmd|ipynb)(#[^)]*)?\)', lambda m: '(' + m[1] + '.md' + (m[2] or '') + ')', body)
    if rel == Path('index.qmd'):
        output['cssclasses'] = ['home']
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('---\n' + yaml.safe_dump(output, allow_unicode=True, sort_keys=False) + '---\n\n' + body.strip() + '\n')
    refs = re.findall(r'!?\[[^\]\n]*\]\(<?([^\n]*?)>?\)', body)
    refs += re.findall(r'!\[\[([^\]]+)\]\]', body)
    refs += re.findall(r'(?:src|href)=["\']([^"\']+)["\']', body)
    for ref in refs:
        copy_asset(ref.split(' "')[0], page)
    report['imported'].append({'source': str(rel), 'note': str(destination.relative_to(CONTENT))})

# Local fonts are website resources, not vault attachments.
fonts = ROOT / 'quartz/static/fonts'
fonts.mkdir(parents=True, exist_ok=True)
for name in ('NotoNaskhArabic-Regular.woff2', 'NotoNaskhArabic-Bold.woff2', 'NotoSansArabicUI-Regular.woff2', 'NotoSansArabicUI-Bold.woff2'):
    shutil.copy2(SOURCE / 'fonts' / name, fonts / name)
shutil.copy2(SOURCE / 'favicon.ico', ROOT / 'quartz/static/favicon.ico')
shutil.copy2(SOURCE / 'kareem.webp', CONTENT / 'kareem.webp')
(ROOT / 'migration').mkdir(exist_ok=True)
if '--notebooks-only' in sys.argv:
    previous = json.loads((ROOT / 'migration/report.json').read_text())
    previous['notebook_outputs'] = report['notebook_outputs']
    report = previous
(ROOT / 'migration/report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(f'Imported {len(report["imported"])} notes. Left {len(report["drafts_left_in_quarto"])} drafts in Quarto. {len(report["missing_assets"])} missing asset references recorded.')
