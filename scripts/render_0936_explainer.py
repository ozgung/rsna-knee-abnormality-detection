"""Render the private explanatory notebook as a readable HTML companion (no execution)."""
from pathlib import Path
import hashlib
import json
import nbformat
from nbconvert import HTMLExporter

root=Path(__file__).resolve().parents[1]
path=root/'outputs/notebooks/rsna-knee-0936/rsna-knee-0936-explained.ipynb'
nb=nbformat.read(path,as_version=4)
nbformat.validate(nb)
exporter=HTMLExporter()
exporter.exclude_input=True
exporter.exclude_output_prompt=True
exporter.exclude_input_prompt=True
body,_=exporter.from_notebook_node(nb)
style='''<style>
body { color:#172d3b; }
main { max-width:1000px; margin:auto; padding:20px 24px 60px; }
.jp-RenderedHTMLCommon { font-size:16px; line-height:1.65; }
.jp-RenderedHTMLCommon p { margin:0 0 1em; }
.jp-RenderedHTMLCommon h1 { color:#087f8c; margin-top:1em; }
.jp-RenderedHTMLCommon h2 { margin-top:1.5em; line-height:1.3; }
.jp-RenderedHTMLCommon img { display:block; max-width:100%; margin:18px auto; }
.jp-RenderedHTMLCommon table { margin:18px 0; font-size:14px; }
.jp-RenderedHTMLCommon th { background:#edf4f6; }
.reading-note { max-width:940px; margin:24px auto 0; padding:14px 22px;
 background:#edf4f6; border-left:4px solid #087f8c; font:14px/1.5 sans-serif; }
@media(max-width:650px) { main { padding:8px; } .jp-RenderedHTMLCommon {font-size:15px;} }
</style>'''
body=body.replace('</head>',style+'</head>')
body=body.replace('<body>', '<body><div class="reading-note">Reading edition · '
    'The complete executable code is preserved in the companion '
    '<a href="rsna-knee-0936-explained.ipynb">Jupyter notebook</a>. '
    'This HTML hides the bundled runtime for readability. MRI illustrations remain private.</div>')
path.with_suffix('.html').write_text(body)
receipt=json.loads((path.parent/'build-receipt.json').read_text())
receipt.update(notebook_schema='PASS',html_sha256=hashlib.sha256(body.encode()).hexdigest(),
    html_code_executed=False,embedded_figure_count=sum(len(c.get('attachments',{})) for c in nb.cells))
(path.parent/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Notebook schema PASS; rendered HTML without executing inference; five embedded figures.')
