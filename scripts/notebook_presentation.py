"""Notebook-local bidi presentation; requires only IPython, no model execution."""
import nbformat


def presentation_cells(root):
    css = (root / "docs" / "notebook-rtl.css").read_text(encoding="utf-8")
    explanation = nbformat.v4.new_markdown_cell(
        '<p dir="ltr"><strong>Notebook display setup:</strong> '
        'Run the setup code cell with <code>Shift+Enter</code> whenever you open the notebook. '
        'This cell only configures the display; prose, code, and formulas remain readable. '
        'No Windows or Chrome language change is required.</p>',
        id="tinyllm-rtl-help",
    )
    source = ('from IPython.display import HTML, display\n\n'
              'display(HTML(r"""\n<span class="tinyllm-rtl-marker" hidden></span>\n<style>\n'
              + css + '\n</style>\n"""))')
    compile(source, "notebook-presentation", "exec")
    cell = nbformat.v4.new_code_cell(source, id="tinyllm-rtl-style",
                                   metadata={"tags": ["presentation"],
                                             "jupyter": {"source_hidden": True}})
    return [explanation, cell]
