"""
Executes the notebook cells in-process to capture stdout, tables, and inline matplotlib plots,
saving a fully rendered .ipynb notebook with all outputs embedded.
"""

import os
import sys
import io
import base64
import subprocess
from pathlib import Path
import nbformat

os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = BASE_DIR / "notebooks" / "sales_prediction.ipynb"
BUILDER_PATH = BASE_DIR / "src" / "build_notebook.py"


def render_notebook():
    print("Rebuilding pristine notebook template...")
    subprocess.run([sys.executable, str(BUILDER_PATH)], check=True)

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Change working directory to notebooks/ so relative paths like ../data work seamlessly
    os.chdir(NOTEBOOK_PATH.parent)

    # Clean suppress for Agg plt.show() warning
    def noop_show(*args, **kwargs):
        pass
    plt.show = noop_show

    exec_globals = {"plt": plt}
    exec_count = 1

    print("Executing notebook cells and capturing outputs...")

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue

        code = cell.source
        cell.outputs = []
        cell.execution_count = exec_count

        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture

        plt.close("all")

        try:
            # Execute code block
            exec(code, exec_globals)
        except Exception as e:
            sys.stdout = old_stdout
            print(f"Error executing cell {idx + 1}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise e
        finally:
            sys.stdout = old_stdout

        captured_text = stdout_capture.getvalue()
        if captured_text:
            cell.outputs.append(nbformat.v4.new_output(
                output_type="stream",
                name="stdout",
                text=captured_text
            ))

        # Capture any matplotlib figures created by plt.show() or open figures
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close(fig)

            cell.outputs.append(nbformat.v4.new_output(
                output_type="display_data",
                data={
                    "image/png": img_b64,
                    "text/plain": "<Figure size>"
                },
                metadata={}
            ))

        exec_count += 1

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"✅ Successfully rendered and saved notebook with all outputs to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    render_notebook()
