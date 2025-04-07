## ✅ 1. **How to Build and Launch Sphinx Docs**

Assuming you’re using **Markdown** + **Sphinx** + **MyST**, here’s how to build your docs locally and preview them:

### 🛠️ Install Dependencies

Create a `docs/requirements.txt` file:

```txt
sphinx
myst-parser
furo
```

Then install:

```bash
pip install -r docs/requirements.txt
```

---

### 📁 Folder Structure (minimal example)

```text
docs/
├── source/
│   ├── index.md
│   ├── installation.md
│   ├── usage.md
│   ├── stages.md
│   ├── cli.md
│   ├── architecture.md
│   └── conf.py
├── build/
```

> Your `index.md` should include a `{toctree}` to organize the structure.

Example `index.md`:

````markdown
# 📚 Documentation

```{toctree}
:maxdepth: 2
:caption: Contents

installation
usage
stages
cli
architecture
api/modules
```
````

---

### 🧱 Generate API Docs (Optional)

If you want to auto-generate doc pages from your Python modules:

```bash
sphinx-apidoc -o docs/source/api/ src/  # or your module folder
```

Then update `index.md` to include `api/modules`.

---

### 🔨 Build the Docs

From the project root:

```bash
cd docs
make html
```

> On Windows (no `make`):

```bash
sphinx-build -b html source build
```

Then open:

```
docs/build/html/index.html
```