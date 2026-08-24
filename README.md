# grammatike / γραμματικὴ τέχνη


> *See [release history](https://github.com/neelsmith/grammatike/blob/main/releases.md)*.

`grammatike` is a python package leveraging LLMs with [dspy](https://dspy.ai) to analyze the syntax of passages of ancient Greek.

It offers an alternative analytic scheme to [Universal Dependencies](https://universaldependencies.org), designed to describe the syntax of ancient Greek in familiar terms that are practical for research and teaching focused on ancient Greek.



## Related work

`grammatike` shares similar goals and design principles with [arsgrammatica](https://github.com/neelsmith/arsgrammatica/tree/main), a parallel package for analyzing Latin syntax.


Released under the [GNU General Public License v3 or later](LICENSE).


## Installing

To use `grammatike` from another project, install it straight from this repository (no PyPI account or release process needed):

```sh
pip install git+https://github.com/neelsmith/grammatike.git
```

That installs whatever's currently on the `main` branch. Pin to a specific branch, tag, or commit by appending `@<ref>`, e.g. `pip install git+https://github.com/neelsmith/grammatike.git@wip` for the development branch, or `@v0.1.0` once a version is tagged. Either way, only `grammatike/` itself is installed as a package -- `dspy` and `pydantic` come along automatically as declared dependencies; the tests and other repo scripts are not part of the installed package and aren't needed to use it.

Working on `grammatike` itself (this repo checked out locally) rather than depending on it from elsewhere: `pip install -e .` from the repo root installs it in editable mode, so source edits take effect immediately without reinstalling. Add `[test]` (or `[dev]`, which also covers `docs/build_api_docs.py` and the `marimo/` notebooks) to also install everything `pytest` needs -- `pip install -e ".[test]"` -- since those are test/tooling dependencies, not runtime ones, and a plain `pip install -e .` deliberately doesn't pull them in. See TESTING.md.


## Using `grammatike`

- [USAGE.md](https://github.com/neelsmith/grammatike/blob/main/USAGE.md)
- [TESTING.md](https://github.com/neelsmith/grammatike/blob/main/TESTING.md)
- [OPTIMIZING.md](https://github.com/neelsmith/grammatike/blob/main/OPTIMIZING.md)
- [BAKEOFF.md](https://github.com/neelsmith/grammatike/blob/main/BAKEOFF.md)
- [DEVELOPMENT.md](https://github.com/neelsmith/grammatike/blob/main/DEVELOPMENT.md) -- how the above fit together into one development loop
- [API documentation](https://neelsmith.github.io/grammatike/grammatike-api-docs.html)

See the [project issue tracker](https://github.com/neelsmith/grammatike/issues) for known gaps and work in progress.


## The analytic scheme

The full syntactic scheme -- verbal expressions, relation labels, token types, and worked Greek examples for each -- is documented in [`syntax_model.md`](https://github.com/neelsmith/grammatike/blob/main/syntax_model.md).