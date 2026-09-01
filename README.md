# grammatike / γραμματικὴ τέχνη


> *See [release history](https://github.com/neelsmith/grammatike/blob/main/releases.md)*.

`grammatike` is a python package leveraging LLMs with [dspy](https://dspy.ai) to analyze the syntax of passages of ancient Greek.

It offers an alternative analytic scheme to [Universal Dependencies](https://universaldependencies.org), designed to describe the syntax of ancient Greek in familiar terms that are practical for research and teaching focused on ancient Greek.

## Related work

`grammatike` shares similar goals and design principles with [arsgrammatica](https://github.com/neelsmith/arsgrammatica/tree/main), a parallel package for analyzing Latin syntax.


Released under the [GNU General Public License v3 or later](LICENSE).




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