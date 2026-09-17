# XGrammar Unicode escape patch

`unicode-escapes.patch` applies to XGrammar `0.2.5.post1`, fetched by the
SHA-256-pinned source archive in the root `CMakeLists.txt`.

The upstream generic JSON string rules accept lone UTF-16 surrogate escapes.
Those values fail nlohmann JSON and Pydantic's JSON parser. Require a high
surrogate escape (`D800`–`DBFF`) to be followed immediately by a low surrogate
escape (`DC00`–`DFFF`), and reject standalone low surrogates. Valid BMP escapes,
paired supplementary characters, ordinary escapes, and hexadecimal letter case
remain supported. This also covers generic object keys.

The patch changes only the `escape` rule in `cpp/grammar.cc` and the
`basic_escape` rule in `cpp/json_schema_converter.cc`. Native constraint tests
cover acceptance, rejection, incomplete-pair masks, and speculative rollback.

[RFC 8259 §8.2](https://www.rfc-editor.org/rfc/rfc8259.html#section-8.2)
describes the interoperability problem: its ABNF permits unpaired surrogates,
while consumers can reject them. Gewell generates Unicode scalar values.

Upstream XGrammar is licensed under
[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).
Its `LICENSE` and `NOTICE` are retained in the downloaded source directory.
