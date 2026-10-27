# Training data

`documents.json` contains 150 short English/Persian documents written and generated for this course. The source, MIT license, generation seed, and hash are recorded in `manifest.json`. Rebuild the same data with `python scripts/build_data.py`.

Documents are split before tokenization: 114 train, 18 validation, and 18 test. Scenario combinations vary across splits, while sentence templates and grammar remain intentionally shared. The data is suitable for learning model mechanics and local pattern learning; it is not a natural-language benchmark.

The vocabulary is built from the training split only. Unknown evaluation characters map to ID zero and their rate is reported. The service rejects unknown input explicitly. Windows are built independently within each split and may cross document boundaries; two newline characters act as the separator. This is acceptable for continuation training, but the boundary is not a special token.

Optional WikiText data is stored in `data/external/` and excluded from Git and release bundles. Its license is separate from this project's MIT code; notebook 07 records the source, revision, and conditions.
