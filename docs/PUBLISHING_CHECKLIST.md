# Publishing checklist (for the authors)

1. Review whether code written by collaborators can be distributed publicly.
2. Select a software license deliberately (MIT, BSD-3-Clause, etc.) **only if authorized by the authors**; do not assume third-party or raw NASA data inherit the same license.
3. Do not include original ICP recordings, patient/animal identifiers, institutional raw files, account credentials, or local access tokens.
4. Check that processed NASA data contain only derived measures, not redistributable raw waveforms.
5. Create a public GitHub repository and upload the CONTENTS of this folder. GitHub does not automatically extract a ZIP uploaded as one file.
6. Confirm that `python scripts/verify_results.py` succeeds in a clean environment.
7. Confirm that figure generation succeeds: `python scripts/plot_paper_figures.py`.
8. When the NASA raw data can be downloaded, test the NASA converter and window-fitting pipeline end-to-end on an independent machine; update this README with exact environment versions and runtime requirements.
9. Publish a fixed release (`v1.0`) only when everything is checked. Optionally archive the tagged release on Zenodo for a persistent DOI.
10. Replace manuscript placeholder URL with the real GitHub release URL; do not say it is publicly available before publishing.
