# Capture screenshots for GitHub

Open https://huggingface.co/spaces/dseshukumar/industrial-asset-health-ai in a desktop browser at 100% zoom. Use Windows Snipping Tool (`Win` + `Shift` + `S`) and save each image as PNG. Create a `screenshots` folder in the GitHub repo.

1. **`question-form.png`**: Open **Ask a diagnostic question**. Load a held-out example so the question and 2–5 choices are filled. Include the title and question-entry fields.
2. **`comparison.png`**: Click **Compare QLoRA and hybrid retrieval**. Capture both prediction cards side by side, together with timings and correctness if visible. Do not crop out labels distinguishing methods.
3. **`benchmark.png`**: Open **Research benchmark** and capture the published accuracy table and benchmark disclaimer.

Do not fabricate screenshots, present static saved outputs as live inference, expose private credentials, or show confidential industrial data. Remove personal browser tabs/bookmarks by capturing just the web application area. Upload the images into GitHub under `screenshots/`, then uncomment the screenshot Markdown lines in `README.md`.
