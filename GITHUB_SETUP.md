# Publish to GitHub

Create an empty private repository named tianjin-dr-preprocessing on GitHub. Do not initialize it with a README. From this extracted folder, run (replace YOUR_USERNAME):

```bash
git init -b main
git add .
git commit -m "Add longitudinal DR preprocessing pipeline"
git remote add origin https://github.com/YOUR_USERNAME/tianjin-dr-preprocessing.git
git push -u origin main
```

Only source code, documentation and tests are included. Dataset images, patient spreadsheets, weights and generated outputs are excluded by .gitignore. The workflow runs unit tests; it does not download medical images or establish segmentation accuracy. See VALIDATION.md for execution limitations.
