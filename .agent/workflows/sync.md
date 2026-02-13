---
description: Validate api/ folder and deploy to Hugging Face Space
---

// turbo-all
1. Validate Python syntax in the api folder
```powershell
cd api; py -m py_compile (Get-ChildItem -Recurse -Filter *.py | % FullName)
```

2. Push updates to Hugging Face
```powershell
cd api; git add .; git commit -m "Manual sync and deploy"; git push hf master:main --force
```
