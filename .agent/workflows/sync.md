---
description: Mirror the root api/ folder to AI_BOT/api and validate syntax
---

// turbo-all
1. Mirror the folder using robocopy
```powershell
robocopy api AI_BOT\api /MIR /XD __pycache__ .pytest_cache .mypy_cache .ruff_cache /XF *.pyc /R:1 /W:1
```

2. Validate the mirrored Python files syntax
```powershell
cd "AI_BOT\api"; py -m py_compile (Get-ChildItem -Recurse -Filter *.py | % FullName)
```
