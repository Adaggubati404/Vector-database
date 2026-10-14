# Vector-database: cleanup checklist (updated after your Oct 13 commits)

Already done: root `.DS_Store` and `.vscode/` deleted, `LICENSE` added.

## Still to do
- [ ] Delete `CLEANUP.md` from the repo. It's a note for you, not part of the project.
- [ ] Delete `vectordb/.DS_Store`, which is still tracked.
- [ ] Optional: untrack the generated `data/meta.json` and `data/vectors.npy`, since `python -m vectordb seed` recreates them.
- [ ] Replace `README.md` with the new version.

```bash
git rm CLEANUP.md vectordb/.DS_Store
git rm -r --cached data        # optional
```

## Lines to add to `.gitignore`
```gitignore
.DS_Store
.vscode/
# only if you untrack data/:
data/
```
