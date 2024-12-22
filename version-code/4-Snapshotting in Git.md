# Snapshotting in Git

## 📸 Introduction to Git Snapshots
Git uses the **snapshot method** to capture the state of the repository when changes are saved using Git commands. This method:
- Saves only the changed files and folders in the snapshot.
- References unchanged files and folders from the previous snapshot.

### How Snapshots Work:
1. **Working Directory**: Where files are created/modified.
2. **Staging Area**: Select files to commit.
3. **Committed Stage**: Saves the project history.

> **Note**: Before running Git commands, ensure you are in a Git repository. Use `git init` to initialize one if needed.

---

## 🗂 File Stages in Git
### **Stages**:
- **Not Staged**: Files created/modified in the working directory but not added to staging.
- **Staged**: Files selected to be saved in the next commit.
- **Committed**: Saved changes as part of the project history.

### **Useful Commands**:
- `git status`: View files in working/staging areas.
- `git log`: View commit history.

---

## ➕ Adding Files (`git add`)
### **Purpose**:
Moves files to the **staging area** for the next commit.

### **Common Usages**:
- `git add <file>`: Add a specific file (e.g., `git add file1.txt`).
- `git add .`: Add all files in the current folder.

### **Risks**:
1. Overly large commits with unrelated changes.
2. Accidental inclusion of sensitive files (e.g., passwords).

### **Undo Added Files**:
- Use `git reset <file>` to move a file from **staging** back to the **working directory**.

---

## ✅ Committing Changes (`git commit`)
### **Purpose**:
Moves files from the **staging area** to the **committed stage** with a descriptive message.

### **Best Practices for Commit Messages**:
- Be short and precise.
- Use present tense.
- Categorize with prefixes (e.g., `feat:`, `fix:`, `docs:`).

### **Examples**:
```bash
git commit -m "feat: create file structure for Git guides"
git commit -m "fix: resolve broken links in documentation"
```

---

## 🔄 Reverting and Amending Commits
- **Revert a Commit**: 
   ```bash
   git revert <commit_id>
   ```
   Creates a new commit to undo changes.

- **Amend a Commit**:
   ```bash
   git commit --amend
   ```
   Updates the last commit with new changes or a corrected message.
   > **Warning**: Do not use after pushing the commit.

---

## 🔀 Switching Commits (`git checkout`)
- Use `git checkout <commit_id>` to move to a specific commit.
- Use `git checkout -` to return to the previous position.

---

## 🗑 Removing Files (`git rm`)
- **Remove from Staging**:
   ```bash
   git rm --cached <file>
   ```
   Moves a file from **staging** to **working directory** without deleting it.

---

## 🔍 Inspecting Changes (`git diff`)
### **Purpose**:
Compares the **working directory** with the repository.

### **Example**:
```bash
# Modify a file
echo "new content" > file.txt

# Compare changes
git diff
```

### **Output Explanation**:
- `-`: Removed lines.
- `+`: Added lines.
- Lines with both indicate replacements.

---

## 📜 Summary
Git’s snapshotting method ensures efficient and incremental storage of changes. By understanding the commands and processes:
1. Manage your repository efficiently.
2. Avoid accidental commits.
3. Maintain a clean and readable project history.
