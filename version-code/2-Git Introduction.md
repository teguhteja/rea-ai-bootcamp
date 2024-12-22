# Git Introduction

## What is Git?

Git is a distributed Version Control System (VCS) for managing files within folders (Repositories). File change history is saved using a series of commits.

With Git, multiple versions of a file can be created without duplicating files or using "Save As." For example, when creating a login feature:

### Example Folder Structure:
```
📁 applications (repository)
⬇🔵: "feat: initialize project" (commit)
⬇🔵: "feat: added view for login and register" (commit)
⬇🔵: "feat: create controller user" (commit)
⬇🔵: "fix: fix error in email input format" (commit)
⬇🔴 Peek at saved information:
  Commit: 6bebc5658521d98f3eeadb42362e43bc072f0273
  Author: User <example@gmail.com>
  Date: Tue Jun 28 2022 18:53:48 GMT+0700
  Message: feat: create authentication and authorization models
⬇🔵: ...
📁 login
```

Changes made to the `login` folder are saved as a series of commits. Each commit acts as a "snapshot" of changes made.

### Commit Information:
- **Commit:** Hash/ID as a unique marker for each commit.
- **Author:** The person who made the commit (useful for team collaboration).
- **Date:** When the commit was made.
- **Message:** Description of what was done in the commit.

---

## Key Differences Between Git and Other VCS

### 1. Snapshots, Not Differences
Most VCS save file changes using a delta-based method. Git, however, saves data as snapshots:
- A snapshot includes all files in the project.
- Unchanged files are linked to previous versions.
- This creates a "snapshot stream," where each commit represents a complete project state.

#### Example:
![Snapshots vs Deltas](https://cdn.prod.website-files.com/5ed2ed8ea90a9c03bf1b5a18/613251677413b8c03d79ddac_upS1_fd17QlxeKZ8oc9kmkM6fcHBOqX8_YjfZpjju-zkrZ-loZjsffcDkgFwd-hBgOHTxgellTfX3BCr9KZbi-iY5FHhNydM7qWRhlr1AwtgWZRgvorWMNH7NOg9Ms9EbAjCrCao%3Ds0.png)

---

### 2. Nearly Every Operation Is Local
Git stores the entire project history on the local disk, making most operations extremely fast:
- Browsing history, commits, or changes is instant and doesn't require server access.
- Work can be done offline, and commits can be synced later when an internet connection is available.

---

### 3. Git Has Integrity
Git ensures no data loss or file corruption using SHA-1 hashes. Each file or directory structure has a unique 40-character hexadecimal hash.

#### Example Hash:
```
24b9da6552252987aa493b52f8696cd6d3b00373
```

---

### 4. Git Generally Only Adds Data
Git rarely overwrites or deletes data. Committed changes are almost impossible to lose, especially when uploaded to platforms like GitHub or GitLab.

---

## The Three States of Git

Git tracks files in three states:
1. **Modified:** File changes not yet staged.
2. **Staged:** File changes marked for the next commit.
3. **Committed:** Changes saved permanently in the repository.

These states correspond to three areas of a Git project:
- **Working Tree:** Local changes.
- **Staging Area:** Changes to be committed.
- **Git Directory:** Committed data.

---

## Installing Git

### Mac and Linux
Git is usually pre-installed. To check:
```bash
git --version
```
If Git isn’t installed, follow [this guide](https://git-scm.com/download/linux) for Linux distributions.

### Windows
Download Git from [here](https://git-scm.com/download/win). Use Git Bash or Command Prompt to interact with Git.

---

## Basic Configuration

### First-Time Git Setup
Set up your Git environment using `git config`:

1. Set your username and email:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "youremail@example.com"
   ```

2. Configure a default text editor (e.g., VS Code):
   ```bash
   git config --global core.editor "code --wait"
   ```

3. Set the default branch name:
   ```bash
   git config --global init.defaultBranch main
   ```

---

### Checking Git Settings
View all settings:
```bash
git config --list
```

Check specific settings:
```bash
git config user.name
```

Example output:
```bash
user.name=Your Name
user.email=youremail@example.com
color.status=auto
color.branch=auto
color.interactive=auto
color.diff=auto
```


### Summary:
This Markdown format ensures clarity and organizes information into logical sections, making it easy to follow and understand Git concepts, installation, and configuration.