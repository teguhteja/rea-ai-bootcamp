
# Git Remote Add and Remote Repositories

## Theme: Remote Repositories (Source: amazonaws.com)

Software development often involves collaboration among multiple team members. To manage projects efficiently, repositories are typically not stored individually on team members' local machines. Instead, repositories are merged and stored on a **remote server**.

### Where to Store the Remote Repository?
You can store the repository:
- On an **office server**.
- On services like **GitHub**, **GitLab**, **Bitbucket**, etc. GitHub is the most popular choice for hosting remote repositories, especially for open-source projects.

---

## Understanding Remote Repositories

### What is a Remote URL?
A **remote URL** is where Git stores your source code remotely. It is different from the local repository and is typically located on a server.

### Types of Remote URLs:
1. **HTTPS URL**: Example: `https://github.com/user/repo.git`
2. **SSH URL**: Example: `git@github.com:user/repo.git`

Git associates the remote URL with a name (default: `origin`), stored in the `.git/config` file in the root directory. Example configuration:

```plaintext
[remote "origin"]
url = https://github.com/user/repo.git
fetch = +refs/heads/*:refs/remotes/origin/*
```

---

## Adding Remote Repositories to Local Git

Use the `git remote add` command to associate a name with a remote URL:

```bash
git remote add origin <REMOTE_URL>
```

### Example:
```bash
[user@localhost]$ git remote add origin https://github.com/user/repo.git
[user@localhost]$ git remote -v
> origin https://github.com/user/repo.git (fetch)
> origin https://github.com/user/repo.git (push)
```

---

## Changing a Remote Repository’s URL

Use the `git remote set-url` command to update the URL.

- **HTTPS URL**: `https://github.com/USERNAME/REPOSITORY.git`
- **SSH URL**: `git@github.com:USERNAME/REPOSITORY.git`

### Example:
```bash
[user@localhost]$ git remote set-url origin git@github.com:USERNAME/REPOSITORY.git
```

---

## Removing a Remote Repository

Use `git remote rm` to remove a remote repository from the local configuration. 

### Example:
```bash
[user@localhost]$ git remote -v
> origin https://github.com/OWNER/REPOSITORY.git (fetch)
> destination https://github.com/FORKER/REPOSITORY.git (fetch)

[user@localhost]$ git remote rm destination
[user@localhost]$ git remote -v
> origin https://github.com/OWNER/REPOSITORY.git (fetch)
```

---

## Managing Multiple Remotes

### Adding Multiple Remotes:
Use unique names for each remote:

```bash
[user@localhost]$ git remote add origin git@github.com:user/repo.git
[user@localhost]$ git remote add upstream git@bitbucket.org:user/repo.git
[user@localhost]$ git remote add custom git@gitlab.com:user/repo.git
```

### Configuring a Primary Remote:
Set the local branch to track a remote branch:
```bash
[user@localhost]$ git checkout BRANCH
[user@localhost]$ git branch -u origin/BRANCH
```

---

## Listing All Remotes

Use `git remote -v` to view all configured remotes:

```bash
[user@localhost]$ git remote -v
origin    git@github.com:user/repo.git (fetch)
upstream  git@bitbucket.org:user/repo.git (fetch)
custom    git@gitlab.com:user/repo.git (fetch)
```

---

## Removing a Remote

To remove a remote:
```bash
[user@localhost]$ git remote remove REMOTE-NAME
```

### Example:
```bash
[user@localhost]$ git remote remove upstream
```
