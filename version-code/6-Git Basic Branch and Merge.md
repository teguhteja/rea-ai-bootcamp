
# Basic Branch and Merge

## Git Branch  
![Git Branch](source: sitepoint.com)

### Branch Concept
When creating a new repository, a main branch is automatically created. You can confirm this by running the following commands:

```bash
git init
git status
```

Output:
```
On branch master
No commits yet
nothing to commit (create/copy files and use "git add" to track)
```

The default branch created is `master`. By creating a new branch, the current state of `master` will be preserved, and you can independently add commits to the new branch. Later, you can merge the commits back into `master`.

### Creating a New Branch
1. **Start with a Commit:**  
   ```bash
   git add .
   git commit -m "Adding authentication"
   ```
   Output:  
   ```
   [master (root-commit) c0e5bfa] Adding authentication
     1 file changed, 1 insertion(+)
     create mode 100644 auth.sh
   ```

2. **View Commit Log:**  
   ```bash
   git log
   ```

3. **Create and Checkout a New Branch:**  
   ```bash
   git checkout -b fix-authentication-bug
   ```
   Output:  
   ```
   Switched to a new branch 'fix-authentication-bug'
   ```

4. **Verify the Branches:**  
   ```bash
   git branch
   ```
   Output:  
   ```
   * fix-authentication-bug
     master
   ```

5. **Compare Logs:**  
   ```bash
   git log
   git log master
   ```

6. **Add a Commit in the New Branch:**  
   ```bash
   git add .
   git commit -m "Fixing bug in auth"
   ```
   Output:  
   ```
   commit c2507da (HEAD -> fix-authentication-bug)
   ```

### HEAD and Branch
The `HEAD` indicates the branch you are currently working on.

---

## Moving Between Branches and Merging

### Merging Branches
To merge a branch into another:  
1. Switch to the target branch:  
   ```bash
   git checkout master
   ```
2. Merge the desired branch:  
   ```bash
   git merge fix-authentication-bug
   ```
   Output:  
   ```
   Updating c0e5bfa..c2507da
   Fast forward
     auth.sh | 1 +
     1 file changed, 1 insertion(+)
   ```

3. Verify the Commit Log:  
   ```bash
   git log
   ```

---

## Git Merge Conflict Resolution
Merging is usually automatic, but conflicts may arise. When a conflict occurs, Git requires manual intervention.  

For more details, refer to the [Git Documentation](https://git-scm.com/doc).
