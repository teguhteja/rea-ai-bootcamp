# Introduction to Version Control

## Before Version Control Systems

When working on projects, we often make revisions. For instance, while creating a proposal or thesis document:

### Example: Writing a Thesis
1. We usually create a 📁 folder, e.g., named `thesis`.
2. Then, we start writing in a 📄 file named `thesis.docx`.
3. When there’s a revision, we create a new version:
   - 📄 `thesis_revisi-1.docx`
   - 📄 `thesis_revision-3.docx`
   - 📄 `thesis_revisi-4.docx`
   - 📄 `thesis_revision-5.docx`
   - 📄 `thesis_revision-6.docx`
   - ... and so on.
4. Eventually, we might end up with 📄 `thesis_final_ready_sidang.docx` 😄.

### Why Do We Do This?
This approach ensures we can:
- Track changes between document revisions.
- Easily revert to a previous version when needed.

For software developers, code revisions often involve tens or hundreds of lines across multiple files. Using a Version Control System (VCS), we can manage all versions of files in a centralized manner.

Imagine revising dozens of files and adding features to your code. If you’re later asked to undo those changes, managing this without a VCS would be challenging. Hence, the importance of VCS!

---

## What is a Version Control System?

A **Version Control System (VCS)** is a system that records changes to a file or set of files over time. It is primarily used for application programs or source code, allowing us to:

- Restore files to a previous state.
- Revert entire projects to earlier states.
- Compare changes over time.
- Identify who made specific changes and when.
- Quickly recover from mistakes or file loss.

---

## Why Use Version Control?

A VCS enables team collaboration without losing or overwriting anyone's work. Here's how it works:

1. A developer makes changes to code in one or more files.
2. The VCS saves a representation of those changes.
3. A history of code changes or versions is created, accessible to all team members, regardless of location.

### Key Benefits:
- **Track Modifications**  
  A VCS logs every change made to the code, useful for debugging and root cause analysis.
  - Example: **Git Log**

- **Compare Earlier Versions**  
  Saved versions allow developers to revert or compare previous states to resolve errors with minimal disruption.
  - Example: **Git Diff**

---

## Best Version Control Systems Today

Some of the best VCS available in the market are (Source: simplilearn.com):
- Git
- Subversion (SVN)
- Mercurial
- Perforce

---

### Summary:
This Markdown content includes headings, subheadings, bullet points, and examples formatted clearly for better readability. It provides a comprehensive introduction to Version Control Systems and their benefits.