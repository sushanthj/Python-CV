---
layout: page
title: Git Concepts
permalink: git_concepts
nav_order: 14
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Before you Begin
{: .fs-9 }

[Reference](https://www.w3schools.com/git/git_getstarted.asp?remote=github){: .btn .fs-5 .mb-4 .mb-md-0}

In the above link, follow the procedures, but instead of using username and password each time, setup the ssh keys and use them more often

*ssh keys are found in ./.ssh folder (or lookup keygen to generate your keys)*

# Basics of generating new content in local and pushing to github

## Process for adding to a github page

git add . \
git commit -m "made new code" \
git push or git push origin develop (if you cloned from develop branch)

## If you want to track a different branch

- git branch --set-upstream-to=origin/master \
  git add . \
  git push

or make a new remote

- git remote add ts_origin_wiki git@github.com:sjayanth21/BR_Wiki.git \
  git push --set-upstream ts_origin_wiki master \
  git push ts_origin_wiki_master


## Working with remotes

Any folder can have a number of remotes like:
origin and ts_origin_github

To make local branch master track a different remote branch (branch in your cloud github repo) do:
git branch --set-upstream-to=origin/master 

or git branch --set-upstream-to=origin/develop

## If you cloned a repo, forked your own branch (using git checkout)

You may need to pull from upstream to update your codebase \
However, running a simple 'git pull' may throw merge conflicts

So do the following
1. Run a 'git fetch' to get the updates on all branches (and if any new branch has been added)
2. In your personal branch commit all changes by doing: git add, commit and push
3. sudo apt install meld
4. Now to get the upstream updates do 'git checkout develop' (whichever is the main branch)
5. Now to put this in your personal branch run 'git checkout feature/sj'
6. Now we do the actual merging using 'git merge develop' (this will merge everythin in deveop into the current branch viz feature/sj)
7. The above step would have thrown some merge conflicts, to solve that run 'git mergetool'
8. The above step opens meld, make all necessary resolutions and save
9. Now our codebase would have been updated to whatever we resolved in meld
10. Now run 'git commit' without any arguments as it is a 'merge commit'
11. Now as usual do 'git push origin feature/sj' to push your updated personal branch to github

## Points to Note

- If you checkout a file 'git checkout blade.py' it resets the file to whatever is the latest from that branch in upstream

- If you want to physically add or change remotes go to the respective folder and do 'nano .git/config'

- the correct syntax for the merge command is: \
'git merge ts_origin/master' \
What this does is that if the current branch is origin/develop it will merge the files of \
current branch i.e origin/develop with ts_origin/master

- Note that even if ts_origin/master is in ts_github account and origin/master is in sushanthj github account, it will still merge as long as remotes exist for both these accounts. If remotes don't exist, you can always add as shown up above 

### Concepts for working with two repos or two repos on two different github accounts:

Basically locally you will have 'master' branch if you do 'git branch' \
This master can track two upstream branches using two different remotes \
One remote is added automatically when you clone the repo \
The next remote will have to be added manually to your other git account or other repo

Then to push the same commit to both branches first do 'git push' \
and see which repo it pushes to (say it pushes to origin/master \
Then do 'git push --set-upstream ts_origin/develop' to push to your second repo \
However, do note that your local branch always tracks to the latest branch you pushed to \
i.e if you do a git pull, it will pull from the latest branch to which you pushed \
in this case it will pull from ts_origin/develop

### Saving a patch file
If you have changes made which you want to save locally and not push to remote, you can save a patch file

```bash
git diff > new_changes.patch
```

Now to apply this patch onto any branch, do:
```bash
git apply new_changes.patch
```

### Saving changes by stashing

Instead of saving a specific file for changes (such as a patch file), you could also stash your changes locally

```
git stash
```
The above command will stash all tracked changes. You could also stash only committed changes. Refer: [stashing](https://www.atlassian.com/git/tutorials/saving-changes/git-stash)

To then apply the stashed changes (one time use only as pop will remove from stash)
```
git stash pop
```

To apply without popping do:
```
git stash apply
```

To remove any particular item in stash:
```
git stash drop
```

To view all entries in stash and then apply specific one do:
```
git stash list
git stash apply n
```

n = stash item number