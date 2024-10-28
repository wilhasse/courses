# Git Guide


Useful commands

## Authenticate

```bash
git remote set-url origin https://wilhasse:***TOKEN***@github.com/wilhasse/courses
git config --global credential.helper store
git push
```

## Basic

```bash
# clone an repository
git clone ***REPO***
# update repository
git pull
# see status , modifiled files
git status
# add all modifications to the commit area
git add .
# add only one file or folder
git add ***FILE***
# commit changes to local repo
git commit -m "***MESSAGE***"
# push modifications to github
git push
```

## Useful

```bash
# save your work stashing modification before switching branches
git satsh save "***YOUR WORK***"
# recovery your changes
git stash apply
# change last commit message
git commit --amend -m "***NEW MESSAGE***"
```

## Working with branches

```bash
# list all branches
git branch -a
# go to specific branch
git checkout ***BRANCH NAME***
# create a new branch , pay attention to where you are the branch starts on the branch you are
git checkout -b ***NEW BRANCH***
# delete local branch
git branch -D ***BRANCH NAME***
# delete branch on github
git push origin --delete ***BRANCH NAME***
# change branch name
git branch -m ***BRANCH*** ***NEW BRANCH***
# change branch on github
git push origin -u ***NEW BRANCH NAME***
# bring commits to current branch
git cherry-pick ***COMMIT RANGE***
```
