# Course

Boot.Dev Git Course

## Ch 1 - Setup

Install

```bash
go install github.com/bootdotdev/bootdev@latest
go env GOPATH
# add path to Library Path adding folder bin
# /etc/profile
bootdev --version
bootdev login
```

Quick config

```bash
git config --get user.name
git config --get user.email
#empty
git config --add --global user.name "myuser"
git config --add --global user.email "myemail"
```

## Ch 2 - Repositories

Create repository

```bash
mkdir webflyx
cd webflyx
git init
```

```bash
# files
git status

# staging all
git add .

# commit
git commit -m "Test"

# change comment
git commit --amend -m "New comment"

# show log
git log
```

## Ch 3 - Internals

```bash
# run git log to see hash of commit
git log
# commit 2c8d99f1243b5489f8c1c185e5fae57701aee301

# see content find two first hash caracter as dir 2c
 ls .git/objects/
 cd .git/objects/2c
# file with rest of caracters
ls -la
# 8d99f1243b5489f8c1c185e5fae57701aee301
```

```bash
# cat-file to see information about commit
git cat-file -p 2c8d99f1243b5489f8c1c185e5fae57701aee301
# tree , author , committer andd comment
# tree 5b21d4f16a4b07a6cde5a3242187f6a5a68b060f

# cat-file in tree hash you can see blob hash and file name
git cat-file -p 5b21d4f16a4b07a6cde5a3242187f6a5a68b060f
# 100644 blob ef7e93fc61a91deecaa551c4707e4c3049af42c9    contents.md
```

## Ch 4 - Config

```bash
# change the default branch to main
git config --global init.defaultBranch main
# add any local information using key-value
git config --add my.info "some"
# get information
git config --get my.info
# remove
git config --unset my.info
# you can have multiple values for the same key, how to remove all of them
git config --unset-all my.info
# remove the entire section
git config --remove-section my
```

## Ch 5 - Branching

```bash
# current branch
git branch
# rename master to main
git branch -m master main
# only create a new branch
git branch my_new_branch
# this command creates and switches to new branch
git switch -c my_new_branch
# only switch to a branch
git switch main
# or way to switch, the old way:
git checkout main
# log in a compact form
git log --oneline
# see all branches dir
ls -la .git/refs/heads/
```

## Ch 6 - Merge

```bash
# show branch graph
git log --oneline --graph --all
# in main merge ddifferences from my_new_branch
git merge my_new_branch
# show graph with merge
git log --oneline --decorate --graph --parents
# delete branch
git branch -d my_new_branch
```

## Ch 7 - Rebase

```bash
# new branch at position COMMITHASH (change it to your commit hash)
git switch -c update_dune COMMITHASH
# rebase main on update_dune
git rebase main
# make rebase default option
git config --global pull.rebase true
```

## Ch 8 - Reset

```bash
# undo the last commit and get the modification unstaged
git reset --soft HEAD~1
# remove the unstaged modification
git reset --hard
# check status
git status
```

## Ch 9 - Remote

```bash
# create local repository
mkdir webflyx_local
cd webflyx_local
git init
# adding a remote inside webflyx_local
git remote add origin ../webflyx
# get the content of a remote repo
git fetch
# see the log from that remote repo
git log origin/update_dune --oneline
# merge the remote repo into local repo
git merge origin/main
# list remote repo
git ls-remote
```

## Ch 10 - Github

```bash
# install git
curl -sS https://webi.sh/gh | sh
# login
gh auth login
# add remote git repo
cd webflyx
git remote add origin https://github.com/wilhasse/webflyx.git
# install git gh
# https://github.com/cli/cli/blob/trunk/docs/install_linux.md
# login
gh auth login
# update local copy
git pull origin main
# merge on a pull
git config pull.rebase false
```

## Ch 11 - Gitignore

```bash
#c reate a file in any path it will ignore from that path
.gitignore
# add things you want to ignore
src/node_modules etc
# Ignore all .txt files
*.txt
# note exclude this file from all .txt
!important.txt
```
