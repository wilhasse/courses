# Course

Boot.Dev Git Course

## Ch 1 - Setup

Install

```bash
go install github.com/bootdotdev/bootdev@latest
go env GOPATH
# add path to Library Path
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
