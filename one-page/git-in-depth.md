# Course

Git In-Depth  
FrontendMasters - Nina Zakharenko

https://frontendmasters.com/courses/git-in-depth/  
https://github.com/nnja/advanced-git

# Modules

## Git Foundations

Init a new repository, inside directory:

```bash
git init
tree .git
C:\GIT-TESTE\.GIT
├───hooks
├───info
├───objects
│   ├───info
│   └───pack
└───refs
    ├───heads
    └───tags
```

First commit

```bash
echo 'Hello World!' > hello.txt
git add hello.txt
git commit -m "initial commit"
//[master (root-commit) 0487c2d] initial commit
// 1 file changed, 1 insertion(+)
// create mode 100644 hello.txt
```

Inspect the objects  
The directory name is the first two digits, the file follow the SHA1 digits

```bash
cd objects\10
dir
// 86f661aad7332d7e343cc49afc46f942c00dfd
git cat-file -t 1086f6
//tree

git cat-file -p 1086f6
//100644 blob d38752edff821f8f38fe3efbe3d80e0ae997f5b1    hello.txt
```

Look at refs

```bash
C:\git-teste>type .git\HEAD
//ref: refs/heads/master

C:\git-teste>type .git\refs\heads\master
//0487c2d932e9d09329c72d09c1f8542c566f6270

C:\git-teste>git log --oneline
//0487c2d (HEAD -> master) initial commit
```

Creating a new branch

```bash
git branch new_branch
tree /f .git\refs\heads
//C:\GIT-TESTE\.GIT\REFS\HEADS
//    master
//    new_branch
```

## Git Areas and Stashing

1. Working Area - where you change your code that are not in the staging area. Untracked files.
2. Staging Area - what files are going to be part of the next commit. The changes between the current commit and the next one.
3. The repository - the files git knows about. It contains all of yout commits

Moving to the staging area

```bash
git add file
git add . // (all files)
git add -p // interactive staging
```

Stash - Save your work without commiting it.

```bash
git stash save "name your stash"
git status
git stash list
git stash show stash@{0}
git stash apply stash@{0}
```

## References, Commits, Branches

What is a branch? It is just a pointer to a particular commit  
Head is how git knows what branch you are currently on  
Tag is a simple pointer to a commit (like branch)
Branch and tag are pointers but branch is dynamic it moves after each commit. Tag ist just a name and keep the same as new commits are added.  
Detached head state - it happens after git checkout to a commit, news commits are not referenced and can be lost if yout don't attach to a new branch.

Where's your HEAD

```bash
git checkout exercise3
type .git\HEAD
// ref: refs/heads/exercise3
git branch
//  exercise2
//* exercise3
//  master
```

Where are your refs?

```bash
git show-ref --heads
// 43388fee19744e8893467331d7853a6475a227b8 refs/heads/exercise2
// e348ebc1187cb3b4066b1e9432a614b464bf9d07 refs/heads/exercise3
// 43388fee19744e8893467331d7853a6475a227b8 refs/heads/master
git cat-file -p 43388fee19744e8893467331d7853a6475a227b8
... Initial commit
```

Detached HEAD

```bash
git log --oneline
git checkout e348ebc
echo "This is a test file" > dangle.txt
git add dangle.txt
git commit -m "This is a dangling commit"
git checkout exercise3
// --> Warning: you are leaving 1 commit behind, not connected to any of your branches
```

## Merging and Rebasing

## History and Diffs

## Fixing Mistakes

## Rebase and Amend

## Forks & Remote Repos

## Danger Zone

## GitHub

## Advanced GitHub
