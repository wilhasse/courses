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

## References, Commits, Branches

## Merging and Rebasing

## History and Diffs

## Fixing Mistakes

## Rebase and Amend

## Forks & Remote Repos

## Danger Zone

## GitHub

## Advanced GitHub