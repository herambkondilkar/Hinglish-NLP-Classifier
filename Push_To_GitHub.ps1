$ErrorActionPreference = 'Stop'
$repoUrl = 'https://github.com/herambkondilkar/Hinglish-NLP-Classifier.git'
$root = $PSScriptRoot
Set-Location $root

Write-Host ''
Write-Host 'Hinglish NLP Classifier - GitHub Publisher' -ForegroundColor Cyan
Write-Host '------------------------------------------' -ForegroundColor Cyan

$required = @(
    'hinglish_desktop_app.py',
    'Launch_Hinglish_Classifier.bat',
    'requirements.txt',
    'models/comi_lingua_dev_tuned_augmented.joblib'
)
foreach ($file in $required) {
    if (-not (Test-Path (Join-Path $root $file))) {
        throw "Required project file is missing: $file`nExtract the complete GitHub-ready project ZIP and run this script from its root folder."
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git for Windows was not found. Install it from https://git-scm.com/download/win, reopen this folder, and run PUSH_TO_GITHUB.bat again.'
}

& git --version
if ($LASTEXITCODE -ne 0) { throw 'Git could not be started.' }

if (-not (& git config user.name)) {
    & git config user.name 'Heramb Kondilkar'
}
if (-not (& git config user.email)) {
    & git config user.email '172323977+herambkondilkar@users.noreply.github.com'
}

if (-not (Test-Path (Join-Path $root '.git'))) {
    & git init -b main
    if ($LASTEXITCODE -ne 0) { throw 'Could not initialize the local Git repository.' }
} else {
    & git branch -M main
    if ($LASTEXITCODE -ne 0) { throw 'Could not set the local branch to main.' }
}

# Do not call `git remote get-url origin` to detect absence: Git writes
# "No such remote" to stderr, which some PowerShell versions treat as a
# terminating NativeCommandError when ErrorActionPreference is Stop.
$remoteNames = @(& git remote)
if ($LASTEXITCODE -ne 0) { throw 'Could not read the configured Git remotes.' }
if ($remoteNames -contains 'origin') {
    & git remote set-url origin $repoUrl
} else {
    & git remote add origin $repoUrl
}
if ($LASTEXITCODE -ne 0) { throw 'Could not configure the GitHub remote.' }

Write-Host ''
Write-Host 'Connecting to GitHub and synchronizing the starting README...' -ForegroundColor Yellow
& git fetch origin main
if ($LASTEXITCODE -ne 0) {
    throw 'Could not fetch the GitHub repository. Check your internet connection and GitHub sign-in, then run this script again.'
}

& git reset --mixed origin/main
if ($LASTEXITCODE -ne 0) { throw 'Could not align the local main branch with GitHub main.' }

& git add -A
if ($LASTEXITCODE -ne 0) { throw 'Git could not stage the project files.' }

& git diff --cached --quiet
$diffExitCode = $LASTEXITCODE
if ($diffExitCode -eq 1) {
    & git commit -m 'Add completed Hinglish NLP Classifier project'
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed. Check that Git user.name and user.email are configured.' }
} elseif ($diffExitCode -eq 0) {
    Write-Host 'No new file changes were found to commit.' -ForegroundColor Yellow
} else {
    throw 'Git could not check the staged changes.'
}

Write-Host ''
Write-Host 'Uploading project to GitHub...' -ForegroundColor Yellow
& git push -u origin main
if ($LASTEXITCODE -ne 0) {
    throw 'GitHub push failed. When prompted, sign in with the GitHub account that owns herambkondilkar/Hinglish-NLP-Classifier, then run PUSH_TO_GITHUB.bat again.'
}

Write-Host ''
Write-Host 'SUCCESS: Project files have been pushed to GitHub.' -ForegroundColor Green
Write-Host 'https://github.com/herambkondilkar/Hinglish-NLP-Classifier' -ForegroundColor Cyan
Write-Host ''
