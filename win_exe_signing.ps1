# Authenticode signing for Windows release packages. Dot-source from make_release.ps1:
# `. (Join-Path $PSScriptRoot "win_exe_signing.ps1")`, then call `Copy-ForSigning` and `Invoke-Sign`.
#
# Smart App Control blocks any unsigned executable, DLL, or script that came from the internet;
# SmartScreen warns on them.
#
# Config. Azure Artifact Signing (default): `signing_metadata.json` next to this file, in the dlib's
# metadata format: `{"Endpoint": ..., "CodeSigningAccountName": ..., "CertificateProfileName": ...}`.
# Authenticate first with `az login`. OV certificate on a hardware token instead: set
# WIN_SIGN_CERT_SHA1 to the certificate's thumbprint. WIN_SIGN_DLIB overrides the dlib search.

$signingMetadata = Join-Path $PSScriptRoot "signing_metadata.json"

function Find-SignTool {
    $found = Get-ChildItem "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" -ErrorAction SilentlyContinue |
        Sort-Object { [version]$_.Directory.Parent.Name } -Descending |
        Select-Object -First 1
    if (-not $found) {
        throw "signtool.exe was not found. Install the Windows SDK, or run: winget install -e --id Microsoft.Azure.ArtifactSigningClientTools"
    }

    $found.FullName
}

function Find-SigningDlib {
    if ($env:WIN_SIGN_DLIB) { return $env:WIN_SIGN_DLIB }

    $roots = "$env:LOCALAPPDATA\Microsoft", "$env:ProgramFiles\Microsoft", "${env:ProgramFiles(x86)}\Microsoft"
    # Packages that ship both architectures keep them in x86/x64 folders; signtool here is x64.
    $found = Get-ChildItem $roots -Filter "Azure.CodeSigning.Dlib.dll" -Recurse -Depth 5 -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -notmatch '\\x86\\' } |
        Select-Object -First 1
    if (-not $found) {
        throw "Azure.CodeSigning.Dlib.dll was not found. Run: winget install -e --id Microsoft.Azure.ArtifactSigningClientTools, or set WIN_SIGN_DLIB to its path."
    }

    $found.FullName
}

# Copies a file or directory into `Destination`, replacing any previous copy, and returns the copy's
# path. For files signing would otherwise modify in place, e.g. ones in the repo or Program Files.
function Copy-ForSigning {
    param([string]$Path, [string]$Destination)

    $target = Join-Path (Resolve-Path -LiteralPath $Destination).Path (Split-Path $Path -Leaf)
    if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
    Copy-Item -LiteralPath $Path -Destination $target -Recurse

    # Scripts are rewritten with a BOM: without one, Windows PowerShell reads them as ANSI, which
    # garbles non-ASCII text and breaks the signature's hash check.
    Get-ChildItem -LiteralPath $target -Recurse -File -Filter *.ps1 | ForEach-Object {
        $text = [IO.File]::ReadAllText($_.FullName, [Text.Encoding]::UTF8)
        [IO.File]::WriteAllText($_.FullName, $text, [Text.UTF8Encoding]::new($true))
    }

    $target
}

# Signs each file in place, and verifies the result. Directories are expanded to the .exe, .dll and
# .ps1 files they contain. Files that already carry a valid signature (e.g. a vendor's) are left alone.
function Invoke-Sign {
    param([string[]]$Paths, [switch]$Unsigned)

    if ($Unsigned) {
        Write-Warning "-Unsigned: not signing. This package will be blocked by Smart App Control."
        return
    }

    $files = $Paths | ForEach-Object {
        if (Test-Path -LiteralPath $_ -PathType Container) {
            Get-ChildItem -LiteralPath $_ -Recurse -File -Include *.exe, *.dll, *.ps1 | ForEach-Object FullName
        } else {
            (Resolve-Path -LiteralPath $_).Path
        }
    }
    $toSign = @($files | Where-Object { (Get-AuthenticodeSignature -LiteralPath $_).Status -ne "Valid" })
    if (-not $toSign) { return }

    $signtool = Find-SignTool

    if ($env:WIN_SIGN_CERT_SHA1) {
        $signArgs = "/sha1", $env:WIN_SIGN_CERT_SHA1, "/tr", "http://timestamp.digicert.com"
    } elseif (Test-Path -LiteralPath $signingMetadata) {
        $signArgs = "/tr", "http://timestamp.acs.microsoft.com", "/dlib", (Find-SigningDlib), "/dmdf", $signingMetadata
    } else {
        throw "No signing config: create $signingMetadata (Azure Artifact Signing), or set WIN_SIGN_CERT_SHA1 (OV certificate). Pass -Unsigned to skip signing."
    }

    # The timestamp keeps signatures valid after the certificate expires; Artifact Signing's last 3 days.
    & $signtool sign /fd SHA256 /td SHA256 @signArgs @toSign
    if ($LASTEXITCODE -ne 0) { throw "signtool sign failed with exit code $LASTEXITCODE." }

    & $signtool verify /pa @toSign
    if ($LASTEXITCODE -ne 0) { throw "signtool verify failed with exit code $LASTEXITCODE." }
}
