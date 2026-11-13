; Windows installer. Built and signed by make_release.ps1, using Inno Setup 6.3 or later:
;   winget install -e --id JRSoftware.InnoSetup
;
; Installs for the current user only, to %LOCALAPPDATA%\Programs\Molchanica: the program keeps
; its prefs, molecules and tools beside its executable, so the folder must be writable without
; admin rights. Same location the old setup_molchanica.ps1 used, so this upgrades those in place.
;
; Defines (ISCC /D):
;   AppVersion  required, e.g. 0.5.1
;   Cufft       path to cufft64_12.dll; omit for the nocuda installer
;   Sign        sign setup and uninstaller with the `molsign` tool given via ISCC /S

#define AppName "Molchanica"
#define ExeName "molchanica.exe"

#ifndef AppVersion
  #error Pass /DAppVersion=<version>
#endif

[Setup]
; Never change this: it's how upgrades and the uninstaller find an existing install.
AppId={{50AED1AE-B50B-4BB8-BDD3-510DA2F709BB}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=David O'Connor
AppPublisherURL=https://www.athanorlab.com
DefaultDirName={autopf}\{#AppName}
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
SetupIconFile=..\resources\icon.ico
UninstallDisplayIcon={app}\{#ExeName}
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes
#ifdef Sign
SignTool=molsign
SignedUninstaller=yes
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; Earlier releases bundled Gemmi here; it's now installed from the Tools panel.
Type: filesandordirs; Name: "{app}\gemmi"

[Files]
Source: "..\target\release\{#ExeName}"; DestDir: "{app}"; Flags: ignoreversion
; For the native ProteinMPNN ddG scanner's weight conversion.
Source: "..\scripts\convert_mpnn_weights.py"; DestDir: "{app}"; Flags: ignoreversion
#ifdef Cufft
; Beside the executable, so it loads without the CUDA toolkit on the Path.
Source: "{#Cufft}"; DestDir: "{app}"; Flags: ignoreversion
#endif

[Icons]
; WorkingDir is only for file dialogs; the program finds its data beside its executable.
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#ExeName}"; WorkingDir: "{app}"; Comment: "Molecule and protein viewer"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#ExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#ExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
const
  CSIDL_COMMON_PROGRAMS = $17;

// What the program writes beside itself. See `data_root` in src/external_tools/paths.rs.
function DataItems: TArrayOfString;
begin
  SetArrayLength(Result, 6);
  Result[0] := 'molchanica_prefs.mca';
  Result[1] := 'managed_molecules';
  Result[2] := 'molecule_diagrams';
  Result[3] := 'gpu_cache';
  Result[4] := 'process_executables';
  Result[5] := 'datasets';
end;

// An all-users install from an older setup script launches a copy that can't save prefs. Removing
// it needs admin rights, so we only point it out.
procedure CurStepChanged(CurStep: TSetupStep);
var
  LegacyDir, LegacyLink, Commands: String;
begin
  if CurStep <> ssPostInstall then
    Exit;

  LegacyDir := ExpandConstant('{commonpf64}\{#AppName}');
  LegacyLink := GetShellFolderByCSIDL(CSIDL_COMMON_PROGRAMS, False) + '\{#AppName}.lnk';

  if DirExists(LegacyDir) and (CompareText(LegacyDir, ExpandConstant('{app}')) <> 0) then
    Commands := Commands + '  Remove-Item -LiteralPath "' + LegacyDir + '" -Recurse -Force' + #13#10;
  if FileExists(LegacyLink) then
    Commands := Commands + '  Remove-Item -LiteralPath "' + LegacyLink + '" -Force' + #13#10;

  if Commands <> '' then
    SuppressibleMsgBox('An earlier all-users install of {#AppName} is still present. Its Start menu entry ' +
      'launches a copy that can''t save preferences. To remove it, run these from an administrator PowerShell:' + #13#10#13#10 +
      Commands, mbInformation, MB_OK, IDOK);
end;

// Uninstall removes only what setup installed. Offer to remove the program's data too, which can be
// many GB of tools.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  App, Path: String;
  Items: TArrayOfString;
  I: Integer;
  Found: Boolean;
begin
  if CurUninstallStep <> usPostUninstall then
    Exit;

  App := ExpandConstant('{app}');
  Items := DataItems;

  Found := False;
  for I := 0 to GetArrayLength(Items) - 1 do
    if FileOrDirExists(App + '\' + Items[I]) then
      Found := True;
  if not Found then
    Exit;

  if SuppressibleMsgBox('Also delete {#AppName}''s preferences, downloaded molecules, and tools installed ' +
    'from the Tools panel? They are in ' + App + '.', mbConfirmation, MB_YESNO or MB_DEFBUTTON2, IDNO) <> IDYES then
    Exit;

  for I := 0 to GetArrayLength(Items) - 1 do begin
    Path := App + '\' + Items[I];
    if DirExists(Path) then
      DelTree(Path, True, True, True)
    else
      DeleteFile(Path);
  end;
  RemoveDir(App);
end;
