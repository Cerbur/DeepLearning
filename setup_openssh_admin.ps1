#Requires -RunAsAdministrator
$ErrorActionPreference = "Continue"

try { Stop-Transcript -ErrorAction SilentlyContinue } catch {}
$logFile = "E:\Code\DeepLearning\setup_openssh.log"
try { Start-Transcript -Path $logFile -Force } catch {}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "      Windows OpenSSH Server 自动化配置工具 (管理员)      " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$targetUser = "10511"
$userHome = "C:\Users\" + $targetUser
$macPubKey = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICpO+CagykOR8NgdqKNUfQv2jpAYuH09B1WacPD9u3WK mac-to-win-dl"

Write-Host ">>> [1/7] 检查基础系统信息..." -ForegroundColor Cyan
Write-Host "目标 SSH 用户: $targetUser"
Write-Host "用户主目录: $userHome"

# 1. 检查并设置网络配置为专用网络 (Private)
Write-Host "`n>>> [2/7] 检查并配置网络连接类别 (切换 WLAN 为 Private)..." -ForegroundColor Cyan
try {
    $wlanProfiles = Get-NetConnectionProfile -ErrorAction SilentlyContinue | Where-Object { $_.InterfaceAlias -eq "WLAN" -or $_.IPv4Connectivity -eq "Internet" }
    if ($wlanProfiles) {
        foreach ($p in $wlanProfiles) {
            Write-Host "当前网络: $($p.Name) (接口: $($p.InterfaceAlias)), 类别: $($p.NetworkCategory)"
            if ($p.NetworkCategory -ne "Private") {
                Write-Host "正在将网络 $($p.Name) 设置为 Private (专用网络)..." -ForegroundColor Yellow
                Set-NetConnectionProfile -InputObject $p -NetworkCategory Private -ErrorAction SilentlyContinue
                Write-Host "已更新网络类别为 Private。" -ForegroundColor Green
            }
        }
    }
} catch {
    Write-Warning "网络连接类别配置跳过: $_"
}

# 2. 安装 OpenSSH.Server
Write-Host "`n>>> [3/7] 检查并安装 OpenSSH.Server..." -ForegroundColor Cyan
$sshCap = Get-WindowsCapability -Online -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "OpenSSH.Server*" }
if (-not $sshCap -or $sshCap.State -ne "Installed") {
    Write-Host "正在安装 OpenSSH.Server Windows 功能 (首次安装预计耗时 15-30 秒)..." -ForegroundColor Yellow
    Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
    Write-Host "OpenSSH.Server 安装成功。" -ForegroundColor Green
} else {
    Write-Host "OpenSSH.Server 已安装，跳过安装步骤。" -ForegroundColor Green
}

# 确保 C:ProgramDatassh 目录存在
if (-not (Test-Path "C:\ProgramData\ssh")) {
    New-Item -ItemType Directory -Path "C:\ProgramData\ssh" -Force | Out-Null
}
# 如果缺少配置文件但存在默认模板，复制模板
if (-not (Test-Path "C:\ProgramData\ssh\sshd_config") -and (Test-Path "C:\Windows\System32\OpenSSH\sshd_config_default")) {
    Copy-Item "C:\Windows\System32\OpenSSH\sshd_config_default" "C:\ProgramData\ssh\sshd_config" -Force
}

# 3. 配置 sshd 服务自启并启动
Write-Host "`n>>> [4/7] 配置 sshd 服务并启动..." -ForegroundColor Cyan
Set-Service -Name sshd -StartupType Automatic
Set-Service -Name "ssh-agent" -StartupType Automatic
Start-Service -Name sshd -ErrorAction SilentlyContinue
Start-Service -Name "ssh-agent" -ErrorAction SilentlyContinue
Write-Host "sshd 服务状态: $((Get-Service -Name sshd).Status)" -ForegroundColor Green

# 4. 配置 Windows 防火墙 (仅允许 Private 网络的 LocalSubnet 访问 TCP 22)
Write-Host "`n>>> [5/7] 配置 Windows 防火墙规则..." -ForegroundColor Cyan
Get-NetFirewallRule -ErrorAction SilentlyContinue | Where-Object {
    ($_.DisplayName -like "*OpenSSH*" -or $_.Name -like "*OpenSSH*" -or $_.Name -eq "OpenSSH-Server-In-TCP") -and
    $_.Name -ne "OpenSSH-Server-In-TCP-Private"
} | ForEach-Object {
    Write-Host "禁用宽泛的默认规则: $($_.DisplayName)"
    Set-NetFirewallRule -InputObject $_ -Enabled False -ErrorAction SilentlyContinue
}

$ruleName = "OpenSSH-Server-In-TCP-Private"
$rule = Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue
$ruleParams = @{
    Name = $ruleName
    DisplayName = "OpenSSH Server (sshd) Private LocalSubnet"
    Description = "Allow SSH on TCP 22 from LocalSubnet in Private network profile only"
    Direction = "Inbound"
    Protocol = "TCP"
    LocalPort = 22
    Profile = "Private"
    RemoteAddress = "LocalSubnet"
    Action = "Allow"
    Enabled = "True"
}
if ($rule) {
    Set-NetFirewallRule @ruleParams
    Write-Host "防火墙规则已更新: $ruleName (仅允许专用网络本地子网访问 TCP 22)" -ForegroundColor Green
} else {
    New-NetFirewallRule @ruleParams | Out-Null
    Write-Host "新建防火墙规则完成: $ruleName (仅允许专用网络本地子网访问 TCP 22)" -ForegroundColor Green
}

# 5. 配置 sshd_config 及 authorized_keys
Write-Host "`n>>> [6/7] 配置 sshd_config 与 authorized_keys 密钥 ACL..." -ForegroundColor Cyan
$sshdConfigPath = "C:\ProgramData\ssh\sshd_config"
if (Test-Path $sshdConfigPath) {
    $configContent = Get-Content $sshdConfigPath -Raw
    $modified = $configContent
    $modified = $modified -replace '(?m)^Match Group administrators', '#Match Group administrators'
    $modified = $modified -replace '(?m)^\s*AuthorizedKeysFile\s+__PROGRAMDATA__/ssh/administrators_authorized_keys', '#       AuthorizedKeysFile __PROGRAMDATA__/ssh/administrators_authorized_keys'
    if ($modified -match '(?m)^#?PubkeyAuthentication\s+') {
        $modified = $modified -replace '(?m)^#?PubkeyAuthentication\s+\w+', 'PubkeyAuthentication yes'
    } else {
        $modified += [Environment]::NewLine + "PubkeyAuthentication yes"
    }
    if ($modified -match '(?m)^#?PasswordAuthentication\s+') {
        $modified = $modified -replace '(?m)^#?PasswordAuthentication\s+\w+', 'PasswordAuthentication yes'
    } else {
        $modified += [Environment]::NewLine + "PasswordAuthentication yes"
    }
    if ($modified -match '(?m)^#?AuthorizedKeysFile\s+') {
        $modified = $modified -replace '(?m)^#?AuthorizedKeysFile\s+.*', 'AuthorizedKeysFile .ssh/authorized_keys'
    } else {
        $modified += [Environment]::NewLine + "AuthorizedKeysFile .ssh/authorized_keys"
    }
    Set-Content -Path $sshdConfigPath -Value $modified -Encoding utf8
    Write-Host "已更新 sshd_config (已注释管理员特殊路径，开启公钥与密码双认证)。" -ForegroundColor Green
}

# 注册表配置 VS Code Remote SSH 默认 Shell
if (-not (Test-Path "HKLM:\SOFTWARE\OpenSSH")) {
    New-Item -Path "HKLM:\SOFTWARE\OpenSSH" -Force | Out-Null
}
Set-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name "DefaultShell" -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -Force
Write-Host "已配置 OpenSSH 默认 Shell: powershell.exe" -ForegroundColor Green

# 写入并配置用户级 ~/.ssh/authorized_keys
$userSshDir = Join-Path $userHome ".ssh"
$userAuthKeys = Join-Path $userSshDir "authorized_keys"
if (-not (Test-Path $userSshDir)) { New-Item -ItemType Directory -Path $userSshDir -Force | Out-Null }
$existingUserKeys = if (Test-Path $userAuthKeys) { Get-Content $userAuthKeys -Raw } else { "" }
if ($existingUserKeys -notmatch [regex]::Escape($macPubKey.Trim())) {
    $merged = if ([string]::IsNullOrWhiteSpace($existingUserKeys)) { $macPubKey } else { $existingUserKeys.TrimEnd() + "`n" + $macPubKey }
    Set-Content -Path $userAuthKeys -Value ($merged + "`n") -Encoding utf8 -NoNewline
}
icacls $userAuthKeys /inheritance:r /grant:r "$($targetUser):(F)" "SYSTEM:(F)" "BUILTIN\Administrators:(F)" | Out-Null
icacls $userSshDir /inheritance:r /grant:r "$($targetUser):(F)" "SYSTEM:(F)" "BUILTIN\Administrators:(F)" | Out-Null
Write-Host "已保护用户级 authorized_keys: $userAuthKeys" -ForegroundColor Green

# 双重保障：写入系统级 C:\ProgramData\ssh\administrators_authorized_keys 并配置 ACL
$adminAuthKeys = "C:\ProgramData\ssh\administrators_authorized_keys"
$existingAdminKeys = if (Test-Path $adminAuthKeys) { Get-Content $adminAuthKeys -Raw } else { "" }
if ($existingAdminKeys -notmatch [regex]::Escape($macPubKey.Trim())) {
    $mergedAdmin = if ([string]::IsNullOrWhiteSpace($existingAdminKeys)) { $macPubKey } else { $existingAdminKeys.TrimEnd() + "`n" + $macPubKey }
    Set-Content -Path $adminAuthKeys -Value ($mergedAdmin + "`n") -Encoding utf8 -NoNewline
}
icacls $adminAuthKeys /inheritance:r /grant:r "SYSTEM:(F)" "BUILTIN\Administrators:(F)" | Out-Null
Write-Host "已保护系统级 administrators_authorized_keys: $adminAuthKeys" -ForegroundColor Green

# 6. 配置语法检查并重启 sshd
Write-Host "`n>>> [7/7] 检查配置语法并重启服务..." -ForegroundColor Cyan
$sshdBin = "C:\Windows\System32\OpenSSH\sshd.exe"
if (Test-Path $sshdBin) {
    & $sshdBin -t
    if ($LASTEXITCODE -eq 0) {
        Write-Host "sshd_config 语法检查通过。" -ForegroundColor Green
    } else {
        Write-Warning "sshd_config 语法检查返回值: $LASTEXITCODE"
    }
}
Restart-Service -Name sshd -Force -ErrorAction SilentlyContinue
Write-Host "sshd 服务已重启完成。" -ForegroundColor Green

Start-Sleep -Seconds 2
$port22 = Get-NetTCPConnection -LocalPort 22 -State Listen -ErrorAction SilentlyContinue
if ($port22) {
    Write-Host "`n[验证成功] TCP 22 端口正在监听中！" -ForegroundColor Green
} else {
    Write-Warning "`n[提示] 请检查服务状态或尝试在管理员终端执行 Test-NetConnection -Port 22"`
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "         OpenSSH Server 全部自动化配置完成！              " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
try { Stop-Transcript } catch {}