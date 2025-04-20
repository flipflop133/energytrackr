#!/usr/bin/env bash
#------------------------------------------------------------------------------#
# energy-measurement.sh                                                       #
#------------------------------------------------------------------------------#
# Description:                                                                 #
#   Configure Linux system for repeatable/stable energy measurements (Intel   #
#   RAPL) by disabling power-saving features and fixing P-states, with        #
#   configurable options via a user-editable config file.                     #
#                                                                              #
# Usage:                                                                       #
#   sudo ./energy-measurement.sh <command>                                      #
#                                                                              #
# Commands:                                                                    #
#   first-setup        Disable intel_pstate and cpuidle, create custom boot    #
#   setup              Apply stable energy measurement settings                #
#   revert-first-setup Re-enable intel_pstate and cpuidle (remove custom entry)#
#   revert-setup       Revert to default power settings                       #
#                                                                              #
# Config file (shell format, optional):                                        #
#   /etc/energy-measurement.conf or ~/.config/energy-measurement.conf          #
#                                                                              #
#   # SERVICES_TO_DISABLE: array of systemd units to mask/stop                #
#   SERVICES_TO_DISABLE=(NetworkManager.service systemd-journald.service       #
#                       wpa_supplicant.service)                                #
#   # WIFI_BT_MODULES: kernel modules to remove                               #
#   WIFI_BT_MODULES=(iwlmvm iwlwifi btusb bluetooth)                          #
#   # CPU_FREQ_KHZ: frequency in kHz to fix CPU/UNC                           #
#   CPU_FREQ_KHZ=2000000                                                       #
#   UNC_FREQ_KHZ=2000000                                                       #
#   # GPU settings (DRM card name without /sys/class/drm/ prefix)             #
#   GPU_CARD=card1                                                             #
#   GPU_MIN_FREQ_MHZ=300                                                       #
#   GPU_MAX_FREQ_MHZ=1100                                                      #
#   # AUDIO power_save values                                                 #
#   SND_POW_SAVE=0                                                             #
#   SND_POW_CTRL=0                                                             #
#   # SATA link power management policy                                       #
#   SATA_POLICY=max_performance                                                #
#------------------------------------------------------------------------------#

set -euo pipefail
trap 'echo "✖ Error at line ${LINENO}" >&2; exit 1' ERR

# Colored logging
RED="\e[31m"
CYAN="\e[96m"
GREEN="\e[32m"
RESET="\e[0m"
log() { echo -e "${CYAN}[INFO]${RESET} $*"; }
warn() { echo -e "${RED}[WARN]${RESET} $*" >&2; }

# Default config variables
declare -a SERVICES_TO_DISABLE=(NetworkManager.service systemd-journald.service
    wpa_supplicant.service)
declare -a WIFI_BT_MODULES=(iwlmvm iwlwifi btusb bluetooth mac80211 cfg80211)
CPU_FREQ_KHZ=2000000
UNC_FREQ_KHZ=2000000
GPU_CARD=card1
GPU_MIN_FREQ_MHZ=300
GPU_MAX_FREQ_MHZ=1100
SND_POW_SAVE=0
SND_POW_CTRL=0
SATA_POLICY=max_performance

# Load config file if exists
for cfg in "/etc/energy-measurement.conf" "$HOME/.config/energy-measurement.conf"; do
    if [[ -r "$cfg" ]]; then
        log "Loading config: $cfg"
        # shellcheck source=/dev/null
        source "$cfg"
    fi
done

# Ensure running as root
is_root() {
    [[ $(id -u) -eq 0 ]] || {
        warn "Please run as root."
        exit 1
    }
}

# Ensure user config exists
ensure_user_config() {
    local user_cfg="$HOME/.config/energy-measurement.conf"
    local default_cfg="$(dirname "$0")/energy-measurement.conf"
    if [[ ! -f "$user_cfg" ]]; then
        if [[ -f "$default_cfg" ]]; then
            mkdir -p "$HOME/.config"
            cp "$default_cfg" "$user_cfg"
            log "Copied default config to $user_cfg"
        else
            warn "Default config not found at $default_cfg"
        fi
    else
        log "User config already exists: $user_cfg"
    fi
}

# Find systemd-boot entry
get_boot_entry() {
    local entry
    entry=$(find /boot/loader/entries/*.conf 2>/dev/null | head -n1)
    [[ -n "$entry" ]] || {
        warn "No systemd-boot entry found."
        exit 1
    }
    echo "$entry"
}

# Copy and prepare custom entry
prepare_custom_entry() {
    local orig=$(get_boot_entry)
    local base=$(basename "$orig" .conf)
    local custom="/boot/loader/entries/${base}-energy.conf"
    if [[ ! -e "$custom" ]]; then
        cp "$orig" "$custom"
        log "Copied $orig to $custom"
    else
        log "Using existing $custom"
    fi
    echo "$custom"
}

# Add/remove kernel parameters
modify_param() {
    local mode=$1 param=$2 file=$3
    if [[ $mode == add ]]; then
        grep -qE "(\s|^)${param}(\s|$)" "$file" ||
            sed -i "/options/ s/$/ ${param}/" "$file"
    else
        sed -i "s/\(\s\|^\)${param}\(\s\|$\)//g" "$file"
    fi
}

# Mask/stop or unmask/start services
manage_services() {
    local action=$1
    for svc in "${SERVICES_TO_DISABLE[@]}"; do
        if command -v systemctl &>/dev/null; then
            if [[ $action == disable ]]; then
                systemctl stop "$svc" &>/dev/null
                systemctl mask "$svc" &>/dev/null
            else
                systemctl unmask "$svc" &>/dev/null
                systemctl start "$svc" &>/dev/null
            fi
            log "${action^}d $svc"
        else
            warn "systemctl not found, skipping $svc"
        fi
    done
}

# Remove or load kernel modules to block/unblock RF
manage_modules() {
    local action=$1
    for mod in "${WIFI_BT_MODULES[@]}"; do
        if [[ $action == disable ]]; then
            modprobe -r "$mod" &>/dev/null || true
        else
            modprobe "$mod" &>/dev/null || true
        fi
    done
}

# Set CPU, uncore, GPU, audio, SATA policies
apply_power_settings() {
    log "Setting CPU governor to userspace and frequency to ${CPU_FREQ_KHZ} kHz"
    echo userspace | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    echo "$CPU_FREQ_KHZ" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq
    echo "$CPU_FREQ_KHZ" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq
    echo "$CPU_FREQ_KHZ" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed

    log "Disabling CPU boost and setting uncore freq to ${UNC_FREQ_KHZ} kHz"
    echo 0 | tee /sys/devices/system/cpu/cpufreq/boost || true
    echo "$UNC_FREQ_KHZ" >/sys/devices/system/cpu/intel_uncore_frequency/*/min_freq_khz 2>/dev/null || true
    echo "$UNC_FREQ_KHZ" >/sys/devices/system/cpu/intel_uncore_frequency/*/max_freq_khz 2>/dev/null || true

    log "Configuring GPU freq on $GPU_CARD to ${GPU_MIN_FREQ_MHZ}-${GPU_MAX_FREQ_MHZ} MHz"
    echo "$GPU_MIN_FREQ_MHZ" >/sys/class/drm/${GPU_CARD}/gt_min_freq_mhz 2>/dev/null || true
    echo "$GPU_MAX_FREQ_MHZ" >/sys/class/drm/${GPU_CARD}/gt_max_freq_mhz 2>/dev/null || true

    log "Disabling audio power save"
    [[ -f /sys/module/snd_hda_intel/parameters/power_save ]] &&
        echo "$SND_POW_SAVE" >/sys/module/snd_hda_intel/parameters/power_save
    [[ -f /sys/module/snd_hda_intel/parameters/power_save_controller ]] &&
        echo "$SND_POW_CTRL" >/sys/module/snd_hda_intel/parameters/power_save_controller

    log "Setting SATA link power management policy to $SATA_POLICY"
    for host in /sys/class/scsi_host/host*; do
        [[ -f "$host/link_power_management_policy" ]] &&
            echo "$SATA_POLICY" >"$host/link_power_management_policy"
    done
}

# Revert settings
revert_power_settings() {
    log "Reverting perf_event_paranoid"
    echo 0 >/proc/sys/kernel/perf_event_paranoid
    log "Re-enabling CPU boost"
    echo 1 | tee /sys/devices/system/cpu/cpufreq/boost || true

    log "Resetting energy_perf_bias to default (6)"
    echo 6 | tee /sys/devices/system/cpu/cpu*/power/energy_perf_bias || true

    log "Reverting audio power save to defaults"
    [[ -f /sys/module/snd_hda_intel/parameters/power_save ]] &&
        echo 1 >/sys/module/snd_hda_intel/parameters/power_save
    [[ -f /sys/module/snd_hda_intel/parameters/power_save_controller ]] &&
        echo 1 >/sys/module/snd_hda_intel/parameters/power_save_controller

    log "Reverting GPU frequencies"
    echo 300 >/sys/class/drm/${GPU_CARD}/gt_min_freq_mhz 2>/dev/null || true
    echo 1100 >/sys/class/drm/${GPU_CARD}/gt_max_freq_mhz 2>/dev/null || true

    log "Setting SATA link power management to standard mode"
    for host in /sys/class/scsi_host/host*; do
        [[ -f "$host/link_power_management_policy" ]] &&
            echo "med_power_with_dipm" >"$host/link_power_management_policy"
    done
}

main() {
    is_root
    case ${1:-} in
    first-setup)
        ensure_user_config
        local custom=$(prepare_custom_entry)
        modify_param add intel_pstate=disable "$custom"
        modify_param add cpuidle.off=1 "$custom"
        modify_param add idle=poll "$custom"
        bootctl update
        log "Custom boot entry prepared: $(basename "$custom")"
        ;;
    setup)
        # Verify that first-setup was applied and system rebooted
        log "Verifying kernel parameters for measurement entry..."
        cmdline=$(cat /proc/cmdline)
        required_params=(intel_pstate=disable cpuidle.off=1 idle=poll)
        for p in "${required_params[@]}"; do
            if ! grep -q "$p" <<<"$cmdline"; then
                warn "Kernel parameter '$p' missing. Have you run 'first-setup' and rebooted?"
                exit 1
            fi
        done
        log "Applying stable measurement configuration"
        echo -1 >/proc/sys/kernel/perf_event_paranoid
        manage_services disable
        rfkill block all &>/dev/null || warn "rfkill not available"
        manage_modules disable
        apply_power_settings
        log "Stable measurement mode enabled"
        ;;
    revert-first-setup)
        local orig=$(get_boot_entry)
        local custom="/boot/loader/entries/$(basename "$orig" .conf)-energy.conf"
        if [[ -f "$custom" ]]; then
            rm -f "$custom"
            bootctl update
            log "Removed custom entry: $(basename "$custom")"
        else
            warn "No custom entry to remove: $custom"
        fi
        ;;
    revert-setup)
        log "Reverting stable measurement settings"
        revert_power_settings
        manage_modules enable
        rfkill unblock all &>/dev/null || warn "rfkill not available"
        manage_services enable
        log "Stable measurement mode reverted"
        ;;
    *)
        echo "Usage: $0 {first-setup|setup|revert-first-setup|revert-setup}" >&2
        exit 1
        ;;
    esac
}

clear
main "$@"
