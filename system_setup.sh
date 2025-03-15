#!/usr/bin/env bash
###############################################################################
# Script Name : system_setup.sh
# Description : Configure linux system for stable energy measurements using
#               Intel RAPL by disabling dynamic power-saving features and
#               locking performance states to reduce variability.
#
#               Usage:
#                 ./system_setup.sh enable   # Apply stable measurement config
#                 ./system_setup.sh disable  # Revert to (best-effort) defaults
#
# Author      : François Bechet
# Version     : 1.0
###############################################################################

set -euo pipefail

#--- Colors --------------------------------------------------------#
RED="\e[31m"
CYAN="\e[96m"
RESET="\e[0m"

#--- Services we want to disable for stable measurement -----------------------#
SERVICES_TO_DISABLE=(
    "NetworkManager.service"
    "systemd-journald.service"
    "systemd-timesyncd.service"
    "wpa_supplicant.service"
    "thermald"
    "power-profiles-daemon"
    "powerd"
    "upower"
)

#--- Modules we want to remove to kill WiFi/Bluetooth -------------------------#
WIFI_BT_MODULES=(
    "iwlmvm"
    "iwlwifi"
    "btusb"
    "bluetooth"
    "mac80211"
    "cfg80211"
)

#----------------------------------------------------------------------------#
#  FUNCTIONS                                                                 #
#----------------------------------------------------------------------------#

#--- Check if running as root ------------------------------------------------#
is_root() {
    if [[ "$(id -u)" -ne 0 ]]; then
        echo -e "${RED}Please run this script as root.${RESET}"
        exit 1
    fi
}

#--- Find the first systemd-boot loader entry file ----------------------------#
get_boot_entry_file() {
    local entry_file
    entry_file=$(find /boot/loader/entries/ -type f -name "*.conf" | head -n 1 || true)
    if [[ -z "$entry_file" ]]; then
        echo "Error: No systemd-boot entry file found." >&2
        exit 1
    fi
    echo "$entry_file"
}

#--- Add kernel parameter if not present --------------------------------------#
add_kernel_param() {
    local param="$1"
    local entry_file="$2"

    if ! grep -qE "(\s|^)${param}(\s|$)" "$entry_file"; then
        echo "Adding '${param}' to kernel parameters in: $entry_file"
        sed -i '/options/ s/$/ '"${param}"'/' "$entry_file"
    else
        echo "'${param}' already present in: $entry_file"
    fi
}

#--- Remove kernel parameter if present ---------------------------------------#
remove_kernel_param() {
    local param="$1"
    local entry_file="$2"

    if grep -qE "(\s|^)${param}(\s|$)" "$entry_file"; then
        echo "Removing '${param}' from kernel parameters in: $entry_file"
        # The next line removes the param whether it’s in the middle or at the edges
        sed -i "s/\(\s\|^\\)${param}\(\s\|$\)//g" "$entry_file"
    else
        echo "'${param}' not found in: $entry_file"
    fi
}

#--- Enable stable measurement ------------------------------------------------#
enable_stable_measurement() {
    echo -e "${RED}Configuring stable power settings for energy measurement...${RESET}"

    # Stop and mask services
    # for svc in "${SERVICES_TO_DISABLE[@]}"; do
    #     systemctl stop "$svc" 2>/dev/null || true
    #     systemctl mask "$svc" 2>/dev/null || true
    # done

    # Block RF and remove modules
    # rfkill block all
    # for mod in "${WIFI_BT_MODULES[@]}"; do
    #     modprobe -r "$mod" 2>/dev/null || true
    # done

    # CPU frequency: set governor userspace and fix freq at 2.0 GHz
    echo "userspace" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    echo 2000000 | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq
    echo 2000000 | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq
    echo 2000000 | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed

    # Disable boost
    echo 0 | tee /sys/devices/system/cpu/cpu*/cpufreq/boost 2>/dev/null || true
    echo 0 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true

    # Uncore freq
    if [[ -d /sys/devices/system/cpu/intel_uncore_frequency/package_00_die_00 ]]; then
        echo 2000000 > /sys/devices/system/cpu/intel_uncore_frequency/package_00_die_00/min_freq_khz || true
        echo 2000000 > /sys/devices/system/cpu/intel_uncore_frequency/package_00_die_00/max_freq_khz || true
    fi

    # Energy perf bias
    echo 0 | tee /sys/devices/system/cpu/cpu*/power/energy_perf_bias 2>/dev/null || true

    # pstate
    if [[ -d /sys/devices/system/cpu/intel_pstate ]]; then
        echo 0 | tee /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
        echo 100 | tee /sys/devices/system/cpu/intel_pstate/min_perf_pct
        echo 100 | tee /sys/devices/system/cpu/intel_pstate/max_perf_pct
    fi

    # GPU freq (assuming card1 is the iGPU)
    if [[ -d /sys/class/drm/card1 ]]; then
        echo 500 > /sys/class/drm/card1/gt_min_freq_mhz 2>/dev/null || true
        echo 500 > /sys/class/drm/card1/gt_max_freq_mhz 2>/dev/null || true
        echo 500 > /sys/class/drm/card1/gt_boost_freq_mhz 2>/dev/null || true
    fi

    # Audio power saving
    if [[ -f /sys/module/snd_hda_intel/parameters/power_save ]]; then
        echo 0 > /sys/module/snd_hda_intel/parameters/power_save
    fi
    if [[ -f /sys/module/snd_hda_intel/parameters/power_save_controller ]]; then
        echo 0 > /sys/module/snd_hda_intel/parameters/power_save_controller
    fi

    # Platform profile (if supported)
    if [[ -f /sys/firmware/acpi/platform_profile ]]; then
        echo "performance" > /sys/firmware/acpi/platform_profile
    fi

    # Disable autosuspend for USB and PCI
    # for device in /sys/bus/usb/devices/*/power/control; do
    #     echo "on" > "$device" 2>/dev/null || true
    # done
    # for device in /sys/bus/pci/devices/*/power/control; do
    #     echo "on" > "$device" 2>/dev/null || true
    # done

    # SATA link power management -> max performance
    for host in /sys/class/scsi_host/host*; do
        if [[ -f "$host/link_power_management_policy" ]]; then
            echo "max_performance" > "$host/link_power_management_policy" 2>/dev/null || true
        fi
    done

    echo -e "\n${CYAN}Stable energy measurement mode applied!${RESET}"
}

#--- Disable (revert) stable measurement --------------------------------------#
disable_stable_measurement() {
    echo -e "${RED}Reverting stable power settings...${RESET}"

    # Unmask & start services
    for svc in "${SERVICES_TO_DISABLE[@]}"; do
        systemctl unmask "$svc" 2>/dev/null || true
        systemctl start "$svc" 2>/dev/null || true
    done

    # Try to load WiFi/BT modules
    for mod in "${WIFI_BT_MODULES[@]}"; do
        modprobe "$mod" 2>/dev/null || true
    done
    rfkill unblock all

    # Return CPU governor to powersave (typical for intel_pstate) or ondemand if available
    # You can adapt this if you prefer a different default
    if [[ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors ]]; then
        if grep -q "powersave" /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors; then
            echo "powersave" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
        elif grep -q "ondemand" /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors; then
            echo "ondemand" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
        fi
    fi

    # Remove forced freq settings (set them to 0 to let the driver handle it)
    echo 0 | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq 2>/dev/null || true
    echo 0 | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq 2>/dev/null || true
    # Some distributions may not allow writing 0 for set freq, so skip it

    # Re-enable boost
    echo 1 | tee /sys/devices/system/cpu/cpu*/cpufreq/boost 2>/dev/null || true
    if [[ -f /sys/devices/system/cpu/cpufreq/boost ]]; then
        echo 1 > /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || true
    fi

    # Reset uncore freq to 0 if it exists
    if [[ -d /sys/devices/system/cpu/intel_uncore_frequency/package_00_die_00 ]]; then
        echo 0 > /sys/devices/system/cpu/intel_uncore_frequency/package_00_die_00/min_freq_khz || true
        echo 0 > /sys/devices/system/cpu/intel_uncore_frequency/package_00_die_00/max_freq_khz || true
    fi

    # Reset energy_perf_bias to default (normal=4 or 6 on many systems)
    # but let's choose 6 as it's typical for 'normal'
    echo 6 | tee /sys/devices/system/cpu/cpu*/power/energy_perf_bias 2>/dev/null || true

    # For intel_pstate, re-allow max range
    if [[ -d /sys/devices/system/cpu/intel_pstate ]]; then
        echo 0 | tee /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null || true
        echo 0 | tee /sys/devices/system/cpu/intel_pstate/min_perf_pct 2>/dev/null || true
        echo 100 | tee /sys/devices/system/cpu/intel_pstate/max_perf_pct 2>/dev/null || true
    fi

    # Revert GPU freq to a typical lower bound (300 MHz) and max to something higher
    # Adjust these if your hardware differs
    if [[ -d /sys/class/drm/card1 ]]; then
        echo 300 > /sys/class/drm/card1/gt_min_freq_mhz 2>/dev/null || true
        echo 1100 > /sys/class/drm/card1/gt_max_freq_mhz 2>/dev/null || true
        echo 1100 > /sys/class/drm/card1/gt_boost_freq_mhz 2>/dev/null || true
    fi

    # Re-enable audio power saving (default is often 1)
    if [[ -f /sys/module/snd_hda_intel/parameters/power_save ]]; then
        echo 1 > /sys/module/snd_hda_intel/parameters/power_save
    fi
    if [[ -f /sys/module/snd_hda_intel/parameters/power_save_controller ]]; then
        echo 1 > /sys/module/snd_hda_intel/parameters/power_save_controller
    fi

    # For platform profile, revert if desired (depends on your distro’s default)
    if [[ -f /sys/firmware/acpi/platform_profile ]]; then
        echo "balanced" > /sys/firmware/acpi/platform_profile
    fi

    # For USB/PCI devices, typical default is "auto" rather than "on"
    for device in /sys/bus/usb/devices/*/power/control; do
        echo "auto" > "$device" 2>/dev/null || true
    done
    for device in /sys/bus/pci/devices/*/power/control; do
        echo "auto" > "$device" 2>/dev/null || true
    done

    # SATA link power management to "med_power_with_dipm" or "min_power" on many distros
    for host in /sys/class/scsi_host/host*; do
        if [[ -f "$host/link_power_management_policy" ]]; then
            echo "med_power_with_dipm" > "$host/link_power_management_policy" 2>/dev/null || true
        fi
    done

    echo -e "\n${CYAN}Stable energy measurement mode reverted (best effort)!${RESET}"
}

#--- Disable Intel pstate + add cpuidle.off=1 + idle=poll ---------------------#
disable_intel_pstate_and_cpuidle() {
    local entry_file
    entry_file=$(get_boot_entry_file)
    echo $entry_file

    echo "Current CPU frequency scaling driver: $(< /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null || true)"

    # For actual disabling of intel_pstate
    add_kernel_param "intel_pstate=disable" "$entry_file"

    # Also add the requested params from the TODO
    add_kernel_param "cpuidle.off=1" "$entry_file"
    add_kernel_param "idle=poll" "$entry_file"

    # Update systemd-boot
    bootctl update
    echo "Kernel parameters updated. A reboot is required for changes to take effect."
}

#--- Re-enable Intel pstate + remove cpuidle.off=1 + idle=poll ----------------#
enable_intel_pstate_and_cpuidle() {
    local entry_file
    entry_file=$(get_boot_entry_file)

    remove_kernel_param "intel_pstate=disable" "$entry_file"
    remove_kernel_param "cpuidle.off=1" "$entry_file"
    remove_kernel_param "idle=poll" "$entry_file"

    bootctl update
    echo "Kernel parameters reverted. A reboot is required for changes to take effect."
}

#--- MAIN ---------------------------------------------------------------------#
usage() {
    echo -e "Usage: $0 {enable|disable}\n"
    echo "  enable  : Apply stable measurement configuration and set kernel params"
    echo "  disable : Revert stable measurement configuration and kernel params"
    exit 1
}

main() {
    is_root

    local cmd="${1:-}"

    case "$cmd" in
        first-setup)
            enable_intel_pstate_and_cpuidle
            ;;
        setup)
            enable_stable_measurement
            ;;
        revert-first-setup)
            disable_intel_pstate_and_cpuidle
            ;;
        revert-setup)
            disable_stable_measurement
            ;;
        *)
            usage
            ;;
    esac
}

clear
main "${@}"
