#!/bin/bash
# Script to configure Lenovo Ideapad 5 for stable energy measurements using Intel RAPL
# This script disables dynamic power-saving features and locks performance states to reduce variability.

set -e  # Exit immediately if a command exits with a non-zero status

# TODO : add this to kernel parameters : cpuidle.off=1, idle=poll
# TODO : look at this /sys/devices/system/cpu/intel_uncore_frequency/
# Disable Intel pstate
disable_intel_pstate() {
    # Check current CPU driver
    DRIVER=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null)

    if [[ -z "$DRIVER" ]]; then
        echo "Error: Unable to determine current CPU frequency scaling driver." >&2
        return 1
    fi

    echo "Current CPU frequency scaling driver: $DRIVER"

    # Only proceed if intel_pstate is being used
    if [[ "$DRIVER" != "intel_pstate" ]]; then
        echo "intel_pstate is not the active driver. No changes will be made."
        return 0
    fi

    # Find systemd-boot entry file
    ENTRY_FILE=$(find /boot/loader/entries/ -name "*.conf" | head -n 1)

    if [[ -z "$ENTRY_FILE" ]]; then
        echo "Error: No systemd-boot entry file found." >&2
        return 1
    fi

    # Append intel_pstate=no_hwp to kernel options if not already present
    if grep -q "intel_pstate=disable" "$ENTRY_FILE"; then
        echo "intel_pstate=disable is already set in $ENTRY_FILE."
    else
        sudo sed -i 's/\(options .*\)/\1 intel_pstate=disable/' "$ENTRY_FILE"
        sudo bootctl update
        echo "intel_pstate=disable added to $ENTRY_FILE. Reboot required."
    fi
}



configure_stable_measurement() {
    echo -e "\e[31mConfiguring stable power settings for energy measurement...\e[0m"

    ###############################
    # Disable power-saving features
    ###############################
    systemctl mask --now thermald
    systemctl mask --now power-profiles-daemon
    systemctl mask --now powerd
    systemctl mask --now upower
    echo "userspace" | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
    echo 2500000 | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq
    echo 2500000 | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_max_freq
    echo 2500000 | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_setspeed
    echo 0 | tee /sys/devices/system/cpu/cpu*/cpufreq/boost
    echo 0 > /sys/devices/system/cpu/cpufreq/boost

    echo 2500000 > /sys/devices/system/cpu/intel_uncore_frequency/package_00_die_00/min_freq_khz
    echo 2500000 > /sys/devices/system/cpu/intel_uncore_frequency/package_00_die_00/max_freq_khz


    echo 0 | tee /sys/devices/system/cpu/cpu*/power/energy_perf_bias

    echo 0 | tee /sys/devices/system/cpu/intel_pstate/no_turbo
    echo 100 | tee /sys/devices/system/cpu/intel_pstate/min_perf_pct
    echo 100 | tee /sys/devices/system/cpu/intel_pstate/max_perf_pct
    
    # Fix GPU frequencies to a stable state
    echo 500 > /sys/class/drm/card1/gt_min_freq_mhz
    echo 500 > /sys/class/drm/card1/gt_max_freq_mhz
    echo 500 > /sys/class/drm/card1/gt_boost_freq_mhz

    # Set energy performance bias to maximum performance
    echo 0 | tee /sys/devices/system/cpu/cpu*/power/energy_perf_bias

    # Disable audio power saving
    echo 0 > /sys/module/snd_hda_intel/parameters/power_save
    echo 0 > /sys/module/snd_hda_intel/parameters/power_save_controller

    # Disable power-efficient workqueues
    #echo N > /sys/module/workqueue/parameters/power_efficient

    # Set platform profile to performance
    echo "performance" > /sys/firmware/acpi/platform_profile

    # Disable auto suspend for USB and PCI devices
    for device in /sys/bus/usb/devices/*/power/control; do
        echo "on" > "$device"
    done

    for device in /sys/bus/pci/devices/*/power/control; do
        echo "on" > "$device"
    done

    # Set SATA link power management to maximum performance
    echo "max_performance" > /sys/class/scsi_host/host0/link_power_management_policy

    echo -e "\n\e[96mStable energy measurement mode applied!\e[0m"
}

main() {
    if [ "$(id -u)" -ne 0 ]; then
        echo -e "\e[31mPlease run this script as root.\e[0m"
        exit 1
    fi
    configure_stable_measurement
    disable_intel_pstate
}

clear
main
