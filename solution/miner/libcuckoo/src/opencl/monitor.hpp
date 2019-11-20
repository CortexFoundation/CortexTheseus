#ifdef PLATFORM_AMD
/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <assert.h>
#include <stdint.h>
#include <unistd.h>

#include <vector>
#include <iostream>
#include <bitset>

#include "rocm_smi/rocm_smi.h"

#define CHK_RSMI_RET(RET) { \
  if (RET != RSMI_STATUS_SUCCESS) { \
    const char *err_str; \
    std::cout << "RSMI call returned " << RET \
      << " at line " << __LINE__ << std::endl; \
      rsmi_status_string(RET, &err_str); \
      std::cout << err_str << std::endl; \
    return RET; \
  } \
}

#define CHK_RSMI_PERM_RET(RET) { \
    if (RET == RSMI_STATUS_PERMISSION) { \
      std::cout << "This command requires root access." << std::endl; \
    } else { \
      CHK_RSMI_RET(RET) \
    } \
}
/*
static void print_test_header(const char *str, uint32_t dv_ind) {
  std::cout << "********************************" << std::endl;
  std::cout << "*** " << str << std::endl;
  std::cout << "********************************" << std::endl;
  std::cout << "Device index: " << dv_ind << std::endl;
}

static const char *
power_profile_string(rsmi_power_profile_preset_masks profile) {
  switch (profile) {
    case RSMI_PWR_PROF_PRST_CUSTOM_MASK:
      return "CUSTOM";
    case RSMI_PWR_PROF_PRST_VIDEO_MASK:
      return "VIDEO";
    case RSMI_PWR_PROF_PRST_POWER_SAVING_MASK:
      return "POWER SAVING";
    case RSMI_PWR_PROF_PRST_COMPUTE_MASK:
      return "COMPUTE";
    case RSMI_PWR_PROF_PRST_VR_MASK:
      return "VR";
    case RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK:
      return "3D FULL SCREEN";
    default:
      return "UNKNOWN";
  }
}

static const char *
perf_level_string(rsmi_dev_perf_level perf_lvl) {
  switch (perf_lvl) {
    case RSMI_DEV_PERF_LEVEL_AUTO:
      return "AUTO";
    case RSMI_DEV_PERF_LEVEL_LOW:
      return "LOW";
    case RSMI_DEV_PERF_LEVEL_HIGH:
      return "HIGH";
    case RSMI_DEV_PERF_LEVEL_MANUAL:
      return "MANUAL";
    default:
      return "UNKNOWN";
  }
}

static rsmi_status_t test_power_profile(uint32_t dv_ind) {
  rsmi_status_t ret;
  rsmi_power_profile_status status;

  print_test_header("Power Profile", dv_ind);

  ret = rsmi_dev_power_profile_presets_get(dv_ind, 0, &status);
  CHK_RSMI_RET(ret)

  std::cout << "The available power profiles are:" << std::endl;
  uint64_t tmp = 1;

  while (tmp <= RSMI_PWR_PROF_PRST_LAST) {
    if ((tmp & status.available_profiles) == tmp) {
      std::cout << "\t" <<
      power_profile_string((rsmi_power_profile_preset_masks)tmp) << std::endl;
    }
    tmp = tmp << 1;
  }
  std::cout << "The current power profile is: " <<
                            power_profile_string(status.current) << std::endl;

  // Try setting the profile to a different power profile
  rsmi_bit_field diff_profiles;
  rsmi_power_profile_preset_masks new_prof;
  diff_profiles = status.available_profiles & (~status.current);

  if (diff_profiles & RSMI_PWR_PROF_PRST_COMPUTE_MASK) {
    new_prof = RSMI_PWR_PROF_PRST_COMPUTE_MASK;
  } else if (diff_profiles & RSMI_PWR_PROF_PRST_VIDEO_MASK) {
    new_prof = RSMI_PWR_PROF_PRST_VIDEO_MASK;
  } else if (diff_profiles & RSMI_PWR_PROF_PRST_VR_MASK) {
    new_prof = RSMI_PWR_PROF_PRST_VR_MASK;
  } else if (diff_profiles & RSMI_PWR_PROF_PRST_POWER_SAVING_MASK) {
    new_prof = RSMI_PWR_PROF_PRST_POWER_SAVING_MASK;
  } else if (diff_profiles & RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK) {
    new_prof = RSMI_PWR_PROF_PRST_3D_FULL_SCR_MASK;
  } else {
    std::cout << "No other non-custom power profiles to set to" << std::endl;
    return ret;
  }

  std::cout << "Setting power profile to " << power_profile_string(new_prof)
                                                        << "..." << std::endl;
  ret = rsmi_dev_power_profile_set(dv_ind, 0, new_prof);
  CHK_RSMI_RET(ret)
  std::cout << "Done." << std::endl;
  rsmi_dev_perf_level pfl;
  ret = rsmi_dev_perf_level_get(dv_ind, &pfl);
  CHK_RSMI_RET(ret)
  std::cout << "Performance Level is now " <<
                                          perf_level_string(pfl) << std::endl;

  ret = rsmi_dev_power_profile_presets_get(dv_ind, 0, &status);
  CHK_RSMI_RET(ret)
  std::cout << "The current power profile is: " <<
                            power_profile_string(status.current) << std::endl;
  std::cout << "Resetting perf level to auto..." << std::endl;

  ret = rsmi_dev_perf_level_set(dv_ind, RSMI_DEV_PERF_LEVEL_AUTO);
  CHK_RSMI_RET(ret)
  std::cout << "Done." << std::endl;

  ret = rsmi_dev_perf_level_get(dv_ind, &pfl);
  CHK_RSMI_RET(ret)
  std::cout << "Performance Level is now " <<
                                          perf_level_string(pfl) << std::endl;

  ret = rsmi_dev_power_profile_presets_get(dv_ind, 0, &status);
  CHK_RSMI_RET(ret)
  std::cout << "The current power profile is: " <<
                            power_profile_string(status.current) << std::endl;

  return ret;
}

static rsmi_status_t test_power_cap(uint32_t dv_ind) {
  rsmi_status_t ret;
  uint64_t orig, min, max, new_cap;

  print_test_header("Power Control", dv_ind);

  ret = rsmi_dev_power_cap_range_get(dv_ind, 0, &max, &min);
  CHK_RSMI_RET(ret)

  ret = rsmi_dev_power_cap_get(dv_ind, 0, &orig);
  CHK_RSMI_RET(ret)

  std::cout << "Original Power Cap: " << orig << " uW" << std::endl;
  std::cout << "Power Cap Range: " << max << " uW to " << min <<
                                                           " uW" << std::endl;
  new_cap = (max + min)/2;

  std::cout << "Setting new cap to " << new_cap << "..." << std::endl;

  ret = rsmi_dev_power_cap_set(dv_ind, 0, new_cap);
  CHK_RSMI_RET(ret)

  ret = rsmi_dev_power_cap_get(dv_ind, 0, &new_cap);
  CHK_RSMI_RET(ret)

  std::cout << "New Power Cap: " << new_cap << " uW" << std::endl;
  std::cout << "Resetting cap to " << orig << "..." << std::endl;

  ret = rsmi_dev_power_cap_set(dv_ind, 0, orig);
  CHK_RSMI_RET(ret)

  ret = rsmi_dev_power_cap_get(dv_ind, 0, &new_cap);
  CHK_RSMI_RET(ret)
  std::cout << "Current Power Cap: " << new_cap << " uW" << std::endl;

  return ret;
}

static rsmi_status_t test_set_overdrive(uint32_t dv_ind) {
  rsmi_status_t ret;
  uint32_t val;

  print_test_header("Overdrive Control", dv_ind);
  std::cout << "Set Overdrive level to 0%..." << std::endl;
  ret = rsmi_dev_overdrive_level_set(dv_ind, 0);
  CHK_RSMI_RET(ret)
  std::cout << "Set Overdrive level to 10%..." << std::endl;
  ret = rsmi_dev_overdrive_level_set(dv_ind, 10);
  CHK_RSMI_RET(ret)
  ret = rsmi_dev_overdrive_level_get(dv_ind, &val);
  CHK_RSMI_RET(ret)
  std::cout << "\t**New OverDrive Level:" << val << std::endl;
  std::cout << "Reset Overdrive level to 0%..." << std::endl;
  ret = rsmi_dev_overdrive_level_set(dv_ind, 0);
  CHK_RSMI_RET(ret)
  ret = rsmi_dev_overdrive_level_get(dv_ind, &val);
  CHK_RSMI_RET(ret)
  std::cout << "\t**New OverDrive Level:" << val << std::endl;

  return ret;
}

static rsmi_status_t test_set_fan_speed(uint32_t dv_ind) {
  rsmi_status_t ret;
  int64_t orig_speed;
  int64_t new_speed;
  int64_t cur_speed;

  print_test_header("Fan Speed Control", dv_ind);

  ret = rsmi_dev_fan_speed_get(dv_ind, 0, &orig_speed);
  CHK_RSMI_RET(ret)
  std::cout << "Original fan speed: " << orig_speed << std::endl;

  if (orig_speed == 0) {
    std::cout << "***System fan speed value is 0. Skip fan test." << std::endl;
    return RSMI_STATUS_SUCCESS;
  }

  new_speed = 1.1 * orig_speed;

  std::cout << "Setting fan speed to " << new_speed << std::endl;

  ret = rsmi_dev_fan_speed_set(dv_ind, 0, new_speed);
  CHK_RSMI_RET(ret)

  sleep(4);

  ret = rsmi_dev_fan_speed_get(dv_ind, 0, &cur_speed);
  CHK_RSMI_RET(ret)

  std::cout << "New fan speed: " << cur_speed << std::endl;

  assert((cur_speed > 0.95 * new_speed && cur_speed < 1.1 * new_speed) ||
      (cur_speed > 0.95 * RSMI_MAX_FAN_SPEED));

  std::cout << "Resetting fan control to auto..." << std::endl;

  ret = rsmi_dev_fan_reset(dv_ind, 0);
  CHK_RSMI_RET(ret)

  sleep(3);

  ret = rsmi_dev_fan_speed_get(dv_ind, 0, &cur_speed);
  CHK_RSMI_RET(ret)

  std::cout << "End fan speed: " << cur_speed << std::endl;

  return ret;
}

static rsmi_status_t test_set_perf_level(uint32_t dv_ind) {
  rsmi_status_t ret;

  rsmi_dev_perf_level pfl, orig_pfl;

  print_test_header("Performance Level Control", dv_ind);

  ret = rsmi_dev_perf_level_get(dv_ind, &orig_pfl);
  CHK_RSMI_RET(ret)
  std::cout << "\t**Original Perf Level:" << perf_level_string(orig_pfl) <<
                                                                    std::endl;

  pfl = (rsmi_dev_perf_level)((orig_pfl + 1) % (RSMI_DEV_PERF_LEVEL_LAST + 1));

  std::cout << "Set Performance Level to " << (uint32_t)pfl << " ..." <<
                                                                   std::endl;
  ret = rsmi_dev_perf_level_set(dv_ind, pfl);
  CHK_RSMI_RET(ret)
  ret = rsmi_dev_perf_level_get(dv_ind, &pfl);
  CHK_RSMI_RET(ret)
  std::cout << "\t**New Perf Level:" << perf_level_string(pfl) << std::endl;
  std::cout << "Reset Perf level to " << orig_pfl << " ..." << std::endl;
  ret = rsmi_dev_perf_level_set(dv_ind, orig_pfl);
  CHK_RSMI_RET(ret)
  ret = rsmi_dev_perf_level_get(dv_ind, &pfl);
  CHK_RSMI_RET(ret)
  std::cout << "\t**New Perf Level:" << perf_level_string(pfl) << std::endl;
  return ret;
}

static rsmi_status_t test_set_freq(uint32_t dv_ind) {
  rsmi_status_t ret;
  rsmi_frequencies f;
  uint32_t freq_bitmask;
  rsmi_clk_type rsmi_clk;

  print_test_header("Clock Frequency Control", dv_ind);
  for (uint32_t clk = (uint32_t)RSMI_CLK_TYPE_FIRST;
                                           clk <= RSMI_CLK_TYPE_LAST; ++clk) {
    rsmi_clk = (rsmi_clk_type)clk;

    ret = rsmi_dev_gpu_clk_freq_get(dv_ind, rsmi_clk, &f);
    CHK_RSMI_RET(ret)

    std::cout << "Initial frequency for clock" << rsmi_clk << " is " <<
                                                      f.current << std::endl;

    // Set clocks to something other than the usual default of the lowest
    // frequency.
    freq_bitmask = 0b01100;  // Try the 3rd and 4th clocks

    std::string freq_bm_str =
               std::bitset<RSMI_MAX_NUM_FREQUENCIES>(freq_bitmask).to_string();

    freq_bm_str.erase(0, std::min(freq_bm_str.find_first_not_of('0'),
                                                       freq_bm_str.size()-1));

    std::cout << "Setting frequency mask for clock " << rsmi_clk <<
        " to 0b" << freq_bm_str << " ..." << std::endl;

    ret = rsmi_dev_gpu_clk_freq_set(dv_ind, rsmi_clk, freq_bitmask);
    CHK_RSMI_RET(ret)

    ret = rsmi_dev_gpu_clk_freq_get(dv_ind, rsmi_clk, &f);
    CHK_RSMI_RET(ret)

    std::cout << "Frequency is now index " << f.current << std::endl;
    std::cout << "Resetting mask to all frequencies." << std::endl;
    ret = rsmi_dev_gpu_clk_freq_set(dv_ind, rsmi_clk, 0xFFFFFFFF);
    CHK_RSMI_RET(ret)

    ret = rsmi_dev_perf_level_set(dv_ind, RSMI_DEV_PERF_LEVEL_AUTO);
    CHK_RSMI_RET(ret)
  }
  return RSMI_STATUS_SUCCESS;
}

static void print_frequencies(rsmi_frequencies *f) {
  assert(f != nullptr);
  for (uint32_t j = 0; j < f->num_supported; ++j) {
    std::cout << "\t**  " << j << ": " << f->frequency[j];
    if (j == f->current) {
      std::cout << " *";
    }
    std::cout << std::endl;
  }
}
*/
int monitor_init(unsigned int deviceCount){

  rsmi_status_t ret;
  ret = rsmi_init(0);
  CHK_RSMI_RET(ret)

  uint32_t num_monitor_devs = 0;

  rsmi_num_monitor_devices(&num_monitor_devs);
  if(deviceCount > num_monitor_devs){
	printf("there is no much device, expected %d, but %d\n", deviceCount, num_monitor_devs);
	return -1;
  }
  return 0;
}

int query_fan_tem(unsigned int device_count, unsigned int *fanSpeeds, unsigned int *temperatures){
  rsmi_status_t ret;
  uint64_t val_ui64;
  int64_t val_i64;
	for(unsigned int i = 0; i < device_count; i++){
	    ret = rsmi_dev_id_get(i, &val_ui64);
	    CHK_RSMI_RET(ret)

	    ret = rsmi_dev_temp_metric_get(i, 0, RSMI_TEMP_CURRENT, &val_i64);
	    CHK_RSMI_RET(ret)
//	    std::cout << "\t**Temperature: " << val_i64/1000 << "C" << std::endl;
	    temperatures[i] = val_i64 / 1000;

	    ret = rsmi_dev_fan_speed_get(i, 0, &val_i64);
	    CHK_RSMI_RET(ret)
	    ret = rsmi_dev_fan_speed_max_get(i, 0, &val_ui64);
	    CHK_RSMI_RET(ret)
		
	    fanSpeeds[i] = 100*val_i64/val_ui64;
//	    std::cout << "\t**Current Fan Speed: ";
//	    std::cout << val_i64/static_cast<float>(val_ui64)*100;
//	    std::cout << "% ("<< val_i64 << "/" << val_ui64 << ")" << std::endl;
	}
	return 0;
}

void monitor_shutdown(){


}
#endif
