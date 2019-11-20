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
#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_MAIN_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_MAIN_H_

#include <vector>
#include <memory>
#include <functional>
#include <set>
#include <string>
#include <cstdint>

#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_monitor.h"
#include "rocm_smi/rocm_smi_power_mon.h"
#include "rocm_smi/rocm_smi_common.h"

namespace amd {
namespace smi {

class RocmSMI {
 public:
    RocmSMI(void);  // direct use of this constructor is deprecated; use
                    // getInstance()

    ~RocmSMI(void);

    static RocmSMI& getInstance(void);
    static std::vector<std::shared_ptr<amd::smi::Device>>&
                                  monitor_devices() {return s_monitor_devices;}
    uint32_t DiscoverDevices(void);
    uint32_t DiscoverAMDPowerMonitors(bool force_update = false);

    // Will execute "func" for every Device object known about, or until func
    // returns non-zero;
    uint32_t IterateSMIDevices(
      std::function<uint32_t(std::shared_ptr<Device>&, void *)> func, void *);

 private:
    std::vector<std::shared_ptr<Device>> devices_;
    std::vector<std::shared_ptr<Monitor>> monitors_;
    std::vector<std::shared_ptr<PowerMon>> power_mons_;

    std::set<std::string> amd_monitor_types_;
    void AddToDeviceList(std::string dev_name);
    void GetEnvVariables(void);
    uint32_t DiscoverAMDMonitors(void);

    static std::vector<std::shared_ptr<amd::smi::Device>> s_monitor_devices;
    RocmSMI_env_vars env_vars_;
};

}  // namespace smi
}  // namespace amd

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_MAIN_H_
