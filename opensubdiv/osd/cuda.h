//
//   Copyright 2014 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#ifndef OSD_CUDA_H
#define OSD_CUDA_H

#include "../osd/opengl.h"

// If dunamic loading is disabled we can just include CUDA runtime and
// link against CUDA runtime, no special initialization is needed in
// this case.
#ifndef OPENSUBDIV_HAS_CUEW
#  include <cuda.h>
#  include <cudaGL.h>
#else  // OPENSUBDIV_HAS_CUEW
#  include <cuew.h>
#endif  // OPENSUBDIV_HAS_CUEW

static inline const char *cuda_GetErrorString(CUresult result) {
   switch (result) {
#define CHECK_VALUE(value) \
    case CUDA_ERROR_ ## value: \
        return "CUDA_ERROR_" # value

       case CUDA_SUCCESS:
           return "SUCCESS";

       CHECK_VALUE(INVALID_VALUE);
       CHECK_VALUE(OUT_OF_MEMORY);
       CHECK_VALUE(NOT_INITIALIZED);
       CHECK_VALUE(DEINITIALIZED);
       //CHECK_VALUE(PROFILER_DISABLED);
       //CHECK_VALUE(PROFILER_NOT_INITIALIZED);
       //CHECK_VALUE(PROFILER_ALREADY_STARTED);
       //CHECK_VALUE(PROFILER_ALREADY_STOPPED);
       CHECK_VALUE(NO_DEVICE);
       CHECK_VALUE(INVALID_DEVICE);
       CHECK_VALUE(INVALID_IMAGE);
       CHECK_VALUE(INVALID_CONTEXT);
       CHECK_VALUE(CONTEXT_ALREADY_CURRENT);
       CHECK_VALUE(MAP_FAILED);
       CHECK_VALUE(UNMAP_FAILED);
       CHECK_VALUE(ARRAY_IS_MAPPED);
       CHECK_VALUE(ALREADY_MAPPED);
       CHECK_VALUE(NO_BINARY_FOR_GPU);
       CHECK_VALUE(ALREADY_ACQUIRED);
       CHECK_VALUE(NOT_MAPPED);
       CHECK_VALUE(NOT_MAPPED_AS_ARRAY);
       CHECK_VALUE(NOT_MAPPED_AS_POINTER);
       CHECK_VALUE(ECC_UNCORRECTABLE);
       CHECK_VALUE(UNSUPPORTED_LIMIT);
       //CHECK_VALUE(CONTEXT_ALREADY_IN_USE);
       //CHECK_VALUE(PEER_ACCESS_UNSUPPORTED);
       CHECK_VALUE(INVALID_SOURCE);
       CHECK_VALUE(FILE_NOT_FOUND);
       CHECK_VALUE(SHARED_OBJECT_SYMBOL_NOT_FOUND);
       CHECK_VALUE(SHARED_OBJECT_INIT_FAILED);
       CHECK_VALUE(OPERATING_SYSTEM);
       CHECK_VALUE(INVALID_HANDLE);
       CHECK_VALUE(NOT_FOUND);
       CHECK_VALUE(NOT_READY);
       CHECK_VALUE(LAUNCH_FAILED);
       CHECK_VALUE(LAUNCH_OUT_OF_RESOURCES);
       CHECK_VALUE(LAUNCH_TIMEOUT);
       CHECK_VALUE(LAUNCH_INCOMPATIBLE_TEXTURING);
       //CHECK_VALUE(PEER_ACCESS_ALREADY_ENABLED);
       //CHECK_VALUE(PEER_ACCESS_NOT_ENABLED);
       //CHECK_VALUE(PRIMARY_CONTEXT_ACTIVE);
       //CHECK_VALUE(CONTEXT_IS_DESTROYED);
       //CHECK_VALUE(ASSERT);
       //CHECK_VALUE(TOO_MANY_PEERS);
       //CHECK_VALUE(HOST_MEMORY_ALREADY_REGISTERED);
       //CHECK_VALUE(HOST_MEMORY_NOT_REGISTERED);
       //CHECK_VALUE(NOT_PERMITTED);
       //CHECK_VALUE(NOT_SUPPORTED);
       CHECK_VALUE(UNKNOWN);
       default:
           return "Unknown";
   }
}

#endif  // OSD_CUDA_H
