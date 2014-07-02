//
//   Copyright 2013 Pixar
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

#include "../osd/cudaComputeController.h"
#include "../osd/cuda.h"
#include "../osd/cudaComputeContext.h"

#include <stdio.h>
#include <string.h>

extern unsigned char datatoc_cudaKernel_fatbin[];

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCudaComputeController::OsdCudaComputeController(int device_idx) {
    cuInit(0);

    CUdevice device;
    // By default use first available device.
    cuda_assert(cuDeviceGet(&device, device_idx));
    cuda_assert(cuCtxCreate(&_context, 0, device));

    cuda_assert(cuModuleLoadData(&_module, datatoc_cudaKernel_fatbin));
}

OsdCudaComputeController::~OsdCudaComputeController() {
    cuda_assert(cuModuleUnload(_module));
    cuda_assert(cuCtxDestroy(_context));
}

void OsdCudaComputeController::OsdCudaComputeDevince(int device_idx) {
    // Keep an eue on this, only change device when all the kernel queues
    // are empty.
    CUdevice device;
    cuda_assert(cuCtxDestroy(_context));
    cuda_assert(cuDeviceGet(&device, device_idx));
    cuda_assert(cuCtxCreate(&_context, 0, device));
}

void
OsdCudaComputeController::ApplyBilinearFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * F_IT = context->GetTable(FarSubdivisionTables::F_IT);
    const OsdCudaTable * F_ITa = context->GetTable(FarSubdivisionTables::F_ITa);
    assert(F_IT);
    assert(F_ITa);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr F_IT_mem = F_IT->GetCudaMemory();
    CUdeviceptr F_ITa_mem = F_ITa->GetCudaMemory();

    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();

    cuda_run_kernel(OsdCudaComputeFace,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &F_IT_mem,
                    &F_ITa_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end);
}

void
OsdCudaComputeController::ApplyBilinearEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * E_IT = context->GetTable(FarSubdivisionTables::E_IT);
    assert(E_IT);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr E_IT_mem = E_IT->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();

    cuda_run_kernel(OsdCudaComputeBilinearEdge,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &E_IT_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end);
}

void
OsdCudaComputeController::ApplyBilinearVertexVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    assert(V_ITa);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr V_ITa_mem = V_ITa->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();

    cuda_run_kernel(OsdCudaComputeBilinearVertex,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &V_ITa_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end);
}

void
OsdCudaComputeController::ApplyCatmarkFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * F_IT = context->GetTable(FarSubdivisionTables::F_IT);
    const OsdCudaTable * F_ITa = context->GetTable(FarSubdivisionTables::F_ITa);
    assert(F_IT);
    assert(F_ITa);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr F_IT_mem = F_IT->GetCudaMemory();
    CUdeviceptr F_ITa_mem = F_ITa->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();

    cuda_run_kernel(OsdCudaComputeFace,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &F_IT_mem,
                    &F_ITa_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end);
}

void
OsdCudaComputeController::ApplyCatmarkQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * F_IT = context->GetTable(FarSubdivisionTables::F_IT);
    assert(F_IT);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr F_IT_mem = F_IT->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();

    cuda_run_kernel(OsdCudaComputeQuadFace,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &F_IT_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end);
}

void
OsdCudaComputeController::ApplyCatmarkTriQuadFaceVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * F_IT = context->GetTable(FarSubdivisionTables::F_IT);
    assert(F_IT);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr F_IT_mem = F_IT->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();

    cuda_run_kernel(OsdCudaComputeTriQuadFace,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &F_IT_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end);
}

void
OsdCudaComputeController::ApplyCatmarkEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * E_IT = context->GetTable(FarSubdivisionTables::E_IT);
    const OsdCudaTable * E_W = context->GetTable(FarSubdivisionTables::E_W);
    assert(E_IT);
    assert(E_W);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr E_IT_mem = E_IT->GetCudaMemory();
    CUdeviceptr E_W_mem = E_W->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();

    cuda_run_kernel(OsdCudaComputeEdge,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &E_IT_mem,
                    &E_W_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end);
}

void
OsdCudaComputeController::ApplyCatmarkRestrictedEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * E_IT = context->GetTable(FarSubdivisionTables::E_IT);
    assert(E_IT);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr E_IT_mem = E_IT->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();

    cuda_run_kernel(OsdCudaComputeRestrictedEdge,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &E_IT_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end);
}

void
OsdCudaComputeController::ApplyCatmarkVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_IT = context->GetTable(FarSubdivisionTables::V_IT);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables::V_W);
    assert(V_ITa);
    assert(V_IT);
    assert(V_W);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr V_ITa_mem = V_ITa->GetCudaMemory();
    CUdeviceptr V_IT_mem = V_IT->GetCudaMemory();
    CUdeviceptr V_W_mem = V_W->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();

    cuda_run_kernel(OsdCudaComputeVertexB,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &V_ITa_mem,
                    &V_IT_mem,
                    &V_W_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end);
}

void
OsdCudaComputeController::ApplyCatmarkVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables::V_W);
    assert(V_ITa);
    assert(V_W);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr V_ITa_mem = V_ITa->GetCudaMemory();
    CUdeviceptr V_W_mem = V_W->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();
    int pass = 0;

    cuda_run_kernel(OsdCudaComputeVertexA,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &V_ITa_mem,
                    &V_W_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end, &pass);
}

void
OsdCudaComputeController::ApplyCatmarkVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables::V_W);
    assert(V_ITa);
    assert(V_W);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr V_ITa_mem = V_ITa->GetCudaMemory();
    CUdeviceptr V_W_mem = V_W->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();
    int pass = 1;

    cuda_run_kernel(OsdCudaComputeVertexA,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &V_ITa_mem,
                    &V_W_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end, &pass);
}

void
OsdCudaComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB1(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_IT = context->GetTable(FarSubdivisionTables::V_IT);
    assert(V_ITa);
    assert(V_IT);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr V_ITa_mem = V_ITa->GetCudaMemory();
    CUdeviceptr V_IT_mem = V_IT->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();

    cuda_run_kernel(OsdCudaComputeRestrictedVertexB1,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &V_ITa_mem,
                    &V_IT_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end);
}

void
OsdCudaComputeController::ApplyCatmarkRestrictedVertexVerticesKernelB2(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_IT = context->GetTable(FarSubdivisionTables::V_IT);
    assert(V_ITa);
    assert(V_IT);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr V_ITa_mem = V_ITa->GetCudaMemory();
    CUdeviceptr V_IT_mem = V_IT->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();

    cuda_run_kernel(OsdCudaComputeRestrictedVertexB2,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &V_ITa_mem,
                    &V_IT_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end);
}

void
OsdCudaComputeController::ApplyCatmarkRestrictedVertexVerticesKernelA(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    assert(V_ITa);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr V_ITa_mem = V_ITa->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();

    cuda_run_kernel(OsdCudaComputeRestrictedVertexA,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &V_ITa_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end);
}

void
OsdCudaComputeController::ApplyLoopEdgeVerticesKernel(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * E_IT = context->GetTable(FarSubdivisionTables::E_IT);
    const OsdCudaTable * E_W = context->GetTable(FarSubdivisionTables::E_W);
    assert(E_IT);
    assert(E_W);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr E_IT_mem = E_IT->GetCudaMemory();
    CUdeviceptr E_W_mem = E_W->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();
    int pass = 1;

    cuda_run_kernel(OsdCudaComputeVertexA,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &E_IT_mem,
                    &E_W_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end, &pass);
}

void
OsdCudaComputeController::ApplyLoopVertexVerticesKernelB(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_IT = context->GetTable(FarSubdivisionTables::V_IT);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables::V_W);
    assert(V_ITa);
    assert(V_IT);
    assert(V_W);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr V_ITa_mem = V_ITa->GetCudaMemory();
    CUdeviceptr V_IT_mem = V_IT->GetCudaMemory();
    CUdeviceptr V_W_mem =V_W->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();

    cuda_run_kernel(OsdCudaComputeLoopVertexB,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &V_ITa_mem,
                    &V_IT_mem,
                    &V_W_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end);
}

void
OsdCudaComputeController::ApplyLoopVertexVerticesKernelA1(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables::V_W);
    assert(V_ITa);
    assert(V_W);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr V_ITa_mem = V_ITa->GetCudaMemory();
    CUdeviceptr V_W_mem = V_W->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();
    int pass = 0;

    cuda_run_kernel(OsdCudaComputeVertexA,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &V_ITa_mem,
                    &V_W_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end, &pass);
}

void
OsdCudaComputeController::ApplyLoopVertexVerticesKernelA2(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaTable * V_ITa = context->GetTable(FarSubdivisionTables::V_ITa);
    const OsdCudaTable * V_W = context->GetTable(FarSubdivisionTables::V_W);
    assert(V_ITa);
    assert(V_W);

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();
    CUdeviceptr varying = _currentBindState.GetOffsettedVaryingBuffer();
    CUdeviceptr V_ITa_mem = V_ITa->GetCudaMemory();
    CUdeviceptr V_W_mem = V_W->GetCudaMemory();
    int vertex_offset = batch.GetVertexOffset();
    int table_offset = batch.GetTableOffset();
    int start = batch.GetStart();
    int end = batch.GetEnd();
    int pass = 1;

    cuda_run_kernel(OsdCudaComputeVertexA,
                    &vertex, &varying,
                    (void *)&_currentBindState.vertexDesc.length,
                    (void *)&_currentBindState.vertexDesc.stride,
                    (void *)&_currentBindState.varyingDesc.length,
                    (void *)&_currentBindState.varyingDesc.stride,
                    &V_ITa_mem,
                    &V_W_mem,
                    &vertex_offset,
                    &table_offset,
                    &start, &end, &pass);
}

void
OsdCudaComputeController::ApplyVertexEdits(
    FarKernelBatch const &batch, OsdCudaComputeContext const *context) const {

    assert(context);

    const OsdCudaHEditTable *edit = context->GetEditTable(batch.GetTableIndex());
    assert(edit);

    const OsdCudaTable * primvarIndices = edit->GetPrimvarIndices();
    const OsdCudaTable * editValues = edit->GetEditValues();

    CUdeviceptr vertex = _currentBindState.GetOffsettedVertexBuffer();

    if (edit->GetOperation() == FarVertexEdit::Add) {
        CUdeviceptr edit_primvar_offset = edit->GetPrimvarOffset();
        CUdeviceptr edit_primvar_width = edit->GetPrimvarWidth();
        CUdeviceptr primvar_indices_mem = primvarIndices->GetCudaMemory();
        CUdeviceptr edit_values_mem = editValues->GetCudaMemory();
        int vertex_offset = batch.GetVertexOffset();
        int table_offset = batch.GetTableOffset();
        int start = batch.GetStart();
        int end = batch.GetEnd();

        cuda_run_kernel(OsdCudaComputeVertexA,
                        &vertex,
                        (void *)&_currentBindState.vertexDesc.length,
                        (void *)&_currentBindState.vertexDesc.stride,
                        &edit_primvar_offset,
                        &edit_primvar_width,
                        &vertex_offset,
                        &table_offset,
                        &start, &end,
                        &primvar_indices_mem,
                        &edit_values_mem);
    } else if (edit->GetOperation() == FarVertexEdit::Set) {
        // XXXX TODO
    }
}

void
OsdCudaComputeController::Synchronize() {

    cuCtxSynchronize();
}

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
