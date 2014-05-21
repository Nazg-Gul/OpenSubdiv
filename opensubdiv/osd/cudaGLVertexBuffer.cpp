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

#include "../osd/cudaGLVertexBuffer.h"
#include "../osd/cuda.h"
#include "../osd/error.h"
#include "../osd/opengl.h"

#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

OsdCudaGLVertexBuffer::OsdCudaGLVertexBuffer(int numElements, int numVertices)
    : _numElements(numElements), _numVertices(numVertices),
      _vbo(0), _devicePtr(0), _cudaResource(0) {
}

OsdCudaGLVertexBuffer::~OsdCudaGLVertexBuffer() {

    unmap();
    cuGraphicsUnregisterResource(_cudaResource);
    glDeleteBuffers(1, &_vbo);
}

OsdCudaGLVertexBuffer *
OsdCudaGLVertexBuffer::Create(int numElements, int numVertices) {
    OsdCudaGLVertexBuffer *instance =
        new OsdCudaGLVertexBuffer(numElements, numVertices);
    if (instance->allocate()) return instance;
    OsdError(OSD_CUDA_GL_ERROR,"OsdCudaGLVertexBuffer::Create failed.\n");
    delete instance;
    return NULL;
}

void
OsdCudaGLVertexBuffer::UpdateData(const float *src, int startVertex, int numVertices) {

    map();
    CUresult err = cuMemcpyHtoD(_devicePtr + _numElements * startVertex,
                                src,
                                _numElements * numVertices * sizeof(float));
    if (err != CUDA_SUCCESS)
        OsdError(OSD_CUDA_GL_ERROR, "OsdCudaGLVertexBuffer::UpdateData failed. : %s\n",
                 cuda_GetErrorString(err));
}

int
OsdCudaGLVertexBuffer::GetNumElements() const {

    return _numElements;
}

int
OsdCudaGLVertexBuffer::GetNumVertices() const {

    return _numVertices;
}

CUdeviceptr
OsdCudaGLVertexBuffer::BindCudaBuffer() {

    map();
    return _devicePtr;
}

GLuint
OsdCudaGLVertexBuffer::BindVBO() {

    unmap();
    return _vbo;
}

bool
OsdCudaGLVertexBuffer::allocate() {

    int size = _numElements * _numVertices * sizeof(float);

    glGenBuffers(1, &_vbo);

#if defined(GL_EXT_direct_state_access)
    if (glNamedBufferDataEXT) {
        glNamedBufferDataEXT(_vbo, size, 0, GL_DYNAMIC_DRAW);
    } else {
#else
    {
#endif
        glBindBuffer(GL_ARRAY_BUFFER, _vbo);
        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    // register vbo as cuda resource
    CUresult err = cuGraphicsGLRegisterBuffer(
        &_cudaResource, _vbo, CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD);

    if (err != CUDA_SUCCESS) return false;
    return true;
}

void
OsdCudaGLVertexBuffer::map() {

    if (_devicePtr) return;
    size_t num_bytes;
    CUdeviceptr ptr;

    CUresult err = cuGraphicsMapResources(1, &_cudaResource, 0);
    if (err != CUDA_SUCCESS)
        OsdError(OSD_CUDA_GL_ERROR, "OsdCudaGLVertexBuffer::map failed.\n%s\n", cuda_GetErrorString(err));
    err = cuGraphicsResourceGetMappedPointer(&ptr, &num_bytes, _cudaResource);
    if (err != CUDA_SUCCESS)
        OsdError(OSD_CUDA_GL_ERROR, "OsdCudaGLVertexBuffer::map failed.\n%s\n", cuda_GetErrorString(err));
    _devicePtr = ptr;
}

void
OsdCudaGLVertexBuffer::unmap() {

    if (_devicePtr == 0) return;
    CUresult err = cuGraphicsUnmapResources(1, &_cudaResource, 0);
    if (err != CUDA_SUCCESS)
        OsdError(OSD_CUDA_GL_ERROR, "OsdCudaGLVertexBuffer::unmap failed.\n%s\n", cuda_GetErrorString(err));
    _devicePtr = 0;
}


}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

