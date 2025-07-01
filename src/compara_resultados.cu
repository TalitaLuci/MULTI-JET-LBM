#include "kernels.cuh"
#include "device_functions.cuh"

__global__ void gpuCollectProfiles(LBMFields lbm, float* axial_profile, float* radial_accum, int* radial_count, const int D) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    if (x >= NX || y >= NY || z >= NZ) return;

    const int xc = NX / 2;
    const int yc = NY / 2;
    const int idx = gpuIdxGlobal3(x,y,z);

    const float phi_val = lbm.phi[idx];

    // --------------------------
    // Perfil axial no centro
    // --------------------------
    if (x == xc && y == yc && phi_val != 0.0f) {
        axial_profile[z] = 1.0f / phi_val;
    }

    // --------------------------
    // Perfil radial médio em z = 12D
    // --------------------------
    const int z0 = 12 * D;
    if (z == z0) {
        const int dx = x - xc;
        const int dy = y - yc;
        const int r_bin = __float2int_rn(sqrtf(dx * dx + dy * dy));  // bin inteiro

        if (r_bin < NZ) {
            atomicAdd(&radial_accum[r_bin], phi_val);
            atomicAdd(&radial_count[r_bin], 1);
        }
    }
}
