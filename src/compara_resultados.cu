#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <map>
#include "constants.cuh"
#include "device_functions.cuh"

void salvarPerfisDeFase(float* d_phi, int NX, int NY, int NZ, int D) {
    size_t size = NX * NY * NZ;
    float* h_phi = new float[size];

    // Copiar phi da GPU para a CPU
    cudaMemcpy(h_phi, d_phi, sizeof(float) * size, cudaMemcpyDeviceToHost);

    // =============================
    // 1. Perfil axial no centro do jato
    // =============================
    std::ofstream out_axial("phi_centerline.csv");

    int xc = NX / 2;
    int yc = NY / 2;

    for (int z = 0; z < NZ; ++z) {
        int idx = z * NY * NX + yc * NX + xc;
        float phi_val = h_phi[idx];
        if (phi_val != 0.0f) {
            out_axial << static_cast<float>(z) / D << "," << 1.0f / phi_val << std::endl;
        }
    }

    out_axial.close();
    std::cout << "[OK] Arquivo phi_centerline.csv gerado.\n";

    // =============================
    // 2. Perfil radial médio em z = 12D
    // =============================
    int z0 = 12 * D;
    std::map<int, std::vector<float>> radial_bins;

    for (int y = 0; y < NY; ++y) {
        for (int x = 0; x < NX; ++x) {
            int idx = z0 * NY * NX + y * NX + x;
            float phi_val = h_phi[idx];

            int dx = x - xc;
            int dy = y - yc;
            int r_bin = static_cast<int>(std::sqrt(dx * dx + dy * dy));

            radial_bins[r_bin].push_back(phi_val);
        }
    }

    std::ofstream out_radial("phi_radial_profile.csv");
    for (const auto& [r_bin, values] : radial_bins) {
        float sum = 0.0f;
        for (float v : values) sum += v;
        float avg = sum / values.size();
        float r_over_z = static_cast<float>(r_bin) / static_cast<float>(z0);
        out_radial << r_over_z << "," << avg << std::endl;
    }

    out_radial.close();
    std::cout << "[OK] Arquivo phi_radial_profile.csv gerado.\n";

    delete[] h_phi;
}
