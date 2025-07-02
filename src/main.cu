#include "kernels.cuh"
#include "host_functions.cuh"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Error: Usage: " << argv[0] << " <velocity set> <ID>" << std::endl;
        return 1;
    }
    std::string VELOCITY_SET = argv[1];
    std::string SIM_ID = argv[2];

    std::string SIM_DIR = createSimulationDirectory(VELOCITY_SET,SIM_ID);
    //computeAndPrintOccupancy();
    initDeviceVars();

    // ================================================================================================== //

    dim3 threadsPerBlock(BLOCK_SIZE_X,BLOCK_SIZE_Y,BLOCK_SIZE_Z);
    dim3 numBlocks((NX + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (NY + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (NZ + threadsPerBlock.z - 1) / threadsPerBlock.z);

    dim3 threadsPerBlockInOut(BLOCK_SIZE_X*2,BLOCK_SIZE_Y*2);  
    dim3 numBlocksInOut((NX + threadsPerBlockInOut.x - 1) / threadsPerBlockInOut.x,
                        (NY + threadsPerBlockInOut.y - 1) / threadsPerBlockInOut.y);
                         
    cudaStream_t mainStream;
    checkCudaErrors(cudaStreamCreate(&mainStream));

    gpuInitFieldsAndDistributions<<<numBlocks,threadsPerBlock,0,mainStream>>> (lbm); 
    getLastCudaError("gpuInitFieldsAndDistributions");

    float* d_axial;
    float* d_radial_sum;
    int* d_radial_count;

    cudaMalloc(&d_axial, sizeof(float) * NZ);
    cudaMalloc(&d_radial_sum, sizeof(float) * NZ);
    cudaMalloc(&d_radial_count, sizeof(int) * NZ);

    cudaMemset(d_axial, 0, sizeof(float) * NZ);
    cudaMemset(d_radial_sum, 0, sizeof(float) * NZ);
    cudaMemset(d_radial_count, 0, sizeof(int) * NZ);

    auto START_TIME = std::chrono::high_resolution_clock::now();
    std::vector<float> axial_accum(NZ, 0.0f);
    std::vector<float> radial_accum(NZ, 0.0f);
    std::vector<int>   radial_count(NZ, 0);

    for (int STEP = 0; STEP <= NSTEPS ; ++STEP) {
        std::cout << "Passo " << STEP << " de " << NSTEPS << " iniciado..." << std::endl;

        // =================================== INFLOW =================================== //

            gpuApplyInflow<<<numBlocksInOut,threadsPerBlockInOut>>> (lbm,STEP); 
            getLastCudaError("gpuApplyInflow");

        // =============================================================================  //
        
        // ========================= COLLISION & STREAMING ========================= //
            
            gpuEvolvePhaseField<<<numBlocks,threadsPerBlock,0,mainStream>>> (lbm); 
            getLastCudaError("gpuEvolvePhaseField");
            gpuMomCollisionStream<<<numBlocks,threadsPerBlock,0,mainStream>>> (lbm); 
            getLastCudaError("gpuMomCollisionStream");

        // ========================================================================= //    

        // =================================== BOUNDARIES =================================== //

            gpuReconstructBoundaries<<<numBlocks,threadsPerBlock,0,mainStream>>> (lbm); 
            getLastCudaError("gpuReconstructBoundaries");
            gpuApplyPeriodicXY<<<numBlocks,threadsPerBlock,0,mainStream>>> (lbm);
            getLastCudaError("gpuApplyPeriodicXY");
            gpuApplyOutflow<<<numBlocksInOut,threadsPerBlockInOut,0,mainStream>>> (lbm);
            getLastCudaError("gpuApplyOutflow");

        // ================================================================================== //
         #define DYNAMIC_SHARED_SIZE 0
         #define AVERAGING_START 800
         #define AVERAGING_END 1000
         #define AVERAGING_SAMPLES ((AVERAGING_END - AVERAGING_START) / MACRO_SAVE + 1)

            gpuCollectProfiles<<<numBlocks, threadsPerBlock, DYNAMIC_SHARED_SIZE, mainStream>>>(
                lbm, d_axial, d_radial_sum, d_radial_count, DIAM
            );
            getLastCudaError("gpuCollectProfiles");

        checkCudaErrors(cudaDeviceSynchronize());

        if (STEP % MACRO_SAVE == 0) {
            copyAndSaveToBinary(lbm.phi, NX * NY * NZ, SIM_DIR, SIM_ID, STEP, "phi");

            std::vector<float> h_axial(NZ);
            std::vector<float> h_radial_sum(NZ);
            std::vector<int>   h_radial_count(NZ);

            cudaMemcpy(h_axial.data(), d_axial, sizeof(float) * NZ, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_radial_sum.data(), d_radial_sum, sizeof(float) * NZ, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_radial_count.data(), d_radial_count, sizeof(int) * NZ, cudaMemcpyDeviceToHost);

            // Acumular os perfis apenas entre os passos definidos
            if (STEP >= AVERAGING_START && STEP <= AVERAGING_END) {
                for (int z = 0; z < NZ; ++z) {
                    if (h_axial[z] > 0.0f) axial_accum[z] += h_axial[z];
                }
                for (int r = 0; r < NZ; ++r) {
                    radial_accum[r] += h_radial_sum[r];
                    radial_count[r] += h_radial_count[r];
                }
            }

            // Limpar buffers da GPU
            cudaMemsetAsync(d_axial, 0, sizeof(float) * NZ, mainStream);
            cudaMemsetAsync(d_radial_sum, 0, sizeof(float) * NZ, mainStream);
            cudaMemsetAsync(d_radial_count, 0, sizeof(int) * NZ, mainStream);
        }

    }
    auto END_TIME = std::chrono::high_resolution_clock::now();

        // Salvar perfil axial médio
    std::ofstream out_axial(SIM_DIR + "/phi_centerline.csv");
    for (int z = 0; z < NZ; ++z) {
        float phi_avg = axial_accum[z] / AVERAGING_SAMPLES;
        if (phi_avg > 1e-6f) {
            out_axial << static_cast<float>(z) / DIAM << "," << 1.0f / phi_avg << "\n";
        }
    }
    out_axial.close();

    // Salvar perfil radial médio
    std::ofstream out_radial(SIM_DIR + "/phi_radial_profile.csv");
    for (int r = 0; r < NZ; ++r) {
        if (radial_count[r] > 0) {
            float avg = radial_accum[r] / radial_count[r];
            float r_over_z = static_cast<float>(r) / (12.0f * DIAM);
            out_radial << r_over_z << "," << avg << "\n";
        }
    }
    out_radial.close();

    checkCudaErrors(cudaStreamDestroy(mainStream));
    cleanupDeviceMemory(lbm);

    cudaFree(d_axial);
    cudaFree(d_radial_sum);
    cudaFree(d_radial_count);

    std::chrono::duration<double> ELAPSED_TIME = END_TIME - START_TIME;
    long long TOTAL_CELLS = static_cast<long long>(NX) * NY * NZ * NSTEPS;
    double MLUPS = static_cast<double>(TOTAL_CELLS) / (ELAPSED_TIME.count() * 1e6);

    std::cout << "\n// =============================================== //\n";
    std::cout << "     Total execution time    : " << ELAPSED_TIME.count() << " seconds\n";
    std::cout << "     Performance             : " << MLUPS << " MLUPS\n";
    std::cout << "// =============================================== //\n" << std::endl;

    generateSimulationInfoFile(SIM_DIR,SIM_ID,VELOCITY_SET,NSTEPS,MACRO_SAVE,TAU,MLUPS);
    getLastCudaError("Final sync");
    return 0;
}
