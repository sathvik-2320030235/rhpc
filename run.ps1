Write-Host "--- Serial ---"
.\kmeans_serial.exe 1000000 100 100 50

Write-Host "`n--- OpenMP ---"
$env:OMP_NUM_THREADS = 4
.\kmeans_omp.exe

Write-Host "`n--- Hybrid MPI + CUDA ---"
$env:OMP_NUM_THREADS = 4
mpiexec -n 2 .\kmeans_parallel.exe 1000000 100 100 50