Write-Host "Compiling Serial..."
g++ -O3 -std=c++11 kmeans_serial.cpp -o kmeans_serial.exe

Write-Host "Compiling OpenMP..."
g++ -O3 -std=c++11 -fopenmp kmeans_omp.cpp -o kmeans_omp.exe

Write-Host "Done!" -ForegroundColor Green