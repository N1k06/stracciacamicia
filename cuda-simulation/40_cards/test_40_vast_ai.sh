# connect to the remote server using SSH
ssh -p <porta> root@<ip>

# download repo and move inside working directory
git clone https://github.com/N1k06/stracciacamicia \
    && cd stracciacamicia/cuda-simulation/40_cards

# try to compile for rtx 5090
nvcc -O3 -arch=sm_120 --ptxas-options=-v straccia_search_40_multigpu.cu -o straccia_search_40

# run quick test for 1 min
./straccia_search_40 multinomial_table.bin 10000000 3000 100000 60 \
    ./checkpoint_test.txt \
    ./hits_test.bin

chmod +x launch_multi_gpu.sh merge_and_status.sh
./launch_multi_gpu.sh 4500 360000 60

# inspect hits and generate report
python inspect_hits_40.py ./hits_40.bin \
    ./hits_40_report.txt