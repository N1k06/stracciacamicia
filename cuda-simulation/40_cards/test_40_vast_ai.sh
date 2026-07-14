# connect to the remote server using SSH
ssh -p <porta> root@<ip>

# download repo and move inside working directory
git clone https://github.com/N1k06/stracciacamicia
cd stracciacamicia/cuda-simulation/40_cards

# try to compile with native architecture (if supported)
nvcc -O3 -arch=native --ptxas-options=-v straccia_search_40.cu -o straccia_search_40
# fallback for rtx 5090
# nvcc -O3 -arch=sm_75 --ptxas-options=-v straccia_search_40.cu -o straccia_search_40

# run quick test for 1 min
./straccia_search_40 multinomial_table.bin 10000000 3000 100000 60 \
    ./checkpoint_test.txt \
    ./hits_test.bin

# run full test for 12 hours
./straccia_search_40 multinomial_table.bin 200000000 3000 100000 36000 \
    ./checkpoint.txt \
    ./hits_40.bin

# inspect hits and generate report
python inspect_hits_40.py ./hits_40.bin \
    ./hits_40_report.txt