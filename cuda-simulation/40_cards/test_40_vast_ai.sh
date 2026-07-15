# connect to the remote server using SSH
ssh -p <porta> root@<ip>
# TERM=xterm-256color ssh -p 46123 root@70.52.83.135 -L 8080:localhost:8080

# download repo and move inside working directory
git clone https://github.com/N1k06/stracciacamicia \
    && cd stracciacamicia/cuda-simulation/40_cards

# try to compile for rtx 5090
nvcc -O3 -arch=sm_120 --ptxas-options=-v straccia_search_40_multigpu.cu -o straccia_search_40

chmod +x launch_multi_gpu.sh merge_and_status.sh

./launch_multi_gpu.sh 4500 360000 60 <checkpoint_number>

# unisci i file di hit delle due GPU e controlla lo stato
./merge_and_status.sh

#riverifica ogni candidato in Python — usa max_turns=4500 (quello reale di questo run!)
python3 inspect_hits_40.py hits_40_merged.bin hits_40_report.txt 4500

# il passo che conta davvero: verifica se sono cicli GENUINI o solo partite lunghe troncate
python3 confirm_cycle.py --counts 28,4,4,4 hits_40_merged.bin \
    --no-header --max-turns 2000000 --report cycles_report.txt