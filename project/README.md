### First training (used default configs)
### Dataset size 256

***PIPELINE***
1. Wait for run_Unet.py to save some Model_XXX.pt files.
2. Run generate.py (Generates the images).
3. Run compute_fmem.py (Calculates $\tau_{mem}$).
4. Generate stats1.npz once.
5. Run compute_FID.py (Calculates $\tau_{gen}$).


Saving dataset statistics (needed for FID evaluation):
python -m pytorch_fid project/data/I_32x32_rgb/ Saves/FID_ref/stats1.npz --save-stats --device cpu

Training:
    python run_Unet.py -n 256 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -t -1

Evaluation:
    GEN
    python generate.py -D ISIC -n 256 -i 0 -s 32 -B 512 -LR 0.0001 -O Adam -W 32 -Ns 100 --device cuda:0

    FMEM (-B to be set at 512 if dataset size >=512)
    python compute_fmem.py -D ISIC -n 256 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B 256 -Ns 1 --gap_threshold 0.333 --device cuda:0

    FID (-B to be set at 512 if dataset size >=512)
    python compute_FID.py -D ISIC -n 256 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B 256 -istat 1 --N1 0 --N2 1 --device cpu




### CONDITIONING
python run_Unet_conditioned.py -n $N_SIZE -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -t -1

python generate_conditioned.py -D ISIC_Conditioned -n 512 -i 0 -s 32 -B 512 -LR 0.0001 -O Adam -W 32 -Ns 100 --device mps -c 1

python compute_fmem_conditioned.py -D ISIC_Conditioned -n 512 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B 512 -Ns 1 --gap_threshold 0.333 --device mps -c 1

python compute_FID_conditioned.py -D ISIC_Conditioned -n 512 -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B 512 -istat 2 --N1 0 --N2 1 --device cpu -c 1

python plot_timescales_conditioned.py -n 256 -classes 2 5