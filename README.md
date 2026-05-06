## Setup environment

on the hpc, locate the project folder, then:
```bash
module load python3/3.12.9
```
create the virtual env
```bash
python3 -m venv .venv
```
activate it:
```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torchaudio #if previously didn't work
```


## Experiments pipeline

> $N_SIZE = 256, 512, 1024, 2048

> $BATCH_SIZE = 256 for N_SIZE =256, 512 otherwise


### Unconditioned
* Train model (the loader is set to pick up a random set of images)
```bash
python run_Unet_conditioned.py -n $N_SIZE -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -t -1
```

* Generate samples

```bash
python generate.py -D ISIC -n $N_SIZE -i 0 -s 32 -B $BATCH_SIZE -LR 0.0001 -O Adam -W 32 -Ns 1000 --device cuda:0
```
> -Ns is the number of samples (default 100, 1000 is for more accuracy but it also takes more space)

* Evaluate FID
```bash
python compute_FID.py -D ISIC -n $N_SIZE -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B $BATCH_SIZE -istat 1 --N1 0 --N2 10 --device cuda:0
```
> -N2 to be set to 1 if we evaluate 100 previously generated samples, set to 10 if we evaluate 1000 samples


* Evaluate Memorization
```bash
python compute_fmem.py -D ISIC -n $N_SIZE -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B $BATCH_SIZE -Ns 10 --gap_threshold 0.333 --device cuda:0
```
> -Ns to be set to 1 if we evaluate 100 previously generated samples, set to 10 if we evaluate 1000 samples


* Plot results (updated the dataset sizes to plot inside the script)
```bash
python plot_timescales.py
```


---


### Conditioned (balanced)
* Train model (the loader is set to pick up a random set of images, with classes balanced)
```bash
python run_Unet_conditioned.py -n $N_SIZE -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -t -1
```

* Generate samples

```bash
python generate_conditioned.py -D ISIC_Conditioned -n $N_SIZE -i 0 -s 32 -B $BATCH_SIZE -LR 0.0001 -O Adam -W 32 -Ns 1000 --device cuda:0 -c 1
```
> -Ns is the number of samples (default 100, 1000 is for more accuracy but it also takes more space)
> -c is the number of the Fitzpatrick class I-VI (1 to 6), 0 means no conditioning

* Evaluate FID
```bash
python compute_fmem_conditioned.py -D ISIC_Conditioned -n $N_SIZE -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B $BATCH_SIZE -Ns 10 --gap_threshold 0.333 --device cuda:0 -c 1 

```
> -N2 to be set to 1 if we evaluate 100 previously generated samples, set to 10 if we evaluate 1000 samples
> -c is the number of the Fitzpatrick class I-VI (1 to 6) to evaluate


* Evaluate Memorization
```bash
python compute_FID_conditioned.py -D ISIC_Conditioned -n $N_SIZE -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B $BATCH_SIZE -istat 1 --N1 0 --N2 10 --device cuda:0 -c 1
```
> -Ns to be set to 1 if we evaluate 100 previously generated samples, set to 10 if we evaluate 1000 samples
> -c is the number of the Fitzpatrick class I-VI (1 to 6) to evaluate


* Plot results (updated the dataset sizes to plot inside the script)
```bash
python plot_timescales_conditioned.py -D ISIC_Conditioned -n $N_SIZE -i 0 -s 32 -LR 0.0001 -O Adam -W 32 -B $BATCH_SIZE -classes 1 2 3 4 5 6
```
> -classes is the number of classes to plot