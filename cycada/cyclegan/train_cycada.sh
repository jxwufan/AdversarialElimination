python train.py --name cycada_mnistfg2mnist_noIdentity \
    --resize_or_crop=None \
    --loadSize=32 --fineSize=32 --which_model_netD n_layers --n_layers_D 3 \
    --model cycle_gan_semantic \
    --lambda_A 1 --lambda_B 1 --lambda_identity 0 \
    --no_flip --batchSize 400 \
    --dataset_mode mnist_mnistfg --dataroot /home/fan/data \
    --which_direction BtoA

