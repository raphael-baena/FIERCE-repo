# clean-train

Optimized code for training usual datasets/model

Examples of use (to reach peak accuracy, not for fastest prototyping):

To train MNIST with 99.64% accuracy (5 minutes):

python main.py --epochs 30 --milestones "[10, 20]" --dataset MNIST --feature-maps 8

To train MNIST with 10% database and 99.31% accuracy (10 minutes):

python main.py --epochs 300 --milestones "[100, 200]" --dataset MNIST --dataset-size 6000 --model wideresnet --feature-maps 4 --skip-epochs 300

To train Fashion-MNIST with 96% accuracy (2 hours):

python main.py --dataset fashion --mixup

To train CIFAR10 with 95.90% accuracy (1 hour):

python main.py --mixup

To train CIFAR100 with 78.55% accuracy (93.54% top-5) (1hour):

python main.py --mixup --dataset cifar100

To train CIFAR100 with 80.12% accuracy (94.70% top-5) (4h):

python main.py --mixup --model wideresnet --feature-maps 16 --dataset CIFAR100

To train Omniglot (few-shot) with 99.85% accuracy (99.39% in 1-shot) (10minutes):

python main.py --dataset omniglotfs --dataset-device cpu --feature-maps 16 --milestones '[10,20]' --epochs 30

To train CUBFS (few-shot) with 85.24% accuracy (68.14% in 1-shot) (2h):

python main.py --dataset cubfs --mixup --rotations

To train CIFARFS (few-shot) with 84.87% accuracy (70.43% in 1-shot) (1h):

python main.py --dataset cifarfs --mixup --rotations --skip-epochs 300

To train CIFARFS (few-shot) with 86.83% accuracy (70.27% in 1-shot) (3h):

python main.py --dataset cifarfs --mixup --model wideresnet --feature-maps 16 --skip-epochs 300

To train MiniImageNet (few-shot) with 80.43% accuracy (64.11% in 1-shot) (2h):

python main.py --dataset miniimagenet --model resnet12 --gamma 0.2 --milestones '[30,60,90]' --epochs 120 --batch-size 128 --preprocessing 'EME'

To train MiniImageNet (few-shot) with 82.43% accuracy (65.63% in 1-shot) (40h):

python main.py --dataset miniimagenet --feature-maps 16 --model S2M2R --lr -0.001 --epochs 600 --milestones '[]' --rotations