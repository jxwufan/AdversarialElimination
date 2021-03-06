import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    elif opt.dataset_mode == 'unaligned_A_labeled':
        from data.unaligned_A_labeled_dataset import UnalignedALabeledDataset
        dataset = UnalignedALabeledDataset()
    elif opt.dataset_mode == 'mnist_svhn':
        from data.mnist_svhn_dataset import MnistSvhnDataset
        dataset = MnistSvhnDataset()
    elif opt.dataset_mode == 'mnist_mnistfg':
        from data.mnist_mnistfg_dataset import MnistMnistfgDataset
        dataset = MnistMnistfgDataset()
    elif opt.dataset_mode == 'mnistfg_test':
        from data.mnistfg_test_dataset import MnistfgTestDataset
        dataset = MnistfgTestDataset()
    elif opt.dataset_mode == 'cifar10_cifar10fg':
        from data.cifar10_cifar10fg_dataset import Cifar10Cifar10fgDataset
        dataset = Cifar10Cifar10fgDataset()
    elif opt.dataset_mode == 'cifar10fg_test':
        from data.cifar10fg_test_dataset import Cifar10fgTestDataset
        dataset = Cifar10fgTestDataset()
    elif opt.dataset_mode == 'cifar10_cifar10bim':
        from data.cifar10_cifar10bim_dataset import Cifar10Cifar10bimDataset
        dataset = Cifar10Cifar10bimDataset()
    elif opt.dataset_mode == 'cifar10bim_test':
        from data.cifar10bim_test_dataset import Cifar10bimTestDataset
        dataset = Cifar10bimTestDataset()
    elif opt.dataset_mode == 'cifar10_cifar10df':
        from data.cifar10_cifar10df_dataset import Cifar10Cifar10dfDataset
        dataset = Cifar10Cifar10dfDataset()
    elif opt.dataset_mode == 'cifar10df_test':
        from data.cifar10df_test_dataset import Cifar10dfTestDataset
        dataset = Cifar10dfTestDataset()
    elif opt.dataset_mode == 'mnist_mnistdf':
        from data.mnist_mnistdf_dataset import MnistMnistdfDataset
        dataset = MnistMnistdfDataset()
    elif opt.dataset_mode == 'mnistdf_test':
        from data.mnistdf_test_dataset import MnistdfTestDataset
        dataset = MnistdfTestDataset()
    elif opt.dataset_mode == 'mnist_mnistbim':
        from data.mnist_mnistbim_dataset import MnistMnistbimDataset
        dataset = MnistMnistbimDataset()
    elif opt.dataset_mode == 'mnistbim_test':
        from data.mnistbim_test_dataset import MnistbimTestDataset
        dataset = MnistbimTestDataset()
    elif opt.dataset_mode == 'svhn_mnist':
        from data.svhn_mnist_dataset import SvhnMnistDataset
        dataset = SvhnMnistDataset()
  
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data
