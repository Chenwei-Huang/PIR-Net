from test_dataset import PIR_Dataset
import EPE_Loss
import models
import torch.utils as utils
import torch
import time
from torch.utils import data
import logging


def Save_list(list1,filename):
    file = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        file.write(str(list1[i]))
        file.write('\n')
    file.close()

def test_model(model, TestLoader, noise):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Start Testing")

    nbatch = 0
    torch.cuda.empty_cache()
    EPE = []
    ACE = []
    model.eval()

    with torch.no_grad():
        start_time = time.time()
        for index, (images, target) in enumerate(TestLoader):
            nbatch += 1
            images = images.to(device)
            target = target.to(device)
            target = target.float()
            outputs = model.forward(images)

            flowepe = EPE_Loss.evaluate(outputs, target)
            ace = EPE_Loss.ACE(outputs, target)
            print('images{}--  EPE={:.3f} , ACE = {:.3f}'.format(nbatch,flowepe.item(),ace))
            EPE.append(flowepe.item())
            ACE.append(ace)

            del flowepe
            del ace

    Avg_EPE = sum(EPE) / len(EPE)
    Avg_ACE = sum(ACE) / len(ACE)
    EPE.append(Avg_EPE)
    ACE.append(Avg_ACE)
    Save_list(EPE, 'list_EPE(finetune)')
    print('Average EPE = {} , Average ACE = {}'.format(Avg_EPE,Avg_ACE))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("************************")
    logger.info("Type of noise: " + noise)
    elapsed_time = time.time() - start_time
    logger.info("Time elapsed: " + str(elapsed_time))
    logger.info("************************")
    print("Average time for per image:", elapsed_time/len(EPE))

    del EPE
    del ACE
    del Avg_EPE
    del Avg_ACE

def main():
    model = models.flownets()
    for i in range(20,21):
    #setmodecpu
        model.load_state_dict(torch.load('checkpoint/CHECKPOINT_FILE_with_epoch_'+str(i), map_location='cpu')['model_state_dict'])
    # model = model.cuda()

        testData = PIR_Dataset("data/source", "data/target", "data/lables", 'Vanilla')
        TestLoader = utils.data.DataLoader(testData, 1)
        print('*********************test on CHECKPOINT_epoch_' + str(i) + '*********************')
        test_model(model, TestLoader, 'Vanilla')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
main()
