
import argparse
import vae

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, default='', help='path to input data')
args = parser.parse_args()

'''load data'''
input_param = {
    'path': os.path.join(args.d, args.f),
    'batchsize': args.batchsize,
    'input_data_type': 'float32',
    'shuffle': True
}
data_handler = dataloader.InputHandle(input_param)

'''load model'''
mdl = vae.BetaVAE()

'''tensorboard monitor'''
tf.summary.scalar('Loss', mdl.loss)
train_summaries = tf.summary.merge_all()

'''train'''
def train():
    
    saver=tf.train.Saver()
    configProt = tf.ConfigProto()
    configProt.gpu_options.allow_growth = True
    configProt.allow_soft_placement = True
    
    with tf.Session(config=configProt) as sess:
    
        for epoch in range(args.numepoch):
            

'''test'''

if __name__=='__main__':
    train()