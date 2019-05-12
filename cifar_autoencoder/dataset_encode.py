from keras.models import load_model
import pickle

autoencoder = load_model('cifar_baseline.h5')
DATASET_DIR = '/home/bill.lv22/AdversarialElimination/'

def pkload(name):
    print "start encode " + name
    return pickle.load(open('/home/bill.lv22/'+ name + '.pkl'))

def pred(data):
    return autoencoder.predict(data)

def dump(lst, fName):
    return pickle.dump(lst, open(DATASET_DIR + '/cifar_autoencoder/'+fName+'_decoded.pkl','wb'))

fg_train, fg_test = pkload('fg')
fg_train_decoded = pred(fg_train)
fg_test_decoded = pred(fg_test)
dump([fg_train, fg_test, fg_train_decoded, fg_test_decoded],'fg')	


x_train, x_test, y_train, y_test = pkload('cifar10')
mnist_train_decoded = pred(x_train)
mnist_test_decoded = pred(x_test)
dump([x_train, x_test, mnist_train_decoded, mnist_test_decoded, y_train, y_test], 'cifar10')

bim_train, bim_test = pkload('bim')
bim_train_decoded = pred(bim_train)
bim_test_decoded = pred(bim_test)
dump([bim_train, bim_test, bim_train_decoded, bim_test_decoded], 'bim')


'''
df_train, df_test = pkload('df')
df_train_decoded = pred(df_train)
df_test_decoded = pred(df_test)
dump([df_train, df_test, df_train_decoded, df_test_decoded], 'df')
'''


#x_train, x_test, y_train, y_test = pickle.load(open(DATASET_DIR+'mnist.pkl'))
