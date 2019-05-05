import pickle
import matplotlib.pyplot as plt

x_adv_test = pickle.load(open("./x_adv_test.pk", 'rb'))
plt.imshow(x_adv_test[0].reshape(28, 28))
plt.show()
