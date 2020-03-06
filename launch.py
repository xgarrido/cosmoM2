import fisher
import deriveeC
import matplotlib.pyplot as plt
corrmat = fisher.plot_subplanck(deriveeC.planck_parameters,1600)[2]
plt.show()

plt.figure()
plt.imshow(corrmat)
plt.colorbar()
plt.show()
