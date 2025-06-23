# import matplotlib.pyplot as plt 

# train_loss = []
# valid_loss = []

# for i in [0,2,3,4]:
# 	with open('out.1.'+str(i)) as infile: # Change to your model training output file:

# 		for line in infile:

# 			if 'Valid Loss' in line:
# 				items = line.strip().split()

# 				train_loss.append(float(items[5].replace(",","")))
# 				valid_loss.append(float(items[8].replace(",","")))

# 	plt.plot(train_loss, label='train')
# 	plt.plot(valid_loss, label='valid')

# 	plt.legend()
# 	plt.show()

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs = axs.flatten()  # Flatten to easily index with i

for idx, i in enumerate([0, 2, 3, 4]):
    train_loss = []
    valid_loss = []

    with open('out.1.' + str(i)) as infile:
        for line in infile:
            if 'Valid Loss' in line:
                items = line.strip().split()
                train_loss.append(float(items[5].replace(",", "")))
                valid_loss.append(float(items[8].replace(",", "")))

    axs[idx].plot(train_loss, label='train')
    axs[idx].plot(valid_loss, label='valid')
    axs[idx].set_title(f'Model out.1.{i}')
    axs[idx].legend()

plt.tight_layout()
plt.show()
