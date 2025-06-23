import torch
import csv
import numpy as np

pep_dict = torch.load('pep_data_plm.pt', weights_only=False)
list_tcr_dict = torch.load('tcr_data_plm.pt', weights_only=False)

data_points = {}
with open('nettcr_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    list_tcr_embed = {}


    for row in reader:
      l = []
      l.append(np.array(pep_dict[row['peptide']]))
      tcrs = list_tcr_dict[(row['TRA_aa'],row['TRB_aa'])]
      l.append(np.array(tcrs['A1']))
      l.append(np.array(tcrs['A2']))
      l.append(np.array(tcrs['A3']))
      l.append(np.array(tcrs['B1']))
      l.append(np.array(tcrs['B2']))
      l.append(np.array(tcrs['B3']))
      data_points[f"{row['peptide']}_{row['A1']}_{row['A2']}_{row['A3']}_{row['B1']}_{row['B2']}_{row['B3']}"] = l
      
      
torch.save(data_points, 'tcrlang.pt', pickle_protocol=5)

### TEST WETHERE THIS ACTUALLY WORKED WTF

data_points = torch.load('tcrlang.pt', weights_only=False)

print(len(data_points))

key, value = next(iter(data_points.items()))

for i in value:
    print(i.shape)
print()