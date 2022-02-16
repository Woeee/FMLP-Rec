import torch
import pickle as pkl

model_lists=['FMLPRec-Beauty------.pt']
model2list={}
for model_path in model_lists:
    data=torch.load(model_path)
    filters = data['item_encoder.layer.0.filterlayer.complex_weight']
    filter_mean = torch.mean(filters, 2)[0]
    model2list[model_path]=filter_mean.cpu().numpy().tolist()

pkl.dump(model2list, open('filter.pkl', 'wb'))