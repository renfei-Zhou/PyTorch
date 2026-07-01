# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# import torchvision
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# from my_helper_functions import accuracy_fn, print_train_time, train_step, test_step, eval_model
# import pandas as pd
# import json
# import matplotlib.pyplot as plt

import mlxtend
print(f"mlxtend version: {mlxtend.__version__}")

# # Necessary data -------------------------------------
# torch.manual_seed(42)
# # Setup training data
# train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=torchvision.transforms.ToTensor(), target_transform=None)
# test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor(), target_transform=None)
# # class name
# class_names = train_data.classes
# # batch size
# BATCH_SIZE = 32
# # Turn dataset into iterables (batches)
# train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
# # Setup iterable
# train_features_batch, train_labels_batch = next(iter(train_dataloader))
# # Device
# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" 
# # Necessary data ends here -------------------------------------


# ### 8. Compare model results and train time
# with open("model_0_results.json", "r", encoding="utf-8") as f:
#     model_0_results = json.load(f)

# with open("model_1_results.json", "r", encoding="utf-8") as f:
#     model_1_results = json.load(f)

# with open("model_2_results.json", "r", encoding="utf-8") as f:
#     model_2_results = json.load(f)

# compare_results = pd.DataFrame([model_0_results,
#                                 model_1_results,
#                                 model_2_results])

# print(f"compare results:\n", compare_results.to_string())

# # Visualize our model results
# compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
# # plt.xlabel("accuracy (%)")
# # plt.ylabel("model")
# # plt.show()



# ### 9. Make and evaluate random predictions with best model
# def make_predictions(model: torch.nn.Module,
#                      data: list,
#                      device: torch.device = device):
#     pred_probs = []
#     model.to(device)
#     model.eval()
#     with torch.inference_mode():
#         for sample in data:
#             # Prepare the sample (add a batch dimension and pass to target device)
#             sample = torch.unsqueeze(sample, dim=0).to(device)

#             # Forward pass (model outputs raw logits)
#             pred_logit = model(sample)

#             # Get prediction probability (logit -> prediction probability)
#             pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

#             # Get pred_prob off the GPU for further calculations
#             pred_probs.append(pred_prob.cpu())

#     # Stack the pred_probs to turn list into a tensor
#     return torch.stack(pred_probs)


# import random
# random.seed(42)
# test_samples = []
# test_labels = []
# for sample, label in random.sample(list(test_data), k=9):
#     test_samples.append(sample)
#     test_labels.append(label)

#     # View the first sample shape
#     shape_test_samples = test_samples[0].shape
#     # plt.imshow(test_samples[0].squeeze(), cmap="gray")
#     # plt.title(class_names[test_labels[0]])
#     # plt.show()


# # Make predictions 
# pred_probs = make_predictions(model=model_2_results,
#                               data=test_samples)

# # View the first two prediction probabilities
# check_first_10_pred_probs = pred_probs[:2]






debug=1