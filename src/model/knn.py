import torch
from .self_smoothing_operator import SSO

class WeightedKNNPredictor():
    def __init__(self, k=10, gaussian_kernel_k=1.0, ssot=10):
        self.gaussian_kernel_k = gaussian_kernel_k
        self.k = k
        self.sso = SSO(k_nearest_neighbors=k, gaussian_kernel_k=gaussian_kernel_k, iteration_num=ssot)

    def __call__(self, batch_feature, data_bank):
        list_predictive_delivery_qty = []
        feature_matrix = torch.cat([batch_feature, data_bank[:, :-1]], dim=0)
        batch_smoothed_similarity_matrix = self.sso(feature_matrix=feature_matrix)
        for iter in range(batch_feature.shape[0]):
            neighbors = data_bank
            sorted_similarity, indices = torch.sort(batch_smoothed_similarity_matrix[iter, batch_feature.shape[0]:], 0, descending=True)
            k_nearest_neighbors_weights = torch.softmax(sorted_similarity[:self.k], dim=0)
            k_nearest_neighbors_label = neighbors[:, -1][indices[:self.k]]
            
            label = (neighbors[:, -1][indices[:self.k]] * torch.softmax(sorted_similarity[:self.k], dim=0)).sum()

        return torch.stack(list_predictive_delivery_qty, dim=0)

if __name__ == '__main__':
    pass