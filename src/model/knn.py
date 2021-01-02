import torch
from .self_smoothing_operator import SSO

class WeightedKNNPredictor():
    def __init__(self, k=10, gaussian_kernel_k=1.0, ssot=10, class_num=2):
        self.gaussian_kernel_k = gaussian_kernel_k
        self.k = k
        self.sso = SSO(k_nearest_neighbors=k, gaussian_kernel_k=gaussian_kernel_k, iteration_num=ssot)
        self.class_num = class_num

    def __call__(self, batch_feature, data_bank):
        """
        :param batch_feature: tensor in shape (n, feature_dim)
        :param data_bank: tensor in shape (data_size, feature_dim + 1), the last dimension of each data is the label id
        :return: probability tensor in shape (n, class_num)
        """
        neighbors = data_bank[torch.randperm(data_bank.shape[0])[2000]]
        feature_matrix = torch.cat([batch_feature, neighbors[:, :-1]], dim=0)
        # batch_smoothed_similarity_matrix = feature_matrix
        batch_smoothed_similarity_matrix = self.sso(feature_matrix=feature_matrix)
        batch_sorted_similarity, batch_indices = torch.sort(
            batch_smoothed_similarity_matrix[:batch_feature.shape[0], batch_feature.shape[0]:],
            dim=1,
            descending=True
        )
        batch_k_nearest_neighbors_weights = torch.softmax(batch_sorted_similarity[:, :self.k], dim=1)
        batch_k_nearest_neighbors_label = neighbors[:, -1][batch_indices[:, :self.k]]

        batch_prob_list = []
        for class_id in range(self.class_num):
            class_wise_prob = (batch_k_nearest_neighbors_label.eq(class_id) * batch_k_nearest_neighbors_weights).sum(1)
            batch_prob_list.append(class_wise_prob)
        batch_prob = torch.stack(batch_prob_list, dim=1)

        return batch_prob