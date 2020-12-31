import torch

class SSO():
    def __init__(self, iteration_num=20, gaussian_kernel_k=0.2, k_nearest_neighbors=10):
        self.T = iteration_num
        self.gaussian_kernel_k = gaussian_kernel_k
        self.k = k_nearest_neighbors

    def _get_similarity_matrix(self, feature_matrix):
        distance_matrix = torch.cdist(feature_matrix, feature_matrix, p=2)
        mean_distance_feature = (torch.sort(distance_matrix[:, :], 1, descending=False)[0])[:, :self.k].mean(dim=1, keepdim=True)
        similarity_matrix = torch.exp(-(distance_matrix**2) * (self.gaussian_kernel_k * (mean_distance_feature**2)).reciprocal())

        return similarity_matrix

    def _apply_sso(self, similarity_matrix):
        diagonal_matrix_inverse = torch.diag(similarity_matrix.sum(dim=1).reciprocal())
        smoothing_kernel = torch.matmul(diagonal_matrix_inverse, similarity_matrix)

        smoothed_similarity_matrix = similarity_matrix
        for _ in range(self.T):
            smoothed_similarity_matrix = smoothed_similarity_matrix.matmul(torch.transpose(smoothing_kernel, 1, 0))

        self_normalization_diagonal_matrix_inverse = torch.diag(torch.diagonal(smoothed_similarity_matrix).reciprocal())
        normalized_smoothed_similarity_matrix = torch.matmul(self_normalization_diagonal_matrix_inverse, smoothed_similarity_matrix)

        return normalized_smoothed_similarity_matrix

    def __call__(self, feature_matrix=None, similarity_matrix=None):
        if feature_matrix is None and similarity_matrix is None:
            raise ValueError("feature_matrix and similarity_matrix shouldn't be None at the same time")
        elif feature_matrix is not None and similarity_matrix is not None:
            raise ValueError("only one of feature_matrix and similarity_matrix to be not None")
        elif feature_matrix:
            with torch.no_grad():
                similarity_matrix = self._get_similarity_matrix(feature_matrix)

        with torch.no_grad():
            batch_smoothed_similarity_matrix = self._apply_sso(similarity_matrix)

        return batch_smoothed_similarity_matrix