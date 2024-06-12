import torch
import torch.nn.functional as F

class OrthogonalityLoss:
    def __init__(self, batch_size, num_feature_maps, feature_shape, device, split=None):

        self.batch_size = batch_size
        self.num_feature_maps = num_feature_maps
        self.feature_shape = feature_shape
        self.device = device
        self.split = split

        self.create_identity_and_mask()

    def create_identity_and_mask(self):
        # Create the identity matrix
        identity_matrix = torch.eye(self.num_feature_maps, device=self.device)  # [num_feature_maps x num_feature_maps]
        identity_matrix = identity_matrix.unsqueeze(0)  # [1 x num_feature_maps x num_feature_maps]
        identity_matrix = identity_matrix.expand(self.batch_size, -1,  -1)  # [batch_size x num_feature_maps x num_feature_maps]

        # Create a mask to zero out diagonal elements and elements not involved in the calculation
        mask = torch.ones(self.num_feature_maps, self.num_feature_maps, device=self.device)
        if self.split is not None:
            mask[:, :self.split] = 0
            mask[self.split:, :] = 0
            mask[:self.split, self.split:] = 1
            mask[self.split:, :self.split] = 1
        mask[range(self.num_feature_maps), range(self.num_feature_maps)] = 0
        mask = mask.unsqueeze(0).expand(self.batch_size, -1, -1)  # [batch_size x num_feature_maps x num_feature_maps]

        self.identity_matrix = identity_matrix
        self.mask = mask

    def __call__(self, H):

        # Reshape H to a 2D tensor for matrix multiplication
        H_reshaped = H.view(self.batch_size,  self.num_feature_maps, -1)  # [batch_size x num_feature_maps x product of feature_shape]

        # Compute H H^T
        H_Ht = torch.matmul(H_reshaped, H_reshaped.transpose(-1, -2))  # [batch_size x num_feature_maps x num_feature_maps]

        # Apply the mask to the difference (H H^T - I)
        masked_diff = (H_Ht - self.identity_matrix) * self.mask

        # Compute the orthogonality loss: || masked_diff ||
        L = torch.norm(masked_diff)

        return L
def orthogonality_loss(H):

    # Get the shape of H
    batch_size, num_feature_maps, *feature_shape = H.shape

    # Reshape H to a 2D tensor for matrix multiplication
    H_reshaped = H.view(batch_size, num_feature_maps, -1)  # [batch_size x num_feature_maps x product of feature_shape]

    # Compute H H^T
    H_Ht = torch.matmul(H_reshaped, H_reshaped.transpose(-1, -2))  # [batch_size x num_feature_maps x num_feature_maps]

    # Create the identity matrix
    identity_matrix = torch.eye(num_feature_maps, device=H.device)  # [num_feature_maps x num_feature_maps]
    identity_matrix = identity_matrix.unsqueeze(0)  # [1 x num_feature_maps x num_feature_maps]
    identity_matrix = identity_matrix.expand(batch_size, -1, -1)  # [batch_size x num_feature_maps x num_feature_maps]

    # Create a mask to zero out diagonal elements
    mask = torch.ones_like(H_Ht)
    mask[..., range(num_feature_maps), range(num_feature_maps)] = 0

    # Apply the mask to the difference (H H^T - I)
    masked_diff = (H_Ht - identity_matrix) * mask

    # Compute the orthogonality loss: || masked_diff ||
    L = torch.norm(masked_diff)

    return L

# Define Main
if __name__ == "__main__":

    orth_loss = OrthogonalityLoss(batch_size=1, num_feature_maps=4, feature_shape=(2, 2, 1), device='cpu', split=2)

    # Manually defining the tensor to have easy-to-check orthogonal feature maps
    H1 = torch.tensor([[
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]]
    ]], dtype=torch.float32).unsqueeze(-1)  # [1 x 4 x 2 x 2 x 1]

    # Calculate the orthogonality loss
    loss = orth_loss(H1) #loss = orthogonality_loss(H1)
    print("Orthogonality Loss:", loss.item())

    H2 = torch.tensor([[
        [[1, 0], [0, 0]],
        [[1, 1], [0, 0]],  # Not orthogonal to the first
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]]
    ]], dtype=torch.float32).unsqueeze(-1)

    # Calculate the orthogonality loss
    loss = orth_loss(H2)   #loss = orthogonality_loss(H2)
    print("Orthogonality Loss:", loss.item())

    H3 = torch.tensor([[
        [[1, 0], [0, 0]],
        [[0, 0], [1, 0]],
        [[1, 1], [0, 0]],
        [[0, 0], [0, 1]]
    ]], dtype=torch.float32).unsqueeze(-1)


    # Calculate the orthogonality loss
    loss = orth_loss(H3) # loss = orthogonality_loss(H1_H2)
    print("Orthogonality Loss:", loss.item())

    print("May be the force with you!")