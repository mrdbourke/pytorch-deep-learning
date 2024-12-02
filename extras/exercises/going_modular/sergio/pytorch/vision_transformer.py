import torch
import torchvision
from torch import nn

# Implementation of a vision transformer following the paper "AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE"

class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
    # 1. Initialize the class with appropriate variables
    def __init__(self,
                 img_size:int=224,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()

        # 2. Make the image size is divisble by the patch size
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        # 3. Create a layer to turn an image into patches
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)

        # 4. Create a layer to flatten the patch feature maps into a single dimension
        self.flatten = nn.Flatten(start_dim=2, # only flatten the feature map dimensions into a single vector. dim=0 is the batch size, dim=1 is the embedding dimension.
                                  end_dim=3)
        
        # 5. Token embedding class
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_dim),
                                        requires_grad=True)
               
        # 6. Position embedding class
        num_patches = (img_size * img_size) // patch_size**2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, embedding_dim),
                                          requires_grad=True)

    # 7. Define the forward method
    def forward(self, x):
        
        # 8. Linear projection of patches 
        x_patched = self.patcher(x)

        # 9. Flatten the linear transformed patches
        x_flattened = self.flatten(x_patched).permute(0, 2, 1) # adjust so the embedding is on the final dimension [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]

        # 10. Create class token and prepend
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        x_flattened_token = torch.cat((class_token, x_flattened), dim=1)

        # 11. Create position embedding
        pos_embedding = self.pos_embedding.expand(batch_size, -1, -1)
        x_flattened_token_pos = x_flattened_token + pos_embedding

        return x_flattened_token_pos



class MultiheadSelfAttentionBlock(nn.Module):
    """Creates a multi-head self-attention block ("MSA block" for short).
    """
    # 1. Initialize the class with hyperparameters from Table 1
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0): # doesn't look like the paper uses any dropout in MSABlocks
        super().__init__()

        # 2. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 3. Create the Multi-Head Attention (MSA) layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True) # does our batch dimension come first?

    # 4. Create a forward() method to pass the data throguh the layers
    def forward(self, x):

        # 5. Normalization layer
        x = self.layer_norm(x)

        # 6. Multihead attention layer
        attn_output, _ = self.multihead_attn(query=x, # query embeddings
                                             key=x, # key embeddings
                                             value=x, # value embeddings
                                             need_weights=False) # do we need the weights or just the layer outputs?
        return attn_output
    

class MLPBlock(nn.Module):
    """Creates a layer normalized multilayer perceptron block ("MLP block" for short)."""
    # 1. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim:int=768, # Hidden Size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base (X4)
                 dropout:float=0.1): # Dropout from Table 3 for ViT-Base
        super().__init__()

        # 2. Create the Norm layer (LN)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # 3. Create the Multilayer perceptron (MLP) layer(s)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(), # "The MLP contains two layers with a GELU non-linearity"
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, # needs to take same in_features as out_features of layer above
                      out_features=embedding_dim), # take back to embedding_dim
            nn.Dropout(p=dropout) # "Dropout, when used, is applied after every dense layer"
        )

    # 4. Create a forward() method to pass the data throguh the layers
    def forward(self, x):
        return self.mlp(self.layer_norm(x))
    

class TransformerEncoderBlock(nn.Module):
    """Creates a Transformer Encoder block."""
    # 1. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 mlp_dropout:float=0.1, # Amount of dropout for dense layers from Table 3 for ViT-Base
                 attn_dropout:float=0): # Amount of dropout for attention layers
        super().__init__()

        # 2. Create MSA block (equation 2)
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)

        # 3. Create MLP block (equation 3)
        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim,
                                   mlp_size=mlp_size,
                                   dropout=mlp_dropout)

    # 4. Create a forward() method
    def forward(self, x):

        # 5. Create residual connection for MSA block (add the input to the output)
        x =  self.msa_block(x) + x

        # 6. Create residual connection for MLP block (add the input to the output)
        x = self.mlp_block(x) + x

        return x
    

# Create a ViT class that inherits from nn.Module
class ViT(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    # 1. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000): # Default for ImageNet but can customize this
        super().__init__() # don't forget the super().__init__()!

        # 2. Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # 3. Create patch embedding layer
        self.patch_embedding = PatchEmbedding(img_size=img_size,
                                              in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)

        # 4. Create Transformer Encoder blocks (we can stack Transformer Encoder blocks using nn.Sequential())
        # Note: The "*" means "all"
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                           num_heads=num_heads,
                                                                           mlp_size=mlp_size,
                                                                           mlp_dropout=mlp_dropout,
                                                                           attn_dropout=attn_dropout) for _ in range(num_transformer_layers)])

        # 5. Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    # 6. Create a forward() method
    def forward(self, x):

        # 7. Create patch embedding (equation 1)
        x = self.patch_embedding(x)

        # 8. Run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # 9. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # 10. Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x
    

# Create a ViT class that inherits from nn.Module using already implemented transformer classes from torchvision
class ViTv2(nn.Module):
    """Creates a Vision Transformer architecture with ViT-Base hyperparameters by default."""
    # 1. Initialize the class with hyperparameters from Table 1 and Table 3
    def __init__(self,
                 img_size:int=224, # Training resolution from Table 3 in ViT paper
                 in_channels:int=3, # Number of channels in input image
                 patch_size:int=16, # Patch size
                 num_transformer_layers:int=12, # Layers from Table 1 for ViT-Base
                 embedding_dim:int=768, # Hidden size D from Table 1 for ViT-Base
                 mlp_size:int=3072, # MLP size from Table 1 for ViT-Base
                 num_heads:int=12, # Heads from Table 1 for ViT-Base
                 attn_dropout:float=0, # Dropout for attention projection
                 mlp_dropout:float=0.1, # Dropout for dense/MLP layers
                 embedding_dropout:float=0.1, # Dropout for patch and position embeddings
                 num_classes:int=1000): # Default for ImageNet but can customize this
        super().__init__() # don't forget the super().__init__()!

        # 2. Create embedding dropout value
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # 3. Create patch embedding layer
        self.patch_embedding = PatchEmbedding(img_size=img_size,
                                              in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        
        # 4. Create a single transformer encoder layer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                    nhead=num_heads,
                                                                    dim_feedforward=mlp_size,
                                                                    dropout=mlp_dropout,
                                                                    activation="gelu",
                                                                    batch_first=True,
                                                                    norm_first=True)
        
        # 5. Crete the stacked transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.transformer_encoder_layer,
                                                         num_layers=num_transformer_layers)
        

        # 6. Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )

    # 7. Create a forward() method
    def forward(self, x):

        # 8. Create patch embedding (equation 1)
        #x = x + self.position_embedding[:, :x.size(1), :]
        x = self.patch_embedding(x)

        # 9. Run embedding dropout (Appendix B.1)
        x = self.embedding_dropout(x)

        # 10. Pass patch, position and class embedding through transformer encoder layers (equations 2 & 3)
        x = self.transformer_encoder(x)

        # 11. Put 0 index logit through classifier (equation 4)
        x = self.classifier(x[:, 0]) # run on each sample in a batch at 0 index

        return x