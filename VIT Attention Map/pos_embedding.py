import torch
import timm
from einops import rearrange
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from src.main.models.VitAttentionScore import VitAttentionScore
from src.main.utils.model_utils import *
from src.main.utils.utils import bind
from src.main.config.config_factory import ConfigFactory


mc, tc = ConfigFactory(model_name="VitAttentionScore")

model_name = "vit_base_patch16_224"
model_official = timm.create_model(model_name, pretrained=True)
model_official.eval()
save_model(model_official)

model_custom = VitAttentionScore(**bind(VitAttentionScore, mc.to_dict()))
model_custom.eval()

for (n_o, p_o), (n_c, p_c) in zip(model_official.named_parameters(), model_custom.named_parameters()):
    if p_o.numel() != p_c.numel():
        break

    p_c.data[:] = p_o.data

with torch.no_grad():

    # reshape the image and change it to float tensor
    img_width, img_height = mc.image_size[1], mc.image_size[2]
    patch_size = mc.patch_size[0]
    patch_num_width = img_width // patch_size
    patch_num_height = img_height // patch_size
    head_num = mc.num_heads

    inp = torch.from_numpy(np.array(Image.open("img1.jpg").resize([img_width, img_height]))) * 1.
    # reshape image to B C H W
    # h: image width = 384, w: image height = 384, c: image channel = 3
    inp = rearrange(inp, "h w c -> 1 c h w")
    # b: batch size = 1, c: channel = 3, p1, p2: patch size = 16, n1, n2: num patch per direction = 24
    
    # get the attention weight
    # shape of pos_embedding: [batch size = 1, cls token + num patch = 1 + 24*24, patch dim = 768]
    _, pos_embedding = model_custom(inp)
    p = pos_embedding[0, 0, :]
    pos_embedding = pos_embedding[0, 1:, :] # 576, 768
    print(pos_embedding.shape)
            
    norm = torch.norm(pos_embedding, 2, dim=1, keepdim=True)
    cosine_similarity = pos_embedding @ pos_embedding.T / (norm @ norm.T)
    cosine_similarity = rearrange(cosine_similarity, "(p1 p2) (h w) -> p1 p2 h w", p2=patch_num_width, w=patch_num_width)
    print(cosine_similarity.shape)
    
    fig, ax = plt.subplots(cosine_similarity.shape[0], cosine_similarity.shape[1])
    images = []
    for i in range(cosine_similarity.shape[0]):
        for j in range(cosine_similarity.shape[1]):
            images.append(ax[i, j].imshow(cosine_similarity[i, j]))
            ax[i, j].label_outer()
            ax[i, j].axis("off")

    # Find the min and max of all colors for use in setting the color scale.
    vmin = min(image.get_array().min() for image in images)
    vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for im in images:
        im.set_norm(norm)

    fig.colorbar(mappable=images[0], ax=ax)
    fig.suptitle("Positional Encoding")
    fig.savefig("src/main/resources/images/PosEmbedding.jpg", dpi=300)
    plt.show()
