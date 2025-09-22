# FusionConformFK
Fusion of Conform and Feynman-Kac (FK) steering and manipulation of latents.

If you get the following error:
<img width="975" height="301" alt="image" src="https://github.com/user-attachments/assets/cf319ce2-7e76-495d-bc9e-97c8a2ca1ac1" />

Please go to the following image and replace the line 31 with this: 
"
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward
"
<img width="975" height="162" alt="image" src="https://github.com/user-attachments/assets/4809c782-ca67-4bc3-baec-f777bd86b535" />
<img width="975" height="202" alt="image" src="https://github.com/user-attachments/assets/063ecf6b-7a1c-457c-8c93-84b6269a9ec9" />


