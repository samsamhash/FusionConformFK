# FusionConformFK
Fusion of Conform and Feynman-Kac (FK) steering and manipulation of latents. Base code is from https://github.com/zacharyhorvitz/Fk-Diffusion-Steering and https://github.com/gemlab-vt/CONFORM
<br/>
Repository with code of the three different cases to be tested in a free account in google colab, just run all the instalations cells first and then you can continue with the tests:
- Fusion_Standar / Its the playground set to test the generation using Conform and Feynman-Kac steering.
- Fusion_LCM / Its the version to test using Latent Consistency Models, dreamshaper-7 for faster generation.
- Fusion_Latent / Its the fusion also using the initial latents from an image to modificate and improve it.
- FusionFK-Conform / PDF with a brief explanation.
  
If you get the following error:
<img width="975" height="301" alt="image" src="https://github.com/user-attachments/assets/cf319ce2-7e76-495d-bc9e-97c8a2ca1ac1" />

Please go to the following image and replace the line 31 with this: 
"
<br/>
from transformers.modeling_utils import PreTrainedModel
<br/>
from transformers.pytorch_utils import apply_chunking_to_forward
<br/>
"
<img width="975" height="162" alt="image" src="https://github.com/user-attachments/assets/4809c782-ca67-4bc3-baec-f777bd86b535" />
<img width="975" height="202" alt="image" src="https://github.com/user-attachments/assets/063ecf6b-7a1c-457c-8c93-84b6269a9ec9" />


