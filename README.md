# AEPredictor
Autoencoder-based Predictors
***
## VAE-based Predictor
* **neu_VED.py**
  * based on/imports a β-VAE implementation:  https://github.com/arpastrana/neu_vae/
  * ref. [Explanation by Rafael Pastrana](https://wandb.ai/arpastrana/beta_vae/reports/Disentangling-Variational-Autoencoders--VmlldzozNDQ3MDk)
  * While (V)AE predicts the encoder input, this predictor predicts other input based on the encoder input.
* **train_VAE.py**
  * a sample use case training program
  * uses dataset: **RotateMNIST.py**, which returns rotated MNIST images with rotation specifications (0, 90, 180, 270, 360 degrees). 
  * The predictor is supposed to predict rotated images from the original images and rotation specifications.
  * sample setting file: **neu_VED_beta.json**
  * sample run with Epoch = 100 (tensorboard images):
 ![TensorBoard_2023-10-02_17 24 30](https://github.com/rondelion/AEDPredictor/assets/11871187/4e366e0d-42d2-4f1e-8656-ff6024599005)
***
## Perceptron-based Predictor
* **SimplePredictor.py**
* **train_SimplePredictor.py**
  * uses dataset: **RotateMNIST.py**: see above.
  * sample setting file: **train_SimplePredictor.json**

## Cerenaut's Simple AE-based Predictor
Uses [simple_autoencoder.py](https://github.com/Cerenaut/cerenaut-pt-core/blob/master/cerenaut_pt_core/components/simple_autoencoder.py) from [cerenaut-pt-core](https://github.com/Cerenaut/cerenaut-pt-core).

* **train_SimpleAEPredictor.py**  
You can use **L1** regularization for **sparseness**.
  * uses dataset: **RotateMNIST.py**: see above.
  * sample setting file: **AEPredictor_simple.json**

