# From MLP to DDPM

Deep learning from scratch, March 2026: from a hand-written NumPy MLP on March 10 to my own DDPM on March 20. Each file is dated by the day I wrote it.

| written | file | model | what it adds |
|---|---|---|---|
| Mar 10 | `myMLP/MLP_numpy3.10.py` | MLP in pure NumPy | forward pass, backpropagation and gradient descent by hand (MNIST) |
| Mar 11 | `myMLP/MLP_torch3.11.py` | MLP in PyTorch | the same network with autograd |
| Mar 12 | `myCNN/CNN_torch3.12.py` | Convolutional network | weight sharing and local receptive fields (MNIST) |
| Mar 14 | `myResNet/ResNet_3.14.py` | Residual network | skip connections, so deeper networks still train |
| Mar 15 | `myUNet/UnetStructure_3.15.py` | U-Net | encoder–decoder for pixel-level segmentation (Oxford-IIIT Pet) |
| Mar 18 | `myUNet/betterUnet3.18.py` | U-Net, revised | a cleaner second version |
| Mar 20 | `DDPM/myDDPM3.20.py` | DDPM | generating images by learning to reverse a noising process (MNIST) |

## Run

```bash
pip install torch torchvision numpy matplotlib pillow
python myCNN/CNN_torch3.12.py
```

Datasets download on first run.

---

Zhuowei Li · later work: [realgain](https://github.com/imlizhuowei/realgain), checking whether self-improving agents really improve
