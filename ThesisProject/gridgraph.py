import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid


# a = torch.Tensor()
# a = read_image('/home/ge75tis/Desktop//t2mt.png')
labels = ['t2m', 'u10', 'v10', 'z', 't', 'tcc', 'tp']

a0 = read_image('/home/ge75tis/Desktop/blank.png')
a1 = read_image('/home/ge75tis/Desktop/LRP180Crop/t2mu10.png')
a2 = read_image('/home/ge75tis/Desktop/LRP180Crop/t2mv10.png')
a3 = read_image('/home/ge75tis/Desktop/LRP180Crop/t2mz.png')
a4 = read_image('/home/ge75tis/Desktop/LRP180Crop/t2mt.png')
a5 = read_image('/home/ge75tis/Desktop/LRP180Crop/t2mtcc.png')
a6 = read_image('/home/ge75tis/Desktop/LRP180Crop/t2mtp.png')


a11 = read_image('/home/ge75tis/Desktop/LRP180Crop/u10t2m.png')
a22 = read_image('/home/ge75tis/Desktop/LRP180Crop/u10v10.png')
a33 = read_image('/home/ge75tis/Desktop/LRP180Crop/u10z.png')
a44 = read_image('/home/ge75tis/Desktop/LRP180Crop/u10t.png')
a55 = read_image('/home/ge75tis/Desktop/LRP180Crop/u10tcc.png')
a66 = read_image('/home/ge75tis/Desktop/LRP180Crop/u10tp.png')

a111 = read_image('/home/ge75tis/Desktop/LRP180Crop/v10t2m.png')
a222 = read_image('/home/ge75tis/Desktop/LRP180Crop/v10u10.png')
a333 = read_image('/home/ge75tis/Desktop/LRP180Crop/v10z.png')
a444 = read_image('/home/ge75tis/Desktop/LRP180Crop/v10t.png')
a555 = read_image('/home/ge75tis/Desktop/LRP180Crop/v10tcc.png')
a666 = read_image('/home/ge75tis/Desktop/LRP180Crop/v10tp.png')

a1111 = read_image('/home/ge75tis/Desktop/LRP180Crop/zt2m.png')
a2222 = read_image('/home/ge75tis/Desktop/LRP180Crop/zu10.png')
a3333 = read_image('/home/ge75tis/Desktop/LRP180Crop/zv10.png')
a4444 = read_image('/home/ge75tis/Desktop/LRP180Crop/zt.png')
a5555 = read_image('/home/ge75tis/Desktop/LRP180Crop/ztcc.png')
a6666 = read_image('/home/ge75tis/Desktop/LRP180Crop/ztp.png')

a11111 = read_image('/home/ge75tis/Desktop/LRP180Crop/tt2m.png')
a22222 = read_image('/home/ge75tis/Desktop/LRP180Crop/tu10.png')
a33333 = read_image('/home/ge75tis/Desktop/LRP180Crop/tv10.png')
a44444 = read_image('/home/ge75tis/Desktop/LRP180Crop/tz.png')
a55555 = read_image('/home/ge75tis/Desktop/LRP180Crop/ttcc.png')
a66666 = read_image('/home/ge75tis/Desktop/LRP180Crop/ttp.png')

aq = read_image('/home/ge75tis/Desktop/LRP180Crop/tcct2m.png')
aw = read_image('/home/ge75tis/Desktop/LRP180Crop/tccu10.png')
ae = read_image('/home/ge75tis/Desktop/LRP180Crop/tccv10.png')
ar = read_image('/home/ge75tis/Desktop/LRP180Crop/tccz.png')
at = read_image('/home/ge75tis/Desktop/LRP180Crop/tcct.png')
az = read_image('/home/ge75tis/Desktop/LRP180Crop/tcctp.png')

ay = read_image('/home/ge75tis/Desktop/LRP180Crop/tpt2m.png')
ax = read_image('/home/ge75tis/Desktop/LRP180Crop/tpu10.png')
ac = read_image('/home/ge75tis/Desktop/LRP180Crop/tpv10.png')
av = read_image('/home/ge75tis/Desktop/LRP180Crop/tpz.png')
ab = read_image('/home/ge75tis/Desktop/LRP180Crop/tpt.png')
an = read_image('/home/ge75tis/Desktop/LRP180Crop/tptcc.png')


# for i in range(7):
#     for j in range(7):
#         if (i == j):
#             print("same")
#         else:





Grid = make_grid([a0, a1, a2, a3, a4, a5, a6, a11, a0, a22, a33, a44, a55, a66
                  , a111, a222, a0, a333, a444, a555, a666, a1111, a2222, a3333, a0, a4444, a5555, a6666,
                  a11111, a22222, a33333, a44444, a0, a55555, a66666,
                  aq, aw, ae, ar, at, a0, az,
                  ay, ax, ac, av, ab, an, a0], nrow=7, padding=25)
img = torchvision.transforms.ToPILImage()(Grid)
print(type(img))
img.show()