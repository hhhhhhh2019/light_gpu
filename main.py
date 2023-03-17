import taichi as ti
import numpy as np
from math import sin, tau, pi

ti.init(arch=ti.gpu)

sw = 1024
sh = 512
pixels = ti.Vector.field(n=3, dtype=float, shape=(sw, sh))


ceils = ti.Vector.field(n=3, dtype=float, shape=(sw, sh))
ceils.fill(0)


for y in range(256):
	for x in range(y):
		ceils[x-y//2 + 400,(256 - y) + 256 - 128].z = 1

#for x in range(128):
#	for y in range(128):
#		if ((x/128*2-1)**2 + (y/128*2-1)**2) ** 0.5 < 1:
#			ceils[x+400, y + 256 - 64].z = -0.5


#for x in range(128):
#	h = sin(pi * x / 128 * 17 * 5)
#	for y in range(128):
#		if ((x/128*2-1)**2 + (y/128*2-1)**2) ** 0.5 < 0.5:
#			ceils[x + 100,y+256-64].x = (h - 0.5 + sin(y/128*pi) ** 2) * sin(y/128*pi)


@ti.kernel
def update():
	for x in range(sw):
		for y in range(sh):
			force = 0.0

			if x == 0:
				force += ceils[x+1,y].x
			elif x == sw - 1:
				force += ceils[x-1,y].x
			else:
				force += ceils[x+1,y].x
				force += ceils[x-1,y].x

			if y == 0:
				force += ceils[x,y+1].x
			elif y == sh - 1:
				force += ceils[x,y-1].x
			else:
				force += ceils[x,y+1].x
				force += ceils[x,y-1].x

			ceils[x,y].y += (force * .25 - ceils[x,y].x) / (ceils[x,y].z + 1.)

	for x in range(sw):
		for y in range(sh):
			h = ceils[x,y].x
			ceils[x,y].x += ceils[x,y].y

			pixels[x,y] = [h,ceils[x,y].z * .5,-h]
			#pixels[x,y] = [h,h,h]

			if x < 30 or y < 30 or x > sw - 30 or y > sh - 30:
				ceils[x,y].y *= 0.95


gui = ti.GUI("light", res=(sw, sh), fast_gui=True)
gui.fps_limit = 10000

while gui.running:
	if gui.frame % 2000 == 0:
		for x in range(128):
			h = sin(pi * x / 128 * 17 * (gui.frame // 2000))
			for y in range(128):
				if ((x/128*2-1)**2 + (y/128*2-1)**2) ** 0.5 < 0.5:
					ceils[x + 100,y+256-64].x = (h - 0.5 + sin(y/128*pi) ** 2) * sin(y/128*pi)

	update()
	gui.set_image(pixels)
	gui.show()
