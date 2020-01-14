@echo off
if not exist datasets mkdir datasets
cd datasets
set FILE=%1

set URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/%FILE%.zip


for %%x in ( ae_photos
			apple2orange
			summer2winter_yosemite
			horse2zebra
			monet2photo
			cezanne2photo
			ukiyoe2photo
			vangogh2photo
			maps
			cityscapes
			cityscapes
			facades
			iphone2dslr_flower
			ae_photos
			) do (
			echo %%x
			if %%x == %FILE% (goto download) else (
				echo Dataset Not Found
				echo Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos
				)
				) 

goto exit

:download
echo Downloading %FILE%
wget %URL%
tar -x -f %FILE%.zip
del %FILE%.zip

:exit
cd..
