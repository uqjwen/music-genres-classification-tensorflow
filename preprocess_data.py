from subprocess import Popen, PIPE, STDOUT
import os
from PIL import Image


from config import audiopath
from config import pixelPerSecond
from config import specpath
from config import slicepath
from config import sliceSize

currentPath = os.path.dirname(os.path.realpath(__file__)) 

def createSpectrogramFromAudio():
	genres = os.listdir(audiopath)
	for genre in genres:
		# print os.listdir(audiopath+genre)
		files = os.listdir(audiopath+genre)
		for filename in files:
			if filename.endswith('.au'):
				newFilename = '.'.join(filename.split('.')[:-1])
				command = "sox '{}{}/{}' -n spectrogram -Y 200 -X {} -m -r -o '{}.png'".\
				format(audiopath,genre,filename,pixelPerSecond, newFilename)
				# print command
				targetPath = "{}{}/".format(specpath, genre)
				# print targetPath
				if not os.path.exists(os.path.dirname(targetPath)):
					try:
						os.makedirs(os.path.dirname(targetPath))
					except OSError as exc: # Guard against race condition
						if exc.errno != errno.EEXIST:
							raise


				p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
				output, errors = p.communicate()
				if errors:
					print errors				

				command = "mv {}.png {}.png".format(newFilename, specpath+genre+'/'+newFilename)
				p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True, cwd=currentPath)
				output, errors = p.communicate()
				if errors:
					print errors


	# print genres
def sliceSpectrogram():
	genres = os.listdir(specpath)
	for genre in genres:
		files = os.listdir(specpath+genre)
		for filename in files:
			img = Image.open(specpath+genre+'/'+filename)
			width, height = img.size
			# print img.size
			samples = int(width/sliceSize)

			for i in range(samples):
				startPixel = i*sliceSize

				imgTmp = img.crop((startPixel, 1, (i+1)*sliceSize, sliceSize+1))

				newFilename = '.'.join(filename.split('.')[:-1])
				newFilename += '.'+str(i)+'.png'


				if not os.path.exists(os.path.dirname(slicepath+genre+'/')):
					try:
						os.makedirs(os.path.dirname(slicepath+genre+'/'))
					except OSError as exc:
						if exc.errno != errno.EEXIST:
							raise

				imgTmp.save(slicepath+genre+'/'+newFilename)



if __name__ == '__main__':
	# createSpectrogramFromAudio()
	# sliceSpectrogram()