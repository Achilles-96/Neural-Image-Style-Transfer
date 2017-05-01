# Neural Image Style Transfer

## Vanilla Style Transfer

<img src="StyleTransfer/InputContentImages/brad_pitt.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/InputStyleImages/blackwhite.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/TestOutputs/FinalOutputs/Test8/output-601.png" width=30% align="middle" style="margin:20;">

Artistic Neural Style transfer tries to transfer the stylistic details from one image onto the content of another image. This has a wide variety of applications in the domain of computer vision.

### Results

<img src="StyleTransfer/InputContentImages/tubingen.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/InputStyleImages/seated-nude.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/TestOutputs/FinalOutputs/Test11/output-1001.png" width=30% align="middle" style="margin:20;">

<img src="StyleTransfer/InputContentImages/vishal.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/InputStyleImages/scenery4.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/TestOutputs/FinalOutputs/Test16/output-1001.png" width=30% align="middle" style="margin:20;">

<img src="StyleTransfer/InputContentImages/bear.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/InputStyleImages/scenery2.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/TestOutputs/FinalOutputs/Test20/output-1000.png" width=30% align="middle" style="margin:20;">


## Localized Style Transfer

<img src="StyleTransfer/InputContentImages/Gogh.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/InputStyleImages/Seth.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/ExtensionOutputs/Gogh_Seth.jpg" width=30% align="middle" style="margin:20;">

A lot of times, we are required to transfer the style of an image to the content which contains multiple objects placed at different locations, so there is no spatial alignment between the content and style images. To overcome this issue where vanilla style transfer does terrible, we introduce localized style transfer where image masks for the style and content are passed along as inputs as well. This supervision allows us to perform style transfer which gives us a much better quality of transfer.

### Results

<img src="StyleTransfer/InputContentImages/Mia.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/InputStyleImages/Freddie.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/ExtensionOutputs/Mia_Freddie.jpg" width=30% align="middle" style="margin:20;">

<img src="StyleTransfer/InputContentImages/Seth.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/InputStyleImages/Gogh.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/ExtensionOutputs/Seth_Gogh.jpg" width=30% align="middle" style="margin:20;">


## Pyramid based Style Transfer

<img src="StyleTransfer/InputContentImages/brad_pitt.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/InputStyleImages/walk.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/ExtensionOutputs/pyramid_brad_pitt_walk_final.jpg" width=30% align="middle" style="margin:20;">

We have noticed that the content of an image is mainly depicted by the edges and to get a true 'style' only transfer, we would be required to get rid of the content color information completely. A few approaches towards this could be to us grayscale content images and to use a laplacian pyramid of content image at various points for the content loss modules. This method does not produce trememdously different results but gives us a lot more flexibility on the kinds of outputs we get. 

### Results

<img src="StyleTransfer/InputContentImages/brad_pitt.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/InputStyleImages/walk2.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/ExtensionOutputs/pyramid_brad_pitt_walk2_final.jpg" width=30% align="middle" style="margin:20;">

<img src="StyleTransfer/InputContentImages/brad_pitt.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/InputStyleImages/seated-nude.jpg" width=30% align="middle" style="margin:20;"><img src="StyleTransfer/ExtensionOutputs/pyramid_brad_pitt_seated-nude_final.jpg" width=30% align="middle" style="margin:20;">


#### Note

A small article on the project and an extension to sketches: https://erilyth.wordpress.com/2016/11/05/neural-style-transfer-sketches/

Credits: jcjohnson's [Neural Style](https://github.com/jcjohnson/neural-style) and alexjc's [Neural Doodle](https://github.com/alexjc/neural-doodle) for sample image masks.
